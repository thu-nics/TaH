"""CPU-only crash/recovery fixtures exercising the real protocol run and merge."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_protocol


def _row(pid, idx, correct=True):
    return dict(problem_id=pid, sample_idx=idx, correct_answer="a",
                predicted_answer="a" if correct else "b", has_answer="True",
                is_correct=str(correct), input_tokens=2, output_tokens=3,
                processing_time=1, mean_passes_per_token=1,
                fraction_tokens_at_least_2_passes=0, pass_stat_tokens=3)


def _sample(pid, idx, text, correct=True, pending=False):
    sample = dict(problem_id=pid, id=pid, sample=idx, output=text)
    if pending:
        sample["grading_status"] = "pending"
    else:
        sample["is_correct"] = correct
    return sample


def _leaf(attempt, rows, samples, status, *, worker=False):
    leaf = attempt / "eval_output/math500_tah/20261014_120000"
    csv_path = leaf / ("job_0/detailed_results.csv" if worker else "detailed_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    run_protocol._write_csv(csv_path, run_protocol.CSV_FIELDS, rows)
    run_protocol.write_json(leaf / "run_status.json", {"status": status})
    if worker:
        for sample in samples:
            run_protocol.write_json(
                leaf / "job_0/details" / sample["problem_id"] / f"sample_{sample['sample']}.json",
                sample)
    else:
        (leaf / "samples.jsonl").write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    return leaf


def _args(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("""name: recovery
eval_config: eval.yaml
groups:
  - name: math
    datasets: [math500]
    repeat_size: 2
""", encoding="utf-8")
    (tmp_path / "eval.yaml").write_text("temperature: 0.6\n", encoding="utf-8")
    monkeypatch.setattr(run_protocol, "load_dataset_items", lambda _: {
        "math500": [{"id": f"math500_{i}"} for i in range(3)]})
    return argparse.Namespace(config=protocol, output_root=tmp_path / "output",
                              model_path="/synthetic-model", backend="tah", seed=42,
                              job_nums=1, tp_size_per_job=1, logger_level="WARNING",
                              plan_only=False)


@pytest.mark.parametrize("old_grade,new_grade", [(True, False), (False, True)])
def test_crash_resume_interrupt_resume_merge(tmp_path, monkeypatch, capsys, old_grade, new_grade):
    args = _args(tmp_path, monkeypatch)
    calls = []

    def fake_eval(argv, cwd, check):
        attempt = Path(argv[argv.index("--output_dir") + 1]).parent
        calls.append(argv)
        if len(calls) == 1:
            _leaf(attempt, [_row("math500_0", 0), _row("math500_1", 0)],
                  [_sample("math500_0", 0, "initial-0"),
                   _sample("math500_1", 0, "initial-1"),
                   _sample("math500_0", 1, "OOM-pending", pending=True)],
                  "running", worker=True)
            # Parent also dies, leaving command/state without exit_code.
            raise SystemExit("host OOM")
        if len(calls) == 2:
            assert argv[argv.index("--data_ids") + 1] == "math500_2"
            _leaf(attempt, [_row("math500_2", 0)],
                  [_sample("math500_2", 0, "resume-0")], "complete")
        elif len(calls) == 3:
            assert argv[argv.index("--data_ids") + 1] == "math500_0,math500_1,math500_2"
            _leaf(attempt, [_row("math500_1", 0)],
                  [_sample("math500_1", 0, "committed-in-interrupted-run"),
                   _sample("math500_0", 0, "Ctrl-C-pending", pending=True),
                   # Kill between the atomic graded JSON and CSV flush.
                   _sample("math500_2", 0, "graded-but-uncommitted", old_grade)],
                  "running", worker=True)
            raise KeyboardInterrupt
        else:
            assert len(calls) == 4
            assert argv[argv.index("--data_ids") + 1] == "math500_0,math500_2"
            _leaf(attempt, [_row("math500_0", 0, new_grade), _row("math500_2", 0, new_grade)],
                  [_sample("math500_0", 0, "final-0", new_grade),
                   _sample("math500_2", 0, "final-2", new_grade)], "complete")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(run_protocol.subprocess, "run", fake_eval)
    with pytest.raises(SystemExit, match="host OOM"):
        run_protocol.main_run(args)
    with pytest.raises(KeyboardInterrupt):
        run_protocol.main_run(args)
    assert run_protocol.main_run(args) == 0
    assert run_protocol.main_run(args) == 0
    assert len(calls) == 4
    state = json.loads((args.output_root / "protocol_state.json").read_text())
    rows, samples, provenance = run_protocol.load_group_records(args.output_root, state["groups"]["math"])
    assert len(rows) == len(samples) == 6
    assert provenance["dropped_count"] == provenance["dropped_duplicate_count"] == 3
    assert len({run_protocol.sample_key(sample) for sample in samples}) == 6
    texts = {run_protocol.sample_key(sample): sample["output"] for sample in samples}
    assert texts[("math500_1", 1)] == "committed-in-interrupted-run"
    assert texts[("math500_0", 1)] == "final-0"
    assert texts[("math500_2", 1)] == "final-2"
    before = {str(p): p.read_bytes() for p in (args.output_root / "groups").rglob("*") if p.is_file()}
    assert run_protocol.main_merge(args) == 0
    canonical = args.output_root / "canonical"
    audit = json.loads((canonical / "merge_provenance.json").read_text())
    assert audit["selected_count"] == 6
    assert audit["dropped_count"] == audit["dropped_duplicate_count"] == 3
    assert audit["groups"]["math"] == provenance
    for dropped in audit["groups"]["math"]["dropped_records"]:
        assert dropped["kept_source"]["attempt_dir"].endswith("0004-resume-sample-1")
        assert dropped["raw"]["file"] != dropped["kept_source"]["raw"]["file"]
    raw_path = next(canonical.rglob("samples.jsonl"))
    merged = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert {run_protocol.sample_key(s): s["output"] for s in merged} == texts
    stable = {str(p): p.read_bytes() for p in canonical.rglob("*")
              if p.name in {"samples.jsonl", "detailed_results.csv", "merge_provenance.json"}}
    assert run_protocol.main_merge(args) == 0
    assert stable == {str(p): p.read_bytes() for p in canonical.rglob("*") if str(p) in stable}
    assert before == {str(p): p.read_bytes() for p in (args.output_root / "groups").rglob("*") if p.is_file()}
    output = capsys.readouterr().out
    assert "3 dropped raw records (3 duplicates)" in output
    assert output.count("DROP math (math500_0, 1)") == 4  # two records, two merges
    assert output.count("DROP math (math500_2, 1)") == 2


def test_normal_run_keeps_every_original_record(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    rows = [_row(f"math500_{i}", idx, i != 1) for i in range(3) for idx in range(2)]
    samples = [_sample(row["problem_id"], row["sample_idx"], f"original-{i}", row["is_correct"] == "True")
               for i, row in enumerate(rows)]
    calls = []

    def fake_eval(argv, cwd, check):
        calls.append(argv)
        _leaf(Path(argv[argv.index("--output_dir") + 1]).parent,
              rows, list(reversed(samples)), "complete")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(run_protocol.subprocess, "run", fake_eval)
    assert run_protocol.main_run(args) == 0
    assert run_protocol.main_run(args) == 0
    assert len(calls) == 1
    assert run_protocol.main_merge(args) == 0
    canonical = args.output_root / "canonical"
    merged = [json.loads(line) for line in next(canonical.rglob("samples.jsonl")).read_text().splitlines()]
    assert merged == samples
    with next(canonical.rglob("detailed_results.csv")).open(newline="") as f:
        assert list(csv.DictReader(f)) == [{k: str(v) for k, v in row.items()} for row in rows]
    audit = json.loads((canonical / "merge_provenance.json").read_text())
    assert audit["selected_count"] == 6
    assert audit["dropped_count"] == audit["dropped_duplicate_count"] == 0


@pytest.mark.parametrize("damage,error", [
    ("duplicate_csv", "duplicate \\(problem, sample\\)"),
    ("duplicate_raw", "duplicate raw sample object"),
    ("missing_raw", "CSV rows without raw samples"),
    ("pending_backed_raw", "CSV row backed by pending"),
    ("orphan_complete", "verified/unknown leaf"),
    ("orphan_unknown", "verified/unknown leaf"),
    ("orphan_success", "successful sub-run"),
    ("unexpected_orphan", "unexpected sample pairs"),
    ("duplicate_committed_attempt", "duplicate \\(problem, sample\\)"),
    ("wrong_remap", "nonzero local sample index"),
])
def test_uncovered_damage_still_fails(tmp_path, damage, error):
    root = tmp_path / "output"
    attempt = root / "attempt"
    rows = [_row("math500_0", 0)]
    samples = [_sample("math500_0", 0, "answer")]
    status, exit_code = "failed", None
    if damage == "duplicate_csv":
        rows *= 2
    elif damage == "duplicate_raw":
        samples *= 2
    elif damage == "missing_raw":
        samples = []
    elif damage == "pending_backed_raw":
        samples = [_sample("math500_0", 0, "answer", pending=True)]
    elif damage.startswith("orphan_") or damage == "unexpected_orphan":
        samples.append(_sample("math500_1", 0, "uncommitted"))
        status = {"orphan_complete": "complete", "orphan_unknown": "unknown"}.get(damage, "failed")
        exit_code = 0 if damage == "orphan_success" else None
    elif damage == "wrong_remap":
        rows[0]["sample_idx"] = 1
    _leaf(attempt, rows, samples, status)
    runs = [{"attempt_dir": "attempt", "exit_code": exit_code}]
    if damage == "wrong_remap":
        runs[0]["sample_index_remap"] = 1
    if damage == "duplicate_committed_attempt":
        _leaf(root / "another", rows, samples, "complete")
        runs.append({"attempt_dir": "another", "exit_code": 0})
    group = {"expected_keys": {"math500": [["math500_0", 0], ["math500_1", 0]]}, "runs": runs}
    if damage == "unexpected_orphan":
        group["expected_keys"]["math500"].pop()
    with pytest.raises(ValueError, match=error):
        run_protocol.load_group_records(root, group)


def test_csv_and_raw_cannot_be_matched_across_leaves(tmp_path):
    attempt = tmp_path / "attempt"
    first = _leaf(attempt, [_row("math500_0", 0)], [], "failed")
    second = first.parent / "20261014_130000"
    first.rename(second)
    _leaf(attempt, [], [_sample("math500_0", 0, "other leaf")], "failed")
    with pytest.raises(ValueError, match="CSV/raw source leaf mismatch"):
        run_protocol.load_run_rows(attempt)


def test_legacy_no_status_records_work_but_unknown_raw_only_leaf_fails(tmp_path):
    attempt = tmp_path / "attempt"
    leaf = _leaf(attempt, [_row("math500_0", 0)],
                 [_sample("math500_0", 0, "legacy")], "complete")
    (leaf / "run_status.json").unlink()
    rows, samples = run_protocol.load_run_rows(attempt)
    assert len(rows) == len(samples) == 1
    orphan = leaf.parent / "20261014_130000"
    orphan.mkdir()
    (orphan / "samples.jsonl").write_text(
        json.dumps(_sample("math500_0", 0, "unknown duplicate")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw samples in unknown leaf"):
        run_protocol.load_run_rows(attempt)
