import csv
import json
import sys
import argparse
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "script/tah/evaluation"))

import compare_runs
import run_protocol
import yaml


FIELDS = [
    "problem_id", "sample_idx", "correct_answer", "predicted_answer", "has_answer",
    "is_correct", "input_tokens", "output_tokens", "processing_time",
    "mean_passes_per_token", "fraction_tokens_at_least_2_passes", "pass_stat_tokens",
]
EXPECTED = {
    "math500": {
        f"math500_{i}": {0, 1} for i in range(4)
    }
}


def _make_run(root: Path, outcomes: list[list[bool]]) -> Path:
    leaf = root / "math500_tah" / "20261013_120000" / "TASK_0_4"
    leaf.mkdir(parents=True)
    rows = []
    for problem_idx, samples in enumerate(outcomes):
        for sample_idx, correct in enumerate(samples):
            rows.append({
                "problem_id": f"math500_{problem_idx}", "sample_idx": sample_idx,
                "correct_answer": "a", "predicted_answer": "a" if correct else "b",
                "has_answer": "True", "is_correct": str(correct), "input_tokens": 10,
                "output_tokens": 20, "processing_time": 1.0,
                "mean_passes_per_token": 1.25, "fraction_tokens_at_least_2_passes": 0.25,
                "pass_stat_tokens": 20,
            })
    with (leaf / "detailed_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (leaf / "run_status.json").write_text(json.dumps({
        "status": "complete", "expected_samples": len(rows), "completed_samples": len(rows),
    }), encoding="utf-8")
    return root


def test_protocol_run_analysis_pairs_problem_scores_and_pass_ratio(tmp_path):
    reference = _make_run(tmp_path / "reference", [[True, True], [False, False], [True, False], [True, False]])
    candidate = _make_run(tmp_path / "candidate", [[True, False], [False, False], [False, False], [True, True]])
    a = compare_runs._run_group([reference], EXPECTED)
    b = compare_runs._run_group([candidate], EXPECTED)
    result = compare_runs._pair_analysis(a, b, "ref", "engine", EXPECTED, 0.03)
    stats = result["datasets"]["math500"]
    assert stats["n_problems"] == 4
    assert stats["mean_difference"] == pytest.approx(-0.125)
    assert stats["paired_se"] > 0
    assert stats["paired_z"] == pytest.approx(stats["mean_difference"] / stats["paired_se"])
    assert stats["ci95"][0] < stats["ci95"][1]
    assert stats["tost_verdict"] == "equivalence_not_shown"
    assert stats["passes_per_token_ratio"] == pytest.approx(1.0)


def test_aggregate_pools_samples_across_seed_run_directories(tmp_path):
    seed_a = _make_run(tmp_path / "seed_a", [[True, False], [False, False], [True, False], [False, False]])
    seed_b = _make_run(tmp_path / "seed_b", [[True, True], [False, True], [False, False], [True, False]])
    pooled = compare_runs._run_group([seed_a, seed_b], EXPECTED)
    summary = compare_runs._summary(pooled["correct_by_dataset"]["math500"])
    assert summary["n_problems"] == 4
    assert summary["accuracy"] == pytest.approx(0.375)
    assert summary["se"] > 0
    assert summary["ci95"][0] < summary["accuracy"] < summary["ci95"][1]
    stats = compare_runs._pass_stats(pooled["rows_by_dataset"]["math500"])
    assert stats["pass_stat_tokens"] == 320
    assert stats["mean_passes_per_token"] == pytest.approx(1.25)


def test_protocol_refuses_missing_pairs(tmp_path):
    run = _make_run(tmp_path / "partial", [[True, True], [False, False], [True, False], [True, False]])
    detail = next(run.rglob("detailed_results.csv"))
    with detail.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with detail.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(ValueError, match="expected_samples mismatch|incomplete math500"):
        compare_runs.read_run(run, EXPECTED)


def test_protocol_refuses_duplicate_pairs(tmp_path):
    run = _make_run(tmp_path / "duplicate", [[True, True], [False, False], [True, False], [True, False]])
    detail = next(run.rglob("detailed_results.csv"))
    with detail.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.append(dict(rows[0]))
    with detail.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=r"duplicate \(problem, sample\)"):
        compare_runs.read_run(run, EXPECTED)


def test_resume_reader_remaps_only_completed_worker_samples(tmp_path):
    attempt = tmp_path / "attempt"
    leaf = attempt / "eval_output" / "math500_tah" / "20261013_120000" / "TASK_0_2"
    details = leaf / "job_0" / "details" / "math500_3"
    details.mkdir(parents=True)
    job_csv = leaf / "job_0" / "detailed_results.csv"
    job_csv.parent.mkdir(parents=True, exist_ok=True)
    with job_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({
            "problem_id": "math500_3", "sample_idx": 0, "is_correct": "True",
            "output_tokens": 3, "mean_passes_per_token": 1.0,
            "fraction_tokens_at_least_2_passes": 0.0, "pass_stat_tokens": 3,
        })
    (details / "sample_0.json").write_text(json.dumps({
        "problem_id": "math500_3", "output": "answer", "is_correct": True,
    }), encoding="utf-8")
    (leaf / "run_status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    rows, samples = run_protocol.load_run_rows(attempt, remap_sample_idx=1)
    assert [(row["problem_id"], row["sample_idx"]) for row in rows] == [("math500_3", 1)]
    assert [(sample["id"], sample["sample"]) for sample in samples] == [("math500_3", 1)]


def test_code_benchmark_uses_evalplus_plus_status(tmp_path):
    run = tmp_path / "code_seed"
    leaf = run / "humaneval_tah_engine" / "20261013_120000"
    leaf.mkdir(parents=True)
    with (leaf / "detailed_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({
            "problem_id": "humaneval_HumanEval/0", "sample_idx": 0,
            "is_correct": "False", "output_tokens": 10,
            "mean_passes_per_token": 1.0,
            "fraction_tokens_at_least_2_passes": 0.0, "pass_stat_tokens": 10,
        })
    (leaf / "run_status.json").write_text(json.dumps({
        "status": "complete", "expected_samples": 1, "completed_samples": 1,
    }), encoding="utf-8")
    (leaf / "code_solutions.eval_results.json").write_text(json.dumps({
        "eval": {"HumanEval/0": [{"task_id": "HumanEval/0", "base_status": "fail", "plus_status": "pass"}]},
    }), encoding="utf-8")

    parsed = compare_runs.read_run(run, {"humaneval": {"humaneval_HumanEval/0": {0}}})
    # EvalPlus's plus score requires both base and extra tests to pass.
    assert parsed["by_dataset"]["humaneval"][("humaneval_HumanEval/0", 0)]["is_correct"] is False


def test_aggregate_writes_markdown_json_and_optional_paper_column(tmp_path, monkeypatch):
    seed_a = _make_run(tmp_path / "aggregate_seed_a", [[True, False], [False, False], [True, False], [False, False]])
    seed_b = _make_run(tmp_path / "aggregate_seed_b", [[True, True], [False, True], [False, False], [True, False]])
    protocol = {"name": "synthetic_paper", "paper_values": {"math500": 0.375}, "paper_average": 0.375}
    monkeypatch.setattr(compare_runs, "expected_from_protocol",
                        lambda _path: (EXPECTED, protocol, {"math500": 2}))
    prefix = tmp_path / "paper_aggregate"
    args = argparse.Namespace(
        protocol_config=tmp_path / "protocol.yaml", run=[str(seed_a), str(seed_b)],
        no_paper_values=False, output_prefix=prefix,
    )

    assert compare_runs.run_aggregate(args) == 0
    result = json.loads(prefix.with_suffix(".json").read_text())
    markdown = prefix.with_suffix(".md").read_text()
    assert result["n_seed_runs"] == 2
    assert result["benchmarks"]["math500"]["accuracy"] == pytest.approx(0.375)
    assert result["benchmarks"]["math500"]["inside_95_ci"] is True
    assert "Paper" in markdown and "Inside 95% CI" in markdown


def test_merge_writes_complete_normal_layout_run(tmp_path):
    output_root = tmp_path / "protocol_output"
    attempt = output_root / "groups" / "math" / "runs" / "0001-initial" / "eval_output"
    leaf = attempt / "math500_tah" / "20261013_120000" / "TASK_0_1"
    leaf.mkdir(parents=True)
    row = {
        "problem_id": "math500_0", "sample_idx": 0, "correct_answer": "a",
        "predicted_answer": "a", "has_answer": "True", "is_correct": "True",
        "input_tokens": 4, "output_tokens": 3, "processing_time": 1.0,
        "mean_passes_per_token": 1.0, "fraction_tokens_at_least_2_passes": 0.0,
        "pass_stat_tokens": 3,
    }
    with (leaf / "detailed_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    (leaf / "samples.jsonl").write_text(json.dumps({
        "problem_id": "math500_0", "id": "math500_0", "sample": 0,
        "output": "answer", "is_correct": True,
    }) + "\n", encoding="utf-8")
    (leaf / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("name: synthetic\n", encoding="utf-8")
    eval_config = tmp_path / "base_eval.yaml"
    eval_config.write_text("temperature: 0.6\n", encoding="utf-8")
    state = {
        "status": "complete", "protocol_config": str(protocol_path),
        "protocol_config_sha256": "synthetic", "eval_config_sha256": "synthetic",
        "protocol_name": "synthetic", "model_path": "/model", "backend": "tah",
        "seed": 1234, "created_at": "2026-10-13T12:00:00+00:00",
        "resolved_eval_config": str(eval_config), "dataset_sizes": {"math500": 500},
        "expected_problem_ids": {"math500": ["math500_0"]},
        "groups": {"math": {
            "datasets": ["math500"], "selected_indices": {"math500": [0]},
            "repeat_size": 1, "expected_keys": {"math500": [["math500_0", 0]]},
            "runs": [{"attempt_dir": str(attempt.relative_to(output_root)),
                      "sample_index_remap": None}],
        }},
    }
    (output_root / "protocol_state.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "protocol_state.json").write_text(json.dumps(state), encoding="utf-8")

    assert run_protocol.main_merge(argparse.Namespace(output_root=output_root)) == 0
    manifest = json.loads((output_root / "canonical/protocol_manifest.json").read_text())
    assert manifest["expected_samples"] == {"math500": 1}
    canonical = next((output_root / "canonical/math500_tah").glob("*/TASK_0_1"))
    assert len(list(csv.DictReader((canonical / "detailed_results.csv").open()))) == 1
    assert json.loads((canonical / "run_status.json").read_text())["status"] == "complete"
    assert json.loads((canonical / "samples.jsonl").read_text())["id"] == "math500_0"


def test_runner_resumes_only_the_missing_problem_sample_pair(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("""name: synthetic_resume
eval_config: eval.yaml
backends: [tah]
groups:
  - name: math_one
    datasets: [math500]
    data_ranges:
      math500: \"0:1\"
    repeat_size: 2
    max_new_tokens: 32
    batch_size: 1
""", encoding="utf-8")
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text("temperature: 0.6\ntop_p: 0.95\ntop_k: 20\nrepeat_size: 1\nmax_new_tokens: 4096\nbatch_size: 16\n", encoding="utf-8")
    monkeypatch.setattr(run_protocol, "load_dataset_items", lambda _datasets: {
        "math500": [{"id": "math500_7", "_source_dataset": "math500"}],
    })
    calls = []

    def fake_eval(argv, cwd, check):
        calls.append(list(argv))
        output = Path(argv[argv.index("--output_dir") + 1])
        config = yaml.safe_load(Path(argv[argv.index("--eval_config") + 1]).read_text())
        remap = config["repeat_size"] == 1
        sample_index = 0 if len(calls) == 1 else 0
        leaf = output / "math500_tah" / f"20261013_12000{len(calls)}" / "TASK_0_1"
        leaf.mkdir(parents=True)
        row = {
            "problem_id": "math500_7", "sample_idx": sample_index,
            "correct_answer": "a", "predicted_answer": "a", "has_answer": "True",
            "is_correct": "True", "input_tokens": 3, "output_tokens": 4,
            "processing_time": 1.0, "mean_passes_per_token": 1.0,
            "fraction_tokens_at_least_2_passes": 0.0, "pass_stat_tokens": 4,
        }
        with (leaf / "detailed_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(row)
        (leaf / "samples.jsonl").write_text(json.dumps({
            "problem_id": "math500_7", "id": "math500_7", "sample": sample_index,
            "output": "answer", "is_correct": True,
        }) + "\n", encoding="utf-8")
        count = 2 if len(calls) == 1 else 1
        (leaf / "run_status.json").write_text(json.dumps({
            "status": "failed" if len(calls) == 1 else "complete",
            "expected_samples": count, "completed_samples": 1,
        }), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(run_protocol.subprocess, "run", fake_eval)
    args = argparse.Namespace(
        config=protocol_path, model_path="/model", backend="tah", seed=1234,
        output_root=tmp_path / "run", job_nums=1, tp_size_per_job=1,
        logger_level="WARNING", plan_only=False,
    )
    with pytest.raises(RuntimeError, match="eval.py failed"):
        run_protocol.main_run(args)
    assert run_protocol.main_run(args) == 0

    state = json.loads((args.output_root / "protocol_state.json").read_text())
    group = state["groups"]["math_one"]
    assert group["status"] == "complete"
    assert group["completed_samples"] == 2
    assert len(calls) == 2
    assert calls[1][calls[1].index("--data_ids") + 1] == "math500_7"
    second_attempt = args.output_root / group["runs"][1]["attempt_dir"]
    derived = yaml.safe_load((second_attempt / "eval_config.yaml").read_text())
    command = json.loads((second_attempt / "command.json").read_text())
    assert derived["repeat_size"] == 1
    assert command["sample_index_remap"] == 1
    assert command["selection"]["data_ids"] == ["math500_7"]


def test_student_t_critical_values_are_two_sided_quantiles():
    assert compare_runs.t_critical(0.975, 1) == pytest.approx(12.7062047, rel=1e-6)
    assert compare_runs.t_critical(0.95, 397) == pytest.approx(1.6487, abs=0.001)
