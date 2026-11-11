"""Code extraction in grading workers reproduces recorded R2 seed-42 code artifacts byte for byte.

Replays recorded engine results through ``run_single_job`` in their original completion
order (fake callback backend, real tokenizer and datasets) and compares the files with
the recorded run. Needs the local checkpoint tokenizer and the seed-42 protocol outputs.
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tah.evaluate import jobs
from tah.evaluate.datasets import load_combined_dataset


ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "models/TaH-plus-1.7B"
RUN = ROOT / "data/engine/paper_protocol_seed42/groups/{group}/runs/0001-initial/eval_output"
GROUPS = {"humaneval": "humaneval_plus", "mbpp": "mbpp_plus"}


def _recorded(dataset):
    dirs = sorted(Path(str(RUN).format(group=GROUPS[dataset])).glob(f"{dataset}_tah_engine/*/"))
    if not dirs or not MODEL.exists():
        pytest.skip("needs models/TaH-plus-1.7B and the R2 seed-42 code group outputs")
    run_dir = dirs[0]
    samples = [json.loads(line) for line in (run_dir / "samples.jsonl").open(encoding="utf-8")]
    return run_dir, samples


def _replay(tmp_path, monkeypatch, dataset, records, workers, pause=0.):
    """Run the harness on ``records`` (in callback order, ``pause`` s apart); return the run directory."""
    by_id = {r["problem_id"]: r for r in records}
    data, field_mapping = load_combined_dataset([dataset])
    args = SimpleNamespace(data_ids=None, data_range=None, dataset_ranges=None)
    selected, ranges = jobs._select_problems(args, data, [dataset])
    selected = [item for item in selected if str(item["id"]) in by_id]
    order = {pid: rank for rank, pid in enumerate(r["problem_id"] for r in records)}

    def infer(prompts, on_result):
        assert len(prompts) == len(selected)
        ranked = sorted(range(len(selected)), key=lambda i: order[str(selected[i]["id"])])
        for i in ranked:
            r = by_id[str(selected[i]["id"])]
            on_result(i, dict(text=r["output"], seconds=r["timing"]["total_seconds"],
                              pass_counts=r["pass_counts"], token_ids=r["token_ids"],
                              prompt_pass_counts=r["prompt_pass_counts"],
                              finish_reason=r["finish_reason"], timing=r["timing"]))
            time.sleep(pause)  # later rows finish grading while a slow extraction runs
    infer.supports_callback = True
    monkeypatch.setattr(jobs, "setup_backend", lambda *a, **kw: (None, infer))
    monkeypatch.setattr(jobs, "cleanup", lambda *a, **kw: None)
    out = tmp_path / f"w{workers}"
    run_dir = jobs._dataset_run_dir(str(out), dataset, "tah_engine", "test", ranges.get(dataset))
    run_dir.mkdir(parents=True)
    (run_dir / "code_solutions.jsonl").write_text("", encoding="utf-8")
    jobs.run_single_job(
        config=dict(batch_size=16, repeat_size=1, grading_workers=workers),
        combined_dataset_name=dataset, dataset_names=[dataset], output_dir=str(out),
        timestamp="test", model_path=str(MODEL), job_id=0, job_nums=1, start_idx=0,
        end_idx=len(selected), tp_size=1, backend="tah_engine", data_range=None,
        dataset_ranges=ranges, problems_data=selected, field_mapping=field_mapping)
    jobs.combine_job_results(run_dir, 1)
    return run_dir


def _filtered_lines(path, keep):
    return [line for line in path.read_bytes().splitlines(keepends=True) if keep(line)]


def _assert_matches_recording(run_dir, recorded_dir, records, dataset):
    ids = {r["problem_id"] for r in records}
    tasks = {pid[len(dataset) + 1:] for pid in ids}
    expected = _filtered_lines(recorded_dir / "code_solutions.jsonl",
                               lambda line: json.loads(line)["task_id"] in tasks)
    assert (run_dir / "code_solutions.jsonl").read_bytes() == b"".join(expected)
    header, *rows = (recorded_dir / "detailed_results.csv").read_bytes().splitlines(keepends=True)
    expected_csv = [header] + [row for row in rows if row.split(b",", 1)[0].decode() in ids]
    assert (run_dir / "detailed_results.csv").read_bytes() == b"".join(expected_csv)
    expected_samples = _filtered_lines(recorded_dir / "samples.jsonl",
                                       lambda line: json.loads(line)["problem_id"] in ids)
    assert (run_dir / "samples.jsonl").read_bytes() == b"".join(expected_samples)


def _callback_order(samples):
    # Recorded code_solutions.jsonl order is the callback (completion) order.
    return sorted(samples, key=lambda r: r["timing"]["total_seconds"])


@pytest.mark.parametrize("dataset", ["humaneval", "mbpp"])
@pytest.mark.parametrize("workers", [0, 4])
def test_uncapped_subset_matches_recorded_bytes(tmp_path, monkeypatch, dataset, workers):
    recorded_dir, samples = _recorded(dataset)
    records = [r for r in _callback_order(samples) if r["finish_reason"] == "eos"][::8]
    run_dir = _replay(tmp_path, monkeypatch, dataset, records, workers)
    _assert_matches_recording(run_dir, recorded_dir, records, dataset)


def test_slow_extraction_keeps_callback_order(tmp_path, monkeypatch):
    """A seconds-long sanitize early in the order must not let later rows overtake it."""
    _, samples = _recorded("humaneval")
    ordered = _callback_order(samples)
    capped = next(r for r in ordered if r["finish_reason"] == "length")
    slow = dict(capped, output=capped["output"][:6000])  # sanitize takes seconds, not minutes
    records = [r for r in ordered if r["finish_reason"] == "eos"][:12]
    records = records[:2] + [slow] + [r for r in records[2:] if r["problem_id"] != slow["problem_id"]]
    files = {}
    for workers in (0, 4):
        run_dir = _replay(tmp_path, monkeypatch, "humaneval", records, workers, pause=.4)
        files[workers] = [(run_dir / name).read_bytes() for name in
                          ("code_solutions.jsonl", "detailed_results.csv", "samples.jsonl",
                           "job_0/detailed_results.csv")]
    assert files[0] == files[4]
    solutions = [json.loads(line)["task_id"] for line in files[4][0].splitlines()]
    assert solutions == [r["problem_id"][len("humaneval_"):] for r in records]


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["humaneval", "mbpp"])
def test_full_group_with_capped_outputs_matches_recorded_bytes(tmp_path, monkeypatch, dataset):
    """Whole recorded group, including 8192-token capped outputs (~12 min of sanitize each)."""
    recorded_dir, samples = _recorded(dataset)
    run_dir = _replay(tmp_path, monkeypatch, dataset, _callback_order(samples), 4)
    _assert_matches_recording(run_dir, recorded_dir, samples, dataset)
    metrics = json.loads((run_dir / "job_0.metrics.json").read_text())
    assert metrics["grading_workers"] == 4
