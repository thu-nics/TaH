"""Multi-job evaluation runner.

Three layers of orchestration:

* :func:`run_single_job` — load a backend, run inference over an assigned
  slice of problems, score each output, write per-problem CSV/JSON files.
* :func:`_run_job_process` — process wrapper that pins ``CUDA_VISIBLE_DEVICES``
  and a fresh NCCL port, then calls ``run_single_job``.
* :func:`allocate_gpus_and_run_jobs` — top-level entry point. Splits the
  selected problems across jobs, fans out one process per job, joins, then
  combines per-job outputs.

:func:`combine_job_results` aggregates the per-job CSV/JSON files into a
single set of files at the run-level directory.

:func:`parse_data_range` parses the ``--data_range`` CLI argument format.
"""
from __future__ import annotations

import csv
import ctypes
import fcntl
import json
import logging as pylog
import os
import shutil
import signal
import socket
import threading
import time
import traceback
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from transformers.utils import logging as hf_logging

from tah.evaluate import codeeval, matheval
from tah.evaluate.code_prompts import extract_tah2_code, make_tah2_prompt
from tah.evaluate.backends import cleanup, setup_backend
from tah.evaluate.datasets import load_combined_dataset
from tah.evaluate.progress import SampleProgress

# Spawn is required for CUDA-capable subprocess workers.
mp.set_start_method("spawn", force=True)


_CSV_FIELDS = (
    "problem_id", "sample_idx", "correct_answer", "predicted_answer",
    "has_answer", "is_correct", "input_tokens", "output_tokens", "processing_time",
    "mean_passes_per_token", "fraction_tokens_at_least_2_passes", "pass_stat_tokens",
)
_CODE_ANSWER_TYPES = frozenset({"humaneval", "mbpp"})
_UNSUPPORTED_CODE_DATASETS = {"livecodebench": "LiveCodeBench has no grader in this repository."}


def _write_run_status(directory: Path, status: dict) -> None:
    """Keep machine-readable completion metadata next to combined artifacts."""
    tmp = directory / "run_status.json.tmp"
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, directory / "run_status.json")


# ────────────────────────────────────────────────────────────────────────────
# Per-problem prompt building + scoring
# ────────────────────────────────────────────────────────────────────────────


def _build_problem(item: dict, idx: int, field_mapping: Dict, detail_dir: Path) -> dict:
    """Pull standard fields out of an item; also create the per-problem dir."""
    pid = str(item.get(field_mapping["id_field"]) or f"problem_{idx}")
    text = str(item.get(field_mapping["question_field"], "")).strip()
    answer = str(item.get(field_mapping["answer_field"], "")).strip()
    answer_type = item.get("_source_answer_type", field_mapping.get("answer_type", "boxed"))
    problem_dir = detail_dir / pid
    problem_dir.mkdir(parents=True, exist_ok=True)
    return {
        "problem_id": pid,
        "dataset_name": item.get("_source_dataset", field_mapping.get("dataset_names", [""])[0]),
        "answer_type": answer_type,
        "original_problem_id": item.get("_original_id", pid),
        "problem_text": text,
        "correct_answer": answer,
        "problem_dir": problem_dir,
        "entry_point": item.get("entry_point"),
        # The loader preserves MBPP assertions verbatim as _original_assertion.
        # HumanEval's test is also carried by TaH2, but its template ignores it.
        "assertion": item.get("assertion", item.get(
            "_original_test" if answer_type == "humaneval" else "_original_assertion", "")),
    }


def _make_prompt(text: str, tokenizer, *, is_code: bool, code_prompt: str = "harness",
                 answer_type: str = "", assertion: str = "") -> str:
    if code_prompt not in ("harness", "tah2"):
        raise ValueError("code_prompt must be 'harness' or 'tah2'")
    if is_code:
        if code_prompt == "tah2":
            return make_tah2_prompt(text, assertion, answer_type, tokenizer)
        return codeeval.make_raw_chat_prompt_for_code_evaluation(
            task_prompt=text, reasoning=False, tokenizer=tokenizer,
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
    )


def _score_one(output_text: str, correct: str, dataset_name: str, *, is_code: bool) -> Tuple[str, bool, bool]:
    """Return ``(predicted_answer, has_answer, is_correct)``.

    Code datasets defer scoring until after generation finishes (run via
    evalplus on the unified solutions file), so we just stamp placeholders.
    """
    if is_code:
        return "pending_code_eval", False, False
    is_correct, predicted = matheval.evaluator_map[dataset_name].rule_judge(output_text, correct)
    if predicted == "No extracted answer":
        return "", False, bool(is_correct)
    return predicted, True, bool(is_correct)


def _extract_code(output_text: str, code_prompt: str, entry_point: str) -> str:
    """Solution submitted to EvalPlus for one code output."""
    if code_prompt == "tah2":
        return extract_tah2_code(output_text)
    return codeeval.sanitize(output_text, entry_point)


def _append_code_solution(path: Path, task_id: str, code: str) -> None:
    """Append a {task_id, solution} entry to the unified jsonl with file lock."""
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps({"task_id": task_id, "solution": str(code)}, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ────────────────────────────────────────────────────────────────────────────
# Per-job runner
# ────────────────────────────────────────────────────────────────────────────


def _dataset_run_dir(output_dir: str, dataset_name: str, backend: str,
                     timestamp: str, data_range=None) -> Path:
    """Return the current per-dataset output directory for this run."""
    out = Path(output_dir) / f"{dataset_name}_{backend}" / timestamp
    if data_range:
        a, b = parse_data_range(data_range)
        out = out / f"TASK_{a}_{b}"
    return out


def _job_output_dir(output_dir: str, dataset_name: str, backend: str,
                    timestamp: str, job_id: int, data_range=None) -> Path:
    return _dataset_run_dir(output_dir, dataset_name, backend, timestamp, data_range) / f"job_{job_id}"


def _build_prompts(
    problems: List[dict], tokenizer, repeat_size: int, is_code: bool,
    *, code_prompt: str = "harness",
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Render each problem ``repeat_size`` times and return
    ``(prompts, prompt_to_problem)`` where ``prompt_to_problem[i] = (problem_idx, sample_idx)``."""
    prompts: List[str] = []
    prompt_to_problem: List[Tuple[int, int]] = []
    for pi, p in enumerate(problems):
        rendered = _make_prompt(p["problem_text"], tokenizer, is_code=is_code,
                                code_prompt=code_prompt, answer_type=p.get("answer_type", ""),
                                assertion=p.get("assertion", ""))
        for s in range(repeat_size):
            prompts.append(rendered)
            prompt_to_problem.append((pi, s))
    return prompts, prompt_to_problem


def _normalise_inference_result(result) -> Tuple[str, float, Optional[List[int]]]:
    """Accept the legacy ``(text, seconds)`` and optional pass-count result."""
    if isinstance(result, dict):
        text = result["text"]
        elapsed = result.get("processing_time", result.get("seconds", 0.0))
        pass_counts = result.get("pass_counts")
    else:
        if len(result) < 2:
            raise ValueError(f"backend result must contain text and elapsed seconds, got {result!r}")
        text, elapsed = result[0], result[1]
        pass_counts = result[2] if len(result) > 2 else None
    if pass_counts is not None:
        pass_counts = [int(n) for n in pass_counts]
    return str(text), float(elapsed), pass_counts


def _pass_summary(pass_counts: Optional[List[int]]) -> Tuple[str, str, int]:
    if not pass_counts:
        return "", "", 0
    n = len(pass_counts)
    return f"{sum(pass_counts) / n:.6f}", f"{sum(v >= 2 for v in pass_counts) / n:.6f}", n


def _process_result(
    prompt_idx: int,
    backend_result,
    prompt_to_problem: List[Tuple[int, int]],
    problems: List[dict],
    tokenizer,
    writers: Dict[str, Tuple[object, csv.DictWriter]],
    all_results: Dict[str, List[dict]],
    scored=None,
    extracted=None,
) -> None:
    """Grade and persist a single result, safe to call from a backend callback.

    ``scored``/``extracted`` come from a grading worker; otherwise both are computed here.
    """
    output_text, proc_time, pass_counts = _normalise_inference_result(backend_result)
    pi, sample_idx = prompt_to_problem[prompt_idx]
    p = problems[pi]
    dataset_name = p["dataset_name"]
    is_code = p["answer_type"] in _CODE_ANSWER_TYPES
    pred, has_ans, ok = scored if scored is not None else _score_one(
        output_text, p["correct_answer"], dataset_name, is_code=is_code)
    detail = {
        "problem_id": p["problem_id"],
        "problem": p["problem_text"],
        "output": output_text,
        "correct_answer": p["correct_answer"],
        "predicted_answer": pred,
        "is_correct": ok,
    }
    if is_code:
        if extracted is None:
            extracted = _extract_code(output_text, p.get("code_prompt", "harness"), p["entry_point"])
        detail["extracted_code"] = extracted
        detail["entry_point"] = p["entry_point"]
        _append_code_solution(p["output_dir"] / "code_solutions.jsonl", p["original_problem_id"], extracted)

    mean_passes, frac_two_plus, pass_tokens = _pass_summary(pass_counts)
    if pass_counts is not None:
        detail["mean_passes_per_token"] = float(mean_passes) if pass_counts else None
        detail["fraction_tokens_at_least_2_passes"] = float(frac_two_plus) if pass_counts else None
        detail["pass_stat_tokens"] = pass_tokens
    if isinstance(backend_result, dict):
        for key in ("token_ids", "pass_counts", "prompt_pass_counts", "finish_reason", "timing"):
            if key in backend_result:
                detail[key] = backend_result[key]
    sample_path = p["problem_dir"] / f"sample_{sample_idx}.json"
    tmp_path = sample_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fd:
        json.dump(detail, fd, ensure_ascii=False, indent=2)
        fd.flush()
        os.fsync(fd.fileno())
    os.replace(tmp_path, sample_path)

    row = {
        "problem_id": p["problem_id"],
        "sample_idx": sample_idx,
        "correct_answer": p["correct_answer"],
        "predicted_answer": pred,
        "has_answer": has_ans,
        "is_correct": ok,
        "input_tokens": len(tokenizer.encode(p["problem_text"])),
        "output_tokens": len(tokenizer.encode(output_text)),
        "processing_time": proc_time,
        "mean_passes_per_token": mean_passes,
        "fraction_tokens_at_least_2_passes": frac_two_plus,
        "pass_stat_tokens": pass_tokens,
    }
    csv_file, writer = writers[dataset_name]
    writer.writerow(row)
    csv_file.flush()
    all_results[dataset_name].append(row)


def _grading_init():
    # CPU graders must not create CUDA contexts or compete for BLAS threads.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)


def _score_timed(output, answer, dataset, is_code):
    started = time.perf_counter()
    score = _score_one(output, answer, dataset, is_code=is_code)
    return score, time.perf_counter() - started


def _grade_timed(output, answer, dataset, is_code, code_prompt, entry_point):
    # Code extraction runs here, not on the backend callback thread: EvalPlus sanitize
    # takes minutes on a length-capped output and would stall generation.
    started = time.perf_counter()
    score = _score_one(output, answer, dataset, is_code=is_code)
    extracted = _extract_code(output, code_prompt, entry_point) if is_code else None
    return score, extracted, time.perf_counter() - started


def _save_pending_result(index, result, mapping, problems):
    """Persist generated text before a process grader has finished."""
    pi, sample = mapping[index]
    p = problems[pi]
    text, seconds, counts = _normalise_inference_result(result)
    detail = dict(problem_id=p["problem_id"], problem=p["problem_text"], output=text,
                  correct_answer=p["correct_answer"], grading_status="pending")
    if isinstance(result, dict):
        detail.update({k: result[k] for k in ("token_ids", "pass_counts", "prompt_pass_counts",
                                             "finish_reason", "timing") if k in result})
    path = p["problem_dir"] / f"sample_{sample}.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_single_job(
    *, config: Dict, combined_dataset_name: str, output_dir: str, timestamp: str,
    model_path: str, job_id: int, job_nums: int, start_idx: int, end_idx: int,
    tp_size: int, backend: str, data_range, problems_data, field_mapping: Dict,
    unified_code_solutions_file: Optional[Path] = None, dataset_names=None,
    dataset_ranges=None,
) -> None:
    """Run one backend instance for this job and write each dataset separately."""
    from transformers import AutoTokenizer

    code_prompt = config.get("code_prompt", "harness")
    if code_prompt not in ("harness", "tah2"):
        raise ValueError("code_prompt must be 'harness' or 'tah2'")
    dataset_names = dataset_names or field_mapping.get("dataset_names") or [combined_dataset_name]
    dataset_ranges = dataset_ranges or {}
    print(f"Job {job_id+1}/{job_nums}: {len(problems_data)} problems, backend={backend}, datasets={dataset_names}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = dict(config, _prompt_count=len(problems_data) * config["repeat_size"])
    grading_workers = config.get("grading_workers", 4 if backend == "tah_engine" else 0)
    if isinstance(grading_workers, bool) or not isinstance(grading_workers, int) or grading_workers < 0:
        raise ValueError("grading_workers must be a nonnegative integer")
    setup_started = time.perf_counter()
    model, infer = setup_backend(backend, config, model_path, tokenizer, tp_size=tp_size)
    setup_seconds = time.perf_counter() - setup_started

    problems = []
    for i, item in enumerate(problems_data):
        ds_name = item.get("_source_dataset", dataset_names[0])
        ds_type = item.get("_source_answer_type", field_mapping.get("answer_types_by_dataset", {}).get(ds_name, field_mapping.get("answer_type", "boxed")))
        item_mapping = dict(field_mapping)
        item_mapping["answer_type"] = ds_type
        item_mapping["dataset_names"] = [ds_name]
        ds_out = _dataset_run_dir(output_dir, ds_name, backend, timestamp, dataset_ranges.get(ds_name))
        job_out = ds_out / f"job_{job_id}"
        (job_out / "details").mkdir(parents=True, exist_ok=True)
        problem = _build_problem(item, item.get("_original_index", start_idx + i), item_mapping, job_out / "details")
        problem["output_dir"] = ds_out
        problem["code_prompt"] = code_prompt
        problems.append(problem)

    prompts: List[str] = []
    prompt_to_problem: List[Tuple[int, int]] = []
    for pi, problem in enumerate(problems):
        rendered = _make_prompt(
            problem["problem_text"], tokenizer, is_code=problem["answer_type"] in _CODE_ANSWER_TYPES,
            code_prompt=code_prompt, answer_type=problem["answer_type"], assertion=problem["assertion"],
        )
        for sample_idx in range(config["repeat_size"]):
            prompts.append(rendered)
            prompt_to_problem.append((pi, sample_idx))
    print(f"Processing batch_size={config['batch_size']} repeat_size={config['repeat_size']} → {len(prompts)} prompts")

    writers: Dict[str, Tuple[object, csv.DictWriter]] = {}
    all_results: Dict[str, List[dict]] = {name: [] for name in dataset_names}
    for ds_name in dataset_names:
        if any(p["dataset_name"] == ds_name for p in problems):
            job_out = _job_output_dir(output_dir, ds_name, backend, timestamp, job_id, dataset_ranges.get(ds_name))
            f = open(job_out / "detailed_results.csv", "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            f.flush()
            writers[ds_name] = (f, writer)

    pool = ProcessPoolExecutor(max_workers=grading_workers, mp_context=mp.get_context("spawn"),
                               initializer=_grading_init) if grading_workers else None
    pending = {}
    grading_work_seconds = 0.0
    grading_inline_seconds = 0.0
    inference_seconds = 0.0
    run_started = time.perf_counter()

    def persist(index, result, score, extracted=None):
        _process_result(index, result, prompt_to_problem, problems, tokenizer, writers, all_results,
                        scored=score, extracted=extracted)

    def drain(futures):
        nonlocal grading_work_seconds
        for future in futures:
            index, result = pending.pop(future)
            score, extracted, seconds = future.result()
            grading_work_seconds += seconds
            persist(index, result, score, extracted)

    def done_prefix():
        # Persist in callback order: a slow code extraction holds back later rows instead
        # of letting them overtake it, so CSV and code_solutions.jsonl order is unchanged.
        ready = []
        for future in pending:
            if not future.done():
                break
            ready.append(future)
        return ready

    def process(index, result):
        nonlocal grading_work_seconds, grading_inline_seconds
        pi, _ = prompt_to_problem[index]
        p = problems[pi]
        text, _, _ = _normalise_inference_result(result)
        args = (text, p["correct_answer"], p["dataset_name"], p["answer_type"] in _CODE_ANSWER_TYPES)
        if pool is not None:
            _save_pending_result(index, result, prompt_to_problem, problems)
            future = pool.submit(_grade_timed, *args, p.get("code_prompt", "harness"), p.get("entry_point"))
            pending[future] = (index, result)
            drain(done_prefix())
        else:
            score, seconds = _score_timed(*args)
            grading_work_seconds += seconds
            grading_inline_seconds += seconds
            persist(index, result, score)

    try:
        with SampleProgress(len(prompts), job_id, getattr(infer, "progress_snapshot", None)) as progress:
            if getattr(infer, "supports_callback", False):
                callback_lock = threading.Lock()
                seen_results = set()

                def on_result(index, result):
                    index = int(index)
                    if not 0 <= index < len(prompts):
                        raise IndexError(f"backend callback prompt index {index} outside 0..{len(prompts)-1}")
                    with callback_lock:
                        if index in seen_results:
                            raise ValueError(f"backend reported prompt {index} more than once")
                        process(index, result)
                        seen_results.add(index)
                        progress.advance()

                inference_started = time.perf_counter()
                infer(prompts, on_result=on_result)
                inference_seconds += time.perf_counter() - inference_started
                missing = len(prompts) - len(seen_results)
                if missing:
                    raise RuntimeError(f"callback backend returned without reporting {missing} of {len(prompts)} prompts")
            else:
                bs = config["batch_size"]
                inference_batches = []
                for ds_name in dataset_names:
                    indexes = [i for i, (pi, _) in enumerate(prompt_to_problem)
                               if problems[pi]["dataset_name"] == ds_name]
                    for start in range(0, len(indexes), bs):
                        inference_batches.append(indexes[start:start + bs])
                for indexes in inference_batches:
                    inference_started = time.perf_counter()
                    batch_outputs = infer([prompts[i] for i in indexes])
                    inference_seconds += time.perf_counter() - inference_started
                    if len(batch_outputs) != len(indexes):
                        raise ValueError(f"backend returned {len(batch_outputs)} outputs for {len(indexes)} prompts")
                    for prompt_idx, result in zip(indexes, batch_outputs):
                        process(prompt_idx, result)
                        progress.advance()
        wait_started = time.perf_counter()
        drain(list(pending))  # insertion (callback) order; result() waits for each
        grading_tail_seconds = time.perf_counter() - wait_started
        metrics = dict(setup_seconds=setup_seconds, inference_call_seconds=inference_seconds,
                       job_wall_seconds=time.perf_counter() - run_started,
                       grading_workers=grading_workers, grading_work_seconds=grading_work_seconds,
                       grading_inline_seconds=grading_inline_seconds,
                       grading_tail_seconds=grading_tail_seconds,
                       **getattr(infer, "telemetry", {}))
        for ds_name in writers:
            directory = _dataset_run_dir(output_dir, ds_name, backend, timestamp, dataset_ranges.get(ds_name))
            (directory / f"job_{job_id}.metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        for ds_name, rows in all_results.items():
            if rows:
                _write_job_stats(_job_output_dir(output_dir, ds_name, backend, timestamp, job_id,
                                                 dataset_ranges.get(ds_name)), rows)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for f, _ in writers.values():
            f.close()
        cleanup(model, backend)
    for ds_name, rows in all_results.items():
        if rows:
            print(f"Job {job_id} done for {ds_name}; accuracy={_overall_accuracy(rows):.4f}")


def _overall_accuracy(rows: List[dict]) -> float:
    return sum(1 for r in rows if r["is_correct"]) / len(rows) if rows else 0.0


def _write_job_stats(out: Path, all_results: List[dict]) -> None:
    """Write evaluation_stats.csv with per-problem stats + a totals row."""
    by_problem: Dict[str, List[dict]] = {}
    for r in all_results:
        by_problem.setdefault(r["problem_id"], []).append(r)

    rows = []
    for pid, rs in by_problem.items():
        n = len(rs)
        c = sum(1 for r in rs if r["is_correct"])
        avg_out = sum(r["output_tokens"] for r in rs) / n if n else 0.0
        mean_pass = _weighted_field(rs, "mean_passes_per_token", "pass_stat_tokens")
        frac_two = _weighted_field(rs, "fraction_tokens_at_least_2_passes", "pass_stat_tokens")
        stat_n = sum(int(r.get("pass_stat_tokens") or 0) for r in rs)
        rows.append((pid, c / n if n else 0.0, c, n, avg_out, mean_pass, frac_two, stat_n))

    total_n = len(all_results)
    total_c = sum(1 for r in all_results if r["is_correct"])
    total_avg_out = sum(r["output_tokens"] for r in all_results) / total_n if total_n else 0.0
    total_mean_pass = _weighted_field(all_results, "mean_passes_per_token", "pass_stat_tokens")
    total_frac_two = _weighted_field(all_results, "fraction_tokens_at_least_2_passes", "pass_stat_tokens")
    total_pass_tokens = sum(int(r.get("pass_stat_tokens") or 0) for r in all_results)

    with open(out / "evaluation_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "accuracy", "correct_count", "total_samples", "avg_output_length",
                    "mean_passes_per_token", "fraction_tokens_at_least_2_passes", "pass_stat_tokens"])
        for pid, acc, c, n, avg, mean_pass, frac_two, stat_n in rows:
            w.writerow([pid, f"{acc:.3f}", c, n, f"{avg:.2f}", _fmt_optional(mean_pass),
                        _fmt_optional(frac_two), stat_n])
        w.writerow([])
        w.writerow(["Total Accuracy", f"{(total_c/total_n if total_n else 0):.3f}", total_c, total_n,
                    f"{total_avg_out:.2f}", _fmt_optional(total_mean_pass), _fmt_optional(total_frac_two),
                    total_pass_tokens])


def _weighted_field(rows: List[dict], value_field: str, weight_field: str) -> Optional[float]:
    weighted = [
        (float(r[value_field]), int(r.get(weight_field) or 0))
        for r in rows
        if r.get(value_field) not in (None, "") and int(r.get(weight_field) or 0) > 0
    ]
    total_weight = sum(weight for _, weight in weighted)
    return sum(value * weight for value, weight in weighted) / total_weight if total_weight else None


def _fmt_optional(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


# ────────────────────────────────────────────────────────────────────────────
# Process wrapper (one per job)
# ────────────────────────────────────────────────────────────────────────────


def _free_port(start: int = 29555, max_tries: int = 100) -> int:
    """Find a port we can bind to from ``start``, wrapping into the 30k range if needed."""
    for i in range(max_tries):
        p = start + i
        if p > 65535:
            p = 30514 + (i % 100)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    raise RuntimeError(f"no free port after {max_tries} tries from {start}")


def _setup_logging(level_name: str) -> None:
    level = level_name.upper()
    hf_logging.set_verbosity(getattr(hf_logging, level, hf_logging.WARNING))
    hf_logging.enable_default_handler()
    hf_logging.enable_propagation()
    pylog.basicConfig(
        level=getattr(pylog, level, pylog.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


_PR_SET_PDEATHSIG = 1  # no Python constant; <linux/prctl.h>


def _install_pdeathsig() -> None:
    """SIGTERM the worker if its parent dies (Linux). Without this, killing the
    eval driver leaves orphan workers reparented to init still pinning the GPU."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


def _run_job_process(job_args: Tuple, result_queue: Queue) -> None:
    """One job per Process; pins CUDA_VISIBLE_DEVICES + NCCL port for the worker."""
    _install_pdeathsig()
    (job_id, config, dataset_names, output_dir, timestamp, model_path,
     job_nums, start_idx, end_idx, tp_size, backend, dataset_ranges, gpu_devices,
     problems_data, field_mapping) = job_args

    _setup_logging(config.get("_logger_level") or "WARNING")

    port = _free_port(start=29555 + job_id * 100)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["SGLANG_NCCL_PORT"] = str(port)

    import torch
    if backend == "tah_engine":
        import random
        import numpy as np
        seed = config.get("_random_seed", 420)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        from tah.model.utils import set_all_seeds
        set_all_seeds(config.get("_random_seed", 420))
    if torch.cuda.is_available():
        torch.cuda.init()

    try:
        print(f"\nJob {job_id}: GPUs={gpu_devices}, NCCL port={port}, indices {start_idx}..{end_idx-1}")
        run_single_job(
            config=config, combined_dataset_name="_".join(dataset_names), dataset_names=dataset_names,
            output_dir=output_dir, timestamp=timestamp, model_path=model_path,
            job_id=job_id, job_nums=job_nums, start_idx=start_idx, end_idx=end_idx,
            tp_size=tp_size, backend=backend, data_range=None, dataset_ranges=dataset_ranges,
            problems_data=problems_data, field_mapping=field_mapping,
        )
        result_queue.put((job_id, True, "ok"))
    except Exception as e:
        error = f"Job {job_id} failed: {e}\n{traceback.format_exc()}"
        for ds_name in dataset_names:
            directory = _dataset_run_dir(output_dir, ds_name, backend, timestamp,
                                         dataset_ranges.get(ds_name))
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"job_{job_id}.error.txt").write_text(error, encoding="utf-8")
        result_queue.put((job_id, False, error))
    finally:
        for k in ("MASTER_PORT", "MASTER_ADDR", "SGLANG_NCCL_PORT", "NCCL_SOCKET_IFNAME"):
            os.environ.pop(k, None)


# ────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ────────────────────────────────────────────────────────────────────────────


def parse_data_range(data_range_list, total_problems: Optional[int] = None) -> Tuple[int, int]:
    """Parse ``--data_range`` argument: ``[end]`` or ``[start, end]`` (end exclusive)."""
    if not data_range_list:
        return 0, total_problems if total_problems is not None else 0
    if len(data_range_list) == 1:
        s, e = 0, data_range_list[0]
    elif len(data_range_list) == 2:
        s, e = data_range_list[0], data_range_list[1]
    else:
        raise ValueError(f"data_range expects 1 or 2 values, got {len(data_range_list)}")
    if total_problems is not None:
        e = min(e, total_problems)
    if s < 0 or e <= s:
        raise ValueError(f"invalid data_range: start={s}, end={e}")
    return s, e


def parse_dataset_ranges(value: Optional[str], dataset_names: List[str]) -> Dict[str, Optional[Tuple[int, int]]]:
    """Parse ``dataset=start:end,dataset=all`` range overrides."""
    ranges: Dict[str, Optional[Tuple[int, int]]] = {}
    if not value:
        return ranges
    for term in value.split(","):
        term = term.strip()
        if not term:
            continue
        if "=" not in term:
            raise ValueError(f"invalid --dataset_ranges entry {term!r}; use dataset=start:end or dataset=all")
        name, spec = (part.strip() for part in term.split("=", 1))
        if name not in dataset_names:
            raise ValueError(f"--dataset_ranges names {name!r}, which is not in --dataset_name {dataset_names}")
        if name in ranges:
            raise ValueError(f"duplicate --dataset_ranges entry for {name!r}")
        if spec.lower() == "all":
            ranges[name] = None
            continue
        try:
            start, end = (int(part) for part in spec.split(":"))
        except Exception as e:
            raise ValueError(f"invalid range {spec!r} for {name}; expected start:end or all") from e
        if start < 0 or end <= start:
            raise ValueError(f"invalid range for {name}: start={start}, end={end}")
        ranges[name] = (start, end)
    return ranges


def _select_problems(args, dataset: List[dict], dataset_names=None) -> Tuple[List[dict], Dict[str, Optional[Tuple[int, int]]]]:
    """Select each dataset's range independently and return tagged items."""
    dataset_names = dataset_names or list(dict.fromkeys(item.get("_source_dataset", "") for item in dataset))
    explicit = parse_dataset_ranges(getattr(args, "dataset_ranges", None), dataset_names)
    ranges: Dict[str, Optional[Tuple[int, int]]] = {}
    if getattr(args, "data_ids", None):
        by_id = {str(item.get("id")): item for item in dataset}
        out = []
        seen = set()
        for pid in (s.strip() for s in str(args.data_ids).split(",") if s.strip()):
            if pid in by_id and pid not in seen:
                item = dict(by_id[pid])
                item["_original_index"] = int(item.get("_original_index", 0))
                out.append(item)
                seen.add(pid)
        if not out:
            raise ValueError("--data_ids matched no problem IDs (use <dataset>_<original_id>)")
        ranges.update({name: None for name in dataset_names})
        return out, ranges

    selected: List[dict] = []
    for name in dataset_names:
        items = [item for item in dataset if item.get("_source_dataset") == name]
        local_range = explicit[name] if name in explicit else getattr(args, "data_range", None)
        if local_range is None:
            ranges[name] = None
            indexes = range(len(items))
        else:
            start, end = parse_data_range(local_range, len(items))
            ranges[name] = (start, end)
            indexes = range(start, end)
        for local_idx in indexes:
            item = dict(items[local_idx])
            item["_original_index"] = local_idx
            selected.append(item)
    if not selected:
        raise ValueError("the selected dataset ranges contain no problems")
    return selected, ranges


def _build_job_args(args, selected, eval_config, dataset_names, timestamp,
                    available_gpus, gpus_per_job, field_mapping, dataset_ranges):
    """Materialise jobs; each job loads one backend and handles all its datasets."""
    job_args_list = []
    for job_id in range(args.job_nums):
        job_problems = selected[job_id::args.job_nums]
        if not job_problems:
            continue
        gpu_start = job_id * gpus_per_job
        job_gpus = available_gpus[gpu_start:gpu_start + gpus_per_job]
        job_args_list.append((
            job_id, eval_config, dataset_names, args.output_dir, timestamp,
            args.model_path, args.job_nums, 0, len(job_problems), gpus_per_job, args.backend,
            dataset_ranges, job_gpus, job_problems, field_mapping,
        ))
    return job_args_list


def _run_jobs(job_args_list) -> Tuple[int, int]:
    """Spawn one Process per job, drain the result queue, return ``(ok, failed)``."""
    queue: Queue = mp.Queue()
    completed = failed = 0

    def _drain_queue() -> None:
        """Process any pending job results from the queue."""
        nonlocal completed, failed
        while not queue.empty():
            jid, ok, msg = queue.get_nowait()
            if ok:
                completed += 1
                print(f"\n✓ Job {jid} ok")
            else:
                failed += 1
                print(f"\n✗ Job {jid} failed: {msg}")

    # Start every job up front; the workers run in parallel and we just wait.
    processes: List[Tuple[Process, int]] = []
    for idx, ja in enumerate(job_args_list):
        p = Process(target=_run_job_process, args=(ja, queue), name=f"sgl-{idx}")
        # ``spawn`` imports this module in the child before calling
        # _run_job_process. Pin the GPU in the environment inherited at
        # process creation time so eager TaH/Transformers imports cannot cache
        # the parent's CUDA device view first.
        previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ja[12]))
        try:
            p.start()
        finally:
            if previous_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
        processes.append((p, ja[0]))
        print(f"Started job {ja[0]}")

    # Poll until every worker is dead; reap dead processes and drain the queue.
    while processes:
        still_active = []
        for p, jid in processes:
            if p.is_alive():
                still_active.append((p, jid))
            else:
                p.join(timeout=1)
        processes = still_active
        _drain_queue()
        if processes:
            time.sleep(0.1)

    _drain_queue()  # final pass for any results that arrived after the last check
    return completed, failed


def allocate_gpus_and_run_jobs(args) -> None:
    """Load one model per job, evaluate all requested datasets, then combine by dataset."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"Running {args.job_nums} jobs with {args.tp_size_per_job} GPU(s) per job")

    # GPU pool comes from CUDA_VISIBLE_DEVICES (or 0..7 if unset).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    available_gpus = [int(x.strip()) for x in visible.split(",") if x.strip()] or list(range(8))
    print(f"Available GPU devices: {available_gpus}")

    dataset_names = [n.strip() for n in args.dataset_name.split(",") if n.strip()]
    unsupported = [name for name in dataset_names if name in _UNSUPPORTED_CODE_DATASETS]
    if unsupported:
        descriptions = "; ".join(f"{name}: {_UNSUPPORTED_CODE_DATASETS[name]}" for name in unsupported)
        raise ValueError(f"Cannot evaluate requested dataset(s): {descriptions} Refusing to apply another dataset's tests.")
    dataset, field_mapping = load_combined_dataset(dataset_names)
    answer_types = field_mapping.get("answer_types_by_dataset", {})
    missing_graders = [
        name for name in dataset_names
        if answer_types.get(name) not in _CODE_ANSWER_TYPES and name not in matheval.evaluator_map
    ]
    if missing_graders:
        raise ValueError(f"No math grader is registered for dataset(s): {missing_graders}")

    selected, dataset_ranges = _select_problems(args, dataset, dataset_names)

    # Load + tag YAML config (worker reads back the tags via config["_..."]).
    with open(args.eval_config, "r") as f:
        eval_config = yaml.safe_load(f)
    engine_options = dict(eval_config.get("engine") or {})
    overrides = getattr(args, "engine_options", None)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("--engine-options must be a JSON object")
        engine_options.update(overrides)
    for attribute, key in (("engine_max_batch_size", "max_batch_size"),
                           ("engine_memory_fraction", "memory_fraction")):
        value = getattr(args, attribute, None)
        if value is not None:
            engine_options[key] = value
    swap_gib = getattr(args, "engine_host_swap_gib", None)
    if swap_gib is not None:
        engine_options["host_swap_bytes"] = int(swap_gib * 1024**3)
    if engine_options:
        eval_config["engine"] = engine_options
    if getattr(args, "grading_workers", None) is not None:
        eval_config["grading_workers"] = args.grading_workers
    if getattr(args, "max_new_tokens", None) is not None:
        eval_config["max_new_tokens"] = args.max_new_tokens
    eval_config["_logger_level"] = args.logger_level
    eval_config["_random_seed"] = getattr(args, "random_seed", 420)

    # Each dataset keeps the established <dataset>_<backend>/<ts>[/TASK]/ layout.
    if args.output_dir is None:
        args.output_dir = Path(args.model_path) / "eval_results"
    args.output_dir = str(args.output_dir)
    dataset_run_dirs = {}
    run_statuses = {}
    for ds_name in dataset_names:
        ds_run_dir = _dataset_run_dir(args.output_dir, ds_name, args.backend, timestamp, dataset_ranges.get(ds_name))
        ds_run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.eval_config, ds_run_dir / Path(args.eval_config).name)
        (ds_run_dir / "effective_config.yaml").write_text(yaml.safe_dump(eval_config, sort_keys=False))
        dataset_run_dirs[ds_name] = ds_run_dir
        run_statuses[ds_name] = {
            "status": "running", "model_path": str(Path(args.model_path).resolve()),
            "backend": args.backend, "dataset": ds_name, "timestamp": timestamp,
            "expected_samples": sum(item["_source_dataset"] == ds_name for item in selected)
                                * eval_config["repeat_size"],
            "completed_samples": 0, "batch_size": eval_config["batch_size"],
            "repeat_size": eval_config["repeat_size"], "random_seed": eval_config["_random_seed"],
        }
        _write_run_status(ds_run_dir, run_statuses[ds_name])
        if answer_types[ds_name] in _CODE_ANSWER_TYPES:
            code_file = ds_run_dir / "code_solutions.jsonl"
            code_file.write_text("", encoding="utf-8")
            print(f"Code solutions file for {ds_name}: {code_file}")
    print(f"Saved evaluation config in {len(dataset_run_dirs)} dataset run director(ies)")

    job_args_list = _build_job_args(
        args, selected, eval_config, dataset_names, timestamp,
        available_gpus, args.tp_size_per_job, field_mapping, dataset_ranges,
    )
    print(f"\nPrepared {len(job_args_list)} jobs for execution")

    completed, failed = _run_jobs(job_args_list)
    failed += max(0, len(job_args_list) - completed - failed)
    print(f"\nDone. ok={completed} failed={failed}")
    if failed > 0:
        print(f"WARNING: {failed} job(s) failed; results may be incomplete.")

    print("\nCombining results from all jobs...")
    try:
        for ds_name, ds_run_dir in dataset_run_dirs.items():
            # Preserve worker artifacts until every worker and coverage check passes.
            combine_job_results(ds_run_dir, args.job_nums, False)
            with (ds_run_dir / "detailed_results.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            actual = [(r["problem_id"], int(r["sample_idx"])) for r in rows]
            expected = {(str(item["id"]), sample) for item in selected
                        if item["_source_dataset"] == ds_name
                        for sample in range(eval_config["repeat_size"])}
            if len(actual) != len(set(actual)) or set(actual) != expected:
                failed = max(1, failed)
            run_statuses[ds_name]["completed_samples"] = len(rows)
            if answer_types[ds_name] in _CODE_ANSWER_TYPES:
                code_solutions_file = ds_run_dir / "code_solutions.jsonl"
                if code_solutions_file.exists() and code_solutions_file.stat().st_size:
                    from tah.evaluate.codeeval import evaluate as code_evaluate
                    n_lines = sum(1 for _ in open(code_solutions_file, encoding="utf-8"))
                    print(f"\nCode eval: {ds_name} on {n_lines} solutions ({code_solutions_file})")
                    grade_started = time.perf_counter()
                    code_evaluate(dataset=ds_name, samples=str(code_solutions_file), allow_subset=True)
                    run_statuses[ds_name]["code_grading_seconds"] = time.perf_counter() - grade_started
                    print(f"Code eval results: {code_solutions_file.with_suffix('.eval_results.json')}")
    except Exception:
        for ds_name, ds_run_dir in dataset_run_dirs.items():
            run_statuses[ds_name].update(status="failed", error=traceback.format_exc())
            _write_run_status(ds_run_dir, run_statuses[ds_name])
        raise
    for ds_name, ds_run_dir in dataset_run_dirs.items():
        run_statuses[ds_name].update(status="failed" if failed else "complete",
                                   completed_jobs=completed, failed_jobs=failed)
        _write_run_status(ds_run_dir, run_statuses[ds_name])
        if not failed and args.del_job_dir:
            for job_id in range(args.job_nums):
                shutil.rmtree(ds_run_dir / f"job_{job_id}", ignore_errors=True)
    if failed:
        raise RuntimeError("Evaluation is incomplete; partial results, status, and worker errors were preserved.")


# ────────────────────────────────────────────────────────────────────────────
# Combining per-job outputs into run-level files
# ────────────────────────────────────────────────────────────────────────────


def combine_job_results(output_dir: Path, job_nums: int, del_job_dir: bool = False) -> None:
    """Aggregate per-job CSV/JSON outputs into combined files under ``output_dir``."""
    all_results: List[dict] = []
    samples_jsonl_path = output_dir / "samples.jsonl"
    samples_jsonl_path.write_text("", encoding="utf-8")  # truncate

    for job_id in range(job_nums):
        job_dir = output_dir / f"job_{job_id}"

        results_file = job_dir / "detailed_results.csv"
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["is_correct"] = row["is_correct"] == "True"
                    row["has_answer"] = row["has_answer"] == "True"
                    row["sample_idx"] = int(row["sample_idx"])
                    row["input_tokens"] = int(row["input_tokens"])
                    row["output_tokens"] = int(row["output_tokens"])
                    row["processing_time"] = float(row["processing_time"])
                    row["mean_passes_per_token"] = float(row["mean_passes_per_token"]) if row.get("mean_passes_per_token") else ""
                    row["fraction_tokens_at_least_2_passes"] = float(row["fraction_tokens_at_least_2_passes"]) if row.get("fraction_tokens_at_least_2_passes") else ""
                    row["pass_stat_tokens"] = int(row.get("pass_stat_tokens") or 0)
                    all_results.append(row)

        details_dir = job_dir / "details"
        if not details_dir.exists():
            continue
        for sample_json in sorted(details_dir.rglob("sample_*.json")):
            with open(sample_json, "r", encoding="utf-8") as f_json:
                obj = json.load(f_json)
            try:
                sample_idx = int(sample_json.stem.split("_")[-1])
            except Exception:
                sample_idx = -1
            obj["id"] = obj.get("problem_id", sample_json.parent.name)
            obj["sample"] = sample_idx
            with open(samples_jsonl_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    by_problem: Dict[str, List[dict]] = {}
    for row in all_results:
        by_problem.setdefault(row["problem_id"], []).append(row)
    problem_stats = {}
    for pid, rows in by_problem.items():
        n = len(rows)
        stat_n = sum(int(r.get("pass_stat_tokens") or 0) for r in rows)
        problem_stats[pid] = {
            "accuracy": sum(1 for r in rows if r["is_correct"]) / n if n else 0.0,
            "correct_count": sum(1 for r in rows if r["is_correct"]),
            "total_samples": n,
            "avg_output_length": sum(r["output_tokens"] for r in rows) / n if n else 0.0,
            "mean_passes_per_token": _weighted_field(rows, "mean_passes_per_token", "pass_stat_tokens"),
            "fraction_tokens_at_least_2_passes": _weighted_field(rows, "fraction_tokens_at_least_2_passes", "pass_stat_tokens"),
            "pass_stat_tokens": stat_n,
        }

    total_n = len(all_results)
    total_c = sum(1 for r in all_results if r["is_correct"])
    total_avg_out = sum(r["output_tokens"] for r in all_results) / total_n if total_n else 0.0
    total_mean_pass = _weighted_field(all_results, "mean_passes_per_token", "pass_stat_tokens")
    total_frac_two = _weighted_field(all_results, "fraction_tokens_at_least_2_passes", "pass_stat_tokens")
    total_pass_tokens = sum(int(r.get("pass_stat_tokens") or 0) for r in all_results)

    stats_file = output_dir / "evaluation_stats.csv"
    with open(stats_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "accuracy", "correct_count", "total_samples", "avg_output_length",
                    "mean_passes_per_token", "fraction_tokens_at_least_2_passes", "pass_stat_tokens"])
        for pid, st in sorted(problem_stats.items()):
            w.writerow([pid, f"{st['accuracy']:.3f}", st["correct_count"], st["total_samples"],
                        f"{st['avg_output_length']:.2f}", _fmt_optional(st["mean_passes_per_token"]),
                        _fmt_optional(st["fraction_tokens_at_least_2_passes"]), st["pass_stat_tokens"]])
        w.writerow([])
        w.writerow([
            "Total Accuracy", f"{(total_c/total_n if total_n else 0):.3f}",
            total_c, total_n, f"{total_avg_out:.2f}", _fmt_optional(total_mean_pass),
            _fmt_optional(total_frac_two), total_pass_tokens,
        ])

    results_file = output_dir / "detailed_results.csv"
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_CSV_FIELDS))
        w.writeheader()
        for r in sorted(all_results, key=lambda x: (x["problem_id"], x["sample_idx"])):
            w.writerow({k: r.get(k, "") for k in _CSV_FIELDS})

    print(f"\nCombined results from {job_nums} jobs")
    print(f"Overall accuracy: {(total_c/total_n if total_n else 0):.4f}")
    print(f"Total problems: {len(problem_stats)}")
    print(f"Combined statistics → {stats_file}")
    print(f"Combined detailed results → {results_file}")

    if del_job_dir:
        for job_id in range(job_nums):
            job_dir = output_dir / f"job_{job_id}"
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        print("Removed per-job directories after combining results")
