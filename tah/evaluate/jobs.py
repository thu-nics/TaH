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
import math
import os
import shutil
import signal
import socket
import time
import traceback
from multiprocessing import Process, Queue
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from tah.evaluate import codeeval, matheval
from tah.evaluate.backends import cleanup, setup_backend
from tah.evaluate.datasets import load_combined_dataset

# Spawn is required for CUDA-capable subprocess workers.
mp.set_start_method("spawn", force=True)


_CSV_FIELDS = (
    "problem_id", "sample_idx", "correct_answer", "predicted_answer",
    "has_answer", "is_correct", "input_tokens", "output_tokens", "processing_time",
)
_CODE_ANSWER_TYPES = frozenset({"livecodebench", "humaneval", "mbpp"})


# ────────────────────────────────────────────────────────────────────────────
# Per-problem prompt building + scoring
# ────────────────────────────────────────────────────────────────────────────


def _build_problem(item: dict, idx: int, field_mapping: Dict, detail_dir: Path) -> dict:
    """Pull standard fields out of an item; also create the per-problem dir."""
    pid = str(item.get(field_mapping["id_field"]) or f"problem_{idx}")
    text = str(item.get(field_mapping["question_field"], "")).strip()
    template = field_mapping.get("prompt_template")
    if template and "{question}" in template:
        text = template.replace("{question}", text)
    answer = str(item.get(field_mapping["answer_field"], "")).strip()
    problem_dir = detail_dir / pid
    problem_dir.mkdir(parents=True, exist_ok=True)
    return {
        "problem_id": pid,
        "original_problem_id": item.get("_original_id", pid),
        "problem_text": text,
        "correct_answer": answer,
        "problem_dir": problem_dir,
        "entry_point": item.get("entry_point"),
    }


def _make_prompt(text: str, tokenizer, *, is_code: bool) -> str:
    if is_code:
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


def _job_output_dir(output_dir: str, combined_dataset_name: str, backend: str,
                    timestamp: str, job_id: int, data_range) -> Path:
    """``<output_dir>/<dataset>_<backend>/<ts>[/TASK_a_b]/job_<id>/``."""
    out = Path(output_dir) / f"{combined_dataset_name}_{backend}" / timestamp
    if data_range:
        a, b = parse_data_range(data_range)
        out = out / f"TASK_{a}_{b}"
    return out / f"job_{job_id}"


def _build_prompts(
    problems: List[dict], tokenizer, repeat_size: int, is_code: bool,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Render each problem ``repeat_size`` times and return
    ``(prompts, prompt_to_problem)`` where ``prompt_to_problem[i] = (problem_idx, sample_idx)``."""
    prompts: List[str] = []
    prompt_to_problem: List[Tuple[int, int]] = []
    for pi, p in enumerate(problems):
        rendered = _make_prompt(p["problem_text"], tokenizer, is_code=is_code)
        for s in range(repeat_size):
            prompts.append(rendered)
            prompt_to_problem.append((pi, s))
    return prompts, prompt_to_problem


def _process_batch(
    batch_outputs: List[Tuple[str, float]],
    batch_offset: int,
    prompts: List[str],
    prompt_to_problem: List[Tuple[int, int]],
    problems: List[dict],
    tokenizer,
    *,
    combined_dataset_name: str,
    is_code: bool,
    unified_code_solutions_file: Optional[Path],
    writer,
    all_results: List[dict],
) -> None:
    """For each (text, time) in ``batch_outputs``, score it, write the
    per-sample json under the problem's dir, append a row to the open CSV
    writer + the in-memory ``all_results`` list."""
    for j, (output_text, proc_time) in enumerate(batch_outputs):
        pi, sample_idx = prompt_to_problem[batch_offset + j]
        p = problems[pi]
        pred, has_ans, ok = _score_one(output_text, p["correct_answer"], combined_dataset_name, is_code=is_code)

        detail = {
            "problem": p["problem_text"],
            "output": output_text,
            "correct_answer": p["correct_answer"],
            "predicted_answer": pred,
            "is_correct": ok,
        }
        if is_code:
            extracted = codeeval.sanitize(output_text, p["entry_point"])
            detail["extracted_code"] = extracted
            detail["entry_point"] = p["entry_point"]
            if unified_code_solutions_file is not None:
                _append_code_solution(unified_code_solutions_file, p["original_problem_id"], extracted)
        with open(p["problem_dir"] / f"sample_{sample_idx}.json", "w", encoding="utf-8") as fd:
            json.dump(detail, fd, ensure_ascii=False, indent=2)

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
        }
        writer.writerow(row)
        all_results.append(row)


def run_single_job(
    *, config: Dict, combined_dataset_name: str, output_dir: str, timestamp: str,
    model_path: str, job_id: int, job_nums: int, start_idx: int, end_idx: int,
    tp_size: int, backend: str, data_range, problems_data, field_mapping: Dict,
    unified_code_solutions_file: Optional[Path],
) -> None:
    """Run inference for one job (one process) and write per-problem files."""
    from transformers import AutoTokenizer

    out = _job_output_dir(output_dir, combined_dataset_name, backend, timestamp, job_id, data_range)
    detail_dir = out / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    print(f"Job {job_id+1}/{job_nums}: {len(problems_data)} problems, backend={backend}, datasets={combined_dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, infer = setup_backend(backend, config, model_path, tokenizer, tp_size=tp_size)

    is_code = field_mapping.get("answer_type", "boxed") in _CODE_ANSWER_TYPES
    problems = [
        _build_problem(item, item.get("_original_index", start_idx + i), field_mapping, detail_dir)
        for i, item in enumerate(problems_data)
    ]
    prompts, prompt_to_problem = _build_prompts(problems, tokenizer, config["repeat_size"], is_code)
    print(f"Processing batch_size={config['batch_size']} repeat_size={config['repeat_size']} → {len(prompts)} prompts")

    results_file = out / "detailed_results.csv"
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    all_results: List[dict] = []
    bs = config["batch_size"]
    for bi in tqdm(range(math.ceil(len(prompts) / bs)),
                   desc=f"Job {job_id} batches", position=job_id, leave=True):
        s, e = bi * bs, min((bi + 1) * bs, len(prompts))
        batch_outputs = infer(prompts[s:e])
        with open(results_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            _process_batch(
                batch_outputs, batch_offset=s,
                prompts=prompts, prompt_to_problem=prompt_to_problem,
                problems=problems, tokenizer=tokenizer,
                combined_dataset_name=combined_dataset_name, is_code=is_code,
                unified_code_solutions_file=unified_code_solutions_file,
                writer=writer, all_results=all_results,
            )

    _write_job_stats(out, all_results)
    print(f"Job {job_id} done; accuracy={_overall_accuracy(all_results):.4f}")
    cleanup(model, backend)


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
        rows.append((pid, c / n if n else 0.0, c, n, avg_out))

    total_n = len(all_results)
    total_c = sum(1 for r in all_results if r["is_correct"])
    total_avg_out = sum(r["output_tokens"] for r in all_results) / total_n if total_n else 0.0

    with open(out / "evaluation_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "accuracy", "correct_count", "total_samples", "avg_output_length"])
        for pid, acc, c, n, avg in rows:
            w.writerow([pid, f"{acc:.3f}", c, n, f"{avg:.2f}"])
        w.writerow([])
        w.writerow(["Total Accuracy", f"{(total_c/total_n if total_n else 0):.3f}", total_c, total_n, f"{total_avg_out:.2f}"])


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
    (job_id, config, combined_dataset_name, output_dir, timestamp, model_path,
     job_nums, start_idx, end_idx, tp_size, backend, data_range, gpu_devices,
     problems_data, field_mapping, unified_code_solutions_file) = job_args

    _setup_logging(config.get("_logger_level") or "WARNING")

    port = _free_port(start=29555 + job_id * 100)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["SGLANG_NCCL_PORT"] = str(port)

    import torch
    from tah.model.utils import set_all_seeds
    set_all_seeds(config.get("_random_seed", 420))
    if torch.cuda.is_available():
        torch.cuda.init()

    try:
        print(f"\nJob {job_id}: GPUs={gpu_devices}, NCCL port={port}, indices {start_idx}..{end_idx-1}")
        run_single_job(
            config=config, combined_dataset_name=combined_dataset_name,
            output_dir=output_dir, timestamp=timestamp, model_path=model_path,
            job_id=job_id, job_nums=job_nums, start_idx=start_idx, end_idx=end_idx,
            tp_size=tp_size, backend=backend, data_range=data_range,
            problems_data=problems_data, field_mapping=field_mapping,
            unified_code_solutions_file=unified_code_solutions_file,
        )
        result_queue.put((job_id, True, "ok"))
    except Exception as e:
        result_queue.put((job_id, False, f"Job {job_id} failed: {e}\n{traceback.format_exc()}"))
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


def _select_problems(args, dataset: List[dict]) -> List:
    """Resolve --data_ids or --data_range into a list of items / indices to evaluate."""
    if getattr(args, "data_ids", None):
        valid_ids = {str(item.get("id")) for item in dataset}
        seen, out = set(), []
        for pid in (s.strip() for s in str(args.data_ids).split(",") if s.strip()):
            if pid in valid_ids and pid not in seen:
                seen.add(pid)
                out.append(pid)
        if not out:
            raise ValueError("--data_ids matched no problem IDs (use <dataset>_<original_id>)")
        return out
    a, b = parse_data_range(args.data_range, len(dataset))
    return list(range(a, b))


def _build_job_args(args, dataset, selected, eval_config, combined_dataset_name, timestamp,
                    available_gpus, gpus_per_job, field_mapping, unified_code_solutions_file):
    """Materialise per-job tuples for _run_job_process. Splits ``selected`` interleaved across jobs."""
    by_id = {str(item.get("id")): item for item in dataset} if getattr(args, "data_ids", None) else None
    job_args_list = []
    for job_id in range(args.job_nums):
        slice_ = selected[job_id::args.job_nums]
        if not slice_:
            continue
        gpu_start = job_id * gpus_per_job
        job_gpus = available_gpus[gpu_start:gpu_start + gpus_per_job]
        if by_id is not None:
            job_problems = [by_id[str(p)] for p in slice_ if str(p) in by_id]
        else:
            job_problems = []
            for i in slice_:
                item = dict(dataset[i])
                item["_original_index"] = i
                job_problems.append(item)
        job_args_list.append((
            job_id, eval_config, combined_dataset_name, args.output_dir, timestamp,
            args.model_path, args.job_nums, 0, len(slice_), gpus_per_job, args.backend,
            args.data_range, job_gpus, job_problems, field_mapping, unified_code_solutions_file,
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
        p.start()
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
    """Top-level: split selected problems across jobs, run in parallel, then combine."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"Running {args.job_nums} jobs with {args.tp_size_per_job} GPU(s) per job")

    # GPU pool comes from CUDA_VISIBLE_DEVICES (or 0..7 if unset).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    available_gpus = [int(x.strip()) for x in visible.split(",") if x.strip()] or list(range(8))
    print(f"Available GPU devices: {available_gpus}")

    dataset_names = [n.strip() for n in args.dataset_name.split(",")]
    dataset, field_mapping = load_combined_dataset(dataset_names)
    combined_dataset_name = "_".join(dataset_names)

    selected = _select_problems(args, dataset)

    # Load + tag YAML config (worker reads back the tags via config["_..."]).
    with open(args.eval_config, "r") as f:
        eval_config = yaml.safe_load(f)
    eval_config["_logger_level"] = args.logger_level
    eval_config["_random_seed"] = getattr(args, "random_seed", 420)

    # Output dir: <output_dir or model_path/eval_results>/<dataset>_<backend>/<ts>[/TASK_a_b]/
    if args.output_dir is None:
        args.output_dir = Path(args.model_path) / "eval_results"
    suffix = ""
    if args.data_range and not getattr(args, "data_ids", None):
        a, b = parse_data_range(args.data_range, len(dataset))
        suffix = f"TASK_{a}_{b}"
    combined_dir = Path(args.output_dir) / f"{combined_dataset_name}_{args.backend}" / timestamp
    if suffix:
        combined_dir = combined_dir / suffix
    combined_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.eval_config, combined_dir / Path(args.eval_config).name)
    print(f"Saved evaluation config to: {combined_dir / Path(args.eval_config).name}")

    is_code = field_mapping.get("answer_type", "boxed") in _CODE_ANSWER_TYPES
    code_solutions_file: Optional[Path] = None
    if is_code:
        code_solutions_file = combined_dir / "code_solutions.jsonl"
        code_solutions_file.write_text("", encoding="utf-8")
        print(f"Code solutions file: {code_solutions_file}")

    job_args_list = _build_job_args(
        args, dataset, selected, eval_config, combined_dataset_name, timestamp,
        available_gpus, args.tp_size_per_job, field_mapping, code_solutions_file,
    )
    print(f"\nPrepared {len(job_args_list)} jobs for execution")

    completed, failed = _run_jobs(job_args_list)
    print(f"\nDone. ok={completed} failed={failed}")
    if failed > 0:
        print(f"WARNING: {failed} job(s) failed; results may be incomplete.")

    print("\nCombining results from all jobs...")
    combine_job_results(combined_dir, len(job_args_list), args.del_job_dir)

    if is_code and code_solutions_file is not None and code_solutions_file.exists():
        from tah.evaluate.codeeval import evaluate as code_evaluate
        evalplus_dataset = field_mapping["answer_type"] if field_mapping["answer_type"] in ("humaneval", "mbpp") else "humaneval"
        n_lines = sum(1 for _ in open(code_solutions_file))
        print(f"\nCode eval: {combined_dataset_name} on {n_lines} solutions ({code_solutions_file})")
        code_evaluate(dataset=evalplus_dataset, samples=str(code_solutions_file))
        print(f"Code eval results: {str(code_solutions_file).replace('.jsonl', '.eval_results.json')}")


# ────────────────────────────────────────────────────────────────────────────
# Combining per-job outputs into run-level files
# ────────────────────────────────────────────────────────────────────────────


def combine_job_results(output_dir: Path, job_nums: int, del_job_dir: bool = False) -> None:
    """Aggregate per-job CSV/JSON outputs into combined files under ``output_dir``."""
    all_results: List[dict] = []
    problem_stats: Dict[str, dict] = {}
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
                    all_results.append(row)

        stats_file = job_dir / "evaluation_stats.csv"
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # header
                for row in reader:
                    if not row or row[0] in ("", "Total Accuracy"):
                        continue
                    problem_stats[row[0]] = {
                        "accuracy": f"{float(row[1]):.3f}",
                        "correct_count": int(row[2]),
                        "total_samples": int(row[3]),
                        "avg_output_length": float(row[4]) if len(row) >= 5 else 0.0,
                    }

        details_dir = job_dir / "details"
        if not details_dir.exists():
            continue
        for problem_dir in details_dir.iterdir():
            if not problem_dir.is_dir():
                continue
            for sample_json in sorted(problem_dir.glob("sample_*.json")):
                with open(sample_json, "r", encoding="utf-8") as f_json:
                    obj = json.load(f_json)
                try:
                    sample_idx = int(sample_json.stem.split("_")[-1])
                except Exception:
                    sample_idx = -1
                obj["id"] = problem_dir.name
                obj["sample"] = sample_idx
                with open(samples_jsonl_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Recompute per-problem avg_output_length using all_results (more authoritative).
    by_pid: Dict[str, List[int]] = {}
    for r in all_results:
        by_pid.setdefault(r["problem_id"], []).append(r["output_tokens"])
    for pid, lens in by_pid.items():
        if pid in problem_stats:
            problem_stats[pid]["avg_output_length"] = sum(lens) / len(lens)

    total_n = len(all_results)
    total_c = sum(1 for r in all_results if r["is_correct"])
    total_avg_out = sum(r["output_tokens"] for r in all_results) / total_n if total_n else 0.0

    stats_file = output_dir / "evaluation_stats.csv"
    with open(stats_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "accuracy", "correct_count", "total_samples", "avg_output_length"])
        for pid, st in sorted(problem_stats.items()):
            w.writerow([pid, st["accuracy"], st["correct_count"], st["total_samples"], f"{st['avg_output_length']:.2f}"])
        w.writerow([])
        w.writerow([
            "Total Accuracy", f"{(total_c/total_n if total_n else 0):.3f}",
            total_c, total_n, f"{total_avg_out:.2f}",
        ])

    results_file = output_dir / "detailed_results.csv"
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_CSV_FIELDS))
        w.writeheader()
        for r in sorted(all_results, key=lambda x: (x["problem_id"], x["sample_idx"])):
            w.writerow({k: r[k] for k in _CSV_FIELDS})

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
