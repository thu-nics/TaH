"""Online evaluation through remote mini-sglang API servers.

Each completed HTTP request is evaluated and saved to disk immediately,
so partial progress is preserved across crashes/interruptions.

The model lives in the mini-sglang server, so this script only needs decode
parameters as plain CLI flags. Both evaluation launchers set their defaults
directly in the bash scripts.

Usage:
    python script2/eval/eval_online.py \
    --model_path output/checkpoint \
    --base_urls http://127.0.0.1:30010 http://127.0.0.1:30011 http://127.0.0.1:30012 http://127.0.0.1:30013 http://127.0.0.1:30014 http://127.0.0.1:30015 http://127.0.0.1:30016 http://127.0.0.1:30017 \
    --datasets aime26 \
    --per_server_concurrency 6 \
    --repeat_size 16 \
    --max_new_tokens 16384
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

# Server URLs are always direct addresses (localhost or peer boxes); a bare
# urlopen honors an inherited http_proxy even for 127.0.0.1, turning every
# request into an instant 503. Bypass
# proxies unconditionally.
_opener = build_opener(ProxyHandler({}))

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from tah2.evaluate.common import (
    CODE_ANSWER_TYPES,
    _maybe_rename_with_accuracy,
    _save_job_stats,
    load_iter_count_distribution,
    load_datasets_with_config,
    parse_data_range,
    save_iter_count_distribution,
    save_results_json,
    update_iter_count_distribution,
)

ONLINE_BACKEND_NAME = "mini_sglang_online"


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def _server_info(base_url: str) -> Dict:
    """What the server actually serves (/v1/models): sampling_seed, iter_mode
    ("adaptive" TaH decider vs "fixed" depth) and fixed_depth. Missing fields
    (older server) come back as None; the seed falls back to this process's env."""
    info: Dict = {"sampling_seed": None, "iter_mode": None, "fixed_depth": None}
    try:
        with _opener.open(f"{base_url.rstrip('/')}/v1/models", timeout=5) as r:
            payload = json.loads(r.read())
        for k in info:
            if payload.get(k) is not None:
                info[k] = payload[k]
    except Exception:
        pass
    if info["sampling_seed"] is None:
        env = os.environ.get("MINISGL_SAMPLING_SEED")
        info["sampling_seed"] = int(env) if env else None
    return info


def _wait_for_server(base_url: str, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/v1/models"
    while time.time() < deadline:
        try:
            with _opener.open(url, timeout=5) as r:
                if 200 <= r.status < 300:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Server at {base_url} not ready within {timeout}s")


def _post_chat_completion(base_url: str, payload: Dict, timeout: int) -> Tuple[str, float, Dict]:
    """POST to /v1/chat/completions; return (text, elapsed, usage_dict).

    mini-sglang's /v1/chat/completions accepts a raw "prompt" string in addition
    to a "messages" list, so we pass the pre-formatted chat template string directly.
    usage carries per-token "iter_counts" (response) and "prompt_iter_counts".
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with _opener.open(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except HTTPError as e:
        raise RuntimeError(f"{url} HTTP {e.code}: {e.read().decode(errors='ignore')}") from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e

    elapsed = time.perf_counter() - t0
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage") or {}
    return content, elapsed, usage


def _iter_counts_to_tracker_records(iter_counts) -> List[Dict]:
    if not iter_counts:
        return []
    return [{"iter_depth": d} for cnt in iter_counts for d in range(max(int(cnt), 1))]


def _is_tah_model(model_path: str) -> bool:
    p = Path(model_path) / "tah_config.json"
    if not p.exists():
        return False
    with open(p, "r", encoding="utf-8") as f:
        return int(json.load(f).get("max_iter", 1)) > 1


def _get_tah_threshold_tag(model_path: str, tah_iter_threshold: Optional[float]) -> Optional[str]:
    if tah_iter_threshold is None:
        return None
    tah_config_path = Path(model_path) / "tah_config.json"
    if not tah_config_path.exists():
        return None
    with open(tah_config_path, "r", encoding="utf-8") as f:
        tah_config = json.load(f)
    if int(tah_config.get("max_iter", 1)) <= 1:
        return None
    return f"thr{tah_iter_threshold}"


def _print_iter_count_distribution(combo_out_dir: Path) -> None:
    iter_dist = load_iter_count_distribution(combo_out_dir / "iter_count_distribution.csv")
    total = sum(iter_dist.values())
    if total <= 0:
        return
    parts = [f"{ic}:{iter_dist[ic]}" for ic in sorted(iter_dist) if iter_dist[ic] > 0]
    avg_iter_count = sum(ic * cnt for ic, cnt in iter_dist.items()) / total
    print(f"iter_count_distribution ({combo_out_dir.name}): {' '.join(parts)} | avg_iter_count={avg_iter_count:.4f}")


def _online_runtime_metrics(
    start_time: float,
    completed: int,
    failed: int,
    concurrency: int,
    output_tokens: int,
) -> Dict:
    wall_time = max(time.perf_counter() - start_time, 0.0)
    return {
        "end_2_end_time": round(wall_time, 4),
        "request_throught": round(completed / wall_time, 4) if wall_time > 0 else 0.0,
        "token_throughtput": round(output_tokens / wall_time, 4) if wall_time > 0 else 0.0,
        "total_concurrency": concurrency,
        "failed_requests": failed,
    }


# ---------------------------------------------------------------------------
# Resume helpers  (true_server has a flat output layout, no job subdirs)
# ---------------------------------------------------------------------------

def _resolve_resume_combo_dir(resume_dir: Path, dataset_name: str, threshold_tag: Optional[str]) -> Optional[Path]:
    """Find the most recent output dir for a dataset under resume_dir."""
    if resume_dir.exists() and (resume_dir / "detailed_results.csv").exists():
        parent_name = resume_dir.parent.name
        grandparent_name = resume_dir.parent.parent.name if resume_dir.parent.parent else ""
        if parent_name == dataset_name and (threshold_tag is None or grandparent_name == threshold_tag):
            return resume_dir

    root = resume_dir / threshold_tag if threshold_tag is not None else resume_dir
    combo_dir = root / dataset_name
    if not combo_dir.exists():
        return None
    candidates = [d for d in combo_dir.iterdir() if d.is_dir()]
    return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None


def _scan_completed_problems(combo_out_dir: Path, repeat_size: int) -> set:
    """Return problem_ids with all repeat_size rows. Partial problems are kept:
    _process_items_realtime skips the (problem_id, sample_idx) rows that exist,
    so a resume only re-runs the missing samples."""
    results_file = combo_out_dir / "detailed_results.csv"
    if not results_file.exists():
        return set()

    counts: Dict[str, int] = {}
    with open(results_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("problem_id")
            if pid:
                counts[pid] = counts.get(pid, 0) + 1

    completed = {pid for pid, n in counts.items() if n == repeat_size}
    partial = sum(repeat_size - n for n in counts.values() if n != repeat_size)
    if partial:
        print(f"Resume: {partial} missing sample(s) across {len(counts) - len(completed)} partial problem(s).")
    # evaluation_stats.csv is rebuilt from the full CSV; results.json is kept so
    # _finalize_outputs can add the earlier session's wall clock to this one.
    (combo_out_dir / "evaluation_stats.csv").unlink(missing_ok=True)
    return completed


def _load_existing_results(combo_out_dir: Path) -> Dict:
    """Reconstruct batch_result from an already-complete detailed_results.csv."""
    all_results = []
    with open(combo_out_dir / "detailed_results.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["sample_idx"] = int(row["sample_idx"])
            row["has_answer"] = row["has_answer"] == "True"
            row["is_correct"] = row["is_correct"] == "True"
            row["input_tokens"] = int(row["input_tokens"])
            row["output_tokens"] = int(row["output_tokens"])
            row["processing_time"] = float(row["processing_time"])
            all_results.append(row)

    by_problem: Dict[str, list] = {}
    for r in all_results:
        by_problem.setdefault(r["problem_id"], []).append(r)
    problem_stats = {
        pid: {
            "accuracy": f"{sum(1 for r in rows if r['is_correct']) / len(rows):.3f}",
            "correct_count": sum(1 for r in rows if r["is_correct"]),
            "total_samples": len(rows),
            "avg_output_length": sum(r["output_tokens"] for r in rows) / len(rows),
        }
        for pid, rows in by_problem.items()
    }
    return {"all_results": all_results, "problem_stats": problem_stats}


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------

def _finalize_outputs(combo_out_dir: Path, batch_result: Dict, config: Dict) -> None:
    """Write evaluation_stats.csv and refresh results.json from current flat outputs."""
    _save_job_stats(combo_out_dir, batch_result)
    fixed_depth = config.get("fixed_depth") if config.get("iter_mode") == "fixed" else None
    if not config["save_iter_counts"]:
        iter_counts_file = combo_out_dir / "iter_counts.jsonl"
        if iter_counts_file.exists():
            iter_counts_file.unlink()
    if fixed_depth:
        avg_iter_count = float(fixed_depth)
        dist = combo_out_dir / "iter_count_distribution.csv"
        if dist.exists():
            dist.unlink()
    else:
        avg_iter_count = save_iter_count_distribution(
            combo_out_dir / "iter_count_distribution.csv",
            load_iter_count_distribution(combo_out_dir / "iter_count_distribution.csv"),
        )
    extra = batch_result.get("runtime_metrics")
    wall_keys = ("end_2_end_time", "request_throught", "token_throughtput", "total_concurrency", "failed_requests")
    results_json = combo_out_dir / "results.json"
    old = json.load(results_json.open(encoding="utf-8")) if results_json.exists() else {}
    if extra and "end_2_end_time" in old:
        # Resumed run: wall clock = earlier session(s) + this one; throughput over
        # the whole CSV (the sessions may have used different fleets).
        with (combo_out_dir / "detailed_results.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        wall = old["end_2_end_time"] + extra["end_2_end_time"]
        extra = {**extra, "end_2_end_time": round(wall, 4),
                 "request_throught": round(len(rows) / wall, 4) if wall else 0.0,
                 "token_throughtput": round(sum(int(r["output_tokens"]) for r in rows) / wall, 4) if wall else 0.0,
                 "wall_clock_sessions": old.get("wall_clock_sessions", 1) + 1}
    elif not extra and old:
        # Re-finalizing without live timing (e.g. re-judging): keep the recorded
        # wall clock -- save_results_json's fallback would report the SUM of
        # per-request times as end_2_end_time.
        extra = {k: old[k] for k in wall_keys + ("wall_clock_sessions",) if k in old}
    extra = dict(extra or {})
    if config.get("iter_mode"):
        extra["iter_mode"] = config["iter_mode"]
    save_results_json(
        combo_out_dir / "results.json",
        combo_out_dir / "detailed_results.csv",
        avg_iter_count=avg_iter_count,
        extra_metrics=extra,
    )


# ---------------------------------------------------------------------------
# Core: real-time per-result evaluation and saving
# ---------------------------------------------------------------------------

def _process_items_realtime(
    dataset_name: str,
    items: List[Dict],
    config: Dict,
    combo_out_dir: Path,
    field_mapping: Dict,
    tokenizer,
    base_urls: List[str],
    model_name: str,
    request_timeout: int,
) -> Dict:
    """Fire all requests concurrently; evaluate and save each result as it completes.

    Returns batch_result dict with all_results and problem_stats.
    """
    import tah2.evaluate.matheval as matheval

    is_code = field_mapping.get("answer_type") in CODE_ANSWER_TYPES
    if is_code:
        import tah2.evaluate.codeeval as codeeval
        evaluator = None
    else:
        evaluator = matheval.evaluator_map[dataset_name]

    results_file = combo_out_dir / "detailed_results.csv"
    samples_file = combo_out_dir / "samples.jsonl"
    iter_counts_file = combo_out_dir / "iter_counts.jsonl"
    iter_dist_file = combo_out_dir / "iter_count_distribution.csv"
    results_json_file = combo_out_dir / "results.json"
    fieldnames = ["problem_id", "sample_idx", "correct_answer", "predicted_answer",
                  "has_answer", "is_correct", "input_tokens", "output_tokens", "processing_time"]
    file_exists = results_file.exists() and results_file.stat().st_size > 0
    samples_exists = samples_file.exists() and samples_file.stat().st_size > 0
    done_samples: set = set()
    if file_exists:  # resume: rows already on disk (sample-level)
        with open(results_file, "r", encoding="utf-8") as f:
            done_samples = {(r["problem_id"], int(r["sample_idx"])) for r in csv.DictReader(f)}
    concurrency = config["per_server_concurrency"] * len(base_urls)
    save_iter_counts = config["save_iter_counts"]
    if not save_iter_counts and iter_counts_file.exists():
        iter_counts_file.unlink()
    iter_dist = load_iter_count_distribution(iter_dist_file)

    # Build task list: one entry per (item, sample_idx)
    tasks: List[Tuple] = []
    for item in items:
        problem_id = str(item.get("id", ""))
        problem_text = str(item.get("question", "")).strip()
        correct_answer = str(item.get("answer", "")).strip()

        if is_code:
            prompt = codeeval.make_raw_chat_prompt_for_code_evaluation(
                task_prompt=problem_text, assertion=item.get("assertion", ""),
                answer_type=field_mapping["answer_type"], reasoning=True, tokenizer=tokenizer,
                starter_code=item.get("_original_starter_code", ""),
            )
        else:
            messages = [{"role": "user", "content": problem_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        payload: Dict = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": config["max_new_tokens"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
        }

        for sample_idx in range(config["repeat_size"]):
            if (problem_id, sample_idx) in done_samples:
                continue
            tasks.append((problem_id, problem_text, correct_answer, sample_idx, payload, item))

    # Shuffle before dispatch.
    random.Random(f"{dataset_name}:{config['repeat_size']}").shuffle(tasks)

    all_results: List[Dict] = []
    completed_results = 0
    correct_results = 0
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for existing in csv.DictReader(f):
                completed_results += 1
                if existing.get("is_correct") == "True":
                    correct_results += 1
    initial_completed_results = completed_results
    run_output_tokens = 0
    runtime_metrics: Dict = {}

    run_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_post_chat_completion, base_urls[i % len(base_urls)], task[4], request_timeout):
            (task[0], task[1], task[2], task[3], task[5])
            for i, task in enumerate(tasks)
        }

        _last_save_ts = time.time()
        fixed_depth = config.get("fixed_depth") if config.get("iter_mode") == "fixed" else None
        avg_iter_count: float = float(fixed_depth) if fixed_depth else 0.0
        if fixed_depth and iter_dist_file.exists():
            iter_dist_file.unlink()
        failed_results = 0
        error_log_file = combo_out_dir / "errors.log"
        with tqdm(total=len(tasks), desc=dataset_name, unit="req") as pbar, \
                open(samples_file, "a" if samples_exists else "w", encoding="utf-8") as samples_fp:
            for future in as_completed(futures):
                problem_id, problem_text, correct_answer, sample_idx, item = futures[future]
                try:
                    output_text, elapsed, usage = future.result()
                    resp_iter_counts = usage.get("iter_counts") or []
                    prompt_iter_counts = usage.get("prompt_iter_counts") or []
                    tracker_records = _iter_counts_to_tracker_records(resp_iter_counts)

                    # Evaluate
                    if is_code:
                        predicted_answer, has_answer, is_correct = "pending_code_eval", False, False
                    else:
                        res = evaluator.rule_judge(output_text, correct_answer)
                        is_correct = res[0]
                        if res[1] == "No extracted answer":
                            predicted_answer, has_answer = "", False
                        else:
                            predicted_answer, has_answer = res[1], True

                    input_tokens = len(tokenizer.encode(problem_text))
                    output_tokens = len(tokenizer.encode(output_text))
                    run_output_tokens += output_tokens

                    row = {
                        "problem_id": problem_id, "sample_idx": sample_idx,
                        "correct_answer": correct_answer, "predicted_answer": predicted_answer,
                        "has_answer": has_answer, "is_correct": is_correct,
                        "input_tokens": input_tokens, "output_tokens": output_tokens,
                        "processing_time": elapsed,
                    }

                    detail = {"problem": problem_text, "output": output_text, "correct_answer": correct_answer,
                              "predicted_answer": predicted_answer, "is_correct": is_correct}
                    if not is_code:
                        detail["evaluation_method"] = getattr(
                            evaluator, "evaluation_method", "rule_judge"
                        )
                    if is_code:
                        from tah2.evaluate.local_eval import extract_code
                        detail["entry_point"] = item.get("entry_point", "")
                        detail["extracted_code"] = extract_code(output_text, detail["entry_point"])
                        detail["original_id"] = item.get("_original_id", problem_id)
                        if item.get("_dataset_version"):
                            detail["dataset_version"] = item["_dataset_version"]
                    detail["id"] = problem_id
                    detail["sample"] = sample_idx
                    samples_fp.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    samples_fp.flush()

                    if save_iter_counts and resp_iter_counts:
                        with open(iter_counts_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "id": problem_id,
                                "sample": sample_idx,
                                "prompt_iter_counts": [int(c) for c in prompt_iter_counts],
                                "response_iter_counts": [int(c) for c in resp_iter_counts],
                            }, ensure_ascii=False) + "\n")
                    if not fixed_depth:
                        update_iter_count_distribution(iter_dist, tracker_records)
                        avg_iter_count = save_iter_count_distribution(iter_dist_file, iter_dist)

                    # Append to detailed_results.csv immediately (sequential loop, no lock needed)
                    with open(results_file, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                            file_exists = True
                        writer.writerow(row)

                    all_results.append(row)
                    completed_results += 1
                    correct_results += int(is_correct)
                except Exception as exc:
                    # A single bad future used to escape and trigger executor.__exit__'s
                    # wait, draining tens of thousands of in-flight requests with the
                    # traceback hidden for ~40min. Catch here so the run continues.
                    failed_results += 1
                    tb = traceback.format_exc()
                    sys.stderr.write(
                        f"[eval_online] problem_id={problem_id!r} sample_idx={sample_idx} "
                        f"failed: {type(exc).__name__}: {exc}\n{tb}"
                    )
                    sys.stderr.flush()
                    try:
                        with open(error_log_file, "a", encoding="utf-8") as ef:
                            ef.write(
                                json.dumps({
                                    "problem_id": problem_id,
                                    "sample_idx": sample_idx,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                    "traceback": tb,
                                }, ensure_ascii=False) + "\n"
                            )
                    except Exception:
                        pass

                # Throttle: re-reading the full detailed_results.csv on every row is O(N²)
                # over the run. Refresh results.json every 200 rows or every 30 s.
                if completed_results and (
                    completed_results % 200 == 0 or (time.time() - _last_save_ts) > 30
                ):
                    save_results_json(
                        results_json_file,
                        results_file,
                        avg_iter_count=avg_iter_count,
                        extra_metrics=_online_runtime_metrics(
                            run_start,
                            completed_results - initial_completed_results,
                            failed_results,
                            concurrency,
                            run_output_tokens,
                        ),
                    )
                    _last_save_ts = time.time()
                pbar.update(1)
                pbar.set_postfix_str(
                    f"correct={correct_results} fail={failed_results}"
                )

        # Final summary after the consumer loop exits — guarantees results.json
        # reflects every row when the dataset finishes.
        runtime_metrics = _online_runtime_metrics(
            run_start,
            completed_results - initial_completed_results,
            failed_results,
            concurrency,
            run_output_tokens,
        )
        if not is_code:
            runtime_metrics["evaluation_method"] = getattr(
                evaluator, "evaluation_method", "rule_judge"
            )
        save_results_json(
            results_json_file,
            results_file,
            avg_iter_count=avg_iter_count,
            extra_metrics=runtime_metrics,
        )
        if failed_results:
            print(
                f"[eval_online] {dataset_name}: {failed_results} request(s) failed; "
                f"see {error_log_file}"
            )

    # Build problem_stats
    by_problem: Dict[str, list] = {}
    for r in all_results:
        by_problem.setdefault(r["problem_id"], []).append(r)
    problem_stats = {
        pid: {
            "accuracy": f"{sum(1 for r in rows if r['is_correct']) / len(rows):.3f}",
            "correct_count": sum(1 for r in rows if r["is_correct"]),
            "total_samples": len(rows),
            "avg_output_length": sum(r["output_tokens"] for r in rows) / len(rows),
        }
        for pid, rows in by_problem.items()
    }
    return {"all_results": all_results, "problem_stats": problem_stats, "runtime_metrics": runtime_metrics}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_online_eval(args):
    eval_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "repeat_size": args.repeat_size,
        "per_server_concurrency": args.per_server_concurrency,
        "save_iter_counts": args.save_iter_counts,
    }

    base_urls = [u.strip().rstrip("/") for u in args.base_urls if u.strip()]
    if not base_urls:
        raise ValueError("At least one --base_urls required.")
    if args.wait_for_server:
        for url in base_urls:
            _wait_for_server(url, timeout=args.wait_timeout)

    model_path = str(Path(args.model_path).expanduser().resolve())
    server_info = _server_info(base_urls[0])
    eval_config["iter_mode"] = server_info["iter_mode"]
    eval_config["fixed_depth"] = server_info["fixed_depth"]
    # Per-token iter counts only mean something when the depth is adaptive
    # (TaH decider). Fixed-depth models (uniform loops or single-pass base)
    # keep a single avg_iter_count = fixed_depth line in results.json instead.
    if eval_config["iter_mode"] == "fixed":
        eval_config["save_iter_counts"] = False
        print(f"iter_mode: fixed (depth {eval_config['fixed_depth']}); per-token iter counts not saved")
    elif eval_config["save_iter_counts"] is None:
        eval_config["save_iter_counts"] = _is_tah_model(model_path)
        if eval_config["save_iter_counts"]:
            print("save_iter_counts: on (TaH checkpoint; pass --no-save_iter_counts to disable)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_out_dir = Path(args.output_dir) if args.output_dir else Path(model_path) / "eval_results"
    resume_root = Path(args.resume_dir) if args.resume_dir else None
    if resume_root is not None and not resume_root.exists():
        raise ValueError(f"Resume directory does not exist: {resume_root}")

    datasets_map: Dict = json.loads(args.datasets_map) if args.datasets_map else {}
    dataset_names = list(datasets_map.keys()) if datasets_map else (args.datasets or [])
    if not dataset_names:
        raise ValueError("Provide --datasets or --datasets_map.")

    model_name = args.model_name or model_path

    print(f"Servers: {base_urls}")
    for dataset_name in dataset_names:
        dataset_list, field_mapping = load_datasets_with_config([dataset_name])
        total = len(dataset_list)
        start, end = parse_data_range(args.data_range, total)
        ds_overrides = datasets_map.get(dataset_name, {})
        combo_config = {**eval_config, **ds_overrides}
        # The decider threshold only names adaptive (TaH) runs; fixed-depth
        # models (uniform loops) go straight under eval_results/<bench>/.
        threshold_tag = (None if eval_config.get("iter_mode") == "fixed"
                         else _get_tah_threshold_tag(model_path, args.tah_iter_threshold))

        # Resolve output dir
        if resume_root is not None:
            combo_out_dir = _resolve_resume_combo_dir(resume_root, dataset_name, threshold_tag)
            if combo_out_dir is None:
                combo_out_dir = base_out_dir / dataset_name / timestamp
                if threshold_tag is not None:
                    combo_out_dir = base_out_dir / threshold_tag / dataset_name / timestamp
            else:
                print(f"Resume: {dataset_name} → {combo_out_dir}")
        else:
            combo_out_dir = base_out_dir / dataset_name / timestamp
            if threshold_tag is not None:
                combo_out_dir = base_out_dir / threshold_tag / dataset_name / timestamp
        combo_out_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        cfg_snap = {
            **combo_config,
            "online_base_urls": base_urls,
            "online_backend": ONLINE_BACKEND_NAME,
            "total_concurrency": combo_config["per_server_concurrency"] * len(base_urls),
            "sampling_seed": server_info["sampling_seed"],
        }
        if dataset_list and dataset_list[0].get("_dataset_version"):
            cfg_snap["dataset_version"] = dataset_list[0]["_dataset_version"]
        with open(combo_out_dir / "eval_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg_snap, f, default_flow_style=False, allow_unicode=True)

        # Resume: find already completed problems
        completed_ids = _scan_completed_problems(combo_out_dir, combo_config["repeat_size"])
        if completed_ids:
            print(f"Resume: {dataset_name} — {len(completed_ids)} problem(s) already done.")

        items = [dict(dataset_list[i], _original_index=i) for i in range(start, end)]
        pending = [item for item in items if str(item.get("id", "")) not in completed_ids]

        if not pending:
            print(f"All problems done for {dataset_name}.")
            batch_result = _load_existing_results(combo_out_dir)
            _finalize_outputs(combo_out_dir, batch_result, combo_config)
            _print_iter_count_distribution(combo_out_dir)
            combo_out_dir = _maybe_rename_with_accuracy(combo_out_dir, field_mapping)
            print(f"✓ {dataset_name}  →  {combo_out_dir}")
            continue

        resumed = (combo_out_dir / "detailed_results.csv").exists()
        # Process with real-time saving
        batch_result = _process_items_realtime(
            dataset_name=dataset_name,
            items=pending,
            config=combo_config,
            combo_out_dir=combo_out_dir,
            field_mapping=field_mapping,
            tokenizer=tokenizer,
            base_urls=base_urls,
            model_name=model_name,
            request_timeout=args.request_timeout,
        )

        # If resumed: reload full results (new + previously completed), keeping
        # this session's runtime metrics for the wall-clock merge.
        if resumed:
            runtime_metrics = batch_result.get("runtime_metrics")
            batch_result = _load_existing_results(combo_out_dir)
            batch_result["runtime_metrics"] = runtime_metrics

        _finalize_outputs(combo_out_dir, batch_result, combo_config)
        _print_iter_count_distribution(combo_out_dir)
        combo_out_dir = _maybe_rename_with_accuracy(combo_out_dir, field_mapping)
        print(f"✓ {dataset_name}  →  {combo_out_dir}")

    print("All online evaluations complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate via online mini-sglang servers with real-time saving.")
    p.add_argument("--model_path", required=True, help="Checkpoint path (used for tokenizer and default output dir).")
    p.add_argument("--base_urls", nargs="+", required=True, help="mini-sglang server base URLs.")
    p.add_argument("--model_name", default=None, help="Model name sent to the API. Defaults to model_path.")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--datasets_map", default=None, help='JSON: {"gsm8k": {"max_new_tokens": 4096}}')
    # decode params (all the server needs; the model config lives server-side)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=16384)
    p.add_argument("--repeat_size", type=int, default=1, help="Samples per question.")
    p.add_argument("--per_server_concurrency", type=int, default=1,
                   help="Max in-flight requests per mini-sglang server. Total concurrency = this * len(base_urls).")
    p.add_argument("--save_iter_counts", action=argparse.BooleanOptionalAction, default=None,
                   help="Write per-sample prompt/response iter-count lists to iter_counts.jsonl. "
                        "Default: on for TaH checkpoints (max_iter > 1), off for plain base.")
    p.add_argument("--tah_iter_threshold", type=float, default=None,
                   help="TaH threshold used only for output path tagging, e.g. thr0.5.")
    p.add_argument("--request_timeout", type=int, default=108000)
    p.add_argument("--data_range", type=int, nargs="+", default=None)
    p.add_argument("--resume_dir", default=None)
    p.add_argument("--wait_for_server", action="store_true")
    p.add_argument("--wait_timeout", type=int, default=600)
    return p.parse_args()


if __name__ == "__main__":
    run_online_eval(parse_args())
