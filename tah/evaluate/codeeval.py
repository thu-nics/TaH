"""HumanEval / MBPP evaluation glue around `evalplus`.

Three entry points the eval driver uses:

* :func:`make_raw_chat_prompt_for_code_evaluation` — turn a problem prompt
  into a chat-templated prompt that primes the model to emit a single
  self-contained Python script in a markdown code block.
* :func:`sanitize` — strip the model output down to just the function body
  for the requested ``entry_point`` (re-exported from
  :mod:`evalplus.sanitize`).
* :func:`evaluate` — run a per-sample untrusted-execution check against the
  evalplus base + plus test suites in a process pool, then write a
  ``*.eval_results.json`` next to the input ``samples.jsonl``.

Public TaH carried a broader ``evaluate`` surface (interactive overwrite
prompt, `gguf_file`/`num_ctx`/`**model_kwargs` plumbing, a "reasoning"
chat-template branch). None of those are used by the cleaned eval driver,
so this module trims to the actually-called interface.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
from termcolor import cprint
from tqdm import tqdm

from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.data.utils import CACHE_DIR
from evalplus.eval import (
    PASS,
    compatible_eval_result,
    estimate_pass_at_k,
    untrusted_check,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec
from evalplus.sanitize import sanitize  # noqa: F401  -- re-exported for the runner

# (status_code, per-input pass/fail booleans) — evalplus's untrusted_check shape.
Result = Tuple[str, List[bool]]

_PROMPT_INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
)
_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
)
# Magic string that splits the assistant template open from the model's start.
_PROMPT_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt_for_code_evaluation(task_prompt: str, reasoning: bool, tokenizer) -> str:
    """Render the task prompt into the model's chat template + a fenced markdown
    block opener so the model continues with the function body inline.

    ``reasoning=True`` wraps the user message only and lets the model think
    freely; ``reasoning=False`` (the default our eval driver passes) primes
    the assistant turn so the model emits just the code.
    """
    if tokenizer.chat_template is None:
        return task_prompt

    user_msg = f"{_PROMPT_INSTRUCTION}\n```\n{task_prompt.strip()}\n```\n"
    if reasoning:
        return tokenizer.apply_chat_template([{"role": "user", "content": user_msg}], tokenize=False)

    primed = f"\n{_RESPONSE_PREFIX}\n```python\n{_PROMPT_SPLITTER}\n```\n"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}, {"role": "assistant", "content": primed}],
        tokenize=False,
    )
    return rendered.split(_PROMPT_SPLITTER)[0]


def _get_groundtruth(problems: Dict, hashcode: str, output_not_none_tasks) -> Dict:
    """Compute (or load from disk cache) the expected outputs for each problem
    by running the canonical solution against the base + plus inputs."""
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        print(f"Loading cached ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Computing expected outputs…")
    t0 = time.time()
    expected = {}
    for task_id, problem in problems.items():
        solution_src = problem["prompt"] + problem["canonical_solution"]
        oracle: Dict[str, Any] = {}
        for which in ("base", "plus"):
            outputs, t = trusted_exec(
                solution_src,
                problem[f"{which}_input"],
                problem["entry_point"],
                record_time=True,
                output_not_none=problem["entry_point"] in output_not_none_tasks,
            )
            oracle[which] = outputs
            oracle[f"{which}_time"] = t
        expected[task_id] = oracle
    print(f"Expected outputs computed in {time.time() - t0:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected, f)
    return expected


def _check_correctness(
    dataset: str, completion_id: int, problem: Dict[str, Any], solution: str,
    expected_output: Dict[str, List], *, base_only: bool, fast_check: bool,
    identifier: str, min_time_limit: float, gt_time_limit_factor: float,
) -> Dict[str, Result]:
    """Run ``solution`` against base (and optionally plus) test inputs."""
    out: Dict[str, Any] = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    for which in (("base",) if base_only else ("base", "plus")):
        out[which] = untrusted_check(
            dataset, solution,
            problem[f"{which}_input"], problem["entry_point"],
            expected=expected_output[which], atol=problem["atol"],
            ref_time=expected_output[f"{which}_time"],
            fast_check=fast_check, min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )
    return out


def _failed_inputs(stat: str, details: Optional[List[bool]], inputs: list, *, full: bool) -> list:
    """Pick which input rows to surface in the per-task failure log."""
    if stat == PASS or not details:
        return []
    if full:
        return [inputs[i] for i, ok in enumerate(details) if not ok]
    return [inputs[len(details) - 1]]  # last failure only


def evaluate(
    dataset: str,
    samples: str,
    *,
    base_only: bool = False,
    parallel: Optional[int] = None,
    test_details: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    output_file: Optional[str] = None,
) -> None:
    """Run evalplus untrusted-check on every sample and write pass@k results.

    ``samples`` is either a path to a ``.jsonl`` (each line ``{task_id,
    solution}``) or a directory containing one. Output JSON path is derived
    from ``samples`` (or set explicitly via ``output_file``).
    """
    n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

    if os.path.isdir(samples):
        result_path = os.path.join(samples, "eval_results.json")
    else:
        assert samples.endswith(".jsonl"), "samples must be a directory or *.jsonl path"
        legacy = samples.replace(".jsonl", "_eval_results.json")
        result_path = legacy if os.path.exists(legacy) else samples.replace(".jsonl", ".eval_results.json")
    if output_file:
        result_path = output_file

    if os.path.isfile(result_path):
        print(f"Loading previous results from {result_path}")
        with open(result_path, "r") as f:
            results = json.load(f)
        results = compatible_eval_result(results)
    else:
        if dataset == "humaneval":
            problems = get_human_eval_plus()
            dataset_hash = get_human_eval_plus_hash()
            output_not_none_tasks: Tuple[str, ...] = ()
        elif dataset == "mbpp":
            problems = get_mbpp_plus()
            dataset_hash = get_mbpp_plus_hash()
            output_not_none_tasks = MBPP_OUTPUT_NOT_NONE_TASKS
        else:
            raise ValueError(f"unsupported code dataset {dataset!r}")

        expected_output = _get_groundtruth(problems, dataset_hash, output_not_none_tasks)
        results = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "hash": dataset_hash, "eval": {}}

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)
            remaining = set()

            print("Reading samples…")
            for sample in tqdm(load_solutions(samples)):
                task_id = sample["task_id"]
                if task_id not in problems:
                    warn(f"task {task_id} in samples but not in dataset")
                    continue
                solution = sample.get("solution") or (problems[task_id]["prompt"] + sample["completion"])
                remaining.add(sample["_identifier"])
                futures.append(executor.submit(
                    _check_correctness,
                    dataset, completion_id[task_id], problems[task_id], solution,
                    expected_output[task_id],
                    base_only=base_only, fast_check=not test_details,
                    identifier=sample["_identifier"],
                    min_time_limit=min_time_limit, gt_time_limit_factor=gt_time_limit_factor,
                ))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remaining), "missing problems in unfinished"
            assert len(completion_id) == len(problems), "missing problems in samples"

            def _watchdog():
                while remaining:
                    last = len(remaining)
                    time.sleep(100)
                    if last == len(remaining) and remaining:
                        warn(f"no samples finished in 100s; {len(remaining)} pending: {remaining}")
            threading.Thread(target=_watchdog, daemon=True).start()

            for future in tqdm(as_completed(futures), total=n_samples):
                res = future.result()
                remaining.discard(res["_identifier"])
                eval_results[res["task_id"]].append(res)

        # Sort completions per task and unpack base/plus statuses + failure details.
        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = []
            for res in task_results:
                base_stat, base_details = res["base"]
                base_fails = _failed_inputs(base_stat, base_details, problems[task_id]["base_input"], full=test_details)
                plus_stat = None
                plus_fails: list = []
                if not base_only:
                    plus_stat, plus_details = res["plus"]
                    plus_fails = _failed_inputs(plus_stat, plus_details, problems[task_id]["plus_input"], full=test_details)
                if dataset == "mbpp":
                    base_fails = mbpp_serialize_inputs(task_id, base_fails)
                    plus_fails = mbpp_serialize_inputs(task_id, plus_fails)
                results["eval"][task_id].append({
                    "task_id": task_id,
                    "solution": res["solution"],
                    "base_status": base_stat,
                    "plus_status": plus_stat,
                    "base_fail_tests": base_fails,
                    "plus_fail_tests": plus_fails,
                })

    # pass@k from base; pass@k+ from base ∩ plus.
    total = np.array([len(r) for r in results["eval"].values()])
    base_correct = np.array([
        sum(r["base_status"] == PASS for r in tres) for tres in results["eval"].values()
    ])
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in (1, 10, 100) if total.min() >= k
    }
    cprint(f"{dataset} (base tests)", "red")
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.3f}", "red")
    results["pass_at_k"] = {"base": pass_at_k}

    if not base_only:
        new_correct = np.array([
            sum(r["base_status"] == r["plus_status"] == PASS for r in tres)
            for tres in results["eval"].values()
        ])
        pass_at_k_plus = {
            f"pass@{k}": estimate_pass_at_k(total, new_correct, k).mean()
            for k in (1, 10, 100) if (total >= k).all()
        }
        cprint(f"{dataset}+ (base + extra tests)", "green")
        for k, v in pass_at_k_plus.items():
            cprint(f"{k}:\t{v:.3f}", "green")
        results["pass_at_k"]["plus"] = pass_at_k_plus

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)
