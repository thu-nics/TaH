"""
eval_offline.py — evaluate one model across multiple (dataset, threshold) combos
                 with a single model load per GPU worker.

Usage example:
    python script/tah2/eval/eval_offline.py \
        --temperature 0.6 --top_p 0.95 --top_k 20 --max_new_tokens 16384 \
        --model_path output/.../checkpoint-2340 \
        --output_dir output/.../custom_eval_results \
        --datasets minerva math500 \
        --thresholds 0.5 0.6 0.7 0.8 0.9 \
        --backend mini_sglang \
        --job_nums 8 --tp_size_per_job 1
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from multiprocessing import Process
from queue import Empty
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from tah2.evaluate.common import load_datasets_with_config, parse_data_range, _maybe_rename_with_accuracy
from tah2.evaluate.worker import _run_persistent_worker
from tah2.evaluate.local_eval import combine_job_results


def _build_batches(dataset, selected_indices, problems_per_batch: int) -> List[List[Dict]]:
    batches = []
    for i in range(0, len(selected_indices), problems_per_batch):
        chunk = selected_indices[i : i + problems_per_batch]
        batch_items = []
        for j in chunk:
            item = dict(dataset[j])
            item["_original_index"] = j
            batch_items.append(item)
        if batch_items:
            batches.append(batch_items)
    return batches


def _scan_completed_jobs(combo_out_dir: Path) -> Tuple[set, int]:
    completed_problem_ids = set()
    next_job_id_start = 0
    if not combo_out_dir.exists():
        return completed_problem_ids, next_job_id_start

    existing_job_dirs = []
    for d in combo_out_dir.iterdir():
        if d.is_dir() and d.name.startswith("job_"):
            try:
                int(d.name.split("_")[1])
                existing_job_dirs.append(d)
            except (IndexError, ValueError):
                continue
    existing_job_dirs.sort(key=lambda x: int(x.name.split("_")[1]))

    for job_dir in existing_job_dirs:
        stats_file = job_dir / "evaluation_stats.csv"
        results_file = job_dir / "detailed_results.csv"
        if stats_file.exists() and results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    problem_id = row.get("problem_id")
                    if problem_id is not None:
                        completed_problem_ids.add(str(problem_id))

            job_id = int(job_dir.name.split("_")[1])
            next_job_id_start = max(next_job_id_start, job_id + 1)

    return completed_problem_ids, next_job_id_start


def _resolve_resume_combo_dir(resume_dir: Path, dataset_name: str, threshold: float, backend: str) -> Optional[Path]:
    dataset_backend_dir = f"{dataset_name}_{backend}"
    threshold_tag = f"thr{threshold}" if threshold is not None else None

    has_job_dirs = any(
        p.is_dir() and p.name.startswith("job_")
        for p in resume_dir.iterdir()
    ) if resume_dir.exists() else False

    direct_match = (
        resume_dir.parent.name == dataset_backend_dir
        and (
            has_job_dirs
            or (resume_dir / "evaluation_stats.csv").exists()
            or (resume_dir / "detailed_results.csv").exists()
        )
    )
    if threshold_tag is not None:
        direct_match = direct_match and (resume_dir.parent.parent.name == threshold_tag)

    if direct_match:
        return resume_dir

    combo_parent = resume_dir / dataset_backend_dir if threshold is None else resume_dir / threshold_tag / dataset_backend_dir
    if not combo_parent.exists():
        return None

    run_dirs = [d for d in combo_parent.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]



def run_server(args):
    mp.set_start_method("spawn", force=True)

    # GPU allocation
    current_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current_cuda is None:
        import torch
        available_gpus = list(range(torch.cuda.device_count()))
    else:
        available_gpus = [gpu.strip() for gpu in current_cuda.split(",") if gpu.strip() and gpu.strip() != "-1"]
    gpus_per_job = args.tp_size_per_job
    max_workers = min(args.job_nums, len(available_gpus) // gpus_per_job)
    if max_workers <= 0:
        raise ValueError("Insufficient GPUs for the requested tp_size_per_job.")
    worker_gpus = {wid: available_gpus[wid * gpus_per_job : (wid + 1) * gpus_per_job] for wid in range(max_workers)}

    # Base eval config
    eval_config = {
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "repeat_size": args.repeat_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "use_tracker": args.use_tracker,
        "_logger_level": args.logger_level,
        "_random_seed": args.random_seed,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_out_dir = Path(args.output_dir) if args.output_dir else (Path(args.model_path) / "eval_results")
    resume_root = Path(args.resume_dir) if getattr(args, "resume_dir", None) else None
    if resume_root is not None and not resume_root.exists():
        raise ValueError(f"Resume directory does not exist: {resume_root}")

    # Resolve dataset list and per-dataset config overrides
    datasets_map: Dict = {}
    if getattr(args, 'datasets_map', None):
        datasets_map = json.loads(args.datasets_map)
    dataset_names_to_process = list(datasets_map.keys()) if datasets_map else (args.datasets or [])
    if not dataset_names_to_process:
        raise ValueError("Either --datasets or --datasets_map must be provided.")

    # Build all (dataset_name, threshold, batches, output_dir, field_mapping, config_override) groups
    thresholds = args.thresholds if args.thresholds else [None]
    print(f"Building task groups for datasets={dataset_names_to_process} × thresholds={thresholds}")
    task_groups: List[Tuple] = []  # (dataset_name, threshold, batches, combo_out_dir, field_mapping, config_override)
    for dataset_name in dataset_names_to_process:
        dataset_list, field_mapping = load_datasets_with_config([dataset_name])
        total = len(dataset_list)
        range_start, range_end = parse_data_range(args.data_range, total)
        selected = list(range(range_start, range_end))

        # Per-dataset config: fall back to global eval_config values
        ds_overrides = datasets_map.get(dataset_name, {})
        ds_batch_size  = ds_overrides.get('batch_size',    eval_config.get('batch_size', 1))
        ds_repeat_size = ds_overrides.get('repeat_size',   eval_config.get('repeat_size', 1))

        # config_override carries only the keys that differ from the global default
        _override_keys = {k: ds_overrides[k] for k in ('max_new_tokens', 'batch_size', 'repeat_size')
                          if k in ds_overrides}
        config_override = _override_keys if _override_keys else None

        # Adjust problems_per_batch to use all workers (using per-dataset batch/repeat sizes)
        ds_ppb = max(1, ds_batch_size // ds_repeat_size)
        est_jobs = math.ceil(len(selected) / ds_ppb)
        ppb = ds_ppb
        if est_jobs < max_workers and selected:
            ppb = max(1, math.ceil(len(selected) / max_workers))

        batches = _build_batches(dataset_list, selected, ppb)

        for threshold in thresholds:
            if resume_root is not None:
                combo_out_dir = _resolve_resume_combo_dir(resume_root, dataset_name, threshold, args.backend)
                if combo_out_dir is None:
                    if threshold is not None:
                        thr_tag = f"thr{threshold}"
                        combo_out_dir = base_out_dir / thr_tag / f"{dataset_name}_{args.backend}" / timestamp
                    else:
                        combo_out_dir = base_out_dir / f"{dataset_name}_{args.backend}" / timestamp
                else:
                    print(f"Resume mode: using existing output dir for {dataset_name} thr={threshold}: {combo_out_dir}")
            else:
                if threshold is not None:
                    thr_tag = f"thr{threshold}"
                    combo_out_dir = base_out_dir / thr_tag / f"{dataset_name}_{args.backend}" / timestamp
                else:
                    combo_out_dir = base_out_dir / f"{dataset_name}_{args.backend}" / timestamp
            combo_out_dir.mkdir(parents=True, exist_ok=True)

            # Save config snapshot (with threshold + per-dataset overrides applied)
            cfg_snap = dict(eval_config)
            if threshold is not None:
                cfg_snap.setdefault("iter_decider_kwargs", {})["threshold"] = threshold
            for k in ('max_new_tokens', 'batch_size', 'repeat_size'):
                if k in ds_overrides:
                    cfg_snap[k] = ds_overrides[k]
            with open(combo_out_dir / "eval_config.json", "w") as f:
                json.dump(cfg_snap, f, indent=2)

            task_groups.append((dataset_name, threshold, batches, combo_out_dir, field_mapping, config_override))

    total_tasks = sum(len(g[2]) for g in task_groups)
    print(f"Total task groups: {len(task_groups)}  |  Total batches: {total_tasks}")

    # Start persistent workers (they load the model once and wait for tasks)
    # dataset_name / output_dir / field_mapping are all overridden per-task; pass placeholders here.
    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    placeholder_field_mapping = {"id_field": "id", "question_field": "question",
                                  "answer_field": "answer", "answer_type": "boxed"}
    workers = []
    for wid in range(max_workers):
        p = Process(
            target=_run_persistent_worker,
            args=(
                wid,
                worker_gpus[wid],
                eval_config,
                dataset_names_to_process[0],  # default dataset_name (overridden per task)
                str(base_out_dir),         # default output_dir   (overridden per task)
                None,                      # timestamp not needed (output_dir given per task)
                args.model_path,
                gpus_per_job,
                args.backend,
                None,                      # data_range
                placeholder_field_mapping, # overridden per task
                None,                      # unified_code_solutions_file
                task_queue,
                result_queue,
            ),
            name=f"worker-{wid}",
        )
        p.start()
        workers.append(p)

    # Process combo groups sequentially (parallel within each group)
    global_job_id = 0
    had_failures = False
    for dataset_name, threshold, batches, combo_out_dir, field_mapping, config_override in task_groups:
        if not batches:
            continue

        combo_label = f"{dataset_name} thr={threshold}" if threshold is not None else dataset_name
        completed_problem_ids, next_job_id_start = _scan_completed_jobs(combo_out_dir)
        if completed_problem_ids:
            print(
                f"Resume mode: {combo_label} found {len(completed_problem_ids)} completed problems, "
                f"next job ID starts from {next_job_id_start}"
            )

        pending_batches = []
        id_field = field_mapping.get("id_field", "id")
        for batch_items in batches:
            filtered_items = []
            for item in batch_items:
                problem_id = str(item.get(id_field, ""))
                if problem_id and problem_id in completed_problem_ids:
                    continue
                filtered_items.append(item)
            if filtered_items:
                pending_batches.append(filtered_items)

        n_batches = len(pending_batches)
        total_job_count = next_job_id_start + n_batches
        if n_batches == 0:
            if completed_problem_ids:
                print(f"All problems already completed for {combo_label}. Running combine_job_results only.")
                combine_job_results(
                    combo_out_dir,
                    next_job_id_start,
                    args.del_job_dir,
                    save_all_trackers=eval_config.get('use_tracker', False),
                )
                combo_out_dir = _maybe_rename_with_accuracy(combo_out_dir, field_mapping)
            else:
                print(f"No pending batches for {combo_label}, skipping.")
            print(f"✓ {combo_label}  →  {combo_out_dir}")
            continue

        # Submit this combo's batches.
        # Directory names use local indices (job_0, job_1, ...) so combine_job_results works.
        # The task carries the global_job_id only for result tracking in the shared queue.
        job_ids_this_combo = []
        for local_idx, batch_items in enumerate(pending_batches):
            jid = global_job_id
            global_job_id += 1
            local_job_id = next_job_id_start + local_idx
            job_output_dir = str(combo_out_dir / f"job_{local_job_id}")
            task_queue.put((jid, batch_items, threshold, job_output_dir, dataset_name, field_mapping, config_override))
            job_ids_this_combo.append(jid)

        # Wait for this combo to finish
        completed = 0
        failed = 0
        pbar = tqdm(total=n_batches, desc=combo_label, unit="batch")
        while completed + failed < n_batches:
            try:
                wid_r, jid_r, success, msg = result_queue.get(timeout=1.0)
                if success:
                    completed += 1
                else:
                    failed += 1
                    tqdm.write(f"✗ job {jid_r} failed: {msg}")
                pbar.update(1)
            except Empty:
                dead = [w for w in workers if not w.is_alive()]
                if dead and all(not w.is_alive() for w in workers):
                    tqdm.write("Error: all workers died")
                    break
        pbar.close()

        if failed or completed < n_batches:
            had_failures = True
            print(f"Evaluation failed for {combo_label}: {completed}/{n_batches} batches completed")
            continue

        # Combine results for this combo
        combine_job_results(combo_out_dir, total_job_count, args.del_job_dir,
                            save_all_trackers=eval_config.get('use_tracker', False))
        combo_out_dir = _maybe_rename_with_accuracy(combo_out_dir, field_mapping)

        print(f"✓ {combo_label}  →  {combo_out_dir}")

    # Shutdown workers
    for _ in range(max_workers):
        task_queue.put(None)
    for w in workers:
        w.join(timeout=30)
        if w.is_alive():
            w.terminate()
            w.join(timeout=5)

    if had_failures:
        raise RuntimeError("Evaluation failed; successful job outputs are preserved for resuming.")
    print("All evaluations complete.")


def main():
    parser = argparse.ArgumentParser(description="Eval server: one model load, many (dataset, threshold) combos.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base output directory for eval results, default is <model_path>/eval_results")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="One or more dataset names, e.g. minerva math500")
    parser.add_argument("--datasets_map", type=str, default=None,
                        help='JSON mapping dataset→per-dataset params, e.g. \'{"gsm8k":{"max_new_tokens":4096,"batch_size":128,"repeat_size":1}}\'')

    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="One or more threshold values, e.g. 0.5 0.6 0.7 (optional for standard models)")
    parser.add_argument("--backend", type=str, default="hf",
                        choices=["hf", "mini_sglang"],
                        help="Inference engine; both support TaH2 and Standard checkpoints.")
    parser.add_argument("--job_nums", type=int, default=8)
    parser.add_argument("--tp_size_per_job", type=int, choices=[1], default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--repeat_size", type=int, default=1)
    parser.add_argument("--use_tracker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data_range", type=int, nargs="+", default=None)
    parser.add_argument("--del_job_dir", type=bool, default=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--logger_level", type=str, default="WARNING")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Resume from previous eval_results root or combo timestamp dir")
    args = parser.parse_args()

    run_server(args)


if __name__ == "__main__":
    main()
