"""CLI entrypoint for the multi-backend, multi-job eval driver.

Wraps :func:`tah.evaluate.allocate_gpus_and_run_jobs`. Run e.g.::

    python script/tah/evaluation/eval.py \\
        --eval_config script/tah/recipes/qwen3_1.7/eval_tah.yaml \\
        --model_path nics-efc/TaH-plus-1.7B \\
        --dataset_name gsm8k --backend tah_engine \\
        --job_nums 1 --tp_size_per_job 1
"""
from __future__ import annotations

import argparse
import json

from transformers.utils import logging as hf_logging

from tah.evaluate import allocate_gpus_and_run_jobs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TaH multi-backend evaluation driver")
    p.add_argument("--eval_config", required=True, help="Path to YAML eval recipe")
    p.add_argument("--model_path", required=True, help="Path or HF id of the model to eval")
    p.add_argument("--backend", choices=("sglang", "hf", "tah", "tah_engine"), default="tah_engine",
                   help="Inference backend (default: tah_engine; tah is the correctness reference)")
    p.add_argument("--engine-options", type=json.loads, default=None,
                   help='JSON object overriding YAML engine options (e.g. {"cuda_graphs": false})')
    p.add_argument("--engine-max-batch-size", type=int, default=None,
                   help="Override engine.max_batch_size: maximum resident requests")
    p.add_argument("--engine-host-swap-gib", type=float, default=None,
                   help="Override engine.host_swap_bytes in GiB (default 8)")
    p.add_argument("--engine-memory-fraction", type=float, default=None,
                   help="Override engine.memory_fraction (default 0.9)")
    p.add_argument("--grading-workers", type=int, default=None,
                   help="Math grading processes; 0 = inline (engine default 4, other backends 0)")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override YAML max_new_tokens for every evaluation request")
    p.add_argument("--dataset_name", required=True,
                   help='Dataset name(s); comma-separated for multi (e.g. "aime24,math500"). '
                        "Must appear in eval_configs/dataset_configs.json.")
    p.add_argument("--job_nums", type=int, default=1, help="Number of parallel jobs to fan out")
    p.add_argument("--tp_size_per_job", type=int, default=1, help="GPUs (tensor-parallel size) per job")
    p.add_argument("--output_dir", default=None,
                   help="Output directory; defaults to <model_path>/eval_results")
    p.add_argument("--data_range", type=int, nargs="+", default=None,
                   help="Subset slice per dataset — [end] or [start, end]")
    p.add_argument("--dataset_ranges", default=None,
                   help='Per-dataset slice overrides, e.g. "math500=0:200,gpqa=all"')
    p.add_argument("--data_ids", default=None,
                   help='Comma-separated specific problem IDs (e.g. "gsm8k_0,gsm8k_5"); overrides --data_range')
    p.add_argument("--del_job_dir", type=bool, default=True,
                   help="Delete per-job directories after combining results")
    p.add_argument("--logger_level", default="WARNING",
                   help="Logger level (DEBUG / INFO / WARNING / ERROR / CRITICAL)")
    p.add_argument("--random_seed", type=int, default=42, help="Per-job random seed")
    return p


def main(args: argparse.Namespace) -> None:
    level = getattr(hf_logging, args.logger_level.upper(), hf_logging.WARNING)
    hf_logging.set_verbosity(level)
    hf_logging.enable_default_handler()
    hf_logging.enable_propagation()
    allocate_gpus_and_run_jobs(args)


if __name__ == "__main__":
    main(_build_parser().parse_args())
