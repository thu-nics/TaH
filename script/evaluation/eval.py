"""CLI entrypoint for the multi-backend, multi-job eval driver.

Wraps :func:`tah.evaluate.allocate_gpus_and_run_jobs`. Run e.g.::

    python script/evaluation/eval.py \\
        --eval_config script/recipes/qwen3_1.7/eval_tah.yaml \\
        --model_path nics-efc/TaH-plus-1.7B \\
        --dataset_name gsm8k --backend tah \\
        --job_nums 8 --tp_size_per_job 1
"""
from __future__ import annotations

import argparse

from transformers.utils import logging as hf_logging

from tah.evaluate import allocate_gpus_and_run_jobs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TaH multi-backend evaluation driver")
    p.add_argument("--eval_config", required=True, help="Path to YAML eval recipe")
    p.add_argument("--model_path", required=True, help="Path or HF id of the model to eval")
    p.add_argument("--backend", choices=("sglang", "hf", "tah"), default="hf",
                   help="Inference backend (default: hf)")
    p.add_argument("--dataset_name", required=True,
                   help='Dataset name(s); comma-separated for multi (e.g. "aime24,math500"). '
                        "Must appear in eval_configs/dataset_configs.json.")
    p.add_argument("--job_nums", type=int, default=1, help="Number of parallel jobs to fan out")
    p.add_argument("--tp_size_per_job", type=int, default=1, help="GPUs (tensor-parallel size) per job")
    p.add_argument("--output_dir", default=None,
                   help="Output directory; defaults to <model_path>/eval_results")
    p.add_argument("--data_range", type=int, nargs="+", default=None,
                   help="Subset slice — [end] or [start, end]")
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
