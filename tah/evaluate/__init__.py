"""TaH eval driver: dataset loading, per-backend setup, multi-job orchestration.

Most callers want :func:`allocate_gpus_and_run_jobs` (the top-level entry
the CLI uses); the module split is otherwise mainly internal:

* ``datasets``  — load + standardise benchmark datasets.
* ``backends``  — per-backend (sglang / hf / tah) model + inference fn.
* ``jobs``      — per-job runner, process orchestration, result aggregation.
* ``matheval``  — math benchmark graders (rule-based via ``math_verify``).
* ``codeeval``  — humaneval/mbpp grading via ``evalplus``.
"""
from tah.evaluate.datasets import load_combined_dataset
from tah.evaluate.jobs import (
    allocate_gpus_and_run_jobs,
    combine_job_results,
    parse_data_range,
    run_single_job,
)

__all__ = [
    "allocate_gpus_and_run_jobs",
    "combine_job_results",
    "load_combined_dataset",
    "parse_data_range",
    "run_single_job",
]
