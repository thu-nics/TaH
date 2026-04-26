"""Backwards-compat shim: ``tah.evaluate.eval_unified`` was the original
single-file driver. The driver has been split into ``datasets.py``,
``backends.py``, and ``jobs.py``; this module re-exports the public entry
points so existing callers keep working.
"""
from tah.evaluate.datasets import load_combined_dataset as load_datasets_with_config  # noqa: F401
from tah.evaluate.jobs import (  # noqa: F401
    allocate_gpus_and_run_jobs,
    combine_job_results,
    parse_data_range,
    run_single_job,
)
