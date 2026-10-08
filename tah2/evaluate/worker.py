"""Persistent worker process for eval_offline.py.

eval_offline.py spawns one _run_persistent_worker per GPU group.
The worker loads the model once, then processes batch tasks from a multiprocessing Queue.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from multiprocessing import Queue

from transformers.utils import logging as hf_logging
import logging as pylog

from .common import _save_job_stats, parse_data_range
from .local_eval import initialize_inference_engine, process_batch_items


def _is_port_available(port: int) -> bool:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def _run_persistent_worker(
    worker_id: int,
    gpu_devices: list,
    config: Dict,
    combined_dataset_name: str,
    output_dir: str,
    timestamp: str,
    model_path: str,
    tp_size: int,
    backend: str,
    data_range,
    field_mapping: Dict,
    unified_code_solutions_file,
    task_queue: Queue,
    result_queue: Queue,
):
    """Load model once; loop processing tasks until sentinel None is received.

    Task tuple: (job_id, batch_items, threshold_override, output_dir_override,
                 dataset_name_override, field_mapping_override, config_override)
    """
    # Configure logging
    lvl = (config.get("_logger_level") or "WARNING").upper()
    hf_logging.set_verbosity(getattr(hf_logging, lvl, hf_logging.WARNING))
    hf_logging.enable_default_handler()
    hf_logging.enable_propagation()
    pylog.basicConfig(level=getattr(pylog, lvl, pylog.WARNING),
                      format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Set GPU + NCCL port
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    unique_port = None
    for retry in range(100):
        candidate = 29555 + worker_id * 100 + retry
        if candidate > 65535:
            candidate = 30514 + ((worker_id + retry) % 100)
        if _is_port_available(candidate):
            unique_port = candidate
            break
    if unique_port is None:
        raise RuntimeError(f"Worker {worker_id}: no available port found")
    os.environ["MASTER_PORT"] = str(unique_port)
    os.environ["SGLANG_NCCL_PORT"] = str(unique_port)

    import torch
    from tah2.utils.modeling import set_all_seeds
    set_all_seeds(config.get("_random_seed", 420))
    if torch.cuda.is_available():
        torch.cuda.init()

    # Resolve base output dir
    if timestamp is None:
        base_output_dir = Path(output_dir)
    else:
        suffix = ""
        if data_range:
            s, e = parse_data_range(data_range)
            suffix = f"TASK_{s}_{e}"
        base_output_dir = Path(output_dir) / f"{combined_dataset_name}_{backend}" / timestamp
        if suffix:
            base_output_dir = base_output_dir / suffix

    # Load model
    try:
        inference_fn, tokenizer, cleanup_fn, tracker, set_threshold_fn, set_gen_params_fn = initialize_inference_engine(
            config=config, model_path=model_path, tp_size=tp_size, backend=backend,
        )
    except Exception as e:
        import traceback
        result_queue.put((worker_id, -1, False, f"Worker {worker_id} failed to load model: {e}\n{traceback.format_exc()}"))
        return

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            job_id, batch_items, threshold_override, output_dir_override, dataset_override, fm_override, cfg_override = task

            if threshold_override is not None:
                set_threshold_fn(threshold_override)
            if cfg_override is not None:
                set_gen_params_fn(**cfg_override)

            job_dir = Path(output_dir_override) if output_dir_override else base_output_dir / f"job_{job_id}"
            task_dataset = dataset_override or combined_dataset_name
            task_fm = fm_override or field_mapping

            try:
                batch_result = process_batch_items(
                    config=config,
                    combined_dataset_name=task_dataset,
                    output_dir=job_dir,
                    job_id=job_id,
                    problems_data=batch_items,
                    field_mapping=task_fm,
                    inference_function=inference_fn,
                    tokenizer=tokenizer,
                    tracker=tracker,
                    unified_code_solutions_file=unified_code_solutions_file,
                )
                _save_job_stats(job_dir, batch_result)
                result_queue.put((worker_id, job_id, True, f"Job {job_id} done"))
            except Exception as e:
                import traceback
                result_queue.put((worker_id, job_id, False, f"Job {job_id} failed: {e}\n{traceback.format_exc()}"))
    finally:
        cleanup_fn()
        for var in ("MASTER_PORT", "MASTER_ADDR", "SGLANG_NCCL_PORT", "NCCL_SOCKET_IFNAME"):
            os.environ.pop(var, None)
