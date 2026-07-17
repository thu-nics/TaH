"""SFT dataset preprocessing for TaH.

Loads the labeller-produced dataset (with ``real_token``, ``mask``,
``mismatch`` columns) and turns each example into the
``(input_ids, attention_mask, labels, iter_count_labels)`` tuple consumed by
:class:`tah.train.CustomTaHDataCollator`.

Public TaH supported many alternative ``iter_count_strategy`` modes (random,
top_entropy, top_ce, ds_divergence, nonmismatch, divergent, all, maxiter)
that the canonical SFT recipes never selected. The cleaned version keeps the
``mismatch`` strategy only — the same one used by both
``sft_tah_step1.yaml`` and ``sft_tah_step2.yaml``.

Two columns the canonical labeller never writes are also dropped (``divergent``,
``ds_divergence``); ``entropy`` and ``cross_entropy`` are still removed before
collation when present (the labeller can be configured to emit them, but the
mismatch strategy doesn't read them).
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Optional

import numpy as np
from accelerate import Accelerator
from datasets import load_from_disk

# Columns produced by the labeller that are inputs to preprocessing but not
# meaningful at training time, so we strip them after building the SFT batch.
_LABELLER_INPUT_COLUMNS = (
    "data_id", "real_text", "real_token", "mask", "mismatch",
    "divergent", "entropy", "cross_entropy", "ds_divergence", "problem_idx",
)


def _build_sft_example(
    input_ids: list,
    prompt_mask: list,
    mismatch_mask: Optional[list],
    max_iter: int,
    max_length: Optional[int],
    truncate: bool,
    query_iter_count: int,
) -> Optional[Dict]:
    """Convert one labelled example into an SFT example.

    Returns None for examples that should be filtered out (no supervised
    tokens, or longer than ``max_length`` when truncation is disabled).
    """
    if not any(prompt_mask):
        return None

    if max_length is not None and len(input_ids) > max_length:
        if not truncate:
            return None
        input_ids = input_ids[:max_length]
        prompt_mask = prompt_mask[:max_length]
        mismatch_mask = mismatch_mask[:max_length] if mismatch_mask is not None else None

    labels = np.array(input_ids, dtype=np.int64)
    prompt_np = np.array(prompt_mask)
    labels[prompt_np == 0] = -100  # ignore prompt tokens for next-token loss

    iter_labels = np.ones(len(input_ids), dtype=np.int32)
    if max_iter > 1 and mismatch_mask is not None:
        mismatch_np = np.array(mismatch_mask)
        if np.any(mismatch_np > 1):
            # The labeller already emitted oracle iter counts directly.
            iter_labels = mismatch_np + 1
        else:
            # Otherwise, mismatch is a 0/1 mask of "hard" positions; assign a
            # uniform random extra iter count to each, restricted to valid
            # (non-prompt) positions.
            mismatch_and_valid = mismatch_np & prompt_np
            idx = np.nonzero(mismatch_and_valid)[0]
            if len(idx) > 0:
                iter_labels[idx] = np.random.randint(2, max_iter + 1, len(idx))
        iter_labels[prompt_np == 0] = query_iter_count

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels.tolist(),
        "iter_count_labels": iter_labels.tolist(),
    }


def _preprocess_for_sft_batch(examples: Dict, *, max_iter: int, max_length: Optional[int], truncate: bool, query_iter_count: int) -> Dict:
    """Vectorise :func:`_build_sft_example` over a HF Datasets batch."""
    n = len(examples["real_token"])
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    if max_iter > 1:
        out["iter_count_labels"] = []
    has_mismatch = "mismatch" in examples
    for i in range(n):
        result = _build_sft_example(
            input_ids=examples["real_token"][i],
            prompt_mask=examples["mask"][i],
            mismatch_mask=examples["mismatch"][i] if has_mismatch else None,
            max_iter=max_iter,
            max_length=max_length,
            truncate=truncate,
            query_iter_count=query_iter_count,
        )
        if result is None:
            continue
        for k in out:
            out[k].append(result[k])
    return out


def _filter_keep(examples: Dict, *, max_length: Optional[int], truncate: bool) -> list[bool]:
    """Drop rows with empty supervision or (when not truncating) over-length."""
    masks, tokens = examples["mask"], examples["real_token"]
    mismatches = examples.get("mismatch")
    enforce_len = bool(max_length) and not truncate
    apply_window = bool(max_length) and truncate
    keep = []
    for i in range(len(tokens)):
        end = max_length if apply_window else len(tokens[i])
        mask_i = masks[i][:end]
        ok = any(mask_i)
        if ok and mismatches is not None:
            mismatch_i = mismatches[i][:end]
            ok = ok and any(mask_i[j] == 1 and mismatch_i[j] == 1 for j in range(len(mask_i)))
        if enforce_len:
            ok = ok and (len(tokens[i]) <= max_length)
        keep.append(ok)
    return keep


def preprocess_dataset(training_config: Dict, data_config: Dict, model_config: Dict, accelerator: Accelerator):
    """Load + preprocess train (and optionally eval) datasets for SFT."""
    del training_config  # unused since the hard_token_relative_weight path was removed
    accelerator.print("Loading dataset…")
    train_ds_path = data_config["train_data_path"]
    eval_ds_path = data_config.get("eval_data_path")
    train_ratio = data_config.get("train_data_ratio", 1.0)
    eval_ratio = data_config.get("eval_data_ratio", 0.05)

    max_iter = model_config.get("max_iter", 1)
    max_length = data_config.get("max_length")
    truncate = (data_config.get("max_length_action", "cutoff") or "cutoff").lower() != "filter"
    query_iter_count = int(data_config.get("query_iter_count", 1))

    with accelerator.main_process_first():
        train_ds = load_from_disk(train_ds_path)
    if train_ratio != 1.0:
        accelerator.print(f"Subsampling train dataset to {train_ratio:.2%}")
        train_ds = train_ds.select(range(int(len(train_ds) * train_ratio)))

    eval_ds = None
    if eval_ds_path:
        try:
            with accelerator.main_process_first():
                eval_ds = load_from_disk(eval_ds_path)
            accelerator.print(f"Loaded separate eval dataset from {eval_ds_path}")
        except Exception as e:
            accelerator.print(f"Warning: could not load eval dataset {eval_ds_path}: {e}")

    if max_length:
        accelerator.print(f"max_length={max_length} ({'truncate' if truncate else 'filter'})")

    # Prefilter: drop rows with no supervision or (when filter mode) over-length.
    filt = partial(_filter_keep, max_length=max_length, truncate=truncate)
    with accelerator.main_process_first():
        train_ds = train_ds.filter(filt, batched=True, batch_size=1000, num_proc=16, desc="Prefiltering train")
        if eval_ds is not None:
            eval_ds = eval_ds.filter(filt, batched=True, batch_size=1000, num_proc=16, desc="Prefiltering eval")

    # Map to SFT examples.
    accelerator.print("Preprocessing datasets…")
    preprocess = partial(
        _preprocess_for_sft_batch, max_iter=max_iter, max_length=max_length,
        truncate=truncate, query_iter_count=query_iter_count,
    )

    def _drop_input_cols(ds):
        return [c for c in _LABELLER_INPUT_COLUMNS if c in ds.column_names]

    with accelerator.main_process_first():
        train_ds = train_ds.map(
            preprocess, batched=True, batch_size=2000, num_proc=16,
            remove_columns=_drop_input_cols(train_ds), desc="SFT train preprocessing",
        )
        if eval_ds is not None:
            eval_ds = eval_ds.map(
                preprocess, batched=True, batch_size=2000, num_proc=16,
                remove_columns=_drop_input_cols(eval_ds), desc="SFT eval preprocessing",
            )

    # If no separate eval set was provided, split a fraction from train.
    if eval_ds is None and eval_ratio > 0:
        accelerator.print(f"Splitting {eval_ratio:.2%} of train as eval")
        split = train_ds.train_test_split(test_size=eval_ratio, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]

    accelerator.print(f"Train dataset size: {len(train_ds)}")
    if eval_ds is not None:
        accelerator.print(f"Eval dataset size: {len(eval_ds)}")
    return train_ds, eval_ds
