"""Prune one or more layers from a Qwen3-style decoder-only model.

Usage:

    python script/preparation/prune.py \\
        --model Qwen/Qwen3-1.7B-Base \\
        --dataset data/processed_data/openr1_math/1_7/eval \\
        --output model/qwen3_1.7_base_pruned \\
        --num_prune 1

For each layer, temporarily remove it from the model and measure perplexity
on a small calibration set; the layers with the smallest ΔPPL are dropped
permanently and the resulting model is saved with ``model.save_pretrained``.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset, load_from_disk
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def _finite_or_inf(x: float) -> float:
    """Return ``x`` if finite, else ``+inf`` so it sorts to the back of "best" lists."""
    return x if math.isfinite(x) else float("inf")


def _get_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_layers_module(model: nn.Module) -> Tuple[nn.Module, str, List[nn.Module]]:
    """Locate the decoder ``ModuleList`` on common HF causal-LM layouts.

    Returns ``(parent_module, attr_name, layers_list)`` where
    ``parent_module.<attr_name>`` is the writable ModuleList we'll
    swap out when pruning.
    """
    for parent_attr in ("model", "transformer"):
        parent = getattr(model, parent_attr, None)
        if parent is None:
            continue
        for layers_attr in ("layers", "h"):
            layers = getattr(parent, layers_attr, None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return parent, layers_attr, list(layers)
    raise RuntimeError("unable to locate decoder layers on the model")


def _prepare_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _read_calibration_from_hf(dataset_arg: str, text_field: str, max_samples: int) -> List[str]:
    """Read up to ``max_samples`` non-empty strings from ``dataset_arg``'s
    ``text_field`` column. Accepts either a local saved-to-disk dataset or an HF
    dataset id."""
    if os.path.isdir(dataset_arg):
        ds = load_from_disk(dataset_arg)
    else:
        ds = load_dataset(dataset_arg)
    if hasattr(ds, "column_names") and text_field not in ds.column_names:
        raise KeyError(f"column {text_field!r} not found; available: {ds.column_names}")
    texts = []
    for value in ds[text_field]:
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
            if len(texts) >= max_samples:
                break
    if not texts:
        raise RuntimeError(f"no usable text found in column {text_field!r}")
    return texts


@torch.no_grad()
def _compute_reference_loss(model: nn.Module, tokenizer, texts: List[str], device: torch.device, batch_size: int, max_length: int) -> float:
    """Compute average token-level loss on calibration texts."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        loss = outputs.loss
        total_loss += float(loss.detach().cpu())
        total_batches += 1

    return total_loss / max(1, total_batches)


@torch.no_grad()
def _compute_loss_with_temp_removed_layer(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    remove_index: int,
) -> float:
    """Temporarily remove one layer, evaluate avg loss, restore the layer."""
    parent, attr, original = _resolve_layers_module(model)
    original_num = getattr(getattr(model, "config", object()), "num_hidden_layers", None)
    try:
        keep = [m for i, m in enumerate(original) if i != remove_index]
        setattr(parent, attr, nn.ModuleList(keep))
        if original_num is not None:
            model.config.num_hidden_layers = len(keep)
        return _compute_reference_loss(model, tokenizer, texts, device, batch_size, max_length)
    finally:
        setattr(parent, attr, nn.ModuleList(original))
        if original_num is not None:
            model.config.num_hidden_layers = original_num


def _prune_layers_inplace(model: nn.Module, remove_indices: List[int]) -> None:
    """Drop the listed layer indices and update ``num_hidden_layers``."""
    parent, attr, layers = _resolve_layers_module(model)
    drop = set(remove_indices)
    keep = [m for i, m in enumerate(layers) if i not in drop]
    setattr(parent, attr, nn.ModuleList(keep))
    if hasattr(getattr(model, "config", None), "num_hidden_layers"):
        model.config.num_hidden_layers = len(keep)


def main() -> int:
    parser = argparse.ArgumentParser(description="prune Qwen/Qwen3 model by layer (approximate selection minimizing error)")
    parser.add_argument("--model", default="Qwen3/Qwen3-4B-Base", help="model name or local path")
    parser.add_argument("--dataset", default="data/processed_data/openr1_math/4/eval", help="Hugging Face dataset ID or local dataset directory (preferred)")
    parser.add_argument("--text_field", default="real_text", help="text field name, default real_text")
    parser.add_argument("--output", default="model/qwen3_4_base_pruned", help="directory to save pruned model")
    parser.add_argument("--num_prune", type=int, default=1, help="number of layers to prune")
    parser.add_argument("--batch_size", type=int, default=4, help="calibration batch size")
    parser.add_argument("--max_samples", type=int, default=16, help="number of samples for importance estimation")
    parser.add_argument("--max_length", type=int, default=4096, help="maximum length of each sample")
    parser.add_argument("--device", default="", help="device, like cuda:0 or cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"], help="model precision")

    args = parser.parse_args()

    device = _get_device(args.device)

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if args.dtype == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = dtype_map[args.dtype]

    tokenizer = _prepare_tokenizer(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    if not (isinstance(args.dataset, str) and len(args.dataset) > 0):
        raise RuntimeError("please provide --dataset to read calibration samples from Hugging Face dataset")
    texts = _read_calibration_from_hf(args.dataset, args.text_field, args.max_samples)

    print("calculating baseline PPL...")
    ref_loss = _compute_reference_loss(model, tokenizer, texts, device, args.batch_size, args.max_length)
    ref_ppl = float(math.exp(ref_loss))
    print(f"baseline loss: {ref_loss:.4f}")
    print(f"baseline PPL: {ref_ppl:.4f}")

    # Try removing each single layer and compute loss/PPL
    _, _, layers_list = _resolve_layers_module(model)
    num_layers = len(layers_list)
    per_layer_loss: List[float] = []
    per_layer_ppl: List[float] = []
    for li in range(num_layers):
        cand_loss = _compute_loss_with_temp_removed_layer(
            model, tokenizer, texts, device, args.batch_size, args.max_length, li
        )
        cand_ppl_raw = float(math.exp(cand_loss))
        cand_ppl = _finite_or_inf(cand_ppl_raw)
        per_layer_loss.append(cand_loss)
        per_layer_ppl.append(cand_ppl)
        delta = _finite_or_inf(cand_ppl - ref_ppl)
        print(f"layer {li} -> PPL: {cand_ppl:.4f}, ΔPPL: {delta:.4f}")

    # Choose layers with minimal impact (smallest increase in loss)
    k = min(args.num_prune, num_layers - 1)
    order = sorted(
        range(num_layers),
        key=lambda i: (_finite_or_inf(per_layer_ppl[i] - ref_ppl), _finite_or_inf(per_layer_ppl[i])),
    )
    remove_indices = sorted(order[:k])
    print(f"planned to prune layer indices (by smallest ΔPPL): {remove_indices}")

    print("executing physical pruning...")
    _prune_layers_inplace(model, remove_indices)

    os.makedirs(args.output, exist_ok=True)

    print("quick verification of PPL after pruning...")
    pruned_loss = _compute_reference_loss(model, tokenizer, texts, device, args.batch_size, args.max_length)
    pruned_ppl = float(math.exp(pruned_loss))
    print(f"PPL after pruning: {pruned_ppl:.4f} (Δ={pruned_ppl - ref_ppl:.4f})")

    print("saving model...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("done")
    return 0

if __name__ == "__main__":
    sys.exit(main())