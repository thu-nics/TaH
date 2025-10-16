import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import sys
import math
from typing import List, Tuple, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk


def _get_device(device_arg: str) -> torch.device:
    """Resolve device from arg."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_layers_module(model: nn.Module) -> Tuple[nn.Module, List[nn.Module]]:
    """Find the decoder layers ModuleList for common HF causal LMs.

    Returns a tuple of (parent_module, layers_list).
    """
    # Common path for modern decoder-only models
    for parent_attr in ["model", "transformer"]:
        parent = getattr(model, parent_attr, None)
        if parent is None:
            continue
        for layers_attr in ["layers", "h"]:
            layers = getattr(parent, layers_attr, None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return parent, list(layers)
    raise RuntimeError("unable to locate model layers")


def _prepare_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def _read_calibration_from_hf(dataset_arg: str, text_field: str, max_samples: int) -> List[str]:
    if load_dataset is None or load_from_disk is None:
        raise RuntimeError("datasets not found")

    texts: List[str] = []

    # Try local disk first
    ds = None
    if os.path.isdir(dataset_arg):
        try:
            ds = load_from_disk(dataset_arg)
        except Exception as e:
            # Fallback: directly read Arrow file to extract text when features schema is incompatible
            try:
                import pyarrow as pa  # type: ignore
                import pyarrow.ipc as pa_ipc  # type: ignore
                # locate data-*.arrow under the directory or its immediate subdirs
                arrow_files = [f for f in os.listdir(dataset_arg) if f.startswith("data-") and f.endswith(".arrow")]
                search_dir = dataset_arg
                if not arrow_files:
                    subdirs = [os.path.join(dataset_arg, d) for d in os.listdir(dataset_arg) if os.path.isdir(os.path.join(dataset_arg, d))]
                    for sd in subdirs:
                        cand = [f for f in os.listdir(sd) if f.startswith("data-") and f.endswith(".arrow")]
                        if cand:
                            search_dir = sd
                            arrow_files = cand
                            break
                if not arrow_files:
                    raise RuntimeError("unable to find data-*.arrow file for fallback reading")
                file_path = os.path.join(search_dir, sorted(arrow_files)[0])
                with pa.memory_map(file_path, "r") as source:
                    reader = pa_ipc.RecordBatchFileReader(source)
                    table = reader.read_all()
                if text_field not in table.column_names:
                    raise KeyError(f"column {text_field} not found in Arrow table: available columns {table.column_names}")
                col_py = table[text_field].to_pylist()
                for value in col_py:
                    if isinstance(value, str) and value.strip():
                        texts.append(value.strip())
                        if len(texts) >= max_samples:
                            break
                if not texts:
                    raise RuntimeError(f"dataset has no available text in column {text_field}")
                return texts
            except Exception as e2:
                raise RuntimeError(f"local dataset loading failed, and Arrow fallback failed: {e2}") from e
    else:
        # Remote or canonical HF dataset id
        ds = load_dataset(dataset_arg)

    # Now ds should be a Dataset-like
    if ds is None:
        raise RuntimeError("unable to load dataset")

    # Extract text_field
    col = ds[text_field]
    for value in col:
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
            if len(texts) >= max_samples:
                break

    if not texts:
        raise RuntimeError(f"dataset has no available text in column {text_field}")
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
    """Temporarily remove one layer, evaluate average loss, and restore."""
    parent, layers_list = _resolve_layers_module(model)
    used_attr: Optional[str] = None
    if hasattr(parent, "layers") and isinstance(getattr(parent, "layers"), nn.ModuleList):
        used_attr = "layers"
    elif hasattr(parent, "h") and isinstance(getattr(parent, "h"), nn.ModuleList):
        used_attr = "h"
    else:
        raise RuntimeError("unable to locate layer container attribute for temporary removal")

    original_modules = list(layers_list)
    original_num_layers: Optional[int] = getattr(getattr(model, "config", object()), "num_hidden_layers", None)

    try:
        keep = [m for i, m in enumerate(original_modules) if i != remove_index]
        setattr(parent, used_attr, nn.ModuleList(keep))
        if original_num_layers is not None:
            model.config.num_hidden_layers = len(keep)
        return _compute_reference_loss(model, tokenizer, texts, device, batch_size, max_length)
    finally:
        setattr(parent, used_attr, nn.ModuleList(original_modules))
        if original_num_layers is not None:
            model.config.num_hidden_layers = original_num_layers


def _prune_layers_inplace(model: nn.Module, remove_indices: List[int]) -> None:
    parent, layers_list = _resolve_layers_module(model)
    keep = [m for i, m in enumerate(layers_list) if i not in set(remove_indices)]

    # Assign back as ModuleList to the parent
    if hasattr(parent, "layers") and isinstance(getattr(parent, "layers"), nn.ModuleList):
        parent.layers = nn.ModuleList(keep)  # type: ignore[attr-defined]
    elif hasattr(parent, "h") and isinstance(getattr(parent, "h"), nn.ModuleList):
        parent.h = nn.ModuleList(keep)  # type: ignore[attr-defined]
    else:
        raise RuntimeError("unable to write back pruned layer list")

    # Update config if present
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(keep)


def main() -> int:
    parser = argparse.ArgumentParser(description="prune Qwen/Qwen3 model by layer (approximate selection minimizing error)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base", help="model name or local path")
    parser.add_argument("--dataset", default="data/processed_data/openr1-math/0_6/eval", help="Hugging Face dataset ID or local dataset directory (preferred)")
    parser.add_argument("--text_field", default="real_text", help="text field name, default real_text")
    parser.add_argument("--output", default="model/qwen3_0.6_pruned", help="directory to save pruned model")
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
    parent, layers_list = _resolve_layers_module(model)
    num_layers = len(layers_list)
    per_layer_loss: List[float] = []
    per_layer_ppl: List[float] = []
    for li in range(num_layers):
        cand_loss = _compute_loss_with_temp_removed_layer(
            model, tokenizer, texts, device, args.batch_size, args.max_length, li
        )
        cand_ppl = float(math.exp(cand_loss))
        per_layer_loss.append(cand_loss)
        per_layer_ppl.append(cand_ppl)
        print(f"layer {li} -> PPL: {cand_ppl:.4f}, ΔPPL: {cand_ppl - ref_ppl:.4f}")

    # Choose layers with minimal impact (smallest increase in loss)
    k = min(args.num_prune, num_layers - 1)
    order = sorted(range(num_layers), key=lambda i: (per_layer_ppl[i] - ref_ppl, per_layer_ppl[i]))
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