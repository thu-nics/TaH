"""Step-0 of the SFT pipeline: produce per-token "mismatch" labels.

For every assistant token in a labelled-conversation dataset, we use a small
language model (the "test model") to predict the next token from the prefix
and compare it to the ground-truth next token. Tokens where the prediction
disagrees are marked ``mismatch=1`` — those are the "hard" tokens the TaH
adapter learns to refine via extra iterations.

Inputs:
  --dataset_path <path>.jsonl|.json
      Each record: ``{"conversations": [{"from": "human"|"assistant", "value": ...}], "system": ...}``
      Or, alternatively, ``{"problem"|"question": ..., "output"|"solution"|"answer": ...}``.

Outputs (under ``--output_path``):
  - HuggingFace ``Dataset`` with columns: ``data_id``, ``real_text``,
    ``real_token``, ``mask`` (1 for assistant tokens), ``mismatch``
    (1 where SLM next-token != GT next-token), and optional ``entropy`` /
    ``cross_entropy`` columns.
  - ``args.json`` capturing the CLI flags this run was invoked with.

Public TaH supported MobileLLM and DeepSeek-R1 chat templates in addition
to Qwen3. The cleaned version only supports Qwen3 — the canonical SFT
recipes use ``Qwen/Qwen3-1.7B`` as the labeller. Trim the others if you
need them by reintroducing the per-model think-token IDs in
:func:`_categorize_masks` and the per-model template in :func:`_format_prompt`.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tah.model.utils import sample_next_token


# ────────────────────────────────────────────────────────────────────────────
# Process management (graceful Ctrl-C, multi-process orchestration)
# ────────────────────────────────────────────────────────────────────────────


_running_processes: List[mp.Process] = []
QWEN3_THINK_TOKEN_ID = 151667  # <think> token in Qwen3 tokenizer


def _signal_handler(signum, frame) -> None:
    """SIGINT/SIGTERM: terminate child workers; force-kill if they hang."""
    print(f"\nReceived signal {signum}, cleaning up processes…")
    for p in _running_processes:
        if not p.is_alive():
            continue
        print(f"Terminating process {p.pid}…")
        p.terminate()
        p.join(timeout=60)
        if p.is_alive():
            print(f"Force killing process {p.pid}…")
            p.kill()
            p.join()
    sys.exit(0)


# ────────────────────────────────────────────────────────────────────────────
# Dataset I/O
# ────────────────────────────────────────────────────────────────────────────


def _load_dataset(path: str, index_range: Optional[Tuple[int, int]], random_num: Optional[int]) -> List[dict]:
    """Load a ``.jsonl`` or ``.json`` file into a list of dicts.

    ``index_range`` slices ``[start, end)``; ``random_num`` then randomly
    subsamples (seeded) when smaller than the slice.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        data = loaded if isinstance(loaded, list) else [loaded]
    else:
        print(f"Warning: unknown extension {ext!r}; trying JSONL")
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} samples from {path}")
    if index_range:
        s, e = index_range
        data = data[s:e]
        print(f"Sliced [{s}:{e}] → {len(data)} samples")
    if random_num and 0 < random_num < len(data):
        import random
        random.seed(42)
        data = random.sample(data, random_num)
        print(f"Random-sampled {random_num} of available")
    return data


def _split_indices(n: int, k: int) -> List[Tuple[int, int]]:
    """Split ``[0, n)`` into ``k`` near-equal contiguous ranges."""
    base, rem = divmod(n, k)
    out, s = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        out.append((s, s + size))
        s += size
    return out


# ────────────────────────────────────────────────────────────────────────────
# Prompt formatting (Qwen3-only)
# ────────────────────────────────────────────────────────────────────────────


def _parse_conversations(conversations: List[dict]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Pull (user, reasoning, response) out of a "conversations" list.

    A response wrapped in ``<think>...</think>`` is split into reasoning and
    final response; otherwise ``(reasoning, response) = (None, None)`` and
    the caller should skip the sample.
    """
    user, asst = None, None
    for c in conversations:
        if c["from"] in ("human", "user"):
            user = c["value"]
        elif c["from"] == "assistant":
            asst = c["value"]
    if not user or not asst:
        return None, None, None
    if "<think>" in asst and "</think>" in asst:
        s = asst.find("<think>") + len("<think>")
        e = asst.find("</think>")
        return user, asst[s:e].strip(), asst[e + len("</think>"):].strip()
    return None, None, None


def _format_prompt(sample: dict, tokenizer) -> Optional[str]:
    """Build the chat-templated prompt that we'll prefill on the SLM.

    Two record formats are accepted:
      * ``{"conversations": [...], "system": ...}``  (tracker-style)
      * ``{"problem"|"question": ..., "output"|"solution"|"answer"|...: ...}``
    """
    if "conversations" in sample:
        user, reasoning, response = _parse_conversations(sample["conversations"])
        if not user or response is None:
            return None
        content = f"{reasoning}\n</think>\n\n{response}" if reasoning else response
        msgs = []
        if sample.get("system"):
            msgs.append({"role": "system", "content": sample["system"]})
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": content}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True,
        )

    question = sample.get("problem") or sample.get("question") or ""
    answer = (
        sample.get("output") or sample.get("solution")
        or sample.get("generation") or sample.get("answer") or ""
    )
    if not question or not answer:
        return None
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )


def _categorize_masks(input_ids: torch.Tensor) -> List[int]:
    """1 for tokens at and after the assistant's <think> token; 0 before.

    The mismatch labels are only meaningful for assistant content (mask=1).
    """
    masks: List[int] = []
    current = 0
    for tok in input_ids[0].tolist():
        if tok == QWEN3_THINK_TOKEN_ID:
            current = 1
        masks.append(current)
    return masks


# ────────────────────────────────────────────────────────────────────────────
# Mismatch scoring
# ────────────────────────────────────────────────────────────────────────────


def _calculate_mismatch(predictions: torch.Tensor, real_tokens: torch.Tensor, data_ids: torch.Tensor) -> torch.Tensor:
    """``mismatch[i] = 1`` iff ``predictions[i] != real_tokens[i+1]`` and
    position ``i`` is not the last token of its sample.

    Sample boundaries are derived from where ``data_ids`` changes. The very
    last token of each sample has no "next token" to predict, so we leave its
    mismatch at 0 by masking it out before the comparison.
    """
    device = predictions.device
    mismatch = torch.zeros_like(predictions, dtype=torch.int32, device=device)
    padded = torch.cat([data_ids, torch.tensor([data_ids[-1] + 1], device=device)])
    sample_ends = torch.where(padded[1:] != padded[:-1])[0]
    valid = torch.ones(len(predictions), dtype=torch.bool, device=device)
    valid[sample_ends] = False
    if valid.any():
        idx = torch.where(valid)[0]
        mismatch[idx] = (predictions[idx] != real_tokens[idx + 1]).int()
    return mismatch.cpu()


# ────────────────────────────────────────────────────────────────────────────
# Per-GPU worker
# ────────────────────────────────────────────────────────────────────────────


def _process_single_gpu(args, device_id: int, data_range: Tuple[int, int], model_name: str) -> None:
    """Run the SLM forward pass on one GPU's slice of the dataset."""
    start_idx, end_idx = data_range
    print(f"GPU {device_id}: range {start_idx}-{end_idx} of {model_name}")

    dataset = _load_dataset(args.dataset_path, (start_idx, end_idx), args.random_num)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=f"cuda:{device_id}", torch_dtype=torch.bfloat16,
        ).eval()
    except Exception as e:
        print(f"GPU {device_id}: model load failed: {e}")
        return

    # Per-token outputs aggregated across the GPU's samples.
    preds, reals, tok_ids, data_ids, masks = [], [], [], [], []
    entropies: Optional[List[torch.Tensor]] = [] if args.save_entropy else None
    ces: Optional[List[torch.Tensor]] = [] if args.save_ce else None

    bs = max(1, int(args.batch_size))
    pbar = tqdm(total=len(dataset), desc=f"GPU {device_id}", position=device_id)
    with torch.no_grad():
        for batch_start in range(0, len(dataset), bs):
            batch_end = min(batch_start + bs, len(dataset))
            batch_ids: List[torch.Tensor] = []
            batch_meta: List[Tuple[int, int]] = []  # (length, global_data_id)

            for offset, sample in enumerate(dataset[batch_start:batch_end]):
                global_id = start_idx + batch_start + offset
                prompt = _format_prompt(sample, tokenizer)
                if prompt is None:
                    continue
                ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                if ids.shape[-1] > args.max_input_length:
                    if args.is_cutoff:
                        ids = ids[: args.max_input_length]
                    else:
                        continue
                batch_ids.append(ids)
                batch_meta.append((ids.shape[-1], global_id))

            if not batch_ids:
                pbar.update(batch_end - batch_start)
                continue

            lengths = [t.shape[0] for t in batch_ids]
            padded = pad_sequence(batch_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attn = torch.arange(padded.shape[1]).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
            input_ids = padded.to(model.device)
            attention_mask = attn.to(model.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.to(torch.float32)

            for b_idx, (seq_len, global_id) in enumerate(batch_meta):
                seq_logits = logits[b_idx, :seq_len, :]
                lp = F.log_softmax(seq_logits, dim=-1) if (args.save_entropy or args.save_ce) else None
                if args.save_entropy:
                    entropies.append((-lp.exp() * lp).sum(dim=-1).cpu())
                pred = sample_next_token(
                    seq_logits, temperature=args.temperature, top_p=args.top_p,
                    top_k=max(args.top_k, 0), do_sample=args.temperature > 0,
                ).cpu()

                token_id = torch.arange(seq_len, dtype=torch.long)
                data_id = torch.full((seq_len,), global_id, dtype=torch.long)
                real = input_ids[b_idx, :seq_len].detach().cpu()
                mask_ = torch.tensor(_categorize_masks(real.unsqueeze(0)), dtype=torch.int32)

                if args.save_ce:
                    # Per-token CE w.r.t. the next ground-truth token, masked
                    # out at sample boundaries (last position has no target).
                    valid = torch.ones(seq_len, dtype=torch.bool, device=lp.device)
                    valid[-1] = False
                    targets = real.to(lp.device)
                    ce = torch.zeros(seq_len, dtype=torch.float32, device=lp.device)
                    if valid.any():
                        idx = torch.where(valid)[0]
                        ce[idx] = -lp[idx, targets[idx + 1]]
                    ces.append(ce.cpu())

                preds.append(pred)
                reals.append(real)
                tok_ids.append(token_id)
                data_ids.append(data_id)
                masks.append(mask_)

            pbar.update(batch_end - batch_start)
            if (batch_start // bs) % 10 == 0:
                torch.cuda.empty_cache()
    pbar.close()

    if not preds:
        print(f"GPU {device_id}: no valid samples")
        return

    cat_preds = torch.cat(preds)
    cat_reals = torch.cat(reals)
    cat_data_ids = torch.cat(data_ids)
    print(f"GPU {device_id}: computing mismatch…")
    mismatch = _calculate_mismatch(cat_preds, cat_reals, cat_data_ids)

    out: dict = {
        "predictions": cat_preds.tolist(),
        "small_token": torch.cat(tok_ids).tolist(),
        "data_id": cat_data_ids.tolist(),
        "mask": torch.cat(masks).tolist(),
        "real_token": cat_reals.tolist(),
        "mismatch": mismatch.tolist(),
    }
    if args.save_entropy:
        out["entropy"] = torch.cat(entropies).tolist()
    if args.save_ce:
        out["cross_entropy"] = torch.cat(ces).tolist()

    out_path = os.path.join(args.output_path, f"results_gpu_{device_id}_{model_name.split('/')[-1]}")
    Dataset.from_dict(out).save_to_disk(out_path)
    print(f"GPU {device_id}: saved → {out_path}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device_id)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# ────────────────────────────────────────────────────────────────────────────
# Per-GPU result merge + final dataset
# ────────────────────────────────────────────────────────────────────────────


def _process_and_convert_dataset(merged: Dataset, model_name: str, output_path: str) -> Dataset:
    """Group per-token results by ``data_id`` into one row per sample, decode
    text, and write the SFT-ready dataset to ``output_path``.

    Also prints a brief stats summary (basic counts, mask split, mismatch
    ratio). Public TaH wrote a multi-page text/JSON/CSV analysis report;
    rich enough to be its own postprocess script — kept the inline summary,
    dropped the report files.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = merged.to_pandas()
    grouped = df.groupby("data_id")
    print(f"Found {len(grouped)} unique data_ids")

    final = []
    for data_id, group in tqdm(grouped, desc="Grouping by data_id"):
        real = group["real_token"].tolist()
        item = {
            "data_id": data_id,
            "real_text": tokenizer.decode(real),
            "real_token": real,
            "mask": group["mask"].tolist(),
            "mismatch": group["mismatch"].tolist(),
        }
        if "entropy" in group.columns:
            item["entropy"] = group["entropy"].tolist()
        if "cross_entropy" in group.columns:
            item["cross_entropy"] = group["cross_entropy"].tolist()
        final.append(item)

    total_tokens = len(df)
    total_mismatch = int(df["mismatch"].sum())
    asst_tokens = int((df["mask"] == 1).sum())
    asst_mismatch = int(df.loc[df["mask"] == 1, "mismatch"].sum())
    print("\n=== Label statistics ===")
    print(f"  samples: {df['data_id'].nunique():,}")
    print(f"  tokens: {total_tokens:,}  (mismatch {total_mismatch:,}, {100*total_mismatch/max(total_tokens,1):.2f}%)")
    print(f"  assistant tokens: {asst_tokens:,}  (mismatch {asst_mismatch:,}, {100*asst_mismatch/max(asst_tokens,1):.2f}%)")
    print("========================")

    processed = Dataset.from_pandas(pd.DataFrame(final))
    processed.save_to_disk(output_path)
    print(f"Processed dataset → {output_path}")
    return processed


def _merge_gpu_results(args, model_name: str) -> Optional[Dataset]:
    """Load and concatenate per-GPU results, then collapse into the final SFT dataset."""
    short = model_name.split("/")[-1]
    parts = []
    for gpu_id in range(args.num_gpu):
        d = os.path.join(args.output_path, f"results_gpu_{gpu_id}_{short}")
        if os.path.exists(d):
            parts.append(Dataset.load_from_disk(d))
            print(f"Loaded GPU {gpu_id} dataset")
    if not parts:
        print("No GPU datasets found to merge")
        return None
    merged = concatenate_datasets(parts)
    final = _process_and_convert_dataset(merged, model_name, args.output_path)
    for gpu_id in range(args.num_gpu):
        d = os.path.join(args.output_path, f"results_gpu_{gpu_id}_{short}")
        if os.path.exists(d):
            shutil.rmtree(d)
    return final


# ────────────────────────────────────────────────────────────────────────────
# Multi-GPU orchestration
# ────────────────────────────────────────────────────────────────────────────


def _terminate_with_timeout(p: mp.Process, soft: int = 30, hard: int = 60) -> None:
    """Best-effort cleanup of one worker process (terminate → kill if still alive)."""
    if not p.is_alive():
        return
    p.terminate()
    p.join(timeout=soft)
    if p.is_alive():
        print(f"Force killing process {p.pid}…")
        p.kill()
        p.join(timeout=hard)


def _process_dataset_multi_gpu(args) -> None:
    """Spawn one worker per GPU per model, wait, then merge results."""
    global _running_processes
    os.makedirs(args.output_path, exist_ok=True)

    full = _load_dataset(args.dataset_path, args.index_range, args.random_num)
    print(f"Total samples: {len(full)}")
    splits = _split_indices(len(full), args.num_gpu)
    print(f"Per-GPU splits: {splits}")

    for model_name in args.test_model_list:
        print(f"\n=== Processing model: {model_name} ===")
        processes = [
            mp.Process(target=_process_single_gpu, args=(args, gpu_id, splits[gpu_id], model_name))
            for gpu_id in range(args.num_gpu)
        ]
        _running_processes = processes
        for p in processes:
            p.start()
        timeout = 24 * 60 * 60
        for i, p in enumerate(processes):
            try:
                p.join(timeout=timeout)
                if p.is_alive():
                    print(f"GPU {i}: process timed out, terminating…")
                    _terminate_with_timeout(p)
                elif p.exitcode != 0:
                    print(f"GPU {i}: exited with code {p.exitcode}")
                else:
                    print(f"GPU {i}: completed")
            except Exception as e:
                print(f"GPU {i}: error during join: {e}")
                _terminate_with_timeout(p)
        for p in processes:
            _terminate_with_timeout(p)
        _running_processes = []

        if _merge_gpu_results(args, model_name) is None:
            print(f"Failed to merge results for {model_name}")

    print("Multi-GPU labelling complete.")


# ────────────────────────────────────────────────────────────────────────────
# CLI entry
# ────────────────────────────────────────────────────────────────────────────


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(description="Multi-GPU SLM prefill → mismatch labelling")
    parser.add_argument("--dataset_path", required=True, help="JSONL or JSON conversations dataset")
    parser.add_argument("--num_gpu", type=int, default=4)
    parser.add_argument("--test_model_list", nargs="+", required=True, help="HF model id(s) for the SLM labeller")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_input_length", type=int, default=32768)
    parser.add_argument("--is_cutoff", type=bool, default=False, help="If True, truncate over-long samples; else drop them")
    parser.add_argument("--index_range", nargs=2, type=int, default=None, help="[start, end) slice of the dataset")
    parser.add_argument("--random_num", type=int, default=None, help="Sub-sample N of available; seed=42")
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_entropy", action="store_true", help="Save per-token softmax entropy")
    parser.add_argument("--save_ce", action="store_true", help="Save per-token cross-entropy w.r.t. next-token labels")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    try:
        _process_dataset_multi_gpu(args)
        with open(os.path.join(args.output_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args.__dict__, f, indent=2)
        print("All processing complete.")
    except Exception:
        for p in _running_processes:
            _terminate_with_timeout(p)
        raise
    finally:
        _running_processes.clear()


if __name__ == "__main__":
    main()
