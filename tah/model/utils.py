"""Helpers used across ``tah/`` — config (de)serialisation, generation, debug
colouring, and a couple of small attribute / param utilities for trainers.

Public TaH bundled a much larger ``utils.py`` (device-map juggling, an unused
class-string-to-type helper, an ``get_attr_recursive`` whose only caller was
``recurrent_transformer.py``); those have been removed.
"""
from __future__ import annotations

import os
import random
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from tah.model.tah_model import TaHForCausalLM


# ────────────────────────────────────────────────────────────────────────────
# Determinism / param introspection (used by SFT_TaH.py).
# ────────────────────────────────────────────────────────────────────────────


def set_all_seeds(seed: int = 42) -> None:
    """Set Python, NumPy, PyTorch (CPU+CUDA) and HF Transformers seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"All random seeds set to {seed}")


def _get_attr_by_path(root_obj, attr_path: str):
    """Resolve dotted attribute path; return None if any hop is missing."""
    obj = root_obj
    for name in attr_path.split("."):
        if not hasattr(obj, name):
            return None
        obj = getattr(obj, name)
    return obj


def freeze_components(model, component_paths, accelerator) -> None:
    """Freeze parameters of components named by dotted paths under the model.

    Path segments may optionally start with ``model.`` (stripped). Missing
    components are reported via ``accelerator.print`` rather than raising,
    matching the public-TaH behaviour.
    """
    if not component_paths:
        return
    for raw_path in component_paths:
        path = raw_path[len("model."):] if raw_path.startswith("model.") else raw_path
        target = _get_attr_by_path(model, path)
        if target is None:
            accelerator.print(f"Warning: freeze_component {raw_path!r} not found on model.")
            continue
        params = list(target.parameters()) if hasattr(target, "parameters") else []
        if not params:
            accelerator.print(f"Warning: freeze_component {raw_path!r} has no parameters to freeze.")
            continue
        for p in params:
            p.requires_grad = False
        accelerator.print(f"Froze component {raw_path!r} ({sum(p.numel() for p in params):,} params).")


def compute_trainable_param_size_gb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 ** 3)


# ────────────────────────────────────────────────────────────────────────────
# Sampling. Used by ``TaHForCasualLM_generate`` below; kept separate so other
# generation code paths can pull just the sampler.
# ────────────────────────────────────────────────────────────────────────────


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    do_sample: bool = True,
) -> torch.Tensor:
    """Sample one token per row from ``(B, V)`` logits with the usual knobs.

    ``do_sample=False`` ⇒ argmax. Otherwise: temperature → softmax → optional
    min_p / top_k / top_p filtering → multinomial.
    """
    if not do_sample:
        return torch.argmax(logits, dim=-1)

    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if min_p > 0.0:
        max_probs = probs.max(dim=-1, keepdim=True).values
        probs = torch.where(probs >= min_p * max_probs, probs, torch.zeros_like(probs))
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if top_k > 0:
        k = min(top_k, probs.size(-1))
        topk_probs, _ = torch.topk(probs, k, dim=-1)
        probs = torch.where(probs < topk_probs[..., -1:], torch.zeros_like(probs), probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = 0
        remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_indices, sorted_remove)
        probs = torch.where(remove, torch.zeros_like(probs), probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
# Generation loop + token coloring (used by the playground demo).
# ────────────────────────────────────────────────────────────────────────────


class IterCountColors:
    """ANSI-colour the per-token iteration counts during generation."""

    _COLORS = {
        1: "\033[0m",   # reset (white)
        2: "\033[92m",  # green
        3: "\033[94m",  # blue
        4: "\033[91m",  # red
        5: "\033[95m",  # magenta
        6: "\033[93m",  # yellow
    }

    @classmethod
    def get_color(cls, n: int) -> str:
        return cls._COLORS.get(n, "\033[96m")  # cyan for >6

    @classmethod
    def print_token(cls, token_text: str, n: int) -> None:
        print(f"{cls.get_color(n)}{token_text}\033[0m", end="", flush=True)

    @classmethod
    def get_legend(cls) -> str:
        return "Color legend: " + ", ".join(
            f"{cls.get_color(n)}={n} iter\033[0m" for n in (1, 2, 3, 4, 5, 6, 7)
        )


def _forward_and_print(
    tah_model: "TaHForCausalLM",
    tokenizer: AutoTokenizer,
    model_inputs: dict,
    cache,
    *,
    new_sequence: bool,
    verbose: bool,
    **kwargs,
):
    """One forward pass; if ``verbose``, print each non-padded token coloured
    by its iteration count."""
    forward_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "past_key_values": cache,
        "use_cache": True,
        "new_sequence": new_sequence,
        **{k: v for k, v in model_inputs.items() if k != "input_ids" and v is not None},
        **kwargs,
    }
    outputs = tah_model(**forward_kwargs)

    if verbose and outputs.iter_count is not None:
        attention_mask = model_inputs.get("attention_mask")
        counts = outputs.iter_count[0]
        if attention_mask is not None:
            valid = attention_mask[0, -counts.shape[0]:] == 1
            tokens = [tokenizer.decode([t]) for t in model_inputs["input_ids"][0][valid]]
            counts = counts[valid]
        else:
            tokens = [tokenizer.decode([t]) for t in model_inputs["input_ids"][0]]
        for tok, c in zip(tokens, counts):
            IterCountColors.print_token(tok, int(c.item()))
    return outputs


def TaHForCasualLM_generate(
    tah_model: "TaHForCausalLM",
    tokenizer: AutoTokenizer,
    model_inputs: dict,
    *,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    verbose: bool = True,
    **kwargs,
):
    """Greedy / sampling generation for ``TaHForCausalLM`` with batched inputs.

    Returns ``(output_tokens, generated_texts)`` — the former is per-batch
    list-of-token-ids, the latter their decoded strings.
    """
    device = model_inputs["input_ids"].device
    B = model_inputs["input_ids"].shape[0]
    tah_model.eval()

    cache = None
    output_tokens: list[list[int]] = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    current_attn = model_inputs.get("attention_mask")

    if verbose:
        print("Input tokens with iteration counts:")
    with torch.no_grad():
        outputs = _forward_and_print(
            tah_model, tokenizer, model_inputs, cache,
            new_sequence=True, verbose=verbose, **kwargs,
        )
        cache = outputs.past_key_values
        if verbose:
            print("\n\nGenerating new tokens:")

        for _ in range(max_new_tokens):
            last_logits = outputs.logits[:, -1, :]
            next_ids = sample_next_token(
                last_logits, temperature=temperature, top_p=top_p,
                top_k=top_k, min_p=min_p, do_sample=do_sample,
            )
            if tokenizer.eos_token_id is not None:
                finished = finished | (next_ids == tokenizer.eos_token_id)
            for i in range(B):
                if not finished[i]:
                    output_tokens[i].append(int(next_ids[i].item()))
            if finished.all():
                break

            next_inputs = {"input_ids": next_ids.unsqueeze(1)}
            if current_attn is not None:
                current_attn = torch.cat(
                    [current_attn, torch.ones(B, 1, dtype=current_attn.dtype, device=device)], dim=1,
                )
                next_inputs["attention_mask"] = current_attn

            outputs = _forward_and_print(
                tah_model, tokenizer, next_inputs, cache,
                new_sequence=False, verbose=verbose, **kwargs,
            )
            cache = outputs.past_key_values

    if verbose:
        print("\033[0m")
    return output_tokens, [tokenizer.decode(toks) if toks else "" for toks in output_tokens]
