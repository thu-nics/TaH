"""Microbenchmarks for tah/model + tah/evaluate hot paths.

Run::

    python tests/bench.py components            # iterated helpers, ~1s each
    python tests/bench.py e2e                   # real TaH-plus-1.7B forward (~5s)
    python tests/bench.py all                   # both, then print a summary
    python tests/bench.py all --json out.json   # also write a json report

Each component is timed as ``(warmup × WARMUP) → measured × ITERS``, with a
``torch.cuda.synchronize`` before the timer start and after the last run, so
the reported numbers exclude lazy-init / first-call compile costs.

The benchmark is the verification surface for the speed-impacting refactors
landing alongside this file: capture a baseline before each change, run the
same command after, look at the printed delta. Component shapes are chosen
to roughly mirror what the wrapper sees during a prefill on TaH-plus-1.7B
(``B=2, T=64, V=151936, H=2048, L=28``).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch


# ────────────────────────────────────────────────────────────────────────────
# Bench infra
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    name: str
    ms: float
    iters: int
    warmup: int
    extra: Dict[str, float] = field(default_factory=dict)


def time_fn(fn: Callable[[], None], *, warmup: int = 5, iters: int = 30) -> float:
    """Median-of-iters wall-clock per call in ms; CUDA-synced if available."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]  # median


def _print_table(results: List[BenchResult]) -> None:
    if not results:
        return
    name_w = max(len(r.name) for r in results) + 2
    print(f"{'name':<{name_w}}{'ms':>10}{'iters':>10}{'extra':>20}")
    print("-" * (name_w + 40))
    for r in results:
        extra = " ".join(f"{k}={v:.3g}" for k, v in r.extra.items()) if r.extra else ""
        print(f"{r.name:<{name_w}}{r.ms:>10.3f}{r.iters:>10}{extra:>20}")


# ────────────────────────────────────────────────────────────────────────────
# Component benchmarks
# ────────────────────────────────────────────────────────────────────────────


def bench_components(device: str) -> List[BenchResult]:
    from tah.model.iter_decider import MLPIterDecider
    from tah.model.loss import IterDeciderLoss, NextTokenPredLoss
    from tah.model.tah_model import (
        additive_logits_update,
        gather_active,
        scatter_back,
        topk_softmax_input_update,
    )

    # Shapes chosen to mirror a realistic TaH-plus-1.7B prefill.
    B, T, V, H, L = 2, 64, 151936, 2048, 28
    K = 100  # input_updater topk

    results: List[BenchResult] = []
    g = torch.Generator(device=device).manual_seed(0)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # ── input_updater ───────────────────────────────────────────────────
    logits = torch.randn(B, T, V, generator=g, device=device, dtype=dtype)
    embed_w = torch.randn(V, H, generator=g, device=device, dtype=dtype)
    results.append(BenchResult(
        name="topk_softmax_input_update",
        ms=time_fn(lambda: topk_softmax_input_update(logits, embed_w, K)),
        iters=30, warmup=5,
        extra={"shape": float(B * T * V), "topk": float(K)},
    ))

    # ── output_updater (additive) ───────────────────────────────────────
    a = torch.randn(B, T, V, generator=g, device=device, dtype=dtype)
    b = torch.randn(B, T, V, generator=g, device=device, dtype=dtype)
    results.append(BenchResult(
        name="additive_logits_update",
        ms=time_fn(lambda: additive_logits_update(a, b)),
        iters=30, warmup=5,
    ))

    # ── gather_active / scatter_back ────────────────────────────────────
    # Mask: half the positions active (typical mid-iteration state).
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[:, :T // 2] = True
    mask = mask[:, torch.randperm(T)]
    embeds = torch.randn(B, T, H, generator=g, device=device, dtype=dtype)
    pos_ids = torch.arange(T, device=device).expand(B, T).clone()
    valid = torch.ones(B, T, dtype=torch.long, device=device)
    results.append(BenchResult(
        name="gather_active",
        ms=time_fn(lambda: gather_active(mask, embeds, pos_ids, valid)),
        iters=30, warmup=5,
        extra={"active_frac": float(mask.float().mean().item())},
    ))

    # scatter_back source comes from gather_active, so warm both up together.
    pad_mask, *gathered = gather_active(mask, embeds)
    src = gathered[0]
    dest = torch.zeros_like(embeds)
    results.append(BenchResult(
        name="scatter_back",
        ms=time_fn(lambda: scatter_back(mask, src=src, dest=dest)),
        iters=30, warmup=5,
    ))
    assignment = torch.zeros(B, T, dtype=torch.bool, device=device)
    assignment[:, T // 4: 3 * T // 4] = True
    results.append(BenchResult(
        name="scatter_back+mask",
        ms=time_fn(lambda: scatter_back(mask, src=src, dest=dest, assignment_mask=assignment)),
        iters=30, warmup=5,
    ))

    # ── MLPIterDecider.forward ──────────────────────────────────────────
    torch.manual_seed(7)
    decider = MLPIterDecider(
        topk=K, hidden_states_size=H,
        hidden_states_layer_nums=[2, 10, 18, 26],
        hidden_dims=[512, 512, 512, 512, 512, 512],
        expansion_factor=4, dropout_rate=0.1, normalize_input=False,
        threshold=0.9, max_iter=2, dtype=dtype,
    ).to(device)
    decider_logits = logits[mask]  # (n_active, V)
    decider_hidden = torch.randn(decider_logits.shape[0], L, H, generator=g, device=device, dtype=dtype)
    results.append(BenchResult(
        name="MLPIterDecider.forward",
        ms=time_fn(lambda: decider(logits=decider_logits, iter_depth=1, all_hidden_states=decider_hidden)),
        iters=30, warmup=5,
        extra={"n_active": float(decider_logits.shape[0])},
    ))

    # ── NextTokenPredLoss.final_loss_func ───────────────────────────────
    labels = torch.randint(0, V, (B, T), generator=g, device=device, dtype=torch.long)
    iter_count = torch.randint(0, 3, (B, T), generator=g, device=device, dtype=torch.long)
    nl = NextTokenPredLoss()
    nl.prepare_loss(B, T, device, torch.float32)
    results.append(BenchResult(
        name="NextTokenPredLoss.final",
        ms=time_fn(lambda: nl.final_loss_func(
            logits=logits, labels_shifted=labels, iter_count=iter_count,
            training=False, num_items_in_batch=int((labels != -100).sum().item()),
        )),
        iters=30, warmup=5,
    ))

    # ── IterDeciderLoss.intra_iter_loss_func ────────────────────────────
    # The wrapper hands the loss a flattened tensor of `valid` continue logits;
    # mirror that here so the bench matches the production call shape.
    n_valid = int(mask.sum().item())
    continue_logits = torch.randn(n_valid, generator=g, device=device, dtype=torch.float32)
    iter_count_labels = torch.randint(0, 3, (B, T), generator=g, device=device, dtype=torch.long)
    valid_mask = mask.long()
    bce = IterDeciderLoss(pos_weight=2.0, skip_last_iter=True, max_iter=2)
    bce.prepare_loss(B, T, device, torch.float32)

    def _intra():
        bce.iter_decider_loss_per_token = torch.zeros(B, T, device=device, dtype=torch.float32)
        bce.intra_iter_loss_func(
            active_logits=None, current_iter_mask=mask, active_labels_shifted=None,
            active_valid_continue_logits=continue_logits, active_valid_mask=valid_mask,
            iter_depth=1, active_iter_count_labels=iter_count_labels,
            iter_decider_threshold=0.5,
        )
    results.append(BenchResult(
        name="IterDeciderLoss.intra",
        ms=time_fn(_intra), iters=30, warmup=5,
        extra={"n_valid": float(n_valid)},
    ))

    return results


# ────────────────────────────────────────────────────────────────────────────
# End-to-end TaH-plus-1.7B forward
# ────────────────────────────────────────────────────────────────────────────


def bench_e2e(device: str) -> List[BenchResult]:
    if device != "cuda":
        print("[skip] e2e bench requires CUDA")
        return []
    from transformers import AutoTokenizer
    from tah.model.tah_model import TaHForCausalLM
    from tah.model.utils import TaHForCasualLM_generate

    ckpt = os.environ.get("TAH_CHECKPOINT", "nics-efc/TaH-plus-1.7B")
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = TaHForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa",
    )

    # Forward latency on a small prompt (representative of one decode step
    # except for KV cache, which we skip here for a clean per-call number).
    text = "Compute 17 + 25. Reply with a single integer."
    inp = tok(text, return_tensors="pt").to(model.device)

    def _forward():
        with torch.no_grad():
            model(**inp, use_cache=False)

    fwd = BenchResult(
        name="TaHForCausalLM.forward",
        ms=time_fn(_forward, warmup=3, iters=10),
        iters=10, warmup=3,
        extra={"input_len": float(inp["input_ids"].shape[1])},
    )

    # Generation: 32 tokens greedy, full TaHForCasualLM_generate path.
    def _generate():
        with torch.no_grad():
            TaHForCasualLM_generate(
                tah_model=model, tokenizer=tok, model_inputs=dict(inp),
                max_new_tokens=32, do_sample=False, verbose=False,
            )

    gen = BenchResult(
        name="TaHForCasualLM_generate(32)",
        ms=time_fn(_generate, warmup=2, iters=5),
        iters=5, warmup=2,
        extra={"new_tokens": 32.0},
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [fwd, gen]


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="TaH benchmarks")
    parser.add_argument("scope", choices=("components", "e2e", "all"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", default=None, help="Write all results to a json file")
    args = parser.parse_args()

    print(f"Device: {args.device}\n")
    all_results: List[BenchResult] = []
    if args.scope in ("components", "all"):
        print("=== components ===")
        cr = bench_components(args.device)
        _print_table(cr)
        print()
        all_results.extend(cr)
    if args.scope in ("e2e", "all"):
        print("=== end-to-end ===")
        er = bench_e2e(args.device)
        _print_table(er)
        print()
        all_results.extend(er)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": args.device, "results": [asdict(r) for r in all_results]}, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
