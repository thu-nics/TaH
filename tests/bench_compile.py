"""Compile vs eager bench for the wrapper's hot helpers.

For each candidate, time eager → compile-warm → compile-steady. ``compile-warm``
is the *first* call (full guard build), ``compile-steady`` is the median of
30 subsequent calls. The deltas tell us whether the helper is worth wrapping
in ``torch.compile`` for inference (steady-state speedup must justify
warmup cost; for serving ~1000s of forwards, even a small steady speedup
amortises).

Run::

    python tests/bench_compile.py
"""
from __future__ import annotations

import time

import torch


def _time(fn, *, iters: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def _first_call_ms(fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    if not torch.cuda.is_available():
        print("compile bench requires CUDA")
        return

    from tah.model.iter_decider import MLPIterDecider
    from tah.model.tah_model import (
        additive_logits_update,
        gather_active,
        scatter_back,
        topk_softmax_input_update,
    )

    device = "cuda"
    dtype = torch.bfloat16
    B, T, V, H, L = 2, 64, 151936, 2048, 28
    K = 100

    g = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(B, T, V, generator=g, device=device, dtype=dtype)
    embed_w = torch.randn(V, H, generator=g, device=device, dtype=dtype)
    a = torch.randn(B, T, V, generator=g, device=device, dtype=dtype)
    b = torch.randn(B, T, V, generator=g, device=device, dtype=dtype)

    print(f"{'name':<35}{'eager':>10}{'compile1':>10}{'compileN':>10}{'speedup':>10}")
    print("-" * 75)

    def _row(name, eager_fn, compile_fn):
        eager_ms = _time(eager_fn)
        compile1_ms = _first_call_ms(compile_fn)
        compileN_ms = _time(compile_fn)
        speedup = eager_ms / compileN_ms if compileN_ms > 0 else float("inf")
        print(f"{name:<35}{eager_ms:>10.3f}{compile1_ms:>10.3f}{compileN_ms:>10.3f}{speedup:>10.2f}x")

    # topk_softmax_input_update
    f = lambda: topk_softmax_input_update(logits, embed_w, K)
    f_c = torch.compile(topk_softmax_input_update)
    _row("topk_softmax_input_update", f, lambda: f_c(logits, embed_w, K))

    # additive_logits_update
    f = lambda: additive_logits_update(a, b)
    f_c = torch.compile(additive_logits_update)
    _row("additive_logits_update", f, lambda: f_c(a, b))

    # gather_active (dynamic max_active, may recompile)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[:, :T // 2] = True
    embeds = torch.randn(B, T, H, generator=g, device=device, dtype=dtype)
    f = lambda: gather_active(mask, embeds)
    f_c = torch.compile(gather_active, dynamic=True)
    _row("gather_active(dynamic)", f, lambda: f_c(mask, embeds))

    # scatter_back (also dynamic max_active)
    pad_mask, gathered = gather_active(mask, embeds)[0], gather_active(mask, embeds)[1]
    src = gathered
    dest = torch.zeros_like(embeds)
    f = lambda: scatter_back(mask, src=src, dest=dest)
    f_c = torch.compile(scatter_back, dynamic=True)
    _row("scatter_back(dynamic)", f, lambda: f_c(mask, src=src, dest=dest))

    # MLPIterDecider.forward
    torch.manual_seed(7)
    decider = MLPIterDecider(
        topk=K, hidden_states_size=H,
        hidden_states_layer_nums=[2, 10, 18, 26],
        hidden_dims=[512, 512, 512, 512, 512, 512],
        expansion_factor=4, dropout_rate=0.1, normalize_input=False,
        threshold=0.9, max_iter=2, dtype=dtype,
    ).to(device).eval()
    decider_logits = logits[mask]
    decider_hidden = torch.randn(decider_logits.shape[0], L, H, generator=g, device=device, dtype=dtype)
    decider_compiled = torch.compile(decider, dynamic=True)
    _row(
        "MLPIterDecider.forward",
        lambda: decider(logits=decider_logits, iter_depth=1, all_hidden_states=decider_hidden),
        lambda: decider_compiled(logits=decider_logits, iter_depth=1, all_hidden_states=decider_hidden),
    )


if __name__ == "__main__":
    main()
