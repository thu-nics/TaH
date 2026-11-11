"""Common-random-number primitives for coupled sampling (R1-O2).

The engine draws token j of a request with seed s from
``request_uniform(seeds=s, counters=j)`` (counter-based SplitMix64). The functions
here give the same uniform on the CPU and an inverse-CDF draw that replaces
``torch.multinomial`` on the reference/oracle side without changing the sampling
distribution. Candidates are ordered by probability (descending), ties by token
ID (ascending): the order of the engine's shortlist, so two nearby distributions
map the same uniform to the same token.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from tah.engine.sampling import RequestRNG, request_uniform

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15  # == -7046029254386353131 as signed int64


def uniform(seed: int, index: int) -> float:
    """Engine uniform for (request seed, generated-token index), pure Python ints."""
    z = (seed + _GAMMA * (index + 1)) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    z ^= z >> 31
    return ((z >> 12) + 0.5) * 2.0 ** -52


def uniforms(seeds, indices, device="cpu") -> torch.Tensor:
    """Vectorized engine uniforms [B] float64, computed by the engine's own function."""
    seeds = torch.as_tensor(seeds, dtype=torch.int64, device=device)
    counters = torch.as_tensor(indices, dtype=torch.int64, device=device)
    return request_uniform(RequestRNG(seeds, counters))


def inverse_cdf_draw(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Token per row of [B, V] nonnegative weights for uniforms u [B] in (0, 1).

    Weights are renormalized in float64; the draw is the first candidate (in
    probability-descending, token-ascending order) whose cumulative weight reaches
    u times the total (the engine's rule). Zero-weight tokens are never drawn.
    """
    if probs.ndim != 2 or u.shape != probs.shape[:1]:
        raise ValueError("probs must be [B, V] and u [B]")
    p = probs.double()
    # Stable sort of ids by descending weight == (weight desc, id asc).
    order = torch.sort(p, dim=-1, descending=True, stable=True).indices
    ranked = p.gather(-1, order)
    cdf = ranked.cumsum(-1)
    target = u.to(device=p.device, dtype=torch.float64)[:, None] * cdf[:, -1:]
    slot = (cdf < target).sum(-1)  # first cdf >= target, as the engine
    # Never step past the last positive weight (guards u*total rounding at the top).
    last = (ranked > 0).sum(-1) - 1
    slot = torch.minimum(slot, last)
    return order.gather(-1, slot[:, None]).squeeze(-1)


@contextmanager
def multinomial_patch(provider, observer=None):
    """Replace torch.multinomial(probs, 1) by inverse_cdf_draw(probs, provider(B)).

    ``provider(batch_size) -> float64 [B]`` returns the uniforms of the current
    call (e.g. ``uniforms(seeds, [step] * B)``). Only num_samples == 1 is allowed.
    """
    original = torch.multinomial

    def draw(probs, num_samples, *args, **kwargs):
        if num_samples != 1 or args or kwargs.get("replacement") or kwargs.get("generator") is not None:
            raise ValueError("coupled multinomial supports num_samples=1 without generator only")
        rows = probs if probs.ndim == 2 else probs[None]
        u = provider(rows.shape[0]).to(rows.device)
        token = inverse_cdf_draw(rows, u)
        if observer is not None:
            observer(rows, u, token)
        return token[:, None] if probs.ndim == 2 else token
    torch.multinomial = draw
    try:
        yield
    finally:
        torch.multinomial = original
