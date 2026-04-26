"""Acc test for ``TaHCache``.

The cache holds per-(layer, iteration) KV plus position/valid metadata, and
exposes "view up to iter N" reads that the wrapper uses to build attention
masks. Cache shape and contents are part of the checkpoint contract with
minisgl-tah, so this test pins the *behaviour*, not just the API.
"""
from __future__ import annotations

import pytest
import torch

from tests._harness import (
    ACC_TOL,
    assert_close,
    capture,
    have_baseline,
    load_baseline,
)


def _cache_inputs(shapes, device):
    B, T, H, L = shapes["B"], shapes["T"], 16, 2  # heads=2, head_dim=H/heads
    n_heads = 2
    head_dim = 8
    g = torch.Generator(device=device).manual_seed(130)

    # Two iterations × two layers, with iter-1 attending to a strict subset.
    def kv(seq_len):
        k = torch.randn(B, n_heads, seq_len, head_dim, generator=g, device=device, dtype=torch.float32)
        v = torch.randn(B, n_heads, seq_len, head_dim, generator=g, device=device, dtype=torch.float32)
        return k, v

    iter0_layer = [kv(T) for _ in range(L)]
    iter1_layer = [kv(T - 2) for _ in range(L)]
    pos_iter0 = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)
    pos_iter1 = torch.arange(T - 2, dtype=torch.long, device=device).unsqueeze(0).expand(B, T - 2)
    valid_iter0 = torch.ones(B, T, dtype=torch.long, device=device)
    valid_iter1 = torch.ones(B, T - 2, dtype=torch.long, device=device)

    return {
        "iter0_layer": iter0_layer,
        "iter1_layer": iter1_layer,
        "pos_iter0": pos_iter0,
        "pos_iter1": pos_iter1,
        "valid_iter0": valid_iter0,
        "valid_iter1": valid_iter1,
        "L": L,
        "B": B,
    }


CACHE_RUNNER = """
def run(payload):
    from tah.model.causal_cache import TaHCache
    cache = TaHCache().to(device='cpu', dtype=torch.float32)
    L = payload['L']
    # Iter 0
    cache.current_iter_depth = 0
    cache.position_ids_to_cache = payload['pos_iter0']
    cache.valid_mask_to_cache = payload['valid_iter0']
    for layer_idx in range(L):
        k, v = payload['iter0_layer'][layer_idx]
        cache.update(k, v, layer_idx)
    # Iter 1 — narrower
    cache.current_iter_depth = 1
    cache.position_ids_to_cache = payload['pos_iter1']
    cache.valid_mask_to_cache = payload['valid_iter1']
    for layer_idx in range(L):
        k, v = payload['iter1_layer'][layer_idx]
        cache.update(k, v, layer_idx)

    out = {}
    for layer_idx in range(L):
        kk0, vv0 = cache.get_cache_upto_iter(layer_idx, 0)
        kk1, vv1 = cache.get_cache_upto_iter(layer_idx, 1)
        out[f'L{layer_idx}_K_upto0'] = kk0
        out[f'L{layer_idx}_V_upto0'] = vv0
        out[f'L{layer_idx}_K_upto1'] = kk1
        out[f'L{layer_idx}_V_upto1'] = vv1
        out[f'L{layer_idx}_pos_upto1'] = cache.get_position_id_upto_iter(layer_idx, 1, init_batch_size=payload['B'])
        out[f'L{layer_idx}_valid_upto1'] = cache.get_valid_mask_upto_iter(layer_idx, 1, init_batch_size=payload['B'])
        out[f'L{layer_idx}_iteridx_upto1'] = cache.get_cache_iter_index_upto_iter(layer_idx, 1)
    return out
"""


@pytest.fixture
def baseline(shapes):
    args = _cache_inputs(shapes, "cpu")
    name = "causal_cache"
    if not have_baseline(name):
        capture(name, CACHE_RUNNER, payload=args)
    return load_baseline(name)


def test_cache_view_acc(baseline, device):
    from tah.model.causal_cache import TaHCache

    def _to_dev(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, list):
            return [_to_dev(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_to_dev(v) for v in x)
        return x

    args = {k: _to_dev(v) for k, v in baseline["args"].items()}

    cache = TaHCache().to(device=device, dtype=torch.float32)
    L = args["L"]
    cache.current_iter_depth = 0
    cache.position_ids_to_cache = args["pos_iter0"]
    cache.valid_mask_to_cache = args["valid_iter0"]
    for layer_idx in range(L):
        k, v = args["iter0_layer"][layer_idx]
        cache.update(k, v, layer_idx)
    cache.current_iter_depth = 1
    cache.position_ids_to_cache = args["pos_iter1"]
    cache.valid_mask_to_cache = args["valid_iter1"]
    for layer_idx in range(L):
        k, v = args["iter1_layer"][layer_idx]
        cache.update(k, v, layer_idx)

    for layer_idx in range(L):
        kk0, vv0 = cache.get_cache_upto_iter(layer_idx, 0)
        kk1, vv1 = cache.get_cache_upto_iter(layer_idx, 1)
        assert_close(f"L{layer_idx}_K_upto0", kk0, baseline["out"][f"L{layer_idx}_K_upto0"], atol=ACC_TOL)
        assert_close(f"L{layer_idx}_V_upto0", vv0, baseline["out"][f"L{layer_idx}_V_upto0"], atol=ACC_TOL)
        assert_close(f"L{layer_idx}_K_upto1", kk1, baseline["out"][f"L{layer_idx}_K_upto1"], atol=ACC_TOL)
        assert_close(f"L{layer_idx}_V_upto1", vv1, baseline["out"][f"L{layer_idx}_V_upto1"], atol=ACC_TOL)
        assert_close(
            f"L{layer_idx}_pos_upto1",
            cache.get_position_id_upto_iter(layer_idx, 1, init_batch_size=args["B"]),
            baseline["out"][f"L{layer_idx}_pos_upto1"],
            atol=ACC_TOL,
        )
        assert_close(
            f"L{layer_idx}_valid_upto1",
            cache.get_valid_mask_upto_iter(layer_idx, 1, init_batch_size=args["B"]),
            baseline["out"][f"L{layer_idx}_valid_upto1"],
            atol=ACC_TOL,
        )
        assert_close(
            f"L{layer_idx}_iteridx_upto1",
            cache.get_cache_iter_index_upto_iter(layer_idx, 1),
            baseline["out"][f"L{layer_idx}_iteridx_upto1"],
            atol=ACC_TOL,
        )
