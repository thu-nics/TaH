"""Acc test for the iter-label generator.

Public TaH had three generators (``Fixed``, ``DynamicMismatch``, ``Max``); only
``FixedIterLabelGenerator`` is used in the canonical recipes. Its only state
is a dense ``(B, T)`` accumulator updated via per-iteration max-merge of the
active slice.

The cleaned wrapper inlines this directly: a single ``full_labels`` tensor in
the forward, two lines to merge the active proposal. The test pins the merge
math against the public class.
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


def _make_inputs(shapes, device):
    B, T = shapes["B"], shapes["T"]
    g = torch.Generator(device=device).manual_seed(125)
    iter1_labels = torch.randint(0, 3, (B, T), generator=g, device=device, dtype=torch.long)
    iter2_labels = torch.randint(0, 3, (B, T), generator=g, device=device, dtype=torch.long)
    # Mask: drop a couple of positions at iter2 (shrink active set)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[0, -1] = False
    mask[1, 0] = False
    return {"iter1_labels": iter1_labels, "iter2_labels": iter2_labels, "mask": mask}


PUBLIC_RUNNER = """
def run(payload):
    from tah.model.iter_label import FixedIterLabelGenerator
    gen = FixedIterLabelGenerator()
    B, T = payload['iter1_labels'].shape
    gen.prepare(B, T, payload['iter1_labels'].device, torch.float32)

    # iter 1 — all positions active
    active_full = torch.ones(B, T, dtype=torch.bool, device=payload['iter1_labels'].device)
    gen.intra_iter_labels(
        active_iter_count_labels=payload['iter1_labels'],
        current_iter_mask=active_full,
    )
    # iter 2 — gather active slice using the user's mask
    mask = payload['mask']
    active_per_seq = mask.sum(1)
    max_len = int(active_per_seq.max())
    SENTINEL = T
    base_idx = torch.arange(T, device=mask.device).expand(B, T).masked_fill(~mask, SENTINEL)
    sorted_idx, _ = torch.sort(base_idx, dim=1, stable=True)
    gather_idx = sorted_idx[:, :max_len].clamp(max=T - 1)
    pad_mask = sorted_idx[:, :max_len].eq(SENTINEL)
    active_slice = torch.gather(payload['iter2_labels'], 1, gather_idx).masked_fill(pad_mask, -100)
    gen.intra_iter_labels(active_iter_count_labels=active_slice, current_iter_mask=mask)
    full = gen.finalize()
    return {'full_labels': full, 'active_slice_iter2': active_slice}
"""


@pytest.fixture
def baseline(shapes):
    inputs = _make_inputs(shapes, "cpu")
    name = "iter_label_fixed"
    if not have_baseline(name):
        capture(name, PUBLIC_RUNNER, payload=inputs)
    return load_baseline(name)


def test_iter_label_inline(baseline, device):
    """The cleaned wrapper inlines: max-merge active labels into the dense view."""
    args = {k: v.to(device) for k, v in baseline["args"].items()}
    iter1_labels = args["iter1_labels"]
    iter2_labels = args["iter2_labels"]
    mask = args["mask"]
    B, T = iter1_labels.shape

    # The inlined behaviour the wrapper will use:
    full_labels = torch.zeros(B, T, dtype=torch.long, device=device)

    # iter 1 — fully active: merge by max
    proposal = iter1_labels.clone()
    proposal[iter1_labels == -100] = 0
    full_labels = torch.maximum(full_labels, proposal)

    # iter 2 — partial active: scatter active proposal into dense, then max-merge
    active_per_seq = mask.sum(1)
    max_len = int(active_per_seq.max())
    SENTINEL = T
    base_idx = torch.arange(T, device=device).expand(B, T).masked_fill(~mask, SENTINEL)
    sorted_idx, _ = torch.sort(base_idx, dim=1, stable=True)
    gather_idx = sorted_idx[:, :max_len].clamp(max=T - 1)
    pad_mask = sorted_idx[:, :max_len].eq(SENTINEL)
    active_slice = torch.gather(iter2_labels, 1, gather_idx).masked_fill(pad_mask, -100)

    # Scatter back: build a dense tmp from active slice + mask
    tmp = torch.zeros_like(full_labels)
    valid = (active_slice != -100)
    proposal_dense = torch.zeros_like(active_slice)
    proposal_dense[valid] = active_slice[valid]
    for b in range(B):
        n = int(active_per_seq[b].item())
        if n:
            tmp[b, mask[b]] = proposal_dense[b, :n]
    full_labels = torch.maximum(full_labels, tmp)

    assert_close("full_labels", full_labels, baseline["out"]["full_labels"], atol=ACC_TOL)
    assert_close("active_slice_iter2", active_slice, baseline["out"]["active_slice_iter2"], atol=ACC_TOL)
