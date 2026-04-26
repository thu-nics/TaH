"""Acc + speed test for the input updater.

Public TaH ships ``TrivialUpdater`` (one class, three modes selected by ctor
args). The canonical recipe uses ``topk=100, use_hidden_states=False`` (i.e. a
top-k softmax over logits, weighted-sum into embedding rows). The cleaned
version exposes a single module function with that single behaviour, so the
test pins exactly the math we care to preserve.
"""
from __future__ import annotations

import pytest
import torch

from tests._harness import (
    ACC_TOL,
    DEVICE,
    assert_close,
    bench,
    capture,
    have_baseline,
    load_baseline,
)


def _make_inputs(shapes, device):
    B, T, V, H, TOPK = shapes["B"], shapes["T"], shapes["V"], shapes["H"], shapes["TOPK"]
    g = torch.Generator(device=device).manual_seed(123)
    logits = torch.randn(B, T, V, generator=g, device=device, dtype=torch.float32)
    prev_inputs = torch.randn(B, T, H, generator=g, device=device, dtype=torch.float32)
    embedding_weight = torch.randn(V, H, generator=g, device=device, dtype=torch.float32)
    hidden_states = torch.randn(B, T, 4, H, generator=g, device=device, dtype=torch.float32)
    return {
        "logits": logits,
        "prev_inputs": prev_inputs,
        "embedding_weight": embedding_weight,
        "hidden_states": hidden_states,
        "topk": TOPK,
    }


PUBLIC_RUNNER = """
def run(payload):
    from tah.model.input_updater import TrivialUpdater
    upd = TrivialUpdater(use_hidden_states=False, topk=payload['topk'])
    out = upd(
        logits=payload['logits'],
        prev_inputs=payload['prev_inputs'],
        embedding_weight=payload['embedding_weight'],
        hidden_states=payload['hidden_states'],
    )
    return {'updated': out}
"""


@pytest.fixture
def baseline(shapes):
    inputs = _make_inputs(shapes, "cpu")  # baseline saved on CPU
    name = "input_updater_topk"
    if not have_baseline(name):
        capture(name, PUBLIC_RUNNER, payload=inputs)
    return load_baseline(name)


def test_input_updater_acc(baseline, shapes, device):
    from tah.model.tah_model import topk_softmax_input_update

    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in baseline["args"].items()}
    out = topk_softmax_input_update(
        logits=inputs["logits"],
        embedding_weight=inputs["embedding_weight"],
        topk=inputs["topk"],
    )
    assert_close("input_updater.updated", out, baseline["out"]["updated"], atol=ACC_TOL)


def test_input_updater_speed(baseline, shapes, device):
    from tah.model.tah_model import topk_softmax_input_update

    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in baseline["args"].items()}

    def cleaned():
        return topk_softmax_input_update(
            logits=inputs["logits"],
            embedding_weight=inputs["embedding_weight"],
            topk=inputs["topk"],
        )

    res = bench("input_updater_topk", cleaned)
    # Sanity floor: inlined helper should always be <1ms on tiny shapes.
    assert res["ms"] < 1.0
