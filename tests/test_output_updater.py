"""Acc + speed test for the output updater.

Public TaH ships ``AdditiveLogitsUpdater`` (residual accumulation across
iterations) and ``NoneUpdater`` (pass-through). Only the additive form is used
in the canonical recipes, so the cleaned package collapses both to a single
``additive_logits_update`` function.
"""
from __future__ import annotations

import pytest
import torch

from tests._harness import (
    ACC_TOL,
    assert_close,
    bench,
    capture,
    have_baseline,
    load_baseline,
)


def _make_inputs(shapes, device):
    B, T, V = shapes["B"], shapes["T"], shapes["V"]
    g = torch.Generator(device=device).manual_seed(124)
    iter0 = torch.randn(B, T, V, generator=g, device=device, dtype=torch.float32)
    iter1 = torch.randn(B, T, V, generator=g, device=device, dtype=torch.float32)
    return {"iter0": iter0, "iter1": iter1}


PUBLIC_RUNNER = """
def run(payload):
    from tah.model.output_updater import AdditiveLogitsUpdater
    upd = AdditiveLogitsUpdater()
    out0 = upd(logits=payload['iter0'], prev_logits=None, iter_depth=0)
    out1 = upd(logits=payload['iter1'], prev_logits=out0, iter_depth=1)
    return {'out0': out0, 'out1': out1}
"""


@pytest.fixture
def baseline(shapes):
    inputs = _make_inputs(shapes, "cpu")
    name = "output_updater_additive"
    if not have_baseline(name):
        capture(name, PUBLIC_RUNNER, payload=inputs)
    return load_baseline(name)


def test_output_updater_acc(baseline, device):
    from tah.model.tah_model import additive_logits_update

    args = {k: v.to(device) for k, v in baseline["args"].items()}
    out0 = additive_logits_update(args["iter0"], prev_logits=None)
    out1 = additive_logits_update(args["iter1"], prev_logits=out0)
    assert_close("out0", out0, baseline["out"]["out0"], atol=ACC_TOL)
    assert_close("out1", out1, baseline["out"]["out1"], atol=ACC_TOL)


def test_output_updater_speed(baseline, device):
    from tah.model.tah_model import additive_logits_update

    args = {k: v.to(device) for k, v in baseline["args"].items()}

    def cleaned():
        a = additive_logits_update(args["iter0"], None)
        return additive_logits_update(args["iter1"], a)

    res = bench("output_updater_additive", cleaned)
    assert res["ms"] < 1.0
