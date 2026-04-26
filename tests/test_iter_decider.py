"""Acc + speed tests for the iter deciders.

Two deciders are kept in the cleaned package — both are used in the canonical
training/eval recipes:

* ``IterLabelDecider``  — step-1 SFT: continue iff oracle iter_count_labels say so.
* ``MLPIterDecider``    — step-2 SFT + eval: learned classifier over hidden + top-k logits.

Public TaH also shipped ``TrivialIterDecider``, ``AlwaysWrapperIterDecider``,
and ``OracleDynamicIterDecider`` — none are used by the released checkpoint and
are removed in tah-release.
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


# ---------------------------------------------------------------- IterLabelDecider

def _label_inputs(shapes, device):
    B, T, V = shapes["B"], shapes["T"], shapes["V"]
    g = torch.Generator(device=device).manual_seed(126)
    logits = torch.randn(B, T, V, generator=g, device=device, dtype=torch.float32)
    labels = torch.randint(0, 3, (B, T), generator=g, device=device, dtype=torch.long)
    labels[0, 0] = -100  # padding-like
    return {"logits": logits, "labels": labels, "max_iter": shapes["MAX_ITER"]}


LABEL_RUNNER = """
def run(payload):
    from tah.model.iter_decider import IterLabelDecider
    dec = IterLabelDecider(max_iter=payload['max_iter'])
    out0 = dec(logits=payload['logits'], iter_depth=0, iter_count_labels=payload['labels'])
    out1 = dec(logits=payload['logits'], iter_depth=1, iter_count_labels=payload['labels'])
    return {
        'd0': out0[0], 'l0': out0[1],
        'd1': out1[0], 'l1': out1[1],
    }
"""


@pytest.fixture
def label_baseline(shapes):
    args = _label_inputs(shapes, "cpu")
    name = "iter_decider_label"
    if not have_baseline(name):
        capture(name, LABEL_RUNNER, payload=args)
    return load_baseline(name)


def test_iter_label_decider_acc(label_baseline, device):
    from tah.model.iter_decider import IterLabelDecider

    args = {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in label_baseline["args"].items()}
    dec = IterLabelDecider(max_iter=args["max_iter"]).to(device)
    out0 = dec(logits=args["logits"], iter_depth=0, iter_count_labels=args["labels"])
    out1 = dec(logits=args["logits"], iter_depth=1, iter_count_labels=args["labels"])
    assert_close("d0", out0[0], label_baseline["out"]["d0"])
    assert_close("l0", out0[1], label_baseline["out"]["l0"])
    assert_close("d1", out1[0], label_baseline["out"]["d1"])
    assert_close("l1", out1[1], label_baseline["out"]["l1"])


# ---------------------------------------------------------------- MLPIterDecider

MLP_KWARGS = dict(
    topk=8,
    hidden_states_size=32,
    hidden_states_layer_nums=[0, 1, 2, 3],
    hidden_dims=[16, 16, 16],
    expansion_factor=2,
    dropout_rate=0.0,
    normalize_input=False,
    threshold=0.5,
    max_iter=2,
    dtype=torch.float32,
)


def _mlp_inputs(shapes, device):
    B, T, V, H, L = shapes["B"], shapes["T"], shapes["V"], shapes["H"], shapes["L"]
    assert V >= MLP_KWARGS["topk"], "vocab too small for topk"
    g = torch.Generator(device=device).manual_seed(127)
    logits = torch.randn(B, T, V, generator=g, device=device, dtype=torch.float32)
    hidden = torch.randn(B, T, L, MLP_KWARGS["hidden_states_size"], generator=g, device=device, dtype=torch.float32)
    return {"logits": logits, "hidden": hidden}


MLP_RUNNER = (
    "def run(payload):\n"
    "    import torch\n"
    "    from tah.model.iter_decider import MLPIterDecider\n"
    "    torch.manual_seed(7)\n"
    f"    dec = MLPIterDecider(**{MLP_KWARGS!r})\n"
    "    state = {k: v.detach().clone() for k, v in dec.state_dict().items()}\n"
    "    out = dec(logits=payload['logits'], iter_depth=0, all_hidden_states=payload['hidden'])\n"
    "    return {'state': state, 'decision': out[0], 'logits': out[1]}\n"
)


@pytest.fixture
def mlp_baseline(shapes):
    args = _mlp_inputs(shapes, "cpu")
    name = "iter_decider_mlp"
    if not have_baseline(name):
        capture(name, MLP_RUNNER, payload=args)
    return load_baseline(name)


def test_mlp_iter_decider_acc(mlp_baseline, device):
    from tah.model.iter_decider import MLPIterDecider

    torch.manual_seed(7)
    dec = MLPIterDecider(**MLP_KWARGS).to(device)
    # Sync weights with the baseline init exactly (avoids any seed/init drift).
    dec.load_state_dict({k: v.to(device) for k, v in mlp_baseline["out"]["state"].items()})

    args = {k: v.to(device) for k, v in mlp_baseline["args"].items()}
    out = dec(logits=args["logits"], iter_depth=0, all_hidden_states=args["hidden"])
    assert_close("decision", out[0], mlp_baseline["out"]["decision"])
    assert_close("logits", out[1], mlp_baseline["out"]["logits"], atol=1e-5)


def test_mlp_iter_decider_speed(mlp_baseline, device):
    from tah.model.iter_decider import MLPIterDecider as Cleaned

    torch.manual_seed(7)
    dec = Cleaned(**MLP_KWARGS).to(device)
    args = {k: v.to(device) for k, v in mlp_baseline["args"].items()}

    def run():
        return dec(logits=args["logits"], iter_depth=0, all_hidden_states=args["hidden"])

    bench("mlp_iter_decider", run)
