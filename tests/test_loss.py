"""Acc tests for the kept loss functions.

* ``NextTokenPredLoss`` — step-1 SFT objective; standard cross-entropy with an
  optional hard/easy reweight knob.
* ``IterDeciderLoss``   — step-2 SFT objective; per-iteration BCE on the iter
  decider's continue logits, supervised by ``iter_count_labels``.

Public TaH also shipped ``ConsistencyLoss`` (unused) which is removed.
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


# ---------------------------------------------------------------- NextTokenPredLoss

def _ce_inputs(shapes, device):
    B, T, V = shapes["B"], shapes["T"], shapes["V"]
    g = torch.Generator(device=device).manual_seed(128)
    logits = torch.randn(B, T, V, generator=g, device=device, dtype=torch.float32)
    labels = torch.randint(0, V, (B, T), generator=g, device=device, dtype=torch.long)
    labels[0, 0] = -100
    iter_count = torch.randint(0, 3, (B, T), generator=g, device=device, dtype=torch.long)
    return {"logits": logits, "labels": labels, "iter_count": iter_count}


CE_RUNNER = """
def run(payload):
    from tah.model.loss import NextTokenPredLoss
    loss = NextTokenPredLoss()
    loss.prepare_loss(payload['logits'].shape[0], payload['logits'].shape[1],
                      payload['logits'].device, torch.float32)
    val = loss.final_loss_func(
        logits=payload['logits'],
        labels_shifted=payload['labels'],
        iter_count=payload['iter_count'],
        training=False,
        num_items_in_batch=int((payload['labels'] != -100).sum().item()),
    )
    return {'loss': val.detach()}
"""


@pytest.fixture
def ce_baseline(shapes):
    args = _ce_inputs(shapes, "cpu")
    name = "loss_next_token"
    if not have_baseline(name):
        capture(name, CE_RUNNER, payload=args)
    return load_baseline(name)


def test_next_token_pred_loss_acc(ce_baseline, device):
    from tah.model.loss import NextTokenPredLoss

    args = {k: v.to(device) for k, v in ce_baseline["args"].items()}
    loss = NextTokenPredLoss()
    loss.prepare_loss(args["logits"].shape[0], args["logits"].shape[1],
                      args["logits"].device, torch.float32)
    val = loss.final_loss_func(
        logits=args["logits"],
        labels_shifted=args["labels"],
        iter_count=args["iter_count"],
        training=False,
        num_items_in_batch=int((args["labels"] != -100).sum().item()),
    )
    assert_close("loss", val, ce_baseline["out"]["loss"], atol=ACC_TOL)


# ---------------------------------------------------------------- IterDeciderLoss

def _bce_inputs(shapes, device):
    B, T = shapes["B"], shapes["T"]
    g = torch.Generator(device=device).manual_seed(129)
    iter_count_labels = torch.randint(0, 3, (B, T), generator=g, device=device, dtype=torch.long)
    iter_count_labels[0, 0] = -100
    valid_mask = torch.ones(B, T, dtype=torch.long, device=device)
    valid_mask[1, -1] = 0  # one padded position — the wrapper masks this out before
                           # calling the iter_decider, so continue_logits has shape
                           # (sum(valid_mask),), not (B*T,).
    n_valid = int(valid_mask.sum().item())
    continue_logits = torch.randn(n_valid, generator=g, device=device, dtype=torch.float32)
    return {
        "continue_logits": continue_logits,
        "iter_count_labels": iter_count_labels,
        "valid_mask": valid_mask,
        "max_iter": shapes["MAX_ITER"],
    }


BCE_RUNNER = """
def run(payload):
    from tah.model.loss import IterDeciderLoss
    B, T = payload['iter_count_labels'].shape
    loss = IterDeciderLoss(pos_weight=2.0, skip_last_iter=True, max_iter=payload['max_iter'])
    loss.prepare_loss(B, T, payload['iter_count_labels'].device, torch.float32)

    # current_iter_mask: all positions active
    current = torch.ones(B, T, dtype=torch.bool, device=payload['iter_count_labels'].device)
    val = loss.intra_iter_loss_func(
        active_logits=None,
        current_iter_mask=current,
        active_labels_shifted=None,
        active_valid_continue_logits=payload['continue_logits'],
        active_valid_mask=payload['valid_mask'],
        iter_depth=1,
        active_iter_count_labels=payload['iter_count_labels'],
        iter_decider_threshold=0.5,
    )
    final = loss.final_loss_func(
        logits=payload['continue_logits'].view(B, T)[:, :1] if False else payload['continue_logits'].new_zeros(B, T, 1),
        labels_shifted=payload['iter_count_labels'],
        iter_count=torch.ones_like(payload['iter_count_labels']),
        iter_count_labels=payload['iter_count_labels'],
        training=True,
        num_items_in_batch=int((payload['iter_count_labels'] != -100).sum().item()),
    )
    return {'final': final.detach()}
"""


@pytest.fixture
def bce_baseline(shapes):
    args = _bce_inputs(shapes, "cpu")
    name = "loss_iter_decider"
    if not have_baseline(name):
        capture(name, BCE_RUNNER, payload=args)
    return load_baseline(name)


def test_iter_decider_loss_acc(bce_baseline, device):
    from tah.model.loss import IterDeciderLoss

    args = {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in bce_baseline["args"].items()}
    B, T = args["iter_count_labels"].shape
    loss = IterDeciderLoss(pos_weight=2.0, skip_last_iter=True, max_iter=args["max_iter"])
    loss.prepare_loss(B, T, args["iter_count_labels"].device, torch.float32)

    current = torch.ones(B, T, dtype=torch.bool, device=device)
    loss.intra_iter_loss_func(
        active_logits=None,
        current_iter_mask=current,
        active_labels_shifted=None,
        active_valid_continue_logits=args["continue_logits"],
        active_valid_mask=args["valid_mask"],
        iter_depth=1,
        active_iter_count_labels=args["iter_count_labels"],
        iter_decider_threshold=0.5,
    )
    final = loss.final_loss_func(
        logits=args["continue_logits"].new_zeros(B, T, 1),
        labels_shifted=args["iter_count_labels"],
        iter_count=torch.ones_like(args["iter_count_labels"]),
        iter_count_labels=args["iter_count_labels"],
        training=True,
        num_items_in_batch=int((args["iter_count_labels"] != -100).sum().item()),
    )
    assert_close("final", final, bce_baseline["out"]["final"], atol=ACC_TOL)
