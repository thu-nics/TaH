"""Loss functions used by ``TaHForCausalLM``.

Two implementations are kept — both used in the canonical recipes. Public TaH
also shipped ``ConsistencyLoss``, which is unused and removed.

* :class:`NextTokenPredLoss` — standard causal-LM cross-entropy. Always
  applied at the end of the iteration loop on the accumulated logits. Step-1
  SFT and eval both use this.

* :class:`IterDeciderLoss` — per-iteration BCE supervised by ``iter_count_labels``.
  Step-2 SFT uses this on top of the iter decider's continue logits to teach
  it which tokens deserve another pass.

Both expose the same three-method protocol expected by ``TaHForCausalLM``:

* ``prepare_loss(B, T, device, dtype)`` — allocate per-token accumulators.
* ``intra_iter_loss_func(...)``         — only on intra-iter losses; called once per iteration.
* ``final_loss_func(...)``              — called once at the end of the forward.

The class-level ``_is_intra_iter_loss: bool`` lets the wrapper decide whether
to invoke ``intra_iter_loss_func``. Lookup by name is via ``LOSS_BY_NAME``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers.utils import logging

from tah.train import fixed_cross_entropy

logger = logging.get_logger(__name__)


class LossFunc:
    _is_intra_iter_loss: bool = False

    def __init__(self, **kwargs):
        self.config = kwargs

    def prepare_loss(self, batch_size: int, query_len: int, device, dtype, **kwargs) -> None:
        pass

    def intra_iter_loss_func(self, *args, **kwargs):
        raise NotImplementedError

    def final_loss_func(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────────────────
# NextTokenPredLoss — used by step-1 SFT and eval everywhere.
# ────────────────────────────────────────────────────────────────────────────


class NextTokenPredLoss(LossFunc):
    """Standard causal-LM cross-entropy on the final accumulated logits."""

    _is_intra_iter_loss: bool = False

    def __init__(self, **kwargs):
        super().__init__()

    def final_loss_func(
        self,
        logits: torch.Tensor,
        labels_shifted: torch.Tensor,
        iter_count: torch.Tensor,
        training: bool,
        **kwargs,
    ) -> torch.Tensor:
        del iter_count, training  # accepted for protocol uniformity
        num_items_in_batch = kwargs.get("num_items_in_batch", None)
        # upcast for numerical parity with HF's cross-entropy path
        flat_logits = logits.view(-1, logits.shape[-1]).float()
        flat_labels = labels_shifted.view(-1).to(flat_logits.device)
        return fixed_cross_entropy(
            flat_logits, flat_labels,
            num_items_in_batch=num_items_in_batch, ignore_index=-100,
        )


# ────────────────────────────────────────────────────────────────────────────
# IterDeciderLoss — per-iteration BCE on continue logits, used by step-2 SFT.
# ────────────────────────────────────────────────────────────────────────────


class IterDeciderLoss(LossFunc):
    """BCE on the iter decider's continue logits, supervised by iter_count_labels.

    At iteration depth ``d``, the per-token target is ``(iter_count_labels > d)``.
    ``skip_last_iter`` skips the loss at ``d == max_iter`` (always-stop step).
    Optional ``pos_weight`` rescales the positive class to handle imbalance.
    """

    _is_intra_iter_loss: bool = True

    def __init__(self, pos_weight: Optional[float] = None, skip_last_iter: bool = True, max_iter: Optional[int] = None, **kwargs):
        super().__init__()
        self.pos_weight = pos_weight
        self.skip_last_iter = bool(skip_last_iter)
        self.max_iter = int(max_iter) if max_iter is not None else None
        if self.skip_last_iter and self.max_iter is None:
            raise ValueError("max_iter must be set when skip_last_iter is True")
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]) if pos_weight is not None else None
        )
        # State allocated by prepare_loss.
        self.iter_decider_loss_per_token: Optional[torch.Tensor] = None
        self._metric_correct_count: Optional[torch.Tensor] = None
        self._metric_total_count: Optional[torch.Tensor] = None

    def prepare_loss(self, batch_size, query_len, device, dtype, **kwargs):
        # Always float32 for losses, regardless of model dtype.
        self.iter_decider_loss_per_token = torch.zeros(batch_size, query_len, device=device, dtype=torch.float32)
        self._metric_correct_count = torch.zeros(1, device=device, dtype=torch.float32)
        self._metric_total_count = torch.zeros(1, device=device, dtype=torch.float32)

    def intra_iter_loss_func(
        self,
        active_logits,
        current_iter_mask: torch.BoolTensor,
        active_labels_shifted,
        active_valid_continue_logits: Optional[torch.Tensor],
        active_valid_mask: torch.LongTensor,
        iter_depth: int,
        active_iter_count_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        del active_logits, active_labels_shifted  # not used by this loss
        if (
            active_iter_count_labels is None
            or active_valid_continue_logits is None
            or not current_iter_mask.any()
            or active_valid_mask.sum() == 0
            or (self.skip_last_iter and int(iter_depth) >= int(self.max_iter))
        ):
            return torch.tensor(0.0, dtype=torch.float32)

        device = active_valid_continue_logits.device

        # Restrict to valid active tokens that have a non-padding iter label.
        valid_active = active_valid_mask == 1
        labels_at_active = active_iter_count_labels[valid_active]
        non_padding = labels_at_active != -100
        if not non_padding.any():
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        targets = (labels_at_active[non_padding] > iter_depth).float()
        used_logits = active_valid_continue_logits[non_padding].float()

        self._record_threshold_accuracy(used_logits, targets, kwargs.get("iter_decider_threshold", 0.5))

        if self.criterion.pos_weight is not None:
            self.criterion.pos_weight = self.criterion.pos_weight.to(device=device)
        loss = self.criterion(used_logits.unsqueeze(-1), targets.unsqueeze(-1))

        # Spread the per-iter loss evenly across contributing positions so that
        # final_loss_func's sum-then-divide gives the correct per-token average,
        # and scatter back into the dense (B, T) accumulator.
        per_token = loss / float(used_logits.numel())
        token_losses = self._scatter_per_token(
            per_token, valid_active, non_padding, current_iter_mask, active_valid_mask, device,
        )
        self.iter_decider_loss_per_token = self.iter_decider_loss_per_token + token_losses
        return token_losses

    def _record_threshold_accuracy(
        self, logits: torch.Tensor, targets: torch.Tensor, threshold,
    ) -> None:
        """Update running counts of correct iter-decider predictions."""
        if isinstance(threshold, torch.Tensor):
            threshold = float(threshold.detach().item())
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > float(threshold)).float()
            self._metric_correct_count += (preds == targets).float().sum()
            self._metric_total_count += float(logits.numel())

    @staticmethod
    def _scatter_per_token(
        per_token: torch.Tensor,
        valid_active: torch.Tensor,
        non_padding: torch.Tensor,
        current_iter_mask: torch.BoolTensor,
        active_valid_mask: torch.LongTensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Place ``per_token`` (a scalar) at every (valid + non_padding) active
        position in a fresh dense ``(B, T)`` tensor."""
        from tah.model.tah_model import scatter_back  # local: avoid import cycle

        active_losses = torch.zeros(active_valid_mask.shape, device=device, dtype=torch.float32)
        # valid_positions[b, k] = True iff the k-th active token in row b is
        # both valid and non-padding (the same subset that contributed to the
        # BCE numerator above).
        valid_positions = torch.zeros_like(active_losses, dtype=torch.bool)
        valid_positions[valid_active] = non_padding
        active_losses[valid_positions] = per_token

        token_losses = torch.zeros(
            current_iter_mask.shape, device=device, dtype=torch.float32,
        )
        scatter_back(current_iter_mask, src=active_losses, dest=token_losses, in_place=True)
        return token_losses

    def final_loss_func(
        self,
        logits: torch.Tensor,
        labels_shifted: torch.Tensor,
        iter_count: torch.Tensor,
        iter_count_labels: Optional[torch.Tensor] = None,
        training: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        del labels_shifted, training
        if self.iter_decider_loss_per_token is None:
            raise RuntimeError("prepare_loss has not been called")

        accumulated = self.iter_decider_loss_per_token
        self.iter_decider_loss_per_token = None  # consume

        valid = (iter_count_labels != -100) & (iter_count > 0) if iter_count_labels is not None else (iter_count > 0)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, dtype=accumulated.dtype)

        # Optional accuracy logging via callback (kept for parity with the trainer).
        callback = kwargs.get("logger_callback")
        with torch.no_grad():
            if callback is not None and self._metric_total_count is not None:
                if not hasattr(callback, "iter_decider_accuracy"):
                    callback.iter_decider_accuracy = 0.0
                num_items = kwargs.get("num_items_in_batch")
                if num_items:
                    callback.iter_decider_accuracy += float(self._metric_correct_count / num_items)
                else:
                    callback.iter_decider_accuracy += float(
                        self._metric_correct_count / torch.clamp(self._metric_total_count, min=1.0)
                    )
        self._metric_correct_count = None
        self._metric_total_count = None

        num_items = kwargs.get("num_items_in_batch")
        if num_items is not None:
            return accumulated[valid].sum() / num_items
        return accumulated[valid].mean()


# ────────────────────────────────────────────────────────────────────────────
# Name-based dispatch (replaces the old registry system).
# ────────────────────────────────────────────────────────────────────────────


LOSS_BY_NAME = {
    "NextTokenPredLoss": NextTokenPredLoss,
    "IterDeciderLoss": IterDeciderLoss,
}


def get_loss_func_class(name: str):
    """Lookup helper for backwards-compatible callers."""
    if name not in LOSS_BY_NAME:
        raise ValueError(f"Unknown loss {name!r}; have {sorted(LOSS_BY_NAME)}")
    return LOSS_BY_NAME[name]
