from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import logging

from tah2.train.loss_utils import fixed_cross_entropy
from tah2.utils.component_registry import (
    capture_init_args,
    get_loss_func_class,
    register_loss_func,
)

logger = logging.get_logger(__name__)


class LossFunc(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

    def prepare_loss(self, batch_size, query_len, device, dtype, **kwargs):
        pass

    def intra_iter_loss_func(self, *args, **kwargs):
        raise NotImplementedError(
            "This loss function does not support intra-iteration loss calculation."
        )

    def final_loss_func(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


@register_loss_func
@capture_init_args
class NextTokenPredLoss(LossFunc):

    def __init__(self, **kwargs):
        super().__init__()
        self._use_target_gather_mix = bool(kwargs.get("use_target_gather_mix", True))

    def final_loss_func(
        self,
        logits: torch.Tensor,
        labels_shifted: torch.Tensor,
        iter_count: torch.Tensor,
        training: bool,
        **kwargs,
    ) -> torch.Tensor:
        num_items_in_batch = kwargs.get("num_items_in_batch", None)
        if self._use_target_gather_mix and logits.dim() == 3 and logits.size(-1) == 1:
            # logits: [B, L, 1] fp32 mixture log-prob at the label id
            # (K-space accumulator with K=1). NLL = -value at valid positions.
            per_token_nll = -logits.squeeze(-1).float()
            valid = labels_shifted.to(per_token_nll.device) != -100
            total = per_token_nll[valid].sum()
            if num_items_in_batch is not None:
                return total / num_items_in_batch
            return total / valid.sum().clamp(min=1)
        vocab_parallel_ops = kwargs.get("vocab_parallel_ops", None)
        if vocab_parallel_ops is not None:
            # logits is a vocab shard [..., V/tp]. Under log-prob mix methods
            # the values are already global log-probs (NLL = -gather); raw
            # logits need the full sharded CE (global logsumexp + gather).
            flat_logits = logits.view(-1, logits.shape[-1])
            flat_labels = labels_shifted.view(-1).to(flat_logits.device)
            if kwargs.get("logits_are_log_probs", False):
                per_token = vocab_parallel_ops.nll_from_log_probs(
                    flat_logits, flat_labels
                )
            else:
                per_token = vocab_parallel_ops.cross_entropy_per_token(
                    flat_logits, flat_labels
                )
            total = per_token.sum()
            if num_items_in_batch is not None:
                return total / num_items_in_batch
            return total / (flat_labels != -100).sum().clamp(min=1)
        vocab_size = logits.shape[-1]
        # Upcast to float to avoid precision issues, mirroring transformers' impl.
        shift_logits = logits.float().view(-1, vocab_size)
        shift_labels = labels_shifted.view(-1).to(shift_logits.device)
        return fixed_cross_entropy(
            shift_logits,
            shift_labels,
            num_items_in_batch=num_items_in_batch,
            ignore_index=-100,
        )


@register_loss_func
@capture_init_args
class IterDeciderLoss(LossFunc):
    """
    Loss function for iter decider that predicts whether each token should continue iterating.
    Uses BCE loss to maximize margin between positive and negative examples.
    Calculates loss at each iteration depth.
    """


    def __init__(
        self,
        dyn_pos_weight: Union[bool, List[bool]] = True,
        cost_sensitive: bool = False,
        pos_weight: Optional[float] = None,
        max_iter: Optional[int] = None,
        bce_margin: float = 0.0,
        **kwargs,
    ):
        """
        Initialize IterDeciderLoss.

        Args:
            dyn_pos_weight: If True, use the dynamic neg/pos ratio (DDP-global) as
                BCE `pos_weight`; if False, use the static `pos_weight` kwarg (or 1.0).
                A list enables this per decider depth, e.g. [true, false] means
                depth 1 uses dynamic pos_weight and depth 2 uses static pos_weight.
            cost_sensitive: If True, multiply per-token BCE by `_pending_cost_scores`
                produced by the label generator.
            pos_weight: Static BCE pos_weight. Used only when `dyn_pos_weight=False`.
            max_iter: Maximum iteration depth.
            bce_margin: Additive margin in logit space (default 0 = standard BCE).
                Training uses BCE on shifted logits ``logit_eff = logit - margin*(2*target-1)``,
                forcing positive samples' logits above +margin and negatives below
                -margin. Hard labels and cost weighting are unchanged. Pushes
                inference-time prob = sigmoid(logit) away from 0.5 by training the
                decider to commit beyond a learned margin.
        The flags are independent and can be freely combined.
        """
        from tah2.model.tah_model import (  # import like this to avoid circular import
            TaHForCausalLM,
        )

        self.assign_active = TaHForCausalLM.assign_active
        super().__init__(**kwargs)
        if isinstance(dyn_pos_weight, (list, tuple)):
            if len(dyn_pos_weight) == 0:
                raise ValueError("dyn_pos_weight list must not be empty.")
            self.dyn_pos_weight_by_depth: Optional[List[bool]] = [
                bool(v) for v in dyn_pos_weight
            ]
            self.dyn_pos_weight = any(self.dyn_pos_weight_by_depth)
        else:
            self.dyn_pos_weight = bool(dyn_pos_weight)
            self.dyn_pos_weight_by_depth = None
        self.cost_sensitive = bool(cost_sensitive)
        self.pos_weight = pos_weight
        self.dynamic_pos_weight_min = 0.02
        self.dynamic_pos_weight_max = 50
        # Optional explicit max_iter (preferred over reading from model at call time)
        self.max_iter: Optional[int] = int(max_iter) if max_iter is not None else None
        bm = float(bce_margin)
        if not (bm >= 0.0):
            raise ValueError(f"bce_margin must be >= 0, got {bce_margin!r}")
        self.bce_margin = bm

        self._metric_entropy_sum_by_depth: Dict[int, torch.Tensor] = {}
        self._metric_entropy_cnt_by_depth: Dict[int, torch.Tensor] = {}

    def _use_dyn_pos_weight(self, iter_depth: int) -> bool:
        if self.dyn_pos_weight_by_depth is None:
            return self.dyn_pos_weight
        idx = min(max(int(iter_depth) - 1, 0), len(self.dyn_pos_weight_by_depth) - 1)
        return self.dyn_pos_weight_by_depth[idx]

    def prepare_loss(self, batch_size, query_len, device, dtype, **kwargs):
        # Metric accumulators (float32 scalars on device)
        self._metric_correct_count = torch.zeros(1, device=device, dtype=torch.float32)
        self._metric_total_count = torch.zeros(1, device=device, dtype=torch.float32)
        self._metric_tp_count = torch.zeros(1, device=device, dtype=torch.float32)
        self._metric_fp_count = torch.zeros(1, device=device, dtype=torch.float32)
        self._metric_fn_count = torch.zeros(1, device=device, dtype=torch.float32)
        # Per-depth decision-entropy accumulators: depth -> (entropy sum, count),
        # emitted as `decider_entropy_d{n}` (batch-mean entropy after iter n).
        self._metric_entropy_sum_by_depth: Dict[int, torch.Tensor] = {}
        self._metric_entropy_cnt_by_depth: Dict[int, torch.Tensor] = {}
        # Accumulate (logits, targets, optional cost scores) across ALL depths.
        self._all_logits_list: list = []
        self._all_targets_list: list = []
        self._all_depths_list: list = []
        self._all_cost_scores_list: list = (
            []
        )  # empty list means cost_sensitive is off or no scores

    def intra_iter_loss_func(
        self,
        active_logits: torch.Tensor,
        current_iter_mask: torch.BoolTensor,
        active_labels_shifted: torch.Tensor,
        active_valid_continue_logits: Optional[torch.Tensor],
        active_valid_mask: torch.LongTensor,
        iter_depth: int,
        active_iter_count_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Accumulate per-depth (logits, targets); BCE is computed globally in final_loss_func."""
        device = active_logits.device
        if (not current_iter_mask.any()) or (active_valid_mask.sum() == 0):
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        if int(iter_depth) >= int(self.max_iter):
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        if active_iter_count_labels is None or active_valid_continue_logits is None:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        active_valid_continue_logits = active_valid_continue_logits.float()
        device = active_valid_continue_logits.device

        # Extract supervised, valid-position targets and logits.
        valid_active_mask = active_valid_mask == 1
        valid_iter_count_labels = active_iter_count_labels[valid_active_mask]
        valid_labels_shifted = active_labels_shifted[valid_active_mask]
        non_padding_mask = valid_labels_shifted != -100
        if not non_padding_mask.any():
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        valid_continue_targets = (valid_iter_count_labels > iter_depth).float()
        final_continue_targets = valid_continue_targets[non_padding_mask]
        final_continue_logits = active_valid_continue_logits[non_padding_mask]

        # Accumulate (logits, targets) for this depth; BCE is computed globally in final_loss_func.
        self._all_logits_list.append(final_continue_logits)
        self._all_targets_list.append(final_continue_targets)
        self._all_depths_list.append(int(iter_depth))

        if self.cost_sensitive:
            label_gen = kwargs.get("label_gen", None)
            pending = None
            if label_gen is not None:
                pending_by_depth = getattr(
                    label_gen, "_pending_cost_scores_by_depth", None
                )
                dense_pending = (
                    pending_by_depth.get(int(iter_depth))
                    if isinstance(pending_by_depth, dict)
                    else None
                )
                if dense_pending is not None:
                    active_pending = label_gen._compact_from_full(
                        dense_pending,
                        current_iter_mask,
                        active_valid_mask.shape[1],
                        pad_value=0.0,
                    )
                    if active_pending is not None:
                        pending = active_pending[valid_active_mask][non_padding_mask]
                if pending is None:
                    pending = getattr(label_gen, "_pending_cost_scores", None)
            if (
                pending is not None
                and pending.shape[0] == final_continue_targets.shape[0]
            ):
                self._all_cost_scores_list.append(
                    pending.to(device=device, dtype=torch.float32)
                )
            else:
                # Fallback: uniform weights so the depth still participates correctly.
                self._all_cost_scores_list.append(
                    torch.ones(
                        final_continue_targets.shape[0],
                        device=device,
                        dtype=torch.float32,
                    )
                )

        # Compute per-depth classification metrics (accuracy / TP / FP / FN) for monitoring.
        with torch.no_grad():
            continue_probs = torch.sigmoid(final_continue_logits)
            pred_positive = continue_probs > 0.5
            target_positive = final_continue_targets > 0.5
            correct = (pred_positive == target_positive).to(torch.float32).sum()
            total = torch.tensor(
                float(pred_positive.numel()), device=device, dtype=torch.float32
            )
            tp = (pred_positive & target_positive).to(torch.float32).sum()
            fp = (pred_positive & (~target_positive)).to(torch.float32).sum()
            fn = ((~pred_positive) & target_positive).to(torch.float32).sum()
            if self._metric_correct_count is not None:
                self._metric_correct_count += correct
                self._metric_total_count += total
                self._metric_tp_count += tp
                self._metric_fp_count += fp
                self._metric_fn_count += fn
            # Binary entropy of the continue prob over the tokens the decider
            # scored at this depth (batch mean = sum/count at logging time).
            entropy = -(
                continue_probs * torch.log(continue_probs.clamp_min(1e-12))
                + (1.0 - continue_probs)
                * torch.log((1.0 - continue_probs).clamp_min(1e-12))
            ).sum()
            depth_key = int(iter_depth)
            zero = torch.zeros((), device=device, dtype=torch.float32)
            self._metric_entropy_sum_by_depth[depth_key] = (
                self._metric_entropy_sum_by_depth.get(depth_key, zero) + entropy
            )
            self._metric_entropy_cnt_by_depth[depth_key] = (
                self._metric_entropy_cnt_by_depth.get(depth_key, zero)
                + total.reshape(())
            )

        return torch.tensor(0.0, device=device, dtype=torch.float32)

    def final_loss_func(
        self,
        logits: torch.Tensor,
        labels_shifted: torch.Tensor,
        iter_count: torch.Tensor,
        iter_count_labels: Optional[torch.Tensor] = None,
        training: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute decider loss from all (logits, targets) accumulated across every
        iteration depth. Weighting is controlled by `self.dyn_pos_weight` (BCE
        pos_weight) and `self.cost_sensitive` (per-token cost multiplier); the
        two flags are independent and can be combined.
        """
        device = logits.device

        # Log classification metrics accumulated across all depths.
        logger_callback = kwargs.get("logger_callback", None)
        with torch.no_grad():
            zero = torch.zeros((), device=device, dtype=torch.float32)

            def _get(attr):
                v = getattr(self, attr, None)
                return (
                    v.reshape(()).to(device=device, dtype=torch.float32)
                    if v is not None
                    else zero
                )

            # Fixed-size stack on every rank (collective alignment): 5 counters
            # + entropy sums/counts for depths 1..max_iter-1.
            n_dec_depths = max(int(self.max_iter) - 1, 0) if self.max_iter else 0
            ent_sums = getattr(self, "_metric_entropy_sum_by_depth", {}) or {}
            ent_cnts = getattr(self, "_metric_entropy_cnt_by_depth", {}) or {}

            def _ent(dct, depth):
                v = dct.get(depth)
                return (
                    v.reshape(()).to(device=device, dtype=torch.float32)
                    if v is not None
                    else zero
                )

            metrics = torch.stack(
                [
                    _get("_metric_correct_count"),
                    _get("_metric_total_count"),
                    _get("_metric_tp_count"),
                    _get("_metric_fp_count"),
                    _get("_metric_fn_count"),
                ]
                + [_ent(ent_sums, d) for d in range(1, n_dec_depths + 1)]
                + [_ent(ent_cnts, d) for d in range(1, n_dec_depths + 1)]
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(
                    metrics,
                    op=dist.ReduceOp.SUM,
                    group=kwargs.get("dp_group", None),
                )
            if logger_callback is not None:
                # Single GPU->CPU sync via .tolist() instead of 5-10 separate .item() calls.
                m = metrics.detach().tolist()
                if m[1] > 0:
                    if training:
                        logger_callback.iter_decider_correct += m[0]
                        logger_callback.iter_decider_total += m[1]
                        logger_callback.iter_decider_tp += m[2]
                        logger_callback.iter_decider_fp += m[3]
                        logger_callback.iter_decider_fn += m[4]
                    else:
                        logger_callback.eval_iter_decider_correct += m[0]
                        logger_callback.eval_iter_decider_total += m[1]
                        logger_callback.eval_iter_decider_tp += m[2]
                        logger_callback.eval_iter_decider_fp += m[3]
                        logger_callback.eval_iter_decider_fn += m[4]
                prefix = "" if training else "eval_"
                sums_acc = getattr(
                    logger_callback,
                    f"{prefix}iter_decider_entropy_sum_by_depth",
                    None,
                )
                cnts_acc = getattr(
                    logger_callback,
                    f"{prefix}iter_decider_entropy_count_by_depth",
                    None,
                )
                if sums_acc is not None and cnts_acc is not None:
                    for i in range(n_dec_depths):
                        ent_sum = m[5 + i]
                        ent_cnt = m[5 + n_dec_depths + i]
                        if ent_cnt > 0:
                            depth = i + 1
                            sums_acc[depth] = sums_acc.get(depth, 0.0) + ent_sum
                            cnts_acc[depth] = cnts_acc.get(depth, 0.0) + ent_cnt

        # Reset metric accumulators.
        self._metric_correct_count = None
        self._metric_total_count = None
        self._metric_tp_count = None
        self._metric_fp_count = None
        self._metric_fn_count = None
        self._metric_entropy_sum_by_depth = {}
        self._metric_entropy_cnt_by_depth = {}

        # Consume accumulated logits/targets/cost scores.
        all_logits_list = self._all_logits_list
        all_targets_list = self._all_targets_list
        all_depths_list = self._all_depths_list
        all_cost_scores_list = self._all_cost_scores_list
        self._all_logits_list = []
        self._all_targets_list = []
        self._all_depths_list = []
        self._all_cost_scores_list = []

        use_cost = self.cost_sensitive and len(all_cost_scores_list) == len(
            all_logits_list
        )
        depth_costs = (
            all_cost_scores_list if use_cost else [None] * len(all_logits_list)
        )

        # dyn_pos_weight class counts: ONE fixed-size all_reduce regardless
        # of how many depths this rank accumulated. Ranks that only ran dummy
        # alignment passes (see TaHForCausalLM._run_dummy_backbone_pass) have
        # shorter depth lists, so a per-depth collective would desync ranks.
        # This must run before the empty-list early return for the same reason.
        global_pos_neg = None
        if self.dyn_pos_weight and dist.is_available() and dist.is_initialized():
            num_depth_slots = (
                (int(self.max_iter) + 1) if self.max_iter is not None else 9
            )
            stats = torch.zeros(
                (num_depth_slots, 2), device=device, dtype=torch.float32
            )
            with torch.no_grad():
                for iter_depth, depth_targets in zip(
                    all_depths_list, all_targets_list
                ):
                    if not self._use_dyn_pos_weight(iter_depth):
                        continue
                    slot = min(int(iter_depth), num_depth_slots - 1)
                    pos = depth_targets.sum().float()
                    stats[slot, 0] += pos
                    stats[slot, 1] += float(depth_targets.numel()) - pos
            dist.all_reduce(
                stats,
                op=dist.ReduceOp.SUM,
                group=kwargs.get("dp_group", None),
            )
            global_pos_neg = stats

        if not all_logits_list:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        # Compute per-depth BCE and sum.
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for iter_depth, depth_logits, depth_targets, depth_cost in zip(
            all_depths_list, all_logits_list, all_targets_list, depth_costs
        ):
            use_dyn_pos_weight = self._use_dyn_pos_weight(iter_depth)
            if use_dyn_pos_weight:
                with torch.no_grad():
                    if global_pos_neg is not None:
                        slot = min(int(iter_depth), global_pos_neg.shape[0] - 1)
                        pos, neg = global_pos_neg[slot, 0], global_pos_neg[slot, 1]
                    else:
                        pos = depth_targets.sum()
                        neg = (
                            torch.tensor(
                                float(depth_targets.numel()),
                                device=device,
                                dtype=depth_targets.dtype,
                            )
                            - pos
                        )
                    if pos > 0 and neg > 0:
                        ratio = (neg / pos).clamp(
                            min=self.dynamic_pos_weight_min,
                            max=self.dynamic_pos_weight_max,
                        )
                    else:
                        ratio = torch.tensor(
                            1.0, device=device, dtype=depth_logits.dtype
                        )
            else:
                if self.pos_weight is not None and float(self.pos_weight) > 0:
                    ratio = torch.tensor(
                        float(self.pos_weight), device=device, dtype=depth_logits.dtype
                    )
                else:
                    ratio = torch.tensor(1.0, device=device, dtype=depth_logits.dtype)

            # Log the actual pos_weight applied this depth.
            # One GPU→CPU sync per depth; depths are
            # capped at max_iter (typically 1 with max_iter=2), so overhead
            # is negligible.
            if logger_callback is not None:
                ratio_value = float(ratio.detach().item())
                if training:
                    logger_callback.iter_decider_pos_weight_sum += ratio_value
                    logger_callback.iter_decider_pos_weight_count += 1.0
                    logger_callback.iter_decider_pos_weight_by_depth_sum[iter_depth] = (
                        logger_callback.iter_decider_pos_weight_by_depth_sum.get(
                            iter_depth, 0.0
                        )
                        + ratio_value
                    )
                    logger_callback.iter_decider_pos_weight_by_depth_count[
                        iter_depth
                    ] = (
                        logger_callback.iter_decider_pos_weight_by_depth_count.get(
                            iter_depth, 0.0
                        )
                        + 1.0
                    )
                else:
                    logger_callback.eval_iter_decider_pos_weight_sum += ratio_value
                    logger_callback.eval_iter_decider_pos_weight_count += 1.0
                    logger_callback.eval_iter_decider_pos_weight_by_depth_sum[
                        iter_depth
                    ] = (
                        logger_callback.eval_iter_decider_pos_weight_by_depth_sum.get(
                            iter_depth, 0.0
                        )
                        + ratio_value
                    )
                    logger_callback.eval_iter_decider_pos_weight_by_depth_count[
                        iter_depth
                    ] = (
                        logger_callback.eval_iter_decider_pos_weight_by_depth_count.get(
                            iter_depth, 0.0
                        )
                        + 1.0
                    )

            if self.bce_margin > 0:
                # Margin BCE: shift logit so positives must exceed +margin,
                # negatives must fall below -margin. sign = 2*target-1 ∈ {-1, +1}.
                logits_eff = depth_logits - self.bce_margin * (
                    2.0 * depth_targets - 1.0
                )
            else:
                logits_eff = depth_logits
            per_decision_loss = F.binary_cross_entropy_with_logits(
                logits_eff,
                depth_targets,
                pos_weight=ratio,
                reduction="none",
            )
            if depth_cost is not None:
                total_loss = total_loss + (depth_cost * per_decision_loss).sum()
            else:
                total_loss = total_loss + per_decision_loss.sum()

        num_items_in_batch = kwargs.get("num_items_in_batch", None)
        if num_items_in_batch is not None:
            return total_loss / num_items_in_batch
        else:
            raise ValueError("num_items_in_batch is not provided in IterDeciderLoss.")


@register_loss_func
@capture_init_args
class CombinedLoss(LossFunc):
    """Weighted sum of multiple loss components."""


    def __init__(self, losses: Optional[list] = None, **kwargs):
        super().__init__(**kwargs)

        if not losses:
            raise ValueError("CombinedLoss requires at least one loss component.")

        self.losses = nn.ModuleList()
        self.loss_names = []
        self.loss_weights = []

        for i, spec in enumerate(losses):
            if not isinstance(spec, dict):
                raise ValueError(
                    f"losses[{i}] must be a dict, got {type(spec).__name__}"
                )
            name = spec.get("name")
            if not name:
                raise ValueError(f"losses[{i}] is missing required key 'name'")
            loss_kwargs = dict(spec.get("kwargs") or {})
            if (
                name == "IterDeciderLoss"
                and "max_iter" not in loss_kwargs
                and "max_iter" in kwargs
            ):
                loss_kwargs["max_iter"] = kwargs["max_iter"]
            weight = spec.get("weight", 1.0)
            if isinstance(weight, torch.Tensor):
                if weight.numel() != 1:
                    raise ValueError(f"losses[{i}].weight must be scalar")
                weight = weight.detach().item()
            try:
                weight = float(weight)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"losses[{i}].weight must be numeric, got {type(weight).__name__}: {weight!r}"
                ) from e

            loss_cls = get_loss_func_class(name)
            self.losses.append(loss_cls(**loss_kwargs))
            self.loss_names.append(name)
            self.loss_weights.append(weight)

        # Propagate the target-gather mixture flag (see NextTokenPredLoss): the
        # runtime reads it off the top-level loss to size the mixture accumulator.
        self._use_target_gather_mix = any(
            getattr(loss, "_use_target_gather_mix", False) for loss in self.losses
        )

    @staticmethod
    def _record_loss_stat(
        logger_callback, loss_name: str, value: float, training: bool
    ):
        if logger_callback is None:
            return
        dict_attr = "loss_sums" if training else "eval_loss_sums"
        sums = getattr(logger_callback, dict_attr, None)
        if sums is None:
            sums = {}
            setattr(logger_callback, dict_attr, sums)
        sums[loss_name] = sums.get(loss_name, 0.0) + value

        safe_name = "".join(
            ch if ch.isalnum() or ch == "_" else "_" for ch in loss_name
        )
        attr = f"{'' if training else 'eval_'}{safe_name}_loss_sum"
        setattr(logger_callback, attr, getattr(logger_callback, attr, 0.0) + value)

    def prepare_loss(self, batch_size, query_len, device, dtype, **kwargs):
        for loss in self.losses:
            loss.prepare_loss(batch_size, query_len, device, dtype, **kwargs)


    def final_loss_func(
        self,
        logits: torch.Tensor,
        labels_shifted: torch.Tensor,
        iter_count: torch.Tensor,
        training: bool,
        **kwargs,
    ) -> torch.Tensor:
        total_loss = None
        logger_callback = kwargs.get("logger_callback", None)
        # Batch the per-loss stat syncs: stack all scalars and do one .tolist().
        stat_tensors: list = []
        stat_names: list = []

        for loss_name, loss, weight in zip(
            self.loss_names, self.losses, self.loss_weights
        ):
            loss_value = loss.final_loss_func(
                logits=logits,
                labels_shifted=labels_shifted,
                iter_count=iter_count,
                training=training,
                **kwargs,
            )

            if logger_callback is not None:
                stat_tensors.append(loss_value.detach().float().reshape(()))
                stat_names.append(loss_name)

            weighted_loss = weight * loss_value
            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )

        if stat_tensors:
            with torch.no_grad():
                values = torch.stack(stat_tensors).tolist()
            for name, val in zip(stat_names, values):
                self._record_loss_stat(logger_callback, name, float(val), training)

        if total_loss is None:
            return torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        return total_loss
