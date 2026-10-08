import contextlib
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator


class DistShim:
    """Minimal stand-in for the Accelerator interface used by shared helpers.

    Lets the torch-native TP path (torchrun, no Accelerator) reuse helpers that
    only need ``print`` / ``is_main_process`` / ``num_processes`` /
    ``main_process_first`` (model loading, dataset preprocessing, gradient
    checkpointing).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.is_main_process = dist.get_rank() == 0
        self.num_processes = dist.get_world_size()

    def print(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)

    @contextlib.contextmanager
    def main_process_first(self):
        if not self.is_main_process:
            dist.barrier()
        try:
            yield
        finally:
            if self.is_main_process:
                dist.barrier()


class MetricLogger:
    """Collects model-side metrics and flushes them into log dicts."""

    LOSS_NAME_ALIASES = {
        "NextTokenPredLoss": "ntp",
        "ForwardKLLoss": "fkl",
        "IterDeciderLoss": "decider",
        "ConsistencyLoss": "consis",
        "GlobalBatchLoadBalancingLoss": "load_balance",
    }

    def __init__(self, trainer_bridge):
        self.trainer = trainer_bridge
        self.max_depth_bins = self._MAX_DEPTH_BINS
        self._reset_train_metrics()
        self._reset_eval_metrics()

    def _reset_train_metrics(self):
        # Raw token sums; emitted as sum / iter_valid_total at log flush.
        self.iter_count_sum = 0.0
        self.iter_label_sum = 0.0
        self.iter_decider_correct = 0.0
        self.iter_decider_total = 0.0
        self.iter_decider_tp = 0.0
        self.iter_decider_fp = 0.0
        self.iter_decider_fn = 0.0
        # Effective BCE pos_weight applied each step. Accumulated across
        # micro-batch steps and depths; emitted as
        # the mean over (steps × depths) in `pop_train_logs`.
        self.iter_decider_pos_weight_sum = 0.0
        self.iter_decider_pos_weight_count = 0.0
        self.iter_decider_pos_weight_by_depth_sum: Dict[int, float] = {}
        self.iter_decider_pos_weight_by_depth_count: Dict[int, float] = {}
        # Per-depth decision entropy -> `decider_entropy_d{n}` = sum / count.
        self.iter_decider_entropy_sum_by_depth: Dict[int, float] = {}
        self.iter_decider_entropy_count_by_depth: Dict[int, float] = {}
        # Per-depth effective experts, kept on-device until log flush.
        self.effective_expert_sum_by_depth: Dict[int, torch.Tensor] = {}
        self.effective_expert_count_by_depth: Dict[int, torch.Tensor] = {}
        self.loss_sums: Dict[str, float] = {}
        # Per-depth counts: key k -> accumulated count of valid tokens with iter_count >= k
        self.iter_ge_sum: Dict[int, float] = {}
        # Per-depth label counts: key k -> accumulated count of valid tokens with iter_label >= k
        self.iter_label_ge_sum: Dict[int, float] = {}
        self.iter_valid_total = 0.0

    def _reset_eval_metrics(self):
        self.eval_iter_count_sum = 0.0
        self.eval_iter_label_sum = 0.0
        self.eval_iter_decider_correct = 0.0
        self.eval_iter_decider_total = 0.0
        self.eval_iter_decider_tp = 0.0
        self.eval_iter_decider_fp = 0.0
        self.eval_iter_decider_fn = 0.0
        self.eval_iter_decider_pos_weight_sum = 0.0
        self.eval_iter_decider_pos_weight_count = 0.0
        self.eval_iter_decider_pos_weight_by_depth_sum: Dict[int, float] = {}
        self.eval_iter_decider_pos_weight_by_depth_count: Dict[int, float] = {}
        self.eval_iter_decider_entropy_sum_by_depth: Dict[int, float] = {}
        self.eval_iter_decider_entropy_count_by_depth: Dict[int, float] = {}
        self.eval_effective_expert_sum_by_depth: Dict[int, torch.Tensor] = {}
        self.eval_effective_expert_count_by_depth: Dict[int, torch.Tensor] = {}
        self.eval_loss_sums: Dict[str, float] = {}
        self.eval_iter_ge_sum: Dict[int, float] = {}
        self.eval_iter_label_ge_sum: Dict[int, float] = {}
        self.eval_iter_valid_total = 0.0

    @staticmethod
    def _append_prf_metrics(
        logs: Dict[str, float], tp: float, fp: float, fn: float, prefix: str = ""
    ):
        tp = float(tp)
        fp = float(fp)
        fn = float(fn)
        if tp + fp + fn <= 0:
            return

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        logs[f"{prefix}iter_decider_precision"] = max(0.0, min(1.0, precision))
        logs[f"{prefix}iter_decider_recall"] = max(0.0, min(1.0, recall))

    def _reduce_sum_if_needed(self, value: float) -> float:
        accelerator = getattr(getattr(self, "trainer", None), "accelerator", None)
        if accelerator is None:
            dp_group = getattr(self, "dp_group", None)
            if dp_group is not None:
                tensor = torch.tensor(float(value), device="cuda", dtype=torch.float32)
                torch.distributed.all_reduce(tensor, group=dp_group)
                return float(tensor.item())
            return float(value)
        if accelerator.num_processes <= 1:
            return float(value)
        tensor = torch.tensor(
            float(value), device=accelerator.device, dtype=torch.float32
        )
        reduced = accelerator.reduce(tensor, reduction="sum")
        return float(reduced.item())

    @classmethod
    def _loss_alias(cls, loss_name: str) -> str:
        return cls.LOSS_NAME_ALIASES.get(loss_name, loss_name)

    def pop_train_logs(self, step_count: float = 1.0) -> Dict[str, float]:
        logs: Dict[str, float] = {}
        step_count = max(float(step_count), 1.0)
        if self.iter_valid_total > 0:
            logs["avg_iter_count"] = self.iter_count_sum / self.iter_valid_total
            logs["avg_iter_label"] = self.iter_label_sum / self.iter_valid_total
            for k, cnt in sorted(self.iter_ge_sum.items()):
                logs[f"iter_ge{k}"] = cnt / self.iter_valid_total
            for k, cnt in sorted(self.iter_label_ge_sum.items()):
                logs[f"iter_label_ge{k}"] = cnt / self.iter_valid_total
        if self.iter_decider_total > 0:
            logs["iter_decider_accuracy"] = (
                self.iter_decider_correct / self.iter_decider_total
            )
        if self.iter_decider_pos_weight_count > 0:
            logs["iter_decider_pos_weight"] = (
                self.iter_decider_pos_weight_sum / self.iter_decider_pos_weight_count
            )
        for depth, weight_sum in sorted(
            self.iter_decider_pos_weight_by_depth_sum.items()
        ):
            count = self.iter_decider_pos_weight_by_depth_count.get(depth, 0.0)
            if count > 0:
                logs[f"iter_decider_pos_weight_d{depth}"] = weight_sum / count
        for depth, ent_sum in sorted(self.iter_decider_entropy_sum_by_depth.items()):
            count = self.iter_decider_entropy_count_by_depth.get(depth, 0.0)
            if count > 0:
                logs[f"decider_entropy_d{depth}"] = ent_sum / count
        self._append_effective_expert_logs(logs, training=True)

        self._append_prf_metrics(
            logs,
            tp=self.iter_decider_tp,
            fp=self.iter_decider_fp,
            fn=self.iter_decider_fn,
            prefix="",
        )

        for loss_name, loss_sum in sorted(self.loss_sums.items()):
            alias = self._loss_alias(loss_name)
            logs[f"loss_{alias}"] = self._reduce_sum_if_needed(loss_sum) / step_count

        self._reset_train_metrics()
        return logs

    def pop_eval_logs(self) -> Dict[str, float]:
        logs: Dict[str, float] = {}
        if self.eval_iter_valid_total > 0:
            logs["eval_avg_iter_count"] = (
                self.eval_iter_count_sum / self.eval_iter_valid_total
            )
            logs["eval_avg_iter_label"] = (
                self.eval_iter_label_sum / self.eval_iter_valid_total
            )
            for k, cnt in sorted(self.eval_iter_ge_sum.items()):
                logs[f"eval_iter_ge{k}"] = cnt / self.eval_iter_valid_total
            for k, cnt in sorted(self.eval_iter_label_ge_sum.items()):
                logs[f"eval_iter_label_ge{k}"] = cnt / self.eval_iter_valid_total
        if self.eval_iter_decider_total > 0:
            logs["eval_iter_decider_accuracy"] = (
                self.eval_iter_decider_correct / self.eval_iter_decider_total
            )
        if self.eval_iter_decider_pos_weight_count > 0:
            logs["eval_iter_decider_pos_weight"] = (
                self.eval_iter_decider_pos_weight_sum
                / self.eval_iter_decider_pos_weight_count
            )
        for depth, weight_sum in sorted(
            self.eval_iter_decider_pos_weight_by_depth_sum.items()
        ):
            count = self.eval_iter_decider_pos_weight_by_depth_count.get(depth, 0.0)
            if count > 0:
                logs[f"eval_iter_decider_pos_weight_d{depth}"] = weight_sum / count
        for depth, ent_sum in sorted(
            self.eval_iter_decider_entropy_sum_by_depth.items()
        ):
            count = self.eval_iter_decider_entropy_count_by_depth.get(depth, 0.0)
            if count > 0:
                logs[f"eval_decider_entropy_d{depth}"] = ent_sum / count
        self._append_effective_expert_logs(logs, training=False)

        self._append_prf_metrics(
            logs,
            tp=self.eval_iter_decider_tp,
            fp=self.eval_iter_decider_fp,
            fn=self.eval_iter_decider_fn,
            prefix="eval_",
        )
        for loss_name, loss_sum in sorted(self.eval_loss_sums.items()):
            alias = self._loss_alias(loss_name)
            logs[f"eval_loss_{alias}"] = self._reduce_sum_if_needed(loss_sum)

        self._reset_eval_metrics()
        return logs

    # ------------------------------------------------------------------
    # Logging helpers called from TaHForCausalLM.forward()
    # ------------------------------------------------------------------

    # Default highest depth bin for iter_ge{k} logging when a training loop does
    # not override ``max_depth_bins`` (which it does, from the model's max_iter).
    _MAX_DEPTH_BINS = 8

    def log_effective_expert_metrics(
        self,
        effective_by_depth: torch.Tensor,
        valid_by_depth: torch.Tensor,
        depths,
        training: bool,
    ) -> None:
        """Accumulate globally-computed, step-level active-expert metrics.

        ``effective_by_depth[d]`` is the mean across MoE layers of the
        inverse-Simpson effective expert count ``1 / sum_e p_e^2`` over the
        global selection frequencies (num_experts when balanced, -> 1 on
        collapse). Detached CUDA scalars are retained until ``pop_*_logs``
        to avoid a forward-path sync.
        """
        prefix = "" if training else "eval_"
        sums: Dict[int, torch.Tensor] = getattr(
            self, f"{prefix}effective_expert_sum_by_depth"
        )
        counts: Dict[int, torch.Tensor] = getattr(
            self, f"{prefix}effective_expert_count_by_depth"
        )
        values = effective_by_depth.detach().float()
        valid = valid_by_depth.detach().float()
        for index, depth in enumerate(depths):
            depth = int(depth)
            contribution = values[index] * valid[index]
            sums[depth] = sums.get(depth, torch.zeros_like(contribution)) + contribution
            counts[depth] = counts.get(
                depth, torch.zeros_like(valid[index])
            ) + valid[index]

    def _append_effective_expert_logs(
        self, logs: Dict[str, float], training: bool
    ) -> None:
        prefix = "" if training else "eval_"
        sums: Dict[int, torch.Tensor] = getattr(
            self, f"{prefix}effective_expert_sum_by_depth"
        )
        counts: Dict[int, torch.Tensor] = getattr(
            self, f"{prefix}effective_expert_count_by_depth"
        )
        depths = sorted(set(sums) & set(counts))
        if not depths:
            return
        # Synchronize all depths once at log flush.
        packed = torch.stack(
            [item for depth in depths for item in (sums[depth], counts[depth])]
        ).tolist()
        for index, depth in enumerate(depths):
            value, count = float(packed[2 * index]), float(packed[2 * index + 1])
            if count > 0:
                logs[f"{prefix}avg_effective_experts_iter{depth}"] = value / count

    def log_iter_metrics(
        self,
        labels_shifted: torch.LongTensor,
        actual_iter_counts: torch.LongTensor,
        finalized_iter_labels,
        device: torch.device,
        training: bool,
    ):
        valid_mask = labels_shifted.detach() != -100
        valid_iter_counts = actual_iter_counts.detach()[valid_mask]
        iter_count_sum = valid_iter_counts.sum().float()
        iter_label_sum = torch.tensor(0.0, device=device, dtype=torch.float)
        if finalized_iter_labels is not None:
            iter_label_sum = finalized_iter_labels.detach()[valid_mask].sum().float()
        valid_total = valid_mask.sum().float()

        # Count tokens with iter_count >= k (actual) and iter_label >= k (label)
        # for k = 2 .. max_depth_bins (= model max_iter under a training loop).
        # Vectorized: broadcast compare against all bin thresholds at once, then
        # one reduction kernel per tensor instead of ``n_bins`` separate ones.
        # n_bins is a rank-invariant config (identical on every rank), so the
        # cross-rank gather/all_reduce below stays aligned.
        n_bins = max(self.max_depth_bins - 1, 1)  # k = 2 .. max_depth_bins
        ks = torch.arange(2, 2 + n_bins, device=device, dtype=valid_iter_counts.dtype)
        ge_counts = (
            (valid_iter_counts.unsqueeze(0) >= ks.unsqueeze(1)).sum(dim=1).float()
        )
        valid_iter_labels = None
        if finalized_iter_labels is not None:
            valid_iter_labels = finalized_iter_labels.detach()[valid_mask].float()
            label_ge_counts = (
                (valid_iter_labels.unsqueeze(0) >= ks.unsqueeze(1).float())
                .sum(dim=1)
                .float()
            )
        else:
            label_ge_counts = torch.zeros(n_bins, device=device, dtype=torch.float)

        accelerator = getattr(getattr(self, "trainer", None), "accelerator", None)
        if accelerator is not None:
            # Gather [iter_count_sum, iter_label_sum, valid_total, ge_counts..., label_ge_counts...] across ranks.
            stacked = torch.cat(
                [
                    torch.stack([iter_count_sum, iter_label_sum, valid_total]),
                    ge_counts,
                    label_ge_counts,
                ]
            )
            gathered = accelerator.gather(stacked)
            sums = gathered.view(-1, len(stacked)).sum(dim=0)
            iter_count_sum, iter_label_sum, valid_total = sums[0], sums[1], sums[2]
            ge_counts = sums[3 : 3 + n_bins]
            label_ge_counts = sums[3 + n_bins :]
        elif getattr(self, "dp_group", None) is not None:
            # Torch-native TP path (no Accelerator): counters are rank-local,
            # so reduce sums and their valid_total denominator over the dp group
            # (NOT world: tp ranks see identical data and would double-count).
            stacked = torch.cat(
                [
                    torch.stack([iter_count_sum, iter_label_sum, valid_total]),
                    ge_counts,
                    label_ge_counts,
                ]
            )
            torch.distributed.all_reduce(stacked, group=self.dp_group)
            iter_count_sum, iter_label_sum, valid_total = (
                stacked[0],
                stacked[1],
                stacked[2],
            )
            ge_counts = stacked[3 : 3 + n_bins]
            label_ge_counts = stacked[3 + n_bins : 3 + 2 * n_bins]

        # Batch every scalar we need into a single sync: the prior impl called
        # ``.item()`` 17 times per microbatch (count_sum, label_sum, valid_total,
        # 7 ge_counts, 7 label_ge_counts) — each one a CPU-GPU sync.  Stack and
        # ``.tolist()`` once instead.
        scalars = torch.cat(
            [
                torch.stack(
                    [
                        iter_count_sum.float(),
                        iter_label_sum.float(),
                        valid_total.float(),
                    ]
                ),
                ge_counts.float(),
                label_ge_counts.float(),
            ]
        )
        scalars_cpu = scalars.tolist()
        count_sum = float(scalars_cpu[0])
        label_sum = float(scalars_cpu[1])
        vt = float(scalars_cpu[2])
        ge_counts_cpu = scalars_cpu[3 : 3 + n_bins]
        label_ge_counts_cpu = scalars_cpu[3 + n_bins : 3 + 2 * n_bins]

        prefix = "" if training else "eval_"
        setattr(
            self,
            f"{prefix}iter_count_sum",
            getattr(self, f"{prefix}iter_count_sum") + count_sum,
        )
        setattr(
            self,
            f"{prefix}iter_label_sum",
            getattr(self, f"{prefix}iter_label_sum") + label_sum,
        )
        if vt > 0:
            setattr(
                self,
                f"{prefix}iter_valid_total",
                getattr(self, f"{prefix}iter_valid_total") + vt,
            )
            ge_sum: Dict[int, float] = getattr(self, f"{prefix}iter_ge_sum")
            label_ge_sum: Dict[int, float] = getattr(self, f"{prefix}iter_label_ge_sum")
            for i in range(n_bins):
                k = i + 2
                cnt = ge_counts_cpu[i]
                if cnt > 0:
                    ge_sum[k] = ge_sum.get(k, 0.0) + cnt
                label_cnt = label_ge_counts_cpu[i]
                if label_cnt > 0:
                    label_ge_sum[k] = label_ge_sum.get(k, 0.0) + label_cnt


def compute_eval_num_items(eval_dataset) -> Optional[int]:
    if eval_dataset is None:
        return None
    total = 0
    for idx in range(len(eval_dataset)):
        example = eval_dataset[idx]
        labels = example.get("labels") if isinstance(example, dict) else None
        if labels is None:
            continue
        labels_tensor = torch.as_tensor(labels)
        total += int((labels_tensor != -100).sum().item())
    return total if total > 0 else None


def compute_eval_global_batch_size(eval_dataset) -> Optional[int]:
    if eval_dataset is None:
        return None
    total = len(eval_dataset)
    return int(total) if total > 0 else None


def count_num_items_in_batch(
    batch_samples: List[Dict[str, torch.Tensor]], accelerator: Accelerator
) -> Optional[torch.Tensor]:
    if not batch_samples or "labels" not in batch_samples[0]:
        return None

    num_items_in_batch = sum(
        (batch["labels"].ne(-100)).sum() for batch in batch_samples
    )
    num_items_in_batch = accelerator.gather(
        num_items_in_batch.to(accelerator.device)
    ).sum()
    return torch.clamp(num_items_in_batch, min=1)


def count_global_batch_size(
    batch_samples: List[Dict[str, torch.Tensor]], accelerator: Accelerator
) -> Optional[torch.Tensor]:
    local_batch_size = 0
    for batch in batch_samples:
        labels = batch["labels"]
        local_batch_size += int(labels.shape[0])

    local_batch_size = torch.tensor(
        local_batch_size, device=accelerator.device, dtype=torch.long
    )
    global_batch_size = accelerator.gather(local_batch_size).sum()
    return torch.clamp(global_batch_size, min=1)


def update_iter_decider_training_state(model, state, num_train_epochs: int):
    iter_decider = getattr(model, "iter_decider", None)
    if iter_decider is None:
        return

    if hasattr(iter_decider, "num_grow_steps") and getattr(
        iter_decider, "num_grow_steps", None
    ) in [None, 0]:
        if getattr(state, "max_steps", None) is not None:
            iter_decider.num_grow_steps = state.max_steps

    if hasattr(iter_decider, "num_epochs") and getattr(
        iter_decider, "num_epochs", None
    ) in [None, 0]:
        iter_decider.num_epochs = int(num_train_epochs)

    if hasattr(iter_decider, "update_training_state") and callable(
        iter_decider.update_training_state
    ):
        current_step = int(getattr(state, "global_step", 0) or 0)
        epoch_value = getattr(state, "epoch", 0)
        current_epoch = int(epoch_value) if epoch_value is not None else 0
        iter_decider.update_training_state(
            current_step=current_step, current_epoch=current_epoch
        )


def update_loss_training_state(model, state):
    loss_objs = []
    if hasattr(model, "train_loss") and model.train_loss is not None:
        loss_objs.append(model.train_loss)
    if hasattr(model, "eval_loss") and model.eval_loss is not None:
        loss_objs.append(model.eval_loss)

    if not loss_objs:
        return

    current_step = int(getattr(state, "global_step", 0) or 0)
    epoch_value = getattr(state, "epoch", 0)
    current_epoch = int(epoch_value) if epoch_value is not None else 0

    def _maybe_update(obj):
        if obj is None:
            return
        if hasattr(obj, "update_training_state") and callable(
            obj.update_training_state
        ):
            try:
                obj.update_training_state(
                    current_step=current_step, current_epoch=current_epoch
                )
            except Exception:
                pass
        for child in getattr(obj, "losses", []):
            _maybe_update(child)

    for loss_obj in loss_objs:
        _maybe_update(loss_obj)


def sync_training_state(model, state, num_train_epochs: int):
    update_iter_decider_training_state(model, state, num_train_epochs)
    update_loss_training_state(model, state)
