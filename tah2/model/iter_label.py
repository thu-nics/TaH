
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Optional, Union

from tah2.utils.component_registry import (
    get_iter_label_generator_class as get_iter_label_generator_class,
    register_iter_label_generator,
)
from tah2.utils.fp32_ops import fp32_logsumexp


class IterLabelGenerator(nn.Module):
    """Base class for generating per-token iter-count labels.

    Contract:
    - prepare(batch_size, seq_len, device, dtype): allocate internal buffers
    - intra_iter_labels(...): return labels for current active tokens, and update internal full labels
    - finalize(): return full (B, S) labels accumulated across iterations
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.full_labels = None

    def prepare(
        self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        self.full_labels = torch.full(
            (batch_size, seq_len), fill_value=0, device=device, dtype=torch.long
        )

    @staticmethod
    def _assign_active(
        current_iter_mask: torch.BoolTensor, src: torch.Tensor, dest: torch.Tensor
    ) -> torch.Tensor:
        """Scatter active `src` back to dense `dest` (vectorized, sync-free)."""
        if current_iter_mask.shape[0] == 0 or src.shape[1] == 0:
            return dest
        B, max_n = src.shape[0], src.shape[1]
        active_counts = current_iter_mask.sum(1)  # kept on GPU
        col_idx = torch.arange(max_n, device=src.device).expand(B, max_n)
        src_valid_mask = col_idx < active_counts.unsqueeze(1)
        dest[current_iter_mask] = src[src_valid_mask]
        return dest


    def intra_iter_labels(
        self,
        active_logits: torch.Tensor,
        active_labels_shifted: Optional[torch.Tensor],
        iter_depth: int,
        current_iter_mask: torch.BoolTensor,
        active_valid_mask: torch.LongTensor,
        prompt_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        **kwargs,
    ) -> Optional[torch.LongTensor]:
        raise NotImplementedError

    def finalize(self) -> Optional[torch.LongTensor]:
        return self.full_labels




@register_iter_label_generator
@register_iter_label_generator("dynamiclabel")
class DynamicIterLabelGenerator(IterLabelGenerator):
    """Posterior CE labels from next-token improvements measured in a side pass."""

    def __init__(
        self,
        max_iter: int = 2,
        abs_improv_coverage: Union[float, List[float]] = 0.99,
        cost_min: float = 1e-6,
        cost_max: float = 1.0,
        memory_lean_fp32_reductions: bool = False,
    ):
        super().__init__()
        self.max_iter = int(max_iter)
        self.memory_lean_fp32_reductions = bool(memory_lean_fp32_reductions)
        coverages = (
            list(abs_improv_coverage)
            if isinstance(abs_improv_coverage, (list, tuple))
            else [abs_improv_coverage]
        )
        if not coverages or any(not 0.0 < float(c) <= 1.0 for c in coverages):
            raise ValueError("abs_improv_coverage must contain values in (0, 1]")
        self.abs_improv_coverages = [float(c) for c in coverages]
        self.cost_min = float(cost_min)
        self.cost_max = float(cost_max)
        if not 0.0 < self.cost_min < self.cost_max:
            raise ValueError("cost bounds must satisfy 0 < cost_min < cost_max")
        self.executed_mask_by_depth: Dict[int, torch.BoolTensor] = {}
        self._iter_scores: Dict[int, torch.Tensor] = {}
        self._iter_score_filled: Dict[int, torch.BoolTensor] = {}
        # Boundary distances weight the cost-sensitive decider BCE.
        self._pending_cost_scores: Optional[torch.Tensor] = None
        self._pending_cost_scores_by_depth: Dict[int, torch.Tensor] = {}


    def _all_gather_flat(self, local: torch.Tensor) -> torch.Tensor:
        """Pool scores over DP ranks, including ranks with no local candidates.

        Every rank calls this a fixed number of times per forward to keep
        the posterior boundary collectives aligned.
        """
        local = local.detach().to(dtype=torch.float64).flatten()
        device = local.device

        if not (dist.is_available() and dist.is_initialized()):
            return local

        group = getattr(self, "dp_group", None)
        world = dist.get_world_size(group)
        local_size = torch.tensor([local.numel()], dtype=torch.long, device=device)
        sizes = [torch.zeros_like(local_size) for _ in range(world)]
        dist.all_gather(sizes, local_size, group=group)
        sizes_cpu = torch.cat(sizes).cpu().tolist()
        max_size = max(sizes_cpu)
        if max_size == 0:
            return torch.empty(0, dtype=torch.float64, device=device)
        padded = torch.zeros(max_size, dtype=torch.float64, device=device)
        if local.numel() > 0:
            padded[: local.numel()] = local
        gathered = [
            torch.zeros(max_size, dtype=torch.float64, device=device)
            for _ in range(world)
        ]
        dist.all_gather(gathered, padded, group=group)
        chunks = [g[:n] for g, n in zip(gathered, sizes_cpu) if n > 0]
        if not chunks:
            return torch.empty(0, dtype=torch.float64, device=device)
        return torch.cat(chunks)

    def prepare(
        self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        super().prepare(batch_size, seq_len, device, dtype)
        self._iter_scores = {}
        self._iter_score_filled = {}
        self.executed_mask_by_depth = {}
        self._pending_cost_scores = None
        self._pending_cost_scores_by_depth = {}

    # ------------------------------------------------------------------
    # Multi-depth score management (for max_iter > 2)
    # ------------------------------------------------------------------

    def _cap_candidates(
        self, candidate_mask: torch.BoolTensor, depth: int
    ) -> torch.BoolTensor:
        """Only label tokens that executed this depth (labels <= actual + 1)."""
        executed = self.executed_mask_by_depth.get(int(depth))
        if executed is None:
            return torch.zeros_like(candidate_mask)
        return candidate_mask & executed

    def iter_scores_list(self) -> List[torch.Tensor]:
        return [self._iter_scores[d] for d in sorted(self._iter_scores)]

    def get_iter_score(self, depth: int) -> Optional[torch.Tensor]:
        return self._iter_scores.get(int(depth))

    def iter_score_filled_mask(self, depth: int) -> Optional[torch.BoolTensor]:
        return self._iter_score_filled.get(int(depth))

    def has_global_tokens(self, mask: torch.BoolTensor) -> bool:
        flag = mask.any().to(dtype=torch.long).reshape(1)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                flag,
                op=dist.ReduceOp.SUM,
                group=getattr(self, "tah_sync_group", None),
            )
        return bool(flag.item())

    def has_global_iter_score(self, depth: int, device: torch.device) -> bool:
        flag = torch.tensor(
            [int(self.get_iter_score(depth) is not None)],
            device=device,
            dtype=torch.long,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                flag,
                op=dist.ReduceOp.SUM,
                group=getattr(self, "tah_sync_group", None),
            )
        return bool(flag.item())

    def ensure_iter_score(self, depth: int, like: torch.Tensor) -> torch.Tensor:
        depth = int(depth)
        score = self._iter_scores.get(depth)
        if score is None:
            score = torch.zeros_like(like, dtype=torch.float32)
            self._iter_scores[depth] = score
            self._iter_score_filled[depth] = torch.zeros_like(like, dtype=torch.bool)
        return score

    def set_iter_score_active(
        self,
        depth: int,
        current_iter_mask: torch.BoolTensor,
        score: torch.Tensor,
    ) -> None:
        """Write active-layout ``score`` into the dense score buffer for ``depth``."""
        depth = int(depth)
        B, S = current_iter_mask.shape
        buf = self.ensure_iter_score(
            depth, torch.empty(B, S, device=score.device, dtype=torch.float32)
        )
        self._assign_active(current_iter_mask, score.float(), buf)
        filled = self._iter_score_filled.get(depth)
        if filled is None:
            filled = torch.zeros(
                B, S, device=current_iter_mask.device, dtype=torch.bool
            )
            self._iter_score_filled[depth] = filled
        self._iter_score_filled[depth] = filled | current_iter_mask.to(
            device=filled.device
        )

    # ------------------------------------------------------------------

    def finalize_posterior_labels(
        self,
        labels_shifted: torch.Tensor,
        ignore_index: int = -100,
    ) -> Optional[torch.LongTensor]:
        """Finalize posterior labels by looping over all depth boundaries.

        For max_iter=2 this is identical to the original (single boundary).
        For max_iter>2 each boundary uses the same abs_improv_coverage
        selection criterion applied to the candidate tokens that survived the
        previous boundary.
        """
        if self.get_iter_score(1) is None:
            return None

        valid_supervision = labels_shifted != ignore_index
        score1 = self.ensure_iter_score(
            1, torch.zeros_like(labels_shifted, dtype=torch.float32)
        )

        labels = torch.ones_like(labels_shifted, dtype=torch.long)
        candidate_mask = valid_supervision

        # Fixed-count boundary loop: the threshold statistic inside
        # _select_posterior_hard_mask all-gathers over the dp-pooled group, and
        # DP replicas gate their cascades independently (TP-scoped sync), so
        # every rank must fire that collective exactly (max_iter - 1) times per
        # forward. Once this rank's cascade is exhausted, the selection still
        # runs on an empty candidate set purely for collective alignment; its
        # result is discarded, so label/cost semantics match the old ``break``.
        done = False
        noop_diff = None
        for depth in range(1, self.max_iter):
            candidate_mask = self._cap_candidates(candidate_mask, depth)
            if not done and not self.has_global_tokens(candidate_mask):
                done = True
            if done:
                if noop_diff is None:
                    noop_diff = torch.zeros_like(
                        labels_shifted, dtype=torch.float32
                    )
                self._select_posterior_hard_mask(
                    score_diff=noop_diff,
                    valid_supervision=torch.zeros_like(valid_supervision),
                    depth=depth,
                )
                continue
            cur_score = self.get_iter_score(depth)
            if cur_score is None:
                cur_score = score1
            next_score = self.get_iter_score(depth + 1)
            if next_score is None:
                next_score = cur_score.clone()  # no improvement → all easy

            score_diff = cur_score - next_score.to(device=cur_score.device, dtype=cur_score.dtype)

            hard_mask, boundary_val = self._select_posterior_hard_mask(
                score_diff=score_diff,
                valid_supervision=candidate_mask,
                depth=depth,
            )
            cost_mask = candidate_mask
            cost = (
                (score_diff[cost_mask].float() - boundary_val)
                .abs()
                .clamp(min=self.cost_min, max=self.cost_max)
            )
            cost_dense = torch.full_like(score_diff, self.cost_min, dtype=torch.float32)
            cost_dense[cost_mask] = cost
            self._pending_cost_scores_by_depth[depth] = cost_dense.detach()
            if depth == 1:
                # Depth-1 costs also expose the flat supervised-token layout.
                self._pending_cost_scores = cost_dense[valid_supervision].detach()

            labels[hard_mask] = depth + 1
            candidate_mask = hard_mask

        labels = labels.clamp(max=self.max_iter)
        labels = labels.masked_fill(~valid_supervision, ignore_index)
        self.full_labels = labels.masked_fill(labels == ignore_index, 0)
        return labels

    def compute_posterior_score(
        self,
        active_logits: torch.Tensor,
        active_labels_shifted: Optional[torch.Tensor],
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Per-token CE on the shifted next-token labels."""
        if active_labels_shifted is None:
            return torch.zeros_like(active_logits[..., 0])
        return self._compute_shifted_token_ce(
            active_logits, active_labels_shifted, ignore_index
        )

    def _select_posterior_hard_mask(
        self,
        score_diff: torch.Tensor,
        valid_supervision: torch.BoolTensor,
        depth: int = 1,
    ) -> tuple[torch.BoolTensor, float]:
        """Select tokens covering the requested fraction of positive CE gain."""
        idx = min(max(int(depth) - 1, 0), len(self.abs_improv_coverages) - 1)
        threshold = self._coverage_threshold(
            score_diff, valid_supervision, self.abs_improv_coverages[idx]
        )
        if threshold is None:
            return torch.zeros_like(valid_supervision, dtype=torch.bool), 0.0
        thr = torch.tensor(threshold, dtype=score_diff.dtype, device=score_diff.device)
        return valid_supervision & (score_diff >= thr), threshold

    def _compute_shifted_token_ce(
        self,
        logits: torch.Tensor,
        labels_shifted: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        ops = getattr(self, "vocab_parallel_ops", None)
        if ops is not None:
            # logits is a vocab shard [..., V/tp]; CE needs the global
            # logsumexp + target gather across the tp group.
            return ops.cross_entropy_per_token(
                logits.reshape(-1, logits.shape[-1]),
                labels_shifted.reshape(-1),
                ignore_index=ignore_index,
            ).reshape_as(labels_shifted)
        if self.memory_lean_fp32_reductions:
            valid = labels_shifted != ignore_index
            safe = labels_shifted.masked_fill(~valid, 0)
            z = fp32_logsumexp(logits)
            picked = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1).float()
            return torch.where(valid, z - picked, torch.zeros_like(z))
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels_shifted.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).reshape_as(labels_shifted)
        return ce

    def _coverage_threshold(
        self,
        score: torch.Tensor,
        valid_mask: torch.BoolTensor,
        coverage: float,
    ) -> Optional[float]:
        """Smallest score s.t. tokens with score >= s cover ``coverage`` fraction
        of the total positive-score sum, pooled across all DP ranks.

        Returns the threshold value (strictly positive by construction), or
        ``None`` if no positive scores exist in the union across ranks.

        All ranks must enter the all_gather unconditionally — passing an empty
        local tensor is the deadlock-safe behavior.
        """
        local = (
            score[valid_mask & (score > 0)].detach().to(dtype=torch.float64).flatten()
        )
        local = torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
        all_pos = self._all_gather_flat(local)
        if all_pos.numel() == 0:
            return None
        sorted_desc, _ = torch.sort(all_pos, descending=True)
        cumsum = torch.cumsum(sorted_desc, dim=0)
        total = cumsum[-1].item()
        if total <= 0.0:
            return None
        target = float(max(0.0, min(1.0, coverage))) * total
        target_tensor = torch.tensor(target, dtype=cumsum.dtype, device=cumsum.device)
        idx = int(torch.searchsorted(cumsum, target_tensor).item())
        idx = max(0, min(idx, sorted_desc.numel() - 1))
        return float(sorted_desc[idx].item())

    @staticmethod
    def _compact_from_full(
        tensor: Optional[torch.Tensor],
        current_iter_mask: torch.BoolTensor,
        target_active_len: int,
        pad_value: float = 0.0,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.dim() < 2:
            return tensor
        bsz = current_iter_mask.shape[0]
        seq_len = current_iter_mask.shape[1]
        if tensor.shape[0] != bsz:
            return tensor
        if tensor.shape[1] == target_active_len:
            return tensor
        if tensor.shape[1] != seq_len:
            return tensor

        # Vectorized inverse of assign_active: gather tensor[b, mask[b]] into
        # out[b, :n_b].  ``dst_valid`` and ``mask`` have matching total counts
        # in row-major order, so the scatter is 1-to-1.
        out_shape = (bsz, target_active_len, *tensor.shape[2:])
        out = tensor.new_full(out_shape, fill_value=pad_value)
        mask = current_iter_mask.to(device=tensor.device)
        active_counts = mask.sum(dim=1)
        col_idx = torch.arange(target_active_len, device=tensor.device).expand(
            bsz, target_active_len
        )
        dst_valid = col_idx < active_counts.unsqueeze(1)
        out[dst_valid] = tensor[mask]
        return out

    def intra_iter_labels(
        self,
        active_logits: torch.Tensor,
        active_labels_shifted: Optional[torch.Tensor],
        iter_depth: int,
        current_iter_mask: torch.BoolTensor,
        active_valid_mask: torch.LongTensor,
        prompt_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        **kwargs,
    ) -> Optional[torch.LongTensor]:
        if active_labels_shifted is None or active_logits is None:
            return None
        self._pending_cost_scores = None
        score = self.compute_posterior_score(
            active_logits, active_labels_shifted, ignore_index
        ).float().detach()
        self.set_iter_score_active(iter_depth, current_iter_mask, score)
        # Labels and decider BCE are finalized after the no-grad side pass.
        return None
