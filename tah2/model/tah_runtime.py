from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from tah2.model.causal_cache import TaHCache
from tah2.model.loss import IterDeciderLoss
from tah2.utils import sparse_ops
from tah2.utils.fp32_ops import fp32_logsumexp


# ---------------------------------------------------------------------------
# Posterior side-pass helpers
# ---------------------------------------------------------------------------


def _dp_any_tokens(label_gen, mask: torch.BoolTensor) -> bool:
    """Align side-pass entry over DP ranks because FSDP forwards use DP collectives.

    Each rank calls this once per depth, even when it has no candidates.
    A rank with no local work must still run a dummy backbone pass.
    """
    flag = mask.any().to(dtype=torch.long).reshape(1)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(
            flag,
            op=dist.ReduceOp.MAX,
            group=getattr(label_gen, "dp_group", None),
        )
    return bool(flag.item())


def _dummy_side_pass_for_alignment(runtime: "TaHForwardRuntime") -> None:
    """Keep parameter-sharded (FSDP) per-layer collectives aligned when this
    rank has nothing to forward in a side pass that other ranks are running.

    Callers gate side passes on ``_dp_any_tokens`` (a DP-wide decision), so a
    rank that bails out locally must still walk the backbone once.
    """
    if not (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    ):
        return
    with torch.no_grad():
        runtime.model._run_dummy_backbone_pass(
            runtime.first_iter_embeds, runtime.position_ids
        )


def _run_side_score_pass(
    runtime: "TaHForwardRuntime",
    depth: int,
    target_mask: torch.BoolTensor,
) -> None:
    """Run one no-grad backbone forward to fill the posterior score at ``depth``
    for the ``target_mask`` (stop) tokens.

    In duo attention mode all tokens stored for this depth are forwarded together
    so that stop-token queries can attend to continue-token iter(depth-1) KV.
    The duo-posterior attention mask encodes this visibility rule using the
    per-depth continue mask stored by capture_posterior_inputs.
    """
    model = runtime.model
    label_gen = model.iter_label_generator
    if runtime.main_cache is None:
        return

    attn_mode = getattr(model, "iter_attention_mode", None)
    # Duo forwards stopped and continuing tokens together so stopped queries
    # can attend to continuing tokens' current-iteration KV.
    needs_full_forward = attn_mode == "duo"
    iter_depth = int(depth) - 1

    stored = runtime.side_inputs_by_depth.get(depth)
    if stored is None:
        _dummy_side_pass_for_alignment(runtime)
        return
    stored_mask, side_input = stored
    forward_mask = stored_mask if needs_full_forward else target_mask
    if not forward_mask.any():
        _dummy_side_pass_for_alignment(runtime)
        return

    with torch.no_grad():
        (
            active_input_embeds,
            active_position_ids,
            active_valid_mask,
            _ac,
            active_labels_shifted,
            _alc,
            _alas,
            _arm,
            active_first_iter_embeds,
        ) = model.to_active(
            forward_mask,
            side_input,
            runtime.position_ids,
            runtime.valid_mask,
            None,
            runtime.labels_shifted,
            None,
            None,
            None,
            runtime.first_iter_embeds,
        )
        if active_valid_mask.shape[1] == 0:
            _dummy_side_pass_for_alignment(runtime)
            return

        # Build run cache: main-forward KV up to iter(depth-2) so the backbone at
        # iter(depth-1) sees the correct prior-iter context without re-running anything.
        run_cache = runtime.main_cache.snapshot_iter(max(iter_depth - 1, 0))
        need_next_input = int(depth) < int(model.max_iter)

        if needs_full_forward:
            # Posterior attention mask (duo): stop-token queries can
            # attend to continue-token iter(depth-1) KV in the current forward
            # batch.  The continue flag drives which entries stay visible.
            continue_mask = runtime.side_pass_continue_mask.get(int(depth))
            mask_impl = "triton" if getattr(model, "_triton_enabled", False) else "sdpa"
            if continue_mask is not None:
                # Compact dense continue_mask to active layout (B, L_fwd)
                fwd_len = active_valid_mask.shape[1]
                active_continue = label_gen._compact_from_full(
                    continue_mask.to(dtype=torch.long),
                    forward_mask,
                    fwd_len,
                    pad_value=0,
                ).to(dtype=torch.bool)
            else:
                active_continue = torch.zeros_like(active_valid_mask, dtype=torch.bool)
            attn_mask = sparse_ops.create_posterior_side_attention_mask(
                iter_attention_mode=attn_mode,
                active_position_ids=active_position_ids,
                active_valid_mask=active_valid_mask,
                active_continue_mask=active_continue,
                cache=run_cache,
                iter_depth=iter_depth,
                dtype=runtime.dtype,
                iter_attention_impl=mask_impl,
            )
            iter_use_cache = need_next_input or runtime.use_cache
        else:
            attn_mask, iter_use_cache = model._build_attention_mask_for_iter(
                iter_depth=iter_depth,
                active_position_ids=active_position_ids,
                active_valid_mask=active_valid_mask,
                cache=run_cache,
                dtype=runtime.dtype,
                use_cache=need_next_input,
            )

        active_outputs = model._process_sparse_iteration(
            sparse_input=active_input_embeds,
            position_ids=active_position_ids,
            valid_mask=active_valid_mask,
            cache_position=None,
            attention_mask=attn_mask,
            iter_depth=iter_depth,
            past_key_values=run_cache,
            use_cache=iter_use_cache,
            output_attentions=runtime.output_attentions,
            output_hidden_states=need_next_input,
            model=model.simple_base_model,
            **runtime.kwargs,
        )

        # Compute scores for forward_mask tokens, write all back.
        # In duo mode continue-token scores are overwritten with new side-pass values,
        # but because their inputs are identical the difference is negligible (bf16 noise).
        active_score = (
            label_gen.compute_posterior_score(
                active_logits=active_outputs.logits.to(device=runtime.device),
                active_labels_shifted=active_labels_shifted,
                ignore_index=-100,
            )
            .float()
        )
        label_gen.set_iter_score_active(depth, forward_mask, active_score)

        all_hidden = (
            model._stack_hidden_states(active_outputs, runtime.device)
            if need_next_input
            else None
        )
        # Prepare inputs for the next depth's side pass (stop tokens at this depth
        # may need score depth+1 filled later in the loop).
        if need_next_input:
            valid_active = active_valid_mask == 1
            if valid_active.any():
                next_values = model.input_updater(
                    prev_inputs=active_first_iter_embeds[valid_active],
                    hidden_states=(
                        all_hidden[valid_active]
                        if all_hidden is not None
                        else None
                    ),
                ).to(device=runtime.device, dtype=runtime.dtype)
                dense_mask = torch.zeros_like(runtime.valid_mask, dtype=torch.bool)
                model.assign_active_with_mask(
                    current_iter_mask=forward_mask,
                    assignment_mask=(runtime.valid_mask == 1),
                    src=valid_active.to(torch.bool),
                    dest=dense_mask,
                )
                runtime.add_side_inputs(depth + 1, dense_mask, next_values)


def _compute_posterior_labels_side_pass(
    runtime: "TaHForwardRuntime",
) -> Optional[torch.LongTensor]:
    """Fill missing posterior scores for all depths by running per-depth no-grad
    backbone passes for stop tokens, then call finalize_posterior_labels."""
    model = runtime.model
    label_gen = model.iter_label_generator
    if label_gen is None or not runtime.use_iter_labeling:
        return None
    if not label_gen.has_global_iter_score(1, runtime.labels_shifted.device):
        return None
    label_gen.executed_mask_by_depth = runtime.active_mask_by_depth

    if runtime.main_cache is None:
        return label_gen.finalize_posterior_labels(
            labels_shifted=runtime.labels_shifted
        )

    if runtime.return_eval_metadata:
        score_mask = runtime.labels_shifted != -100
        for next_depth in range(2, int(model.max_iter) + 1):
            filled = label_gen.iter_score_filled_mask(next_depth)
            missing = (
                score_mask
                if filled is None
                else score_mask & (~filled.to(device=score_mask.device))
            )
            if _dp_any_tokens(label_gen, missing):
                _run_side_score_pass(runtime, next_depth, missing)
            runtime.side_inputs_by_depth.pop(next_depth, None)
        return label_gen.finalize_posterior_labels(
            labels_shifted=runtime.labels_shifted
        )

    score1 = label_gen.ensure_iter_score(
        1, torch.zeros_like(runtime.labels_shifted, dtype=torch.float32)
    )
    candidate_mask = runtime.labels_shifted != -100
    # Intermediate boundaries select candidates for the next side pass. The
    # final boundary is selected only in label finalization. Each DP rank
    # fires the same number of selection and side-pass collectives, including
    # ranks whose candidate cascade ended earlier.
    done = False
    noop_diff = None
    for depth in range(1, int(model.max_iter)):
        candidate_mask = label_gen._cap_candidates(candidate_mask, depth)
        if not done and not label_gen.has_global_tokens(candidate_mask):
            done = True
        if not done:
            cur_score = label_gen.get_iter_score(depth)
            score_exists_globally = label_gen.has_global_iter_score(
                depth, runtime.device
            )
            if cur_score is None:
                if not score_exists_globally:
                    done = True
                else:
                    cur_score = label_gen.ensure_iter_score(depth, score1)
        next_depth = depth + 1
        if done:
            missing = torch.zeros_like(candidate_mask)
        else:
            filled = label_gen.iter_score_filled_mask(next_depth)
            missing = (
                candidate_mask
                if filled is None
                else candidate_mask & (~filled.to(device=candidate_mask.device))
            )
        # dp-aligned side-pass entry (see _dp_any_tokens): fires exactly once
        # per depth on every rank, done or not — a done rank enters with an
        # empty missing set and walks the dummy/empty path inside.
        if _dp_any_tokens(label_gen, missing):
            _run_side_score_pass(runtime, next_depth, missing)
        runtime.side_inputs_by_depth.pop(next_depth, None)

        if done:
            if noop_diff is None:
                noop_diff = torch.zeros_like(
                    runtime.labels_shifted, dtype=torch.float32
                )
            if depth < int(model.max_iter) - 1:
                label_gen._select_posterior_hard_mask(
                    score_diff=noop_diff,
                    valid_supervision=torch.zeros_like(candidate_mask),
                    depth=depth,
                )
            continue

        if depth == int(model.max_iter) - 1:
            break
        next_score = label_gen.get_iter_score(next_depth)
        if next_score is None:
            next_score = label_gen.ensure_iter_score(next_depth, cur_score)
        hard_mask, _ = label_gen._select_posterior_hard_mask(
            score_diff=cur_score - next_score.to(device=cur_score.device, dtype=cur_score.dtype),
            valid_supervision=candidate_mask,
            depth=depth,
        )
        candidate_mask = hard_mask

    return label_gen.finalize_posterior_labels(labels_shifted=runtime.labels_shifted)


def _run_posterior_deferred_decider_loss(
    runtime: "TaHForwardRuntime",
    labels: Optional[torch.LongTensor],
) -> None:
    """Supervise saved decider logits after the posterior labels are available."""
    if labels is None:
        return
    loss_kwargs = dict(runtime.kwargs, label_gen=runtime.model.iter_label_generator)
    if hasattr(runtime.model, "logger_callback"):
        loss_kwargs["logger_callback"] = runtime.model.logger_callback
    for module in runtime.loss_func.modules():
        if not isinstance(module, IterDeciderLoss):
            continue
        for depth, logits in runtime.decider_logits_by_depth.items():
            iter_mask = runtime.active_mask_by_depth[depth]
            if not iter_mask.any():
                continue
            # Dense masked positions have the same flat token order as the
            # compact layout. No gather/scatter round trip or dummy logits needed.
            valid = runtime.valid_mask * iter_mask
            module.intra_iter_loss_func(
                active_logits=logits,
                current_iter_mask=iter_mask,
                active_labels_shifted=runtime.labels_shifted,
                active_valid_continue_logits=logits[valid == 1],
                active_valid_mask=valid,
                iter_depth=depth,
                active_iter_count_labels=labels,
                **loss_kwargs,
            )


@dataclass
class TaHForwardRuntime:
    model: Any
    loss_func: nn.Module
    labels_shifted: Optional[torch.LongTensor]
    valid_mask: torch.LongTensor
    position_ids: torch.LongTensor
    first_iter_embeds: torch.Tensor
    use_cache: bool
    output_attentions: Optional[bool]
    output_hidden_states: Optional[bool]
    return_eval_metadata: bool
    kwargs: Dict[str, Any]
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        batch_size, query_len = self.valid_mask.shape
        model = self.model

        self.collect_eval_metadata = bool(
            self.output_hidden_states or self.return_eval_metadata
        )
        self.use_iter_labeling = (
            (model.training or self.return_eval_metadata)
            and (model.iter_label_generator is not None)
            and (self.labels_shifted is not None)
        )
        self.first_iter_hidden_states = None

        # Score depth d uses updater outputs from depth d-1. A side pass
        # fills CE scores missing because tokens stopped in the main forward.
        self.side_inputs_by_depth: Dict[int, Tuple[torch.BoolTensor, torch.Tensor]] = {}
        self.side_pass_continue_mask: Dict[int, torch.BoolTensor] = {}
        self.main_cache: Optional[TaHCache] = None
        self.decider_logits_by_depth: Dict[int, torch.Tensor] = {}
        self.active_mask_by_depth: Dict[int, torch.BoolTensor] = {}

        # Accumulate mixture log-probabilities; supervised training gathers
        # target probabilities into a small [B, L, 1] buffer instead of [B, L, V].
        vocab_size = model.config.vocab_size
        # Vocab-parallel TP: every logits tensor in this forward is a local
        # shard [..., V/tp]; size the mixture accumulator accordingly.
        self.vocab_parallel_ops = getattr(model, "vocab_parallel_ops", None)
        if self.vocab_parallel_ops is not None:
            vocab_size = self.vocab_parallel_ops.local_size
            if getattr(model, "memory_lean_fp32_reductions", False):
                raise NotImplementedError(
                    "vocab-parallel TP does not support memory_lean_fp32_reductions"
                )
        hidden_size = model.config.hidden_size
        self.use_target_gather_mix = (
            getattr(self.loss_func, "_use_target_gather_mix", False)
            and self.labels_shifted is not None
        )
        width = 1 if self.use_target_gather_mix else vocab_size
        self.final_output_logits = torch.full(
            (batch_size, query_len, width), float("-inf"), device=self.device,
            dtype=torch.float32 if self.use_target_gather_mix else self.dtype,
        )
        self.final_output_hidden = (
            torch.zeros(
                batch_size, query_len, hidden_size, device=self.device, dtype=self.dtype
            )
            if self.output_hidden_states
            else None
        )
        # even_mix weights each active iteration by 1 and normalizes by
        # actual_iter_counts in finalize_logits, so they don't need it.
        #
        # fp32, not self.dtype: remaining_mass is a running product of per-iter
        # continue probs (∏_{j<d} g_j). In bf16 the geometric decay loses ~1%
        # relative precision per step and compounds with depth — by iter 8 only
        # ~2-3 significant digits survive, corrupting the deep-iter mixture
        # weights. The V-space mixture accumulator stays bf16 (B*L*V memory), but
        # this O(B*L) mass buffer is cheap to keep in fp32.
        self.remaining_mass = (
            self.valid_mask.to(device=self.device, dtype=torch.float32)
            if model.weighted_hidden_method == "stop_prob_mix"
            else None
        )

    # ------------------------------------------------------------------
    # Per-iteration hooks
    # ------------------------------------------------------------------


    def capture_step(
        self,
        iter_depth: int,
        current_iter_mask: torch.BoolTensor,
        active_outputs,
        active_valid_continue_logits: Optional[torch.Tensor],
    ) -> None:
        if self.output_hidden_states and iter_depth == 1:
            last_hidden = active_outputs.hidden_states[-1]
            self.first_iter_hidden_states = torch.zeros_like(self.first_iter_embeds)
            self.model.assign_active_with_mask(
                current_iter_mask=current_iter_mask,
                assignment_mask=(self.valid_mask == 1),
                src=last_hidden,
                dest=self.first_iter_hidden_states,
            )
        if (
            not (self.use_iter_labeling or self.collect_eval_metadata)
            or active_valid_continue_logits is None
            or iter_depth >= self.model.max_iter
        ):
            return
        mask = current_iter_mask & (self.valid_mask == 1)
        logits = torch.zeros_like(self.valid_mask, dtype=torch.float32)
        logits[mask] = active_valid_continue_logits.float()
        self.decider_logits_by_depth[iter_depth] = logits
        if self.use_iter_labeling:
            self.active_mask_by_depth[iter_depth] = mask

    def add_side_inputs(
        self,
        depth: int,
        mask: torch.BoolTensor,
        values: torch.Tensor,
    ) -> None:
        """Merge detached updater outputs into the input buffer for one score depth."""
        if values.numel() == 0:
            return
        stored = self.side_inputs_by_depth.get(depth)
        if stored is None:
            stored_mask = torch.zeros_like(self.valid_mask, dtype=torch.bool)
            inputs = torch.zeros_like(self.first_iter_embeds)
        else:
            stored_mask, inputs = stored
        inputs[mask] = values.detach().to(device=self.device, dtype=self.dtype)
        self.side_inputs_by_depth[depth] = (stored_mask | mask, inputs)

    def capture_posterior_inputs(
        self,
        iter_depth: int,
        all_hidden: torch.Tensor,
        active_first_iter_embeds: torch.Tensor,
        current_iter_mask: torch.BoolTensor,
        active_valid_mask: torch.LongTensor,
        active_finished_mask: torch.BoolTensor,
        cache: TaHCache,
    ) -> None:
        """Prepare hypothetical next-iteration inputs, including stopped tokens.

        Score depths are 1-based: after the first iteration this prepares the
        second iteration's inputs. Duo needs continuing tokens as KV context
        too; other attention modes only need inputs for stopped tokens.
        """
        if not self.use_iter_labeling or iter_depth >= self.model.max_iter:
            return
        self.main_cache = cache
        valid = active_valid_mask == 1
        continued = torch.zeros_like(self.valid_mask, dtype=torch.bool)
        self.model.assign_active(current_iter_mask, valid & ~active_finished_mask, continued)
        self.side_pass_continue_mask[iter_depth + 1] = continued

        target = valid if self.model.iter_attention_mode == "duo" else valid & active_finished_mask
        if not target.any():
            return
        with torch.no_grad():
            values = self.model.input_updater(
                prev_inputs=active_first_iter_embeds[target],
                hidden_states=all_hidden[target],
            )
        mask = torch.zeros_like(self.valid_mask, dtype=torch.bool)
        self.model.assign_active(current_iter_mask, target, mask)
        self.add_side_inputs(iter_depth + 1, mask, values)


    def accumulate_iter(
        self,
        current_iter_mask: torch.BoolTensor,
        active_valid_mask: torch.LongTensor,
        active_finished_mask: torch.BoolTensor,
        active_labels_shifted: Optional[torch.Tensor],
        active_outputs,
        all_hidden: Optional[torch.Tensor],
        active_remaining_mass: Optional[torch.Tensor],
        active_valid_continue_logits: Optional[torch.Tensor],
        active_valid_continue_decision: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Accumulate weighted hidden/logits for this iteration. Returns active_continue_prob."""
        model = self.model
        weighted_method = model.weighted_hidden_method

        if all_hidden is None:
            raise ValueError(
                "TaH weighted hidden aggregation requires output hidden states."
            )
        active_last_hidden = all_hidden[..., -1, :]

        active_continue_prob = None
        if weighted_method == "stop_prob_mix":
            # Keep continue prob + token weight in fp32 (see remaining_mass note
            # in __post_init__): these feed the running mass product, so a bf16
            # round-trip here is exactly what compounds across depth. Downstream
            # consumers re-cast as needed (log_w does `.float()`; the bf16 hidden
            # buffer auto-casts on assign), so fp32 here is safe everywhere.
            active_continue_prob = torch.zeros_like(
                active_remaining_mass, dtype=torch.float32
            )
            if active_valid_continue_logits is not None:
                valid_continue_prob = torch.sigmoid(
                    active_valid_continue_logits.to(torch.float32)
                )
            else:
                valid_continue_prob = active_valid_continue_decision.to(
                    dtype=torch.float32
                )
            active_continue_prob[active_valid_mask == 1] = valid_continue_prob

            active_token_weight = active_remaining_mass * (1.0 - active_continue_prob)
            valid_finished_mask = active_finished_mask & (active_valid_mask == 1)
            active_token_weight[valid_finished_mask] = active_remaining_mass[
                valid_finished_mask
            ]
        elif weighted_method == "even_mix":
            # Uniform mixture across actually-executed iters: contribute weight=1
            # at every active iter; finalize_logits divides by actual_iter_counts
            # per token so the per-token total mass is exactly 1.
            active_token_weight = active_last_hidden.new_full(
                active_last_hidden.shape[:2], 1.0
            )
        else:
            raise ValueError(f"Unknown weighted_hidden_method: {weighted_method!r}")

        if self.use_target_gather_mix:
            log_weight = torch.log(active_token_weight.float().clamp(min=1e-10)).unsqueeze(-1)
            self._accumulate_target_mix(
                active_outputs.logits, active_labels_shifted, current_iter_mask, log_weight
            )
        else:
            # Inference accumulates full-vocabulary log-probabilities. Filter
            # padding before log_softmax; clone because autograd saves the
            # previous logaddexp output for backward.
            _avm = active_valid_mask == 1
            flat_logits = active_outputs.logits[_avm]  # (n_active, V) — V/tp local shard under vocab parallel
            # Autocast promotes this log_softmax to fp32 (~2x (n, V));
            # memory-sensitive recipes should bypass this branch via
            # NextTokenPredLoss.use_target_gather_mix.
            if self.vocab_parallel_ops is not None:
                flat_log_p = self.vocab_parallel_ops.log_softmax(flat_logits)
            else:
                flat_log_p = torch.log_softmax(flat_logits, dim=-1)
            flat_log_w = (
                torch.log(active_token_weight.float().clamp(min=1e-10))
                .to(self.dtype)[_avm]
                .unsqueeze(-1)
            )
            if self.vocab_parallel_ops is not None:
                # Replicated per-token weight entering sharded vocab
                # math: backward must SUM the per-shard partial grads
                # (each rank only backprops tokens whose target id is
                # in its shard) or the decider gradient is silently
                # token-partitioned across tp ranks.
                flat_log_w = self.vocab_parallel_ops.copy_to_shards(flat_log_w)
            flat_term = flat_log_p + flat_log_w  # (n_active, V)

            mask = current_iter_mask & (self.valid_mask == 1)
            new_logits = self.final_output_logits.clone()
            new_logits[mask] = torch.logaddexp(new_logits[mask], flat_term).to(new_logits.dtype)
            self.final_output_logits = new_logits
        if self.final_output_hidden is not None:
            model.add_active_with_mask(
                current_iter_mask=current_iter_mask,
                assignment_mask=(self.valid_mask == 1),
                src=active_last_hidden * active_token_weight.unsqueeze(-1),
                dest=self.final_output_hidden,
            )
        return active_continue_prob

    def _accumulate_target_mix(
        self,
        active_logits: torch.Tensor,
        active_labels_shifted: torch.Tensor,
        current_iter_mask: torch.BoolTensor,
        log_weight: torch.Tensor,
    ) -> None:
        """Accumulate target-token log-probabilities in fp32 without a full-vocab mixture."""
        target_ids = active_labels_shifted.clamp(min=0).unsqueeze(-1)
        if self.vocab_parallel_ops is not None:
            normalizer = self.vocab_parallel_ops.logsumexp_lean(active_logits)
            target_logits = self.vocab_parallel_ops.gather_values(active_logits, target_ids)
        else:
            normalizer = fp32_logsumexp(active_logits)
            target_logits = active_logits.gather(-1, target_ids)
        log_prob = target_logits.float() - normalizer.unsqueeze(-1)
        term = log_weight + log_prob
        dense_term = self.final_output_logits.new_full(self.final_output_logits.shape, float("-inf"))
        self.model.assign_active_with_mask(
            current_iter_mask=current_iter_mask,
            assignment_mask=(self.valid_mask == 1),
            src=term,
            dest=dense_term,
        )
        self.final_output_logits = torch.logaddexp(self.final_output_logits, dense_term)


    def update_remaining_mass(
        self,
        current_iter_mask: torch.BoolTensor,
        next_iter_mask: torch.BoolTensor,
        active_remaining_mass: torch.Tensor,
        active_continue_prob: torch.Tensor,
    ) -> None:
        if self.model.weighted_hidden_method != "stop_prob_mix":
            return
        updated = active_remaining_mass * active_continue_prob
        self.remaining_mass = torch.zeros_like(self.remaining_mass)
        self.model.assign_active_with_mask(
            current_iter_mask=current_iter_mask,
            assignment_mask=next_iter_mask,
            src=updated,
            dest=self.remaining_mass,
        )

    def finalize_logits(
        self,
        actual_iter_counts: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return mixture log-probabilities; normalize even_mix by executed depth."""
        if self.model.weighted_hidden_method == "even_mix":
            if actual_iter_counts is None:
                raise ValueError("even_mix requires actual_iter_counts")
            # Padding has zero iterations and is masked downstream.
            counts = actual_iter_counts.clamp(min=1)
            log_counts = torch.log(counts.to(self.final_output_logits.dtype)).unsqueeze(-1)
            self.final_output_logits = self.final_output_logits - log_counts
            if self.final_output_hidden is not None:
                inv_counts = (1.0 / counts.to(self.final_output_hidden.dtype)).unsqueeze(-1)
                self.final_output_hidden = self.final_output_hidden * inv_counts
        return self.final_output_logits

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> Optional[torch.Tensor]:
        if not self.use_iter_labeling:
            return None
        _compute_posterior_labels_side_pass(self)
        labels = self.model.iter_label_generator.finalize()
        _run_posterior_deferred_decider_loss(self, labels)
        return labels

    def output_fields(self) -> Dict[str, Any]:
        return {
            "first_iter_hidden_states": self.first_iter_hidden_states,
            "decider_logits": (
                list(self.decider_logits_by_depth.values()) or None
                if self.collect_eval_metadata else None
            ),
            "iter_scores": (
                self.model.iter_label_generator.iter_scores_list()
                if self.collect_eval_metadata and self.model.iter_label_generator is not None
                else None
            ),
        }
