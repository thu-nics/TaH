import math
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from tah2.utils.component_registry import (
    get_iter_decider_class as get_iter_decider_class,
    capture_init_args,
    register_iter_decider,
)
from tah2.utils.fp32_ops import fp32_logsumexp

POSITIVE_INFINITY_LOGITS = 10.0
MINUS_INFINITY_LOGITS = -10.0
NEUTRAL_LOGITS = 0.0


class IterDecider(nn.Module):
    """Base class for deciding whether to continue iterating a token.

    All IterDecider implementations must efficiently handle inputs of arbitrary shape (..., vocab_size)
    where (...) can be any number of leading dimensions (batch, sequence, etc.).
    """

    def __init__(self, threshold: float = 0.5, max_iter: int = 3):
        super().__init__()
        # store as buffer to allow assignment on subclasses without property conflicts
        self.register_buffer(
            "threshold", torch.tensor(float(threshold), dtype=torch.float32)
        )
        self.max_iter = max_iter

    def forward(self, logits: torch.Tensor, iter_depth: int, **kwargs) -> torch.Tensor:
        """
        Decide whether to continue iterating a token.

        Args:
            logits: The logits of the token, shape (..., vocab_size) where (...)
                   represents arbitrary leading dimensions
            iter_depth: The iteration depth of the token that has been processed.
            Optional kwargs:
                - hidden_states: The hidden states of the token, shape (..., hidden_size) where (...)

        Returns:
            A float tensor of shape (...) with values between 0 and 1,
            indicating the probability of continuing iteration.
            The output preserves all leading dimensions from the input.
        """
        raise NotImplementedError


@register_iter_decider
@capture_init_args
class TrivialIterDecider(IterDecider):
    """Trivial iteration decider that always ends.

    Efficiently handles arbitrary input shapes (..., vocab_size) by returning
    a boolean tensor of shape (...,) filled with False values.
    """

    def __init__(self, max_iter: int = 1, memory_lean_fp32_reductions: bool = False,):
        super().__init__(max_iter=max_iter)

    def forward(
        self,
        logits: torch.Tensor,
        iter_depth: int,
        active_valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Select valid positions if full tensors are provided
        if active_valid_mask is not None and logits.dim() == 3:
            logits = logits[active_valid_mask == 1]  # (N, V)

        decision = torch.zeros(
            logits.shape[:-1], dtype=torch.bool, device=logits.device
        )
        logits_out = torch.full(
            decision.shape, NEUTRAL_LOGITS, dtype=logits.dtype, device=logits.device
        )
        return decision, logits_out


@register_iter_decider
@capture_init_args
class AlwaysIterDecider(IterDecider):
    """Continue valid tokens until the configured fixed depth."""

    def __init__(
        self,
        max_iter: int = 2,
        threshold: float = 0.5,
        memory_lean_fp32_reductions: bool = False,
    ):
        super().__init__(max_iter=max_iter, threshold=threshold)

    def forward(
        self,
        logits: torch.Tensor,
        iter_depth: int,
        active_valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.BoolTensor, torch.FloatTensor]:
        if active_valid_mask is not None and logits.dim() == 3:
            logits = logits[active_valid_mask == 1]
        should_continue = iter_depth < self.max_iter
        decision = torch.full(
            logits.shape[:-1], should_continue, dtype=torch.bool, device=logits.device
        )
        logits_out = torch.full(
            decision.shape,
            POSITIVE_INFINITY_LOGITS if should_continue else MINUS_INFINITY_LOGITS,
            dtype=logits.dtype,
            device=logits.device,
        )
        return decision, logits_out


@register_iter_decider
@capture_init_args
class Qwen3MLPIterDecider(IterDecider):
    """
    Lightweight FFN-based iteration decider using Qwen3's SwiGLU MLP.
    Input: first-layer (input embedding) + last-layer hidden states + top-k logits → fuse → FFN → score.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 2048,
        topk: int = 2048,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        threshold: float = 0.5,
        max_iter: int = 2,
        dtype: torch.dtype = torch.bfloat16,
        uniform_init: bool = False,
        stochastic_sampling: bool = False,
        base_warmup_steps: int = 0,
        memory_lean_fp32_reductions: bool = False,
        **kwargs,
    ):
        super().__init__(threshold=threshold, max_iter=max_iter)
        self.topk = topk
        self.memory_lean_fp32_reductions = bool(memory_lean_fp32_reductions)
        # If True, sample the continue decision from Bernoulli(sigmoid(logit)) during
        # training instead of applying a hard threshold.
        self.stochastic_sampling = bool(stochastic_sampling)
        self.base_warmup_steps = int(base_warmup_steps)
        self._current_step: int = 0

        self.norm_first = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_last = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_l = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_proj = nn.Linear(
            hidden_size * 3, hidden_size, bias=False, dtype=dtype
        )

        mlp_cfg = SimpleNamespace(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        self.mlp = Qwen3MLP(mlp_cfg)
        self.norm_out = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.score = nn.Linear(hidden_size, 1, bias=False, dtype=dtype)

        if uniform_init:
            # sigmoid(logit) ~ Uniform(0,1): need logit ~ Logistic(0,1) with std = pi/sqrt(3).
            # After norm_out each component has unit variance => std_w = pi / sqrt(3 * hidden_size).
            self.score.weight.data.normal_(
                mean=0.0, std=math.pi / math.sqrt(3 * hidden_size)
            )
        else:
            self.score.weight.data.normal_(mean=0.0, std=0.01)
        self.to(dtype)

    def forward(
        self,
        logits: torch.Tensor,
        iter_depth: int,
        all_hidden_states: Optional[torch.Tensor] = None,
        active_valid_mask: Optional[torch.Tensor] = None,
        first_iter_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.BoolTensor, torch.FloatTensor]:
        if active_valid_mask is not None and logits.dim() == 3:
            logits = logits[active_valid_mask == 1]
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states[active_valid_mask == 1]
            if first_iter_embeds is not None and first_iter_embeds.dim() == 3:
                first_iter_embeds = first_iter_embeds[active_valid_mask == 1]

        N = logits.shape[0]
        if iter_depth >= self.max_iter:
            return (
                torch.zeros(N, dtype=torch.bool, device=logits.device),
                torch.full(
                    (N,),
                    MINUS_INFINITY_LOGITS,
                    dtype=logits.dtype,
                    device=logits.device,
                ),
            )

        if all_hidden_states is None:
            raise ValueError("Qwen3MLPIterDecider requires all_hidden_states.")

        # Use first-iteration input embeddings as the anchor; fall back to current-iteration
        # embedding layer output (all_hidden_states[..., 0, :]) when not provided.
        first_embed = (
            first_iter_embeds
            if first_iter_embeds is not None
            else all_hidden_states[..., 0, :]
        )
        h = torch.cat(
            [
                self.norm_first(first_embed),
                self.norm_last(all_hidden_states[..., -1, :]),
            ],
            dim=-1,
        )

        vocab_parallel_ops = getattr(self, "vocab_parallel_ops", None)
        if vocab_parallel_ops is not None:
            # logits is a vocab shard; top-k over the GLOBAL softmax (values
            # only). Stays fp32, matching the unsharded branch below where
            # autocast promotes torch.softmax to fp32.
            k = min(self.topk, vocab_parallel_ops.vocab_size)
            topk_probs = vocab_parallel_ops.softmax_topk_values(logits, k)
        elif self.memory_lean_fp32_reductions:
            k = min(self.topk, logits.size(-1))
            topk_logits, _ = torch.topk(logits, k=k, dim=-1)
            topk_probs = torch.exp(
                topk_logits.float() - fp32_logsumexp(logits).unsqueeze(-1)
            ).to(logits.dtype)
        else:
            k = min(self.topk, logits.size(-1))
            topk_probs, _ = torch.topk(torch.softmax(logits, dim=-1), k=k, dim=-1)
        if topk_probs.size(-1) < self.topk:
            topk_probs = F.pad(topk_probs, (0, self.topk - topk_probs.size(-1)))
        l = self.norm_l(topk_probs)

        x = self.input_proj(torch.cat([h, l], dim=-1))
        x = self.norm_out(self.mlp(x))

        decision_logits = self.score(x).squeeze(-1)
        prob = torch.sigmoid(decision_logits.float())
        if (
            self.training
            and self.base_warmup_steps > 0
            and self._current_step < self.base_warmup_steps
        ):
            # Linear threshold ramp 0 -> self.threshold over base_warmup_steps.
            # iter_depth >= max_iter is already early-exited above.
            ramp = float(self._current_step) / float(self.base_warmup_steps)
            effective_thr = self.threshold * ramp
            decision_mask = prob > effective_thr
        elif self.stochastic_sampling:
            decision_mask = torch.bernoulli(prob).bool()
        else:
            decision_mask = prob > self.threshold
        return decision_mask, decision_logits

    def update_training_state(
        self, current_step: int = 0, current_epoch: int = 0, **kwargs
    ) -> None:
        """Called once per training step from `sync_training_state`."""
        self._current_step = int(current_step)
