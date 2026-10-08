from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from tah2.utils.component_registry import (
    get_input_updater_class as get_input_updater_class,
    register_input_updater,
    capture_init_args,
)


class InputUpdater(nn.Module, ABC):
    """Build the next iteration's inputs from original embeddings and hidden states."""

    @abstractmethod
    def forward(self, prev_inputs: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@register_input_updater
@capture_init_args
class IdentityUpdater(InputUpdater):
    """Parameter-free placeholder for single-iteration standard training."""

    def __init__(self):
        super().__init__()

    def forward(self, prev_inputs: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return prev_inputs


@register_input_updater
@capture_init_args
class Qwen3MLPUpdater(InputUpdater):
    """
    Concat prev_inputs and last-layer hidden states, project to hidden_size,
    then pass through a Qwen3 MLP sub-block to produce the iter>=1 input::

        # custom 2-stream merging (no Qwen3 analog — needed because we fuse
        # the previous iter input and the iter0 hidden state into one stream):
        prev_normed = norm_prev(prev_inputs)               # like input_layernorm
        last_hidden = norm_last(hidden[-1])
        projected   = projection(cat(prev_normed, last_hidden))

        # Qwen3 MLP sub-block (post_attention_layernorm + mlp):
        update = mlp(norm_pre_mlp(projected))
        out    = norm_out(update)                          # iter>=1 input

    The updater's output IS the iter>=1 input directly — there is no
    residual carrier here, so a post-MLP ``norm_out`` is needed to fix the
    RMS to roughly what ``embed_tokens`` produces at iter=0, since layer 0's
    attention reads it raw.

    Norms:

      * ``norm_pre_mlp`` plays the role of ``post_attention_layernorm`` from
        a Qwen3 decoder block and is the core part of the sub-block;
        defaults to ``True``.
      * ``norm_out`` sets the magnitude of the iter>=1 input; off by default
        and only useful when downstream consumers assume unit-RMS embeddings.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        enable_norm_pre_mlp: bool = True,
        enable_norm_out: bool = True,
    ):
        super().__init__()
        self.projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        mlp_config = SimpleNamespace(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        self.mlp = Qwen3MLP(mlp_config)
        self.norm_prev = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_last = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_pre_mlp = (
            Qwen3RMSNorm(hidden_size, eps=rms_norm_eps) if enable_norm_pre_mlp else nn.Identity()
        )
        self.norm_out = (
            Qwen3RMSNorm(hidden_size, eps=rms_norm_eps) if enable_norm_out else nn.Identity()
        )

    def forward(
        self,
        prev_inputs: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: (..., num_layers+1, hidden_size) -> last layer: (..., hidden_size)
        last_hidden = self.norm_last(hidden_states[..., -1, :])
        prev_normed = self.norm_prev(prev_inputs)
        # Concat normed prev_inputs and last hidden states along feature dim
        concat = torch.cat([prev_normed, last_hidden], dim=-1)  # (..., 2 * hidden_size)
        projected = self.projection(concat)                       # (..., hidden_size)
        projected = self.norm_pre_mlp(projected)                  # ↔ post_attention_layernorm
        update = self.mlp(projected)                              # (..., hidden_size)
        return self.norm_out(update)
