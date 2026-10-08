"""TaH DUO and uniform Qwen3 inference.

Each token reuses its RoPE position while the updater feeds the previous hidden
state back into the transformer. Recurrent KV lives in separate depth tables:
DUO retains it across tokens; causal uniform releases it after each sample.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, Tuple

import torch
import torch.nn.functional as F

from tah2.minisgl.core import get_global_ctx
from tah2.minisgl.layers import (
    BaseOP,
    LinearReplicated,
    ParallelLMHead,
    RMSNorm,
    silu_and_mul,
)
from tah2.minisgl.utils import init_logger, nvtx_annotate, torch_dtype

from .base import BaseLLMModel
from .qwen3 import Qwen3Model

if TYPE_CHECKING:
    from .config import ModelConfig

logger = init_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# TaH Components
# ──────────────────────────────────────────────────────────────────────


class Qwen3MLPIterDecider(BaseOP):
    """Lightweight FFN-based decider adapted from the training reference.

    Input features:
      - normalized first-layer input embeddings ``(N, H)``
      - normalized last-layer hidden states ``(N, H)``
      - normalized top-k probability features ``(N, topk)``
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        topk: int,
        threshold: float = 0.5,
        rms_norm_eps: float = 1e-6,
    ):
        assert topk > 0, "topk must be > 0"
        self._threshold_value = float(threshold)
        self._threshold_tensor = torch.tensor(threshold, dtype=torch.float32)
        self._topk = int(topk)

        self.norm_first = RMSNorm(size=hidden_size, eps=rms_norm_eps)
        self.norm_last = RMSNorm(size=hidden_size, eps=rms_norm_eps)
        self.norm_l = RMSNorm(size=hidden_size, eps=rms_norm_eps)
        self.input_proj = LinearReplicated(hidden_size * 3, hidden_size, has_bias=False)
        self.gate_up_proj = LinearReplicated(hidden_size, intermediate_size * 2, has_bias=False)
        self.down_proj = LinearReplicated(intermediate_size, hidden_size, has_bias=False)
        self.norm_out = RMSNorm(size=hidden_size, eps=rms_norm_eps)
        self.score = LinearReplicated(hidden_size, 1, has_bias=False)
        self._init_parameters_if_materialized()

    def _init_parameters_if_materialized(self) -> None:
        if self.score.weight.is_meta:
            return
        with torch.no_grad():
            self.norm_first.weight.fill_(1.0)
            self.norm_last.weight.fill_(1.0)
            self.norm_l.weight.fill_(1.0)
            self.input_proj.weight.zero_()
            self.gate_up_proj.weight.zero_()
            self.down_proj.weight.zero_()
            self.norm_out.weight.fill_(1.0)
            self.score.weight.zero_()

    @property
    def _threshold(self) -> float:
        return self._threshold_value

    @_threshold.setter
    def _threshold(self, value: float) -> None:
        self._threshold_value = float(value)
        self._threshold_tensor.fill_(value)

    @nvtx_annotate("TaHQwen3MLPIterDecider")
    def forward(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        return_continue_probs: bool = False,
        **_: object,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("Qwen3MLPIterDecider requires hidden_states.")
        if input_embeds is None:
            input_embeds = hidden_states
        if hidden_states.shape[0] != logits.shape[0] or input_embeds.shape[0] != logits.shape[0]:
            raise ValueError(
                "Qwen3MLPIterDecider expects hidden_states/input_embeds/logits to share batch dim."
            )

        # top-k probs feeds the decider — matches the training-time feature.
        k = min(self._topk, logits.size(-1))
        topk_probs = torch.topk(torch.softmax(logits, dim=-1), k=k, dim=-1).values.to(
            dtype=hidden_states.dtype
        )
        if k < self._topk:
            topk_probs = F.pad(topk_probs, (0, self._topk - k))

        f = self.norm_first.forward(input_embeds)
        h = self.norm_last.forward(hidden_states)
        l = self.norm_l.forward(topk_probs)
        x = self.input_proj.forward(torch.cat([f, h, l], dim=-1))
        x = self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))
        x = self.norm_out.forward(x)
        # Reuse the sigmoid output for both the threshold check and the
        # stop_prob weighted-hidden aggregation downstream.
        continue_probs = torch.sigmoid(self.score.forward(x).squeeze(-1).float())
        needs_iter = continue_probs > self._threshold_tensor
        if return_continue_probs:
            return needs_iter, continue_probs
        return needs_iter


class TaHInputUpdater(BaseOP):
    """Qwen3MLPUpdater adapted for serving.

    Computation::

        prev  = rms_norm(prev_embeds)
        last  = rms_norm(hidden)
        x     = linear(concat(prev, last))   # (2H → H)
        x     = rms_norm(x)                  # if enable_norm_pre_mlp
        delta = silu_gated_mlp(x)             # H → 2I → I → H
        delta = rms_norm(delta)              # if enable_norm_out
        out   = delta

    ``norm_pre_mlp`` / ``norm_out`` mirror the training reference's optional Qwen3RMSNorms that
    align this block with the Qwen3 decoder idiom (post-projection norm
    before the MLP, output norm to keep iter>=1 embeddings at unit RMS).
    Both are gated by checkpoint flags; when disabled they are a no-op and
    their weight tensors are absent from the state-dict.

    Weight-compatible with the training reference's ``Qwen3MLPUpdater`` checkpoint
    (``input_updater.bin``).  The gate/up projections are stored as a
    merged ``gate_up_proj`` following mini-sglang conventions.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        enable_norm_pre_mlp: bool = False,
        enable_norm_out: bool = False,
    ):
        self.enable_norm_pre_mlp = enable_norm_pre_mlp
        self.enable_norm_out = enable_norm_out
        self.norm_prev = RMSNorm(size=hidden_size, eps=1e-6)
        self.norm_last = RMSNorm(size=hidden_size, eps=1e-6)
        self.projection = LinearReplicated(hidden_size * 2, hidden_size, has_bias=False)
        self.gate_up_proj = LinearReplicated(hidden_size, intermediate_size * 2, has_bias=False)
        self.down_proj = LinearReplicated(intermediate_size, hidden_size, has_bias=False)
        if enable_norm_pre_mlp:
            self.norm_pre_mlp = RMSNorm(size=hidden_size, eps=1e-6)
        if enable_norm_out:
            self.norm_out = RMSNorm(size=hidden_size, eps=1e-6)

    @nvtx_annotate("TaHUpdater")
    def forward(self, prev_embeds: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prev_embeds: ``(N, H)`` – embeddings from the previous iteration.
            hidden:      ``(N, H)`` – last-layer hidden state.
        Returns:
            ``(N, H)`` – new embeddings for the next iteration.
        """
        prev = self.norm_prev.forward(prev_embeds)
        last = self.norm_last.forward(hidden)
        concat = torch.cat([prev, last], dim=-1)  # (N, 2H)
        projected = self.projection.forward(concat)  # (N, H)
        if self.enable_norm_pre_mlp:
            projected = self.norm_pre_mlp.forward(projected)
        gate_up = self.gate_up_proj.forward(projected)  # (N, 2I)
        x = silu_and_mul(gate_up)  # (N, I)
        delta = self.down_proj.forward(x)  # (N, H)
        if self.enable_norm_out:
            delta = self.norm_out.forward(delta)
        return delta


# ──────────────────────────────────────────────────────────────────────
# TaH Model
# ──────────────────────────────────────────────────────────────────────


class TaHQwen3ForCausalLM(BaseLLMModel):
    """TaH-wrapped Qwen3 for mini-sglang serving.

    Exposes fine-grained methods so the engine can orchestrate the
    iteration loop externally and capture recurrent decode in CUDA graphs.

    Public API:
        forward()               – drop-in iter=0 forward, returns logits
        forward_with_hidden()   – iter=0, returns (hidden, logits, embeds)
        compute_iter_embeds()   – input updater
        load_tah_weights(dir)   – load tah_config / updater / iter_decider
    """

    def __init__(self, config: ModelConfig):
        # ── base Qwen3 ──
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        # ── TaH components ──
        self.tah_updater = None  # initialized from checkpoint shapes
        self.tah_decider = None  # loaded from the checkpoint before graph capture
        # Shared depth cap for prefill and decode; the decider only predicts continuation.
        self._tah_max_iter = 2
        self.weighted_hidden_method: str | None = None
        self.iter_attention_mode: str = "causal"  # "causal" | "duo"
        super().__init__()

    # ──────────── forward paths ────────────

    def forward(self) -> torch.Tensor:
        """Standard iter=0 forward.  Drop-in compatible with ``Qwen3ForCausalLM``."""
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(hidden)

    def forward_with_hidden(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """iter=0 forward that also returns intermediate state for TaH.

        Returns:
            ``(hidden, logits, embeds)`` where

            - *hidden*  ``(T, H)``  – last-layer hidden state
            - *logits*  ``(T, V)``  (decode) or ``(bs, V)``  (prefill)
            - *embeds*  ``(T, H)``  – input embeddings (for ``input_updater``)
        """
        batch = get_global_ctx().batch
        embeds = self.model.embed_tokens.forward(batch.input_ids)
        # The first decoder layer's fused_add_rmsnorm aliases ``embeds`` as the
        # residual stream and mutates it in place through every later layer.
        # ``input_updater`` and the decider both need the *original* lookup, so
        # clone the input fed to the layers and keep ``embeds`` intact for the
        # caller.
        hidden = self._layers_forward(embeds.clone())
        logits = self.lm_head.forward(hidden)
        return hidden, logits, embeds

    def forward_from_embeds_with_hidden(
        self, embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward from embeddings, returning both hidden state and logits.

        Used by the multi-iteration loop so the next iteration can
        compute new embeddings from the latest hidden state.

        Returns:
            ``(hidden, logits)``
        """
        hidden = self._layers_forward(embeds)
        logits = self.lm_head.forward(hidden)
        return hidden, logits

    def _layers_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run transformer layers + final norm, bypassing ``embed_tokens``."""
        residual: torch.Tensor | None = None
        for layer in self.model.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.model.norm.forward(x, residual)[0]

    # ──────────── TaH iteration helpers ────────────

    def compute_iter_embeds(self, prev_embeds: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Compute new embeddings for iter>=1 via input updater."""
        return self.tah_updater.forward(prev_embeds, hidden)

    @property
    def tah_max_iter(self) -> int:
        return self._tah_max_iter

    # ──────────── weight loading ────────────

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        """Load base Qwen3 weights only.

        TaH-specific weights (``input_updater.bin``) are loaded via
        :meth:`load_tah_weights` after this call.
        """
        _pfx = _join_prefix

        # base model + lm_head
        self.model.load_state_dict(state_dict, prefix=_pfx(prefix, "model"), _internal=True)
        self.lm_head.load_state_dict(state_dict, prefix=_pfx(prefix, "lm_head"), _internal=True)

        # tah_updater / tah_decider loaded separately → skip here
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def load_tah_weights(self, checkpoint_dir: str, device: torch.device) -> None:
        """Load TaH-specific artefacts from *checkpoint_dir*.

        Reads:
            - ``tah_config.json``     → max_iter, decider threshold config
            - ``input_updater.bin``   → updater weights
            - ``iter_decider.bin``    → trained gate or fixed-depth decider
        """
        # ── config ──
        tah_cfg: Dict[str, object] = {}
        config_path = os.path.join(checkpoint_dir, "tah_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                tah_cfg = json.load(f)
            self._tah_max_iter = tah_cfg.get("max_iter", 2)
            weighted_hidden_method = tah_cfg.get("weighted_hidden_method")
            if weighted_hidden_method not in ("stop_prob_mix", "even_mix"):
                raise ValueError(
                    f"Unsupported TaH weighted_hidden_method: {weighted_hidden_method}"
                )
            self.weighted_hidden_method = weighted_hidden_method
            iter_attention_mode = tah_cfg.get("iter_attention_mode", "causal")
            if iter_attention_mode not in ("causal", "duo"):
                raise ValueError(
                    f"Unsupported TaH iter_attention_mode: {iter_attention_mode}; "
                    "supported: 'causal', 'duo'"
                )
            self.iter_attention_mode = iter_attention_mode

        # ── updater weights ──
        updater_path = os.path.join(checkpoint_dir, "input_updater.bin")
        if os.path.exists(updater_path):
            data = torch.load(updater_path, map_location=device, weights_only=False)
            init_args = data.get("init_args", {}) or {}
            updater_class = str(data.get("class", "Qwen3MLPUpdater"))
        else:
            updater_class = None
        if updater_class != "Qwen3MLPUpdater":
            raise ValueError("TaH requires a Qwen3MLPUpdater checkpoint")
        updater_kwargs = (
            tah_cfg.get("input_updater_kwargs", {}) if isinstance(tah_cfg, dict) else {}
        )
        hidden_size = int(
            init_args.get(
                "hidden_size",
                updater_kwargs.get("hidden_size", self.model.embed_tokens.weight.shape[1]),
            )
        )
        intermediate_size = int(
            init_args.get(
                "intermediate_size",
                updater_kwargs.get(
                    "intermediate_size", self.model.layers.op_list[0].mlp.down_proj.weight.shape[1]
                ),
            )
        )
        enable_norm_pre_mlp = bool(
            init_args.get(
                "enable_norm_pre_mlp",
                updater_kwargs.get("enable_norm_pre_mlp", True),
            )
        )
        enable_norm_out = bool(
            init_args.get(
                "enable_norm_out",
                updater_kwargs.get("enable_norm_out", True),
            )
        )
        with torch.device(str(device)), torch_dtype(self.model.embed_tokens.weight.dtype):
            self.tah_updater = TaHInputUpdater(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                enable_norm_pre_mlp=enable_norm_pre_mlp,
                enable_norm_out=enable_norm_out,
            )

        raw_sd = data.get("state_dict", {})
        converted = _convert_updater_state_dict(raw_sd)
        converted = _cast_state_dict_like(converted, self.tah_updater.state_dict())
        self.tah_updater.load_state_dict(converted)
        logger.info(
            "Loaded input_updater.bin as Qwen3MLPUpdater(hidden_size=%d, intermediate_size=%d, "
            "enable_norm_pre_mlp=%s, enable_norm_out=%s)",
            hidden_size,
            intermediate_size,
            enable_norm_pre_mlp,
            enable_norm_out,
        )

        # ── optional iter decider weights ──
        decider_path = os.path.join(checkpoint_dir, "iter_decider.bin")
        if os.path.exists(decider_path):
            data = torch.load(decider_path, map_location=device, weights_only=False)
            decider_class = str(data.get("class", ""))
            if decider_class == "AlwaysIterDecider":
                self.tah_decider = AlwaysIterDecider()
            elif decider_class != "Qwen3MLPIterDecider":
                raise ValueError(f"Unsupported TaH decider: {decider_class}")
            else:
                init_args = data.get("init_args", {}) or {}
                hidden_size = int(
                    init_args.get("hidden_size", self.model.embed_tokens.weight.shape[1])
                )
                intermediate_size = int(init_args.get("intermediate_size", hidden_size))
                topk = int(init_args.get("topk", hidden_size))
                rms_norm_eps = float(init_args.get("rms_norm_eps", 1e-6))
                threshold = float(init_args.get("threshold", 0.5))

                with torch.device(str(device)), torch_dtype(self.model.embed_tokens.weight.dtype):
                    self.tah_decider = Qwen3MLPIterDecider(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        topk=topk,
                        threshold=threshold,
                        rms_norm_eps=rms_norm_eps,
                    )

                raw_sd = data.get("state_dict", {})
                converted = _convert_qwen3_mlp_decider_state_dict(raw_sd)
                converted = _cast_state_dict_like(converted, self.tah_decider.state_dict())
                self.tah_decider.load_state_dict(converted)
                logger.info(
                    "Loaded iter_decider.bin as Qwen3MLPIterDecider(topk=%d, threshold=%.6f)",
                    self.tah_decider._topk,
                    self.tah_decider._threshold,
                )

        if self.tah_decider is None:
            raise FileNotFoundError(f"Missing iter_decider.bin in {checkpoint_dir}")


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _convert_updater_state_dict(
    raw: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert the training reference ``Qwen3MLPUpdater`` state-dict → mini-sglang layout.

    Key changes:
        - ``mlp.gate_proj.weight`` + ``mlp.up_proj.weight`` → ``gate_up_proj.weight``
        - ``mlp.down_proj.weight`` → ``down_proj.weight``
    """
    converted: Dict[str, torch.Tensor] = {}

    # direct mappings (unchanged key names). norm_pre_mlp / norm_out are only present
    # when the training reference checkpoint was trained with the corresponding enable_* flag.
    for key in (
        "norm_prev.weight",
        "norm_last.weight",
        "projection.weight",
        "norm_pre_mlp.weight",
        "norm_out.weight",
    ):
        if key in raw:
            converted[key] = raw[key]

    # mlp.down_proj → down_proj
    if "mlp.down_proj.weight" in raw:
        converted["down_proj.weight"] = raw["mlp.down_proj.weight"]

    # merge gate + up → gate_up  (gate first, then up – matches mini-sglang convention)
    if "mlp.gate_proj.weight" in raw and "mlp.up_proj.weight" in raw:
        converted["gate_up_proj.weight"] = torch.cat(
            [raw["mlp.gate_proj.weight"], raw["mlp.up_proj.weight"]], dim=0
        )

    return converted


def _convert_qwen3_mlp_decider_state_dict(
    raw: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert the training reference ``Qwen3MLPIterDecider`` state-dict → mini-sglang decider layout."""
    converted: Dict[str, torch.Tensor] = {}

    # the training reference Qwen3MLPIterDecider keys.
    if "norm_first.weight" in raw:
        converted["norm_first.weight"] = raw["norm_first.weight"]
    if "norm_last.weight" in raw:
        converted["norm_last.weight"] = raw["norm_last.weight"]
    if "norm_l.weight" in raw:
        converted["norm_l.weight"] = raw["norm_l.weight"]
    if "input_proj.weight" in raw:
        converted["input_proj.weight"] = raw["input_proj.weight"]
    if "mlp.down_proj.weight" in raw:
        converted["down_proj.weight"] = raw["mlp.down_proj.weight"]
    if "norm_out.weight" in raw:
        converted["norm_out.weight"] = raw["norm_out.weight"]
    if "score.weight" in raw:
        converted["score.weight"] = raw["score.weight"]

    if "mlp.gate_proj.weight" in raw and "mlp.up_proj.weight" in raw:
        converted["gate_up_proj.weight"] = torch.cat(
            [raw["mlp.gate_proj.weight"], raw["mlp.up_proj.weight"]], dim=0
        )

    return converted


def _cast_state_dict_like(
    state_dict: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Cast input tensors to target tensor dtype/device by key."""
    casted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key not in target:
            continue
        ref = target[key]
        casted[key] = value.to(device=ref.device, dtype=ref.dtype)
    return casted


__all__ = [
    "TaHQwen3ForCausalLM",
    "Qwen3MLPIterDecider",
]


class AlwaysIterDecider(BaseOP):
    """Fixed-depth uniform recurrence, with the same graph signature as the MLP gate."""

    def forward(self, logits, return_continue_probs=False, **kwargs):
        needs = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        probs = torch.ones(logits.shape[0], dtype=torch.float32, device=logits.device)
        return (needs, probs) if return_continue_probs else needs
