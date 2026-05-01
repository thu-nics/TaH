"""TaH: Selective Latent Iterations for Reasoning Language Models.

Wraps a Hugging Face causal LM (Qwen3, etc.) so that a learned subset of
tokens runs additional internal forward passes ("iterations") to refine the
prediction. The decision to iterate is per-token, made by:

* :class:`tah.model.iter_decider.IterLabelDecider` — step-1 SFT: continue iff
  the dataset's oracle ``iter_count_labels`` say so.
* :class:`tah.model.iter_decider.MLPIterDecider` — step-2 SFT + eval/serving:
  a small classifier over hidden states + top-k logits.

Each iteration writes its KV into a separate slot of the per-layer
:class:`tah.model.causal_cache.TaHCache` so that future iterations can see
prior ones (causally) without disturbing the iter-0 cache.

Single-implementation interfaces from the public TaH layout are inlined here:

* ``topk_softmax_input_update``  — was ``input_updater.TrivialUpdater``.
* ``additive_logits_update``     — was ``output_updater.AdditiveLogitsUpdater``.
* The dense max-merge of per-iteration ``iter_count_labels``
  — was ``iter_label.FixedIterLabelGenerator``.
* LoRA setup / per-iteration enable
  — was ``adapter.setup_adapter`` / ``configure_lora_for_iteration``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from tah.model.causal_cache import TaHCache
from tah.model.iter_decider import (
    ITER_DECIDER_BY_NAME,
    IterDecider,
    load_iter_decider,
    save_iter_decider,
)
from tah.model.loss import LOSS_BY_NAME, LossFunc
from tah.model.tah_config import TaHConfig

logger = logging.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Inlined helpers (single-implementation slots from public TaH).
# ────────────────────────────────────────────────────────────────────────────


def topk_softmax_input_update(
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Map per-token logits to a soft input embedding.

    Top-k softmax over ``logits`` (last dim is vocab), look up the
    corresponding embedding rows, return the probability-weighted sum. Last
    dim of ``logits`` is consumed; all leading dims are preserved.
    """
    k = min(topk, logits.size(-1))
    topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)
    topk_probs = torch.softmax(topk_values, dim=-1)
    topk_embeds = embedding_weight[topk_indices]  # (..., k, H)
    out = torch.sum(topk_probs.unsqueeze(-1) * topk_embeds, dim=-2)
    # Trainer eval runs under autocast which can promote softmax/sum to fp32;
    # the caller scatters the result into a bf16 buffer so cast back.
    return out.to(embedding_weight.dtype)


def additive_logits_update(
    logits: torch.Tensor,
    prev_logits: Optional[torch.Tensor],
) -> torch.Tensor:
    """Residual accumulation of output logits across iterations."""
    return logits if prev_logits is None else prev_logits + logits


# ────────────────────────────────────────────────────────────────────────────
# Active-token sparse <-> dense scatter helpers.
#
# Each iteration of the wrapper processes only those positions whose
# `current_iter_mask` is True. We pack them into a dense (B, max_active, …)
# block (right-padded), run the base model on that block, and scatter results
# back into the dense (B, T, …) buffers.
# ────────────────────────────────────────────────────────────────────────────


def gather_active(
    current_iter_mask: torch.BoolTensor,
    *tensors,
    pad_value=0,
) -> Tuple[torch.Tensor, ...]:
    """Vectorised stable-sort gather: pack active positions to the left.

    For each batch row, the True positions of ``current_iter_mask`` are
    gathered in order from each input tensor and right-padded to the
    per-batch maximum active length. Inputs may be ``None`` (passed
    through). Floating and long tensors get filled with ``pad_value`` at
    pad positions; the caller may overwrite (e.g. ``-100`` for labels).
    Returns ``(pad_mask, *gathered)`` where ``pad_mask`` is a
    ``(B, max_active)`` bool that's True at padding positions.
    """
    B, S = current_iter_mask.shape
    device = current_iter_mask.device
    SENTINEL = S
    base_idx = torch.arange(S, device=device).expand(B, S).masked_fill(~current_iter_mask, SENTINEL)
    sorted_idx, _ = torch.sort(base_idx, dim=1, stable=True)
    max_len = int(current_iter_mask.sum(1).max().item()) if current_iter_mask.any() else 0
    gather_idx = sorted_idx[:, :max_len]
    pad_mask = gather_idx.eq(SENTINEL)
    gather_idx_clamped = gather_idx.clamp(max=max(S - 1, 0))

    out = [pad_mask]
    for t in tensors:
        if t is None:
            out.append(None)
            continue
        if t.dim() == 2:  # (B, S)
            g = torch.gather(t, 1, gather_idx_clamped).masked_fill(pad_mask, pad_value)
        else:  # (B, S, ...)
            extra = t.shape[2:]
            g_idx = gather_idx_clamped.view(B, max_len, *([1] * len(extra))).expand(B, max_len, *extra)
            g = torch.gather(t, 1, g_idx).masked_fill(
                pad_mask.view(B, max_len, *([1] * len(extra))), pad_value,
            )
        out.append(g)
    return tuple(out)


def scatter_back(
    current_iter_mask: torch.BoolTensor,
    src: torch.Tensor,
    dest: torch.Tensor,
    *,
    in_place: bool = False,
    assignment_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """Scatter a per-batch right-padded ``src`` back into a dense ``dest``.

    ``src`` is shape ``(B, max_active, ...)`` with row ``b`` valid up to
    ``current_iter_mask[b].sum()``; the rest is padding. The valid prefix of
    each row is placed at the True positions of ``current_iter_mask[b]`` in
    ``dest`` (which has shape ``(B, S, ...)``).

    When ``assignment_mask`` is supplied, only positions where both
    ``current_iter_mask`` AND ``assignment_mask`` are True are updated; the
    rest of ``dest`` is preserved.

    Vectorised — no per-batch Python loop.
    """
    B, max_active = src.shape[:2]
    active_counts = current_iter_mask.sum(1)
    # Pad positions in ``src``: column k of row b is padding iff k >= active_counts[b].
    pad_mask = torch.arange(max_active, device=src.device).unsqueeze(0) >= active_counts.unsqueeze(1)
    valid_src = src[~pad_mask]  # (sum(current_iter_mask), ...) — same row-major order as the mask

    out = dest if in_place else dest.clone()
    if assignment_mask is None:
        out[current_iter_mask] = valid_src
        return out

    # Two-stage: first write all active values into a dense intermediate, then
    # copy only the positions enabled by the assignment_mask.
    intermediate = torch.zeros_like(out)
    intermediate[current_iter_mask] = valid_src
    final_mask = current_iter_mask & assignment_mask
    out[final_mask] = intermediate[final_mask]
    return out


# ────────────────────────────────────────────────────────────────────────────
# Output dataclass and config helpers.
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class TaHCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Adds per-token ``iter_count`` (and optional generated label tensor)."""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    iter_count: Optional[torch.LongTensor] = None
    iter_count_labels: Optional[torch.LongTensor] = None


# ────────────────────────────────────────────────────────────────────────────
# Config (de)serialisation. ``torch.dtype`` and bare ``type`` objects sneak
# into iter_decider_kwargs (e.g. dtype=torch.bfloat16); these helpers let us
# round-trip them through json without losing identity.
# ────────────────────────────────────────────────────────────────────────────

_DTYPE_BY_STR = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}


def _config_to_serialisable(obj):
    """Walk a config dict; encode ``torch.dtype`` and ``type`` as small dicts."""
    if isinstance(obj, dict):
        return {k: _config_to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_config_to_serialisable(v) for v in obj]
    if isinstance(obj, type):
        return {"__type__": True, "__module__": obj.__module__, "__name__": obj.__name__}
    if isinstance(obj, torch.dtype):
        return {"__dtype__": True, "__str__": str(obj)}
    return obj


def _config_from_serialisable(obj):
    """Inverse of :func:`_config_to_serialisable`."""
    import importlib
    if isinstance(obj, dict):
        if obj.get("__type__") is True:
            return getattr(importlib.import_module(obj["__module__"]), obj["__name__"])
        if obj.get("__dtype__") is True:
            return _DTYPE_BY_STR.get(obj["__str__"], torch.float32)
        return {k: _config_from_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_config_from_serialisable(v) for v in obj]
    return obj


def _resolve_attr_path(root, dotted: str):
    obj = root
    for seg in dotted.split("."):
        if not seg or seg == "self":
            continue
        obj = getattr(obj, seg)
    return obj


def _with_max_iter(kwargs: Optional[dict], max_iter: int) -> dict:
    """Return ``kwargs`` with ``max_iter`` filled in (existing values win)."""
    out = dict(kwargs or {})
    out.setdefault("max_iter", max_iter)
    return out


def _build_loss(name: str, kwargs: dict, max_iter: int) -> LossFunc:
    """Instantiate a loss class by name. Both shipped classes
    (NextTokenPredLoss, IterDeciderLoss) accept ``max_iter`` via kwargs."""
    return LOSS_BY_NAME[name](**_with_max_iter(kwargs, max_iter))


def _build_iter_decider(name: str, kwargs: dict, max_iter: int) -> IterDecider:
    """Instantiate an iter_decider class by name with ``max_iter`` filled in."""
    return ITER_DECIDER_BY_NAME[name](**_with_max_iter(kwargs, max_iter))


def _resolve_tah_config(checkpoint_dir: str, override: Optional[TaHConfig]) -> TaHConfig:
    """Pick the TaHConfig to use when loading a checkpoint.

    Order of precedence:
      1. ``checkpoint_dir/tah_config.json`` (filtered to known fields,
         and with serialised ``type`` / ``torch.dtype`` sentinels restored).
      2. Non-default fields of ``override`` overlay the saved config when both
         are present (fields with value ``None`` or ``{}`` are treated as "use
         saved").
      3. ``TaHConfig()`` if neither is available — emits a warning since the
         resulting model won't match any checkpoint shape.
    """
    cfg_path = os.path.join(checkpoint_dir, "tah_config.json")
    saved: Optional[TaHConfig] = None
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            raw = _config_from_serialisable(json.load(f))
        valid = {f.name for f in fields(TaHConfig)}
        saved = TaHConfig(**{k: v for k, v in raw.items() if k in valid})

    if override is not None and saved is not None:
        merged = asdict(saved)
        for k, v in asdict(override).items():
            if v is not None and v != {}:
                merged[k] = v
        return TaHConfig(**merged)
    if override is not None:
        return override
    if saved is not None:
        return saved
    logger.warning("No tah_config.json in %s and no tah_config given; using defaults", checkpoint_dir)
    return TaHConfig()


# ────────────────────────────────────────────────────────────────────────────
# The TaH wrapper.
# ────────────────────────────────────────────────────────────────────────────


class TaHForCausalLM(PreTrainedModel):
    """Selective-iteration wrapper around a HF causal LM.

    During each forward pass, every token starts at ``iter_depth = 0`` (the
    base model's regular forward). For tokens where ``iter_decider`` votes
    "continue", the wrapper builds a soft input embedding from the current
    logits (top-k softmax mix) and re-runs the base model with LoRA enabled.
    Up to ``max_iter`` such rounds are performed; finished tokens accumulate
    their final logits via additive residuals, in-flight tokens carry on.

    Notes:
      * The base model is wrapped with PEFT LoRA in ``__init__``. The adapter
        is enabled only at ``iter_depth >= 1``; iter-0 runs the base weights.
      * Per-iteration KV caches are kept in a single :class:`TaHCache` so that
        attention masks can be built that allow each iteration to see all
        prior iterations of the same positions plus the iter-0 prefix.
      * ``iter_attention_mode`` is fixed to "duo" (the only mode used in the
        canonical recipes); other modes from public TaH have been removed.
    """

    def __init__(self, base_model: PreTrainedModel, config: Optional[TaHConfig] = None):
        # SDPA is the only attention impl we exercise in the recurrent loop.
        base_model._supports_sdpa = True
        super().__init__(base_model.config)
        self.config = base_model.config
        self.supports_gradient_checkpointing = True

        if config is None:
            config = TaHConfig()
        self.tah_config = config

        # Validate embedding key early.
        try:
            _resolve_attr_path(base_model, config.embedding_key)
        except AttributeError as e:
            raise ValueError(f"embedding_key {config.embedding_key!r} not found in base model") from e
        self.embedding_key = config.embedding_key

        self.max_iter = config.max_iter
        self.input_topk = int(config.input_updater_kwargs.get("topk", 100))

        # Iter decider — kept as an nn.Module subclass since two impls are used.
        self.iter_decider = _build_iter_decider(
            config.iter_decider, config.iter_decider_kwargs, self.max_iter
        )

        # Optional eval-time decider override (always a path like "iter_decider"
        # in the canonical recipes; we resolve once during from_pretrained too).
        self.eval_iter_decider = self._resolve_eval_iter_decider(config)

        # Loss objects — one for train, one for eval. Step-1 uses the same
        # NextTokenPredLoss for both; step-2 uses IterDeciderLoss for train,
        # NextTokenPredLoss for eval.
        self.train_loss = _build_loss(config.train_loss, config.train_loss_kwargs, self.max_iter)
        self.eval_loss = (
            _build_loss(config.eval_loss, config.eval_loss_kwargs, self.max_iter)
            if config.eval_loss else self.train_loss
        )

        # Base model attaches AFTER PEFT wrap.
        self.simple_base_model = base_model
        self._setup_lora(config)

        # LoRA enabled-state cache, so we don't toggle every step needlessly.
        self._lora_enabled: Optional[bool] = None

    # ── lora ──────────────────────────────────────────────────────────────

    def _setup_lora(self, config: TaHConfig) -> None:
        """Wrap the base model with PEFT LoRA. Adapter is gated per-iteration."""
        if config.adapter != "lora":
            raise ValueError(f"adapter must be 'lora', got {config.adapter!r}")
        # base_grad / adapter_grad are TaH-specific knobs the upstream PEFT
        # LoraConfig doesn't accept; copy and pop before forwarding.
        peft_kwargs = dict(config.adapter_kwargs)
        base_grad = peft_kwargs.pop("base_grad", True)
        adapter_grad = peft_kwargs.pop("adapter_grad", True)
        self.simple_base_model = get_peft_model(self.simple_base_model, LoraConfig(**peft_kwargs))
        self._set_lora_grad_flags(base_grad, adapter_grad)

    def _set_lora_grad_flags(self, base_grad: bool, adapter_grad: bool) -> None:
        """Always reapplied: PEFT freezes all non-LoRA params in
        ``get_peft_model``, so even ``(True, True)`` needs us to re-enable the
        base. An earlier no-op early-return silently broke step-1 SFT."""
        for name, p in self.simple_base_model.base_model.named_parameters():
            p.requires_grad = adapter_grad if "lora" in name.lower() else base_grad

    def _set_lora_enabled(self, enabled: bool) -> None:
        if self._lora_enabled is enabled:
            return
        if enabled:
            self.simple_base_model.base_model.enable_adapter_layers()
        else:
            self.simple_base_model.base_model.disable_adapter_layers()
        self._lora_enabled = enabled

    def _resolve_eval_iter_decider(self, config: TaHConfig) -> IterDecider:
        name = config.eval_iter_decider
        if not name:
            return self.iter_decider
        if isinstance(name, str) and name.startswith("iter_decider"):
            return _resolve_attr_path(self, name)
        return _build_iter_decider(name, config.eval_iter_decider_kwargs, self.max_iter)

    # ── handles & device ──────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return self.simple_base_model.device

    @property
    def embed_tokens(self):
        # PEFT places the original model under .base_model.model.
        return _resolve_attr_path(self.simple_base_model.base_model.model, self.embedding_key)

    @property
    def _active_iter_decider(self) -> IterDecider:
        """Decider used at the current train/eval mode."""
        if self.training:
            return self.iter_decider
        return self.eval_iter_decider or self.iter_decider

    # ── small forward-pass helpers ────────────────────────────────────────

    @staticmethod
    def _stack_hidden_states(outputs, device: torch.device) -> Optional[torch.Tensor]:
        """Convert HF ``output_hidden_states`` (tuple of (B, T, H)) into ``(B, T, L, H)``."""
        hs = getattr(outputs, "hidden_states", None)
        if not hs:
            return None
        layer_stack = torch.stack([h.to(device=device) for h in hs], dim=0)  # (L, B, T, H)
        return layer_stack.permute(1, 2, 0, 3)  # (B, T, L, H)

    def _force_one_continuation(
        self,
        decision: torch.Tensor,
        active_labels_shifted: Optional[torch.Tensor],
        active_valid_mask: torch.Tensor,
        iter_depth: int,
    ) -> torch.Tensor:
        """If labels exist and the decider says "stop everywhere", force one
        labeled position to continue so the iter-decider keeps getting
        gradient. No-op when there's no label or no remaining iter budget.
        """
        if (
            active_labels_shifted is None
            or decision is None
            or decision.numel() == 0
            or iter_depth >= self.max_iter
        ):
            return decision
        label_mask = (active_labels_shifted != -100)[active_valid_mask == 1]
        if label_mask.any() and not decision[label_mask].any():
            candidates = torch.nonzero(label_mask, as_tuple=False).flatten()
            chosen = candidates[torch.randint(0, candidates.numel(), (1,), device=decision.device)]
            decision[chosen] = True
        return decision

    @staticmethod
    def _max_merge_iter_labels(
        full: torch.Tensor,
        active: torch.Tensor,
        current_iter_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Update the dense ``full`` iter-labels view with ``active`` proposals.

        ``-100`` positions in ``active`` are treated as 0 (no proposal); the
        merge takes the per-position max so each token's label monotonically
        accumulates across iterations.
        """
        proposal = torch.zeros_like(active)
        valid = active != -100
        proposal[valid] = active[valid]
        tmp = torch.zeros_like(full)
        scatter_back(current_iter_mask, src=proposal, dest=tmp, in_place=True)
        return torch.maximum(full, tmp)

    @staticmethod
    def _forward_kwargs(kwargs: dict) -> dict:
        """Filter caller-supplied forward kwargs down to those the loss / callback
        plumbing actually reads."""
        return {k: v for k, v in kwargs.items() if k in ("global_step", "num_items_in_batch")}

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TaHCache] = None,
        labels: Optional[torch.LongTensor] = None,
        iter_count_labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        new_sequence: bool = False,  # forwarded to oracle deciders if any
        **kwargs,
    ) -> TaHCausalLMOutputWithPast:
        """One TaH forward over a batch.

        Args:
            input_ids: ``(B, T)``
            attention_mask: ``(B, T_total)``; if longer than T, the last T
                entries are taken as the per-query mask.
            position_ids: optional ``(B, T)``; if absent, computed from the
                cumulative sum of ``attention_mask`` plus the cache prefix.
            labels: ``(B, T)`` for cross-entropy; ``-100`` is ignored.
            iter_count_labels: ``(B, T)`` oracle labels for the iter decider.
            use_cache: whether to return ``past_key_values`` for use in
                subsequent forwards (decoding).
            new_sequence: signals oracle-style deciders that the batch is fresh.

        Returns:
            :class:`TaHCausalLMOutputWithPast`.
        """
        # Public TaH disabled these unconditionally; we keep the contract.
        assert not kwargs.get("output_attentions"), "TaH does not support output_attentions"
        assert not kwargs.get("output_hidden_states"), "TaH does not support output_hidden_states"

        # Causal-LM shifted labels (next-token prediction).
        if labels is not None:
            labels_shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            labels_all_shifted = F.pad(input_ids.clone(), (0, 1), value=-100)[..., 1:].contiguous()
        else:
            labels_shifted = None
            labels_all_shifted = None

        B, T = input_ids.shape
        V = self.config.vocab_size
        device = input_ids.device

        input_embeds = self.embed_tokens(input_ids)  # (B, T, H)
        dtype = input_embeds.dtype

        cumulative_logits = torch.zeros(B, T, V, device=device, dtype=dtype)
        final_output_logits = torch.zeros(B, T, V, device=device, dtype=dtype)
        actual_iter_counts = torch.zeros(B, T, dtype=torch.long, device=device)

        cache = past_key_values if past_key_values is not None else TaHCache().to(device=device, dtype=dtype)

        if attention_mask is not None:
            valid_mask = attention_mask[:, -T:].clone().to(dtype=torch.long)
            assert valid_mask.shape == (B, T), f"attention_mask shape {attention_mask.shape} bad for T={T}"
        else:
            valid_mask = torch.ones_like(input_ids, dtype=torch.long)

        if position_ids is None:
            cache_iter0_valid = cache.get_valid_mask_upto_iter(layer_idx=0, upto_iter_idx=0, init_batch_size=B).to(device)
            position_ids = (
                torch.cumsum(torch.cat((cache_iter0_valid, valid_mask), dim=-1), dim=-1)[:, -T:] - 1
            ).clamp(min=0)
        else:
            position_ids = position_ids.clone()

        loss_func = self.train_loss if self.training else self.eval_loss
        loss_func.prepare_loss(B, T, device, dtype)

        # FixedIterLabelGenerator behaviour, inlined: the caller-supplied
        # `iter_count_labels` are the per-token oracle counts; we maintain a
        # dense max-merge view across iterations for analysis/loss.
        track_iter_labels = (iter_count_labels is not None) and (labels_shifted is not None)
        if track_iter_labels:
            full_iter_labels = torch.zeros(B, T, dtype=torch.long, device=device)
        else:
            full_iter_labels = None

        # Per-iteration loop ------------------------------------------------
        current_iter_mask = torch.ones_like(input_ids, dtype=torch.bool)
        finished_mask = torch.zeros_like(current_iter_mask, dtype=torch.bool)
        iter_depth = 0

        while iter_depth < self.max_iter and current_iter_mask.any():
            self._set_lora_enabled(iter_depth >= 1)

            # ── Phase 1: pack active positions + run the base model once.
            (
                pad_mask,
                active_input_embeds, active_cumulative_logits, active_position_ids,
                active_valid_mask, active_labels_shifted, active_iter_count_labels,
                active_labels_all_shifted,
            ) = gather_active(
                current_iter_mask,
                input_embeds, cumulative_logits, position_ids,
                valid_mask, labels_shifted, iter_count_labels, labels_all_shifted,
            )
            # Label-typed tensors need the loss ignore index at pad positions
            # rather than the default 0.
            if active_labels_shifted is not None:
                active_labels_shifted = active_labels_shifted.masked_fill(pad_mask, -100)
            if active_labels_all_shifted is not None:
                active_labels_all_shifted = active_labels_all_shifted.masked_fill(pad_mask, -100)

            if active_valid_mask.shape[1] == 0:
                break

            sdpa_attn_mask = self._build_attention_mask(
                active_position_ids, active_valid_mask, cache, iter_depth, dtype=dtype
            )
            cache.current_iter_depth = iter_depth
            cache.position_ids_to_cache = active_position_ids
            cache.valid_mask_to_cache = active_valid_mask
            outputs = self.simple_base_model(
                inputs_embeds=active_input_embeds,
                position_ids=active_position_ids,
                attention_mask=sdpa_attn_mask,
                past_key_values=cache,
                use_cache=True if iter_depth < self.max_iter - 1 else use_cache,
                output_hidden_states=True,
            )

            iter_depth += 1

            # ── Phase 2: residual-accumulate logits + scatter back to dense.
            updated_active_logits = additive_logits_update(
                outputs.logits.to(device=device), active_cumulative_logits,
            )
            cumulative_logits = scatter_back(
                current_iter_mask, src=updated_active_logits, dest=cumulative_logits,
            )

            # ── Phase 3: ask the iter decider whether to keep iterating.
            all_hidden = self._stack_hidden_states(outputs, device)
            decider = self._active_iter_decider
            valid_active = active_valid_mask == 1
            decision, continue_logits = decider(
                logits=updated_active_logits[valid_active],
                iter_depth=iter_depth,
                all_hidden_states=all_hidden[valid_active] if all_hidden is not None else None,
                labels_shifted=active_labels_all_shifted[valid_active] if active_labels_all_shifted is not None else None,
                iter_count_labels=active_iter_count_labels[valid_active] if active_iter_count_labels is not None else None,
            )
            decision = self._force_one_continuation(decision, active_labels_shifted, active_valid_mask, iter_depth)
            if continue_logits is not None:
                continue_logits = continue_logits.to(device=device)

            # ── Phase 4: convert the per-active decision into per-token finished
            # / next-iter masks; bump iter counts for everyone we just processed.
            active_finished_mask = torch.ones_like(active_valid_mask, dtype=torch.bool)
            active_finished_mask[valid_active] = ~decision
            scatter_back(current_iter_mask, src=active_finished_mask, dest=finished_mask, in_place=True)
            actual_iter_counts[current_iter_mask] += 1

            # ── Phase 5: optional intra-iter loss accumulation (IterDeciderLoss).
            if labels_shifted is not None and loss_func._is_intra_iter_loss:
                loss_func.intra_iter_loss_func(
                    active_logits=updated_active_logits,
                    current_iter_mask=current_iter_mask,
                    active_labels_shifted=active_labels_shifted,
                    active_valid_continue_logits=continue_logits,
                    active_valid_mask=active_valid_mask,
                    iter_depth=iter_depth,
                    active_iter_count_labels=active_iter_count_labels,
                    iter_decider_threshold=decider.threshold,
                    model=self,
                    **self._forward_kwargs(kwargs),
                )

            # ── Phase 6: write tokens that just finalised into the output buffer.
            if active_finished_mask.any():
                scatter_back(
                    current_iter_mask, src=updated_active_logits, dest=final_output_logits,
                    in_place=True, assignment_mask=finished_mask,
                )

            # ── Phase 7: max-merge iter-count labels into the dense view.
            if track_iter_labels and active_iter_count_labels is not None:
                full_iter_labels = self._max_merge_iter_labels(
                    full_iter_labels, active_iter_count_labels, current_iter_mask,
                )

            # ── Phase 8: prepare next-iter inputs by feeding logits → embeddings.
            next_iter_mask = (~finished_mask) & current_iter_mask & (valid_mask == 1)
            if next_iter_mask.any():
                active_next = (~active_finished_mask) & valid_active
                # Clone before the in-place index_put: when base embeddings are
                # trainable (step-1 default), autograd has saved active_input_embeds
                # for backward through the simple_base_model call above; mutating
                # it in place would trip the saved-tensor version check.
                active_input_embeds = active_input_embeds.clone()
                active_input_embeds[active_next] = topk_softmax_input_update(
                    logits=updated_active_logits[active_next],
                    embedding_weight=self.embed_tokens.weight,
                    topk=self.input_topk,
                ).to(device=device)
                input_embeds = torch.zeros_like(input_embeds)
                scatter_back(
                    current_iter_mask, src=active_input_embeds, dest=input_embeds,
                    in_place=True, assignment_mask=next_iter_mask,
                )

            current_iter_mask = next_iter_mask
            if not current_iter_mask.any():
                break

        # Final cross-entropy / iter-decider loss.
        loss = None
        if labels_shifted is not None:
            loss_kwargs = self._forward_kwargs(kwargs)
            loss_kwargs["iter_count_labels"] = full_iter_labels if full_iter_labels is not None else iter_count_labels
            loss_kwargs["model"] = self
            if hasattr(self, "logger_callback"):
                loss_kwargs["logger_callback"] = self.logger_callback
            loss = loss_func.final_loss_func(
                logits=final_output_logits,
                labels_shifted=labels_shifted,
                iter_count=actual_iter_counts,
                training=self.training,
                **loss_kwargs,
            )
            # Optional avg-iter-count logging.
            if hasattr(self, "logger_callback"):
                num_items = kwargs.get("num_items_in_batch")
                with torch.no_grad():
                    if num_items is not None:
                        valid_iter = labels_shifted.detach() != -100
                        iter_sum = actual_iter_counts.detach()[valid_iter].float().sum()
                        self.logger_callback.avg_iter_count += float((iter_sum / num_items).item())
                    else:
                        self.logger_callback.avg_iter_count = float(actual_iter_counts.detach().float().mean().item())

        return TaHCausalLMOutputWithPast(
            loss=loss,
            logits=final_output_logits,
            past_key_values=cache if use_cache else None,
            iter_count=actual_iter_counts,
            iter_count_labels=full_iter_labels,
        )

    # ── attention mask ────────────────────────────────────────────────────

    def _build_attention_mask(
        self,
        active_position_ids: torch.Tensor,
        active_valid_mask: torch.LongTensor,
        cache: TaHCache,
        iter_depth: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Build the SDPA attention mask for one iteration (``"duo"`` mode).

        At depth ``i``, position ``p`` may attend to KV slot ``(cp, ci)``
        iff ``cp <= p`` AND ``ci <= i`` AND that KV slot is valid. Returns a
        ``(B, 1, query_len, total_kv)`` additive mask (0 = unmasked,
        ``min_dtype`` = masked) or ``None`` if there is nothing to attend to.
        """
        B, T = active_position_ids.shape
        device = active_position_ids.device

        if cache is not None and cache.has_layer(layer_idx=0):
            iter_index_cache = cache.get_cache_iter_index_upto_iter(layer_idx=0, upto_iter_idx=iter_depth)
            pos_cache = cache.get_position_id_upto_iter(layer_idx=0, upto_iter_idx=iter_depth, init_batch_size=B)
            valid_cache = cache.get_valid_mask_upto_iter(layer_idx=0, upto_iter_idx=iter_depth, init_batch_size=B)
            kv_cache_len = iter_index_cache.shape[-1]
        else:
            iter_index_cache = torch.empty((0,), device=device, dtype=torch.long)
            pos_cache = torch.empty((B, 0), device=device, dtype=torch.long)
            valid_cache = torch.empty((B, 0), device=device, dtype=torch.long)
            kv_cache_len = 0

        kv_total = kv_cache_len + T
        if kv_total == 0:
            return None
        min_dtype = torch.finfo(dtype).min

        kv_pos = torch.cat((pos_cache, active_position_ids), dim=-1)[:, None, :]                # (B, 1, kv_total)
        kv_valid = torch.cat((valid_cache, active_valid_mask), dim=-1)[:, None, :]              # (B, 1, kv_total)
        kv_iter = torch.cat(
            (iter_index_cache, torch.full((T,), iter_depth, dtype=torch.long, device=device)),
            dim=-1,
        )[None, None, :]                                                                          # (1, 1, kv_total)

        q_pos = active_position_ids[:, :, None]                                                  # (B, T, 1)
        q_iter = torch.full_like(q_pos, iter_depth)                                              # (B, T, 1)

        attn_bool = (kv_pos <= q_pos) & (kv_iter <= q_iter) & (kv_valid == 1)
        attn = torch.full((B, T, kv_total), min_dtype, device=device, dtype=dtype)
        attn[attn_bool] = 0.0
        return attn[:, None, :, :]

    # ── persistence ───────────────────────────────────────────────────────

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """Save base model + LoRA + iter_decider + ``tah_config.json``."""
        os.makedirs(save_directory, exist_ok=True)

        # LoRA adapter directory
        lora_dir = os.path.join(save_directory, "lora")
        os.makedirs(lora_dir, exist_ok=True)
        self.simple_base_model.save_pretrained(lora_dir, **kwargs)

        # Base model with cleaned keys (strip PEFT's `.base_layer` and `lora_*`).
        base_model = self.simple_base_model.base_model.model
        original_state_dict = base_model.state_dict

        def cleaned_state_dict():
            sd = original_state_dict()
            return {
                k.replace(".base_layer", ""): v
                for k, v in sd.items() if "lora" not in k.lower()
            }

        base_model.state_dict = cleaned_state_dict
        try:
            base_model.save_pretrained(save_directory, **kwargs)
        finally:
            base_model.state_dict = original_state_dict

        save_iter_decider(self.iter_decider, save_directory)

        # iter_decider_kwargs may contain torch.dtype / type objects (e.g.
        # ``dtype=torch.bfloat16``); stringify them before json.dump.
        with open(os.path.join(save_directory, "tah_config.json"), "w", encoding="utf-8") as f:
            json.dump(_config_to_serialisable(asdict(self.tah_config)), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        tah_config: Optional[TaHConfig] = None,
        **kwargs,
    ) -> "TaHForCausalLM":
        """Load a saved TaH model.

        Accepts either a local directory or a Hugging Face Hub repo id; in
        the latter case the snapshot is fetched (or its cached copy is
        located) so we can read ``tah_config.json`` and ``iter_decider.bin``
        from a real path on disk.

        Resolution order for the final TaH config:
          1. ``tah_config.json`` from the checkpoint, if present.
          2. Fields of the ``tah_config`` argument that are non-None override
             those values (lets callers tweak inference-time knobs).
          3. ``TaHConfig()`` defaults if nothing else is available.
        """
        if not os.path.isdir(pretrained_model_name_or_path):
            from huggingface_hub import snapshot_download
            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)

        final_cfg = _resolve_tah_config(pretrained_model_name_or_path, tah_config)

        # Pop a one-shot iter_decider override (the recipe form is
        # `iter_decider_kwargs.load_path: …`).
        iter_decider_path = None
        if "load_path" in (final_cfg.iter_decider_kwargs or {}):
            iter_decider_path = final_cfg.iter_decider_kwargs.pop("load_path")

        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model = cls(base_model, config=final_cfg)

        # Re-attach LoRA weights.
        adapter_path = os.path.join(pretrained_model_name_or_path, "lora")
        if os.path.isdir(adapter_path):
            model.simple_base_model.load_adapter(adapter_path, adapter_name="default")
            model._set_lora_grad_flags(
                base_grad=final_cfg.adapter_kwargs.get("base_grad", True),
                adapter_grad=final_cfg.adapter_kwargs.get("adapter_grad", True),
            )

        # Load iter_decider weights. ``load_iter_decider`` always returns a
        # CPU module; move it to the base model's device so the iter loop
        # doesn't trip on a device mismatch.
        if iter_decider_path is not None:
            model.iter_decider = load_iter_decider(
                iter_decider_path,
                class_name=final_cfg.iter_decider,
                init_args=final_cfg.iter_decider_kwargs,
            )
        elif os.path.isfile(os.path.join(pretrained_model_name_or_path, "iter_decider.bin")):
            model.iter_decider = load_iter_decider(
                pretrained_model_name_or_path,
                class_name=final_cfg.iter_decider,
                init_args=final_cfg.iter_decider_kwargs,
            )
        model.iter_decider = model.iter_decider.to(device=model.device, dtype=model.dtype)
        model.eval_iter_decider = model._resolve_eval_iter_decider(final_cfg)
        if model.eval_iter_decider is not model.iter_decider:
            model.eval_iter_decider = model.eval_iter_decider.to(device=model.device, dtype=model.dtype)
        return model
