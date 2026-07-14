"""Persistent configuration for ``TaHForCausalLM``.

Public TaH carried a config field per pluggable component (input_updater,
output_updater, iter_label_generator, iter_attention_mode). In tah-release
those slots have a single implementation each, inlined into the wrapper, so
their config fields are no longer load-bearing and are dropped here.

Old checkpoints whose ``tah_config.json`` still contains those keys load fine:
``TaHForCausalLM.from_pretrained`` filters the JSON to fields known by this
dataclass before instantiating it. Conversely, dropping the fields means new
saves don't carry inert names that suggest configurability where there is
none.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TaHConfig:
    # Wrapper-level settings ------------------------------------------------
    embedding_key: str = "model.embed_tokens"
    max_iter: Optional[int] = None

    # Iter decider — one of {"IterLabelDecider", "MLPIterDecider"}.
    iter_decider: Optional[str] = None
    iter_decider_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Optional alias used at eval/inference time. Either a class name (built
    # afresh) or an attribute path like "iter_decider" (alias of the trained
    # decider). When None, the trained decider is reused.
    eval_iter_decider: Optional[str] = None
    eval_iter_decider_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Input updater is fixed to top-k softmax over logits + embedding-row mix.
    # Only ``topk`` from this dict is read by the wrapper; the rest is kept as
    # a dict for forwards-compat with old saved configs.
    input_updater_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Adapter is fixed to LoRA. ``adapter_kwargs`` are forwarded to
    # ``peft.LoraConfig`` after popping the TaH-specific
    # ``base_grad`` / ``adapter_grad`` knobs.
    adapter: str = "lora"
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Losses — one of {"NextTokenPredLoss", "IterDeciderLoss"}.
    train_loss: Optional[str] = None
    train_loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_loss: Optional[str] = None
    eval_loss_kwargs: Dict[str, Any] = field(default_factory=dict)
