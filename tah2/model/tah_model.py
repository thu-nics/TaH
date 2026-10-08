"""Minimal TaH recurrent transformer implementation."""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import dispatch_model
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from tah2.model.causal_cache import TaHCache, _TaHCacheReadOnly
from tah2.model.input_updater import get_input_updater_class
from tah2.model.iter_decider import get_iter_decider_class
from tah2.model.iter_label import get_iter_label_generator_class
from tah2.model.loss import get_loss_func_class
from tah2.model.tah_config import TaHConfig
from tah2.model.tah_runtime import TaHForwardRuntime
from tah2.utils import sparse_ops
from tah2.utils.model_io import (
    load_input_updater,
    load_iter_decider,
    save_input_updater,
    save_iter_decider,
)
from tah2.utils.modeling import (
    dict_string_to_type,
    get_attr_recursive,
    get_device_map,
    type_to_dict_string,
)

logger = logging.get_logger(__name__)


def _tah_gc_layer_call(self, *args, **kwargs):
    """Replacement ``__call__`` for decoder layers under TaH gradient checkpointing.

    Unlike HF's ``GradientCheckpointingLayer.__call__``, this preserves
    ``past_key_values`` and ``use_cache`` in the kwargs and uses
    ``context_fn`` to set the TaH cache read-only during recomputation.
    """
    if self.gradient_checkpointing and self.training:
        # Remote models use either singular or plural cache kwargs.
        cache = kwargs.get("past_key_values", kwargs.get("past_key_value", None))

        def _ctx_fn():
            if cache is not None and hasattr(cache, "disable_update"):
                return contextlib.nullcontext(), _TaHCacheReadOnly(cache)
            return contextlib.nullcontext(), contextlib.nullcontext()

        return self._gradient_checkpointing_func(
            functools.partial(nn.Module.__call__, self, **kwargs),
            *args,
            context_fn=_ctx_fn,
        )
    return nn.Module.__call__(self, *args, **kwargs)


@dataclass
class TaHCausalLMOutputWithPast(CausalLMOutputWithPast):
    # Keep HF output contract, plus TaH-specific iteration statistics.
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    iter_count: Optional[torch.LongTensor] = None
    iter_count_labels: Optional[torch.LongTensor] = None
    # Optional evaluation metadata.
    first_iter_hidden_states: Optional[torch.FloatTensor] = (
        None  # (B, L, hidden_dim) hidden states from first iteration
    )
    decider_logits: Optional[List[torch.FloatTensor]] = (
        None  # per-depth dense (B, L) continue logits
    )
    iter_scores: Optional[List[torch.FloatTensor]] = (
        None  # per-depth posterior CE scores
    )


class TaHForCausalLM(PreTrainedModel):
    # HF's init-time attn-dispatch check reads these flags on both the
    # wrapper class and the base model instance.
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True

    def __init__(
        self, base_model: PreTrainedModel, config: Optional[TaHConfig] = None, **kwargs
    ):
        base_model._supports_sdpa = True
        base_model._supports_flash_attn_2 = True
        base_model._supports_flash_attn_3 = True
        super().__init__(base_model.config)
        self.config = base_model.config
        self.supports_gradient_checkpointing = True

        self.tah_config = config or TaHConfig()
        self.simple_base_model = base_model

        # Resolve embedding layer once; updater uses this path repeatedly.
        try:
            get_attr_recursive(base_model, self.tah_config.embedding_key)
        except AttributeError as e:
            raise ValueError(
                f"Embedding_key {self.tah_config.embedding_key} not found in base model"
            ) from e
        self.embedding_key = self.tah_config.embedding_key
        self.max_iter = self.tah_config.max_iter
        self.iter_attention_mode = self.tah_config.iter_attention_mode
        if self.iter_attention_mode not in ("causal", "duo", "same_iter"):
            raise ValueError("iter_attention_mode must be 'causal', 'duo', or 'same_iter'")
        self.iter_attention_impl = getattr(
            self.tah_config, "iter_attention_impl", "sdpa"
        )
        self.weighted_hidden_method = self.tah_config.weighted_hidden_method
        if self.weighted_hidden_method not in ("stop_prob_mix", "even_mix"):
            raise ValueError("weighted_hidden_method must be 'stop_prob_mix' or 'even_mix'")
        self.memory_lean_fp32_reductions = bool(
            getattr(self.tah_config, "memory_lean_fp32_reductions", False)
        )

        # Attention routing.  Hybrid modes switch per-iter in _process_sparse_iteration;
        # pure "triton" registers a custom "tah_sdpa" interface that reads the
        # sentinel-mask attributes built by sparse_ops.
        self._fa2_hybrid_enabled = self.iter_attention_impl in (
            "fa2_hybrid",
            "triton_hybrid",
        )
        self._triton_enabled = self.iter_attention_impl in ("triton", "triton_hybrid")

        if self._triton_enabled:
            from tah2.kernels.hf_adapter import register_tah_attention_impl

            register_tah_attention_impl("tah_sdpa")

        if self._fa2_hybrid_enabled:
            base_model.config._attn_implementation = "flash_attention_2"
        elif self._triton_enabled:
            base_model.config._attn_implementation = "tah_sdpa"

        # Build pluggable TaH components from config.
        self.iter_decider = self._build_iter_decider(
            self.tah_config.iter_decider,
            self.tah_config.iter_decider_kwargs,
        )
        self.eval_iter_decider = self._resolve_eval_iter_decider(
            self.tah_config.eval_iter_decider,
            getattr(self.tah_config, "eval_iter_decider_kwargs", {}) or {},
        )
        self.input_updater = self._build_input_updater(
            self.tah_config.input_updater,
            self.tah_config.input_updater_kwargs,
        )
        self.train_loss = self._build_loss(
            self.tah_config.train_loss, self.tah_config.train_loss_kwargs
        )
        self.eval_loss = (
            self._build_loss(
                self.tah_config.eval_loss, self.tah_config.eval_loss_kwargs
            )
            if self.tah_config.eval_loss
            else self.train_loss
        )

        self.iter_label_generator = None
        if self.tah_config.iter_label_generator:
            iter_label_generator_kwargs = dict(self.tah_config.iter_label_generator_kwargs)
            iter_label_generator_kwargs.setdefault("max_iter", self.max_iter)
            iter_label_generator_kwargs.setdefault(
                "memory_lean_fp32_reductions", self.memory_lean_fp32_reductions
            )
            self.iter_label_generator = get_iter_label_generator_class(
                self.tah_config.iter_label_generator
            )(**iter_label_generator_kwargs)

        self.components = [
            "input_updater",
            "iter_decider",
            "eval_iter_decider",
            "iter_label_generator",
            "train_loss",
            "eval_loss",
        ]

        # Optional: dispatch wrapper and components across devices.
        device_map = kwargs.pop("device_map", None)
        if device_map is not None:
            self._dispatch_with_device_map(device_map)

    def _get_decoder_layers(self):
        inner_model = getattr(self.simple_base_model, "model", self.simple_base_model)
        return getattr(inner_model, "layers", [])

    def _get_decoder_layers_or_raise(self):
        """Return decoder layers or fail before enabling HF checkpointing."""
        inner_model = getattr(self.simple_base_model, "model", self.simple_base_model)
        layers = getattr(inner_model, "layers", None)
        if not layers:
            raise RuntimeError(
                "TaH gradient checkpointing expects decoder layers at "
                "`simple_base_model.model.layers`, but none were found. "
                f"base_model={type(self.simple_base_model).__name__}, "
                f"inner_model={type(inner_model).__name__}. "
                "Refusing to enable gradient checkpointing because HF would "
                "otherwise fall back to disabling cache inputs."
            )
        return layers

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable per-layer gradient checkpointing on the base model, patching decoder
        layers so they preserve the TaH cache instead of stripping it (default HF
        behaviour) and use ``context_fn`` to set the cache to read-only during
        checkpoint recomputation."""
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        # Non-reentrant checkpointing preserves the recurrent cache context.
        gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        layers = self._get_decoder_layers_or_raise()

        # Remote models may omit HF's checkpointing flags.
        inner_model = getattr(self.simple_base_model, "model", self.simple_base_model)
        self.simple_base_model.supports_gradient_checkpointing = True
        if inner_model is not self.simple_base_model:
            inner_model.supports_gradient_checkpointing = True
        for layer in layers:
            if not hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = False

        if hasattr(self.simple_base_model, "gradient_checkpointing_enable"):
            self.simple_base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs
            )

        # Patch each decoder layer so its __call__ does NOT strip past_key_values /
        # use_cache.  The default ``GradientCheckpointingLayer.__call__`` forcibly sets
        # them to None/False which breaks TaH cross-iteration attention.
        #
        # Python resolves __call__ on the *class*, not the instance, so we create a
        # per-layer-type subclass with our override and swap __class__. The patched
        # class is shared per original class and keeps its name, so class-name-based
        # tooling (e.g. FSDP transformer_layer_cls_to_wrap) still resolves and
        # isinstance checks hold across all layers.
        patched_classes: Dict[type, type] = {}
        for layer in layers:
            if hasattr(layer, "_tah_original_class"):
                continue  # already patched
            orig_cls = type(layer)
            patched_cls = patched_classes.get(orig_cls)
            if patched_cls is None:
                patched_cls = type(
                    orig_cls.__name__,
                    (orig_cls,),
                    {"__call__": _tah_gc_layer_call},
                )
                patched_classes[orig_cls] = patched_cls
            layer._tah_original_class = orig_cls
            layer.__class__ = patched_cls

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing and restore original layer classes."""
        if hasattr(self.simple_base_model, "gradient_checkpointing_disable"):
            self.simple_base_model.gradient_checkpointing_disable()

        for layer in self._get_decoder_layers():
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = False
            if hasattr(layer, "_tah_original_class"):
                layer.__class__ = layer._tah_original_class
                del layer._tah_original_class

    def _build_iter_decider(
        self, class_name: str, kwargs_dict: Optional[Dict[str, Any]]
    ):
        decider_kwargs = dict(kwargs_dict or {})
        decider_kwargs["max_iter"] = self.max_iter
        decider_cls = get_iter_decider_class(class_name)
        init_params = inspect.signature(decider_cls.__init__).parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values()
        )
        if accepts_kwargs or "memory_lean_fp32_reductions" in init_params:
            decider_kwargs.setdefault(
                "memory_lean_fp32_reductions", self.memory_lean_fp32_reductions
            )
        return decider_cls(**decider_kwargs)

    def _build_input_updater(
        self, class_name: str, kwargs_dict: Optional[Dict[str, Any]]
    ):
        return get_input_updater_class(class_name)(**(kwargs_dict or {}))

    def _build_loss(self, class_name: str, kwargs_dict: Optional[Dict[str, Any]]):
        loss_kwargs = dict(kwargs_dict or {})
        loss_kwargs["max_iter"] = self.max_iter
        return get_loss_func_class(class_name)(**loss_kwargs)

    def _resolve_eval_iter_decider(
        self, spec: Any, eval_kwargs: Optional[Dict[str, Any]] = None
    ):
        # Allow using the train decider, a path reference, or a separate class.
        if isinstance(spec, (list, tuple)):
            for one_spec in spec:
                resolved = self._resolve_eval_iter_decider(one_spec, eval_kwargs)
                if resolved is not None:
                    return resolved
            return self.iter_decider
        if spec is None:
            return self.iter_decider
        if isinstance(spec, str):
            if spec.startswith("iter_decider") or spec.startswith("self."):
                return self._resolve_attr_path(spec)
            resolved_kwargs = dict(eval_kwargs or {})
            resolved_kwargs.setdefault("max_iter", self.max_iter)
            resolved_kwargs.setdefault(
                "memory_lean_fp32_reductions", self.memory_lean_fp32_reductions
            )
            return get_iter_decider_class(spec)(**resolved_kwargs)
        return spec

    def _resolve_attr_path(self, path: str):
        cur = self
        for seg in path.split("."):
            if not seg or seg == "self":
                continue
            cur = getattr(cur, seg)
        return cur

    def _dispatch_with_device_map(self, device_map):
        # Reuse project helper to keep wrapper/base-model mapping consistent.
        mapped = get_device_map(self, device_map, self.dtype)
        kwargs = {
            "device_map": mapped,
            "offload_dir": None,
            "offload_index": None,
            "offload_buffers": False,
            "skip_keys": self.simple_base_model._skip_keys_device_placement,
        }
        dispatch_model(self, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.simple_base_model.device

    @property
    def embed_tokens(self):
        return get_attr_recursive(self.simple_base_model, self.embedding_key)

    def _move_components(self, method_name: str, *args, **kwargs):
        self.simple_base_model = getattr(self.simple_base_model, method_name)(
            *args, **kwargs
        )
        for name in self.components:
            component = getattr(self, name, None)
            if component is not None and hasattr(component, method_name):
                setattr(self, name, getattr(component, method_name)(*args, **kwargs))
        return self

    def to(self, *args, **kwargs):
        return self._move_components("to", *args, **kwargs)

    def cuda(self, device=None):
        return self._move_components("cuda", device)

    def cpu(self):
        return self._move_components("cpu")

    @staticmethod
    def _shift_labels(labels: Optional[torch.LongTensor], input_ids: torch.LongTensor):
        # Causal LM alignment: token t predicts label at t+1.
        if labels is None:
            return None, None
        labels_shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
        labels_all_shifted = F.pad(input_ids.clone(), (0, 1), value=-100)[
            ..., 1:
        ].contiguous()
        return labels_shifted, labels_all_shifted

    def _init_position_ids(
        self,
        batch_size: int,
        query_len: int,
        valid_mask: torch.LongTensor,
        cache: TaHCache,
        position_ids: Optional[torch.LongTensor],
        device: torch.device,
    ):
        # Continue absolute positions after cached prefix if caller does not provide them.
        if position_ids is not None:
            return position_ids.clone()
        return torch.clamp(
            torch.cumsum(
                torch.cat(
                    (
                        cache.get_valid_mask_upto_iter(
                            layer_idx=0, upto_iter_idx=0, init_batch_size=batch_size
                        ).to(device),
                        valid_mask,
                    ),
                    dim=-1,
                ),
                dim=-1,
            )[:, -query_len:]
            - 1,
            min=0,
        )

    def _build_attention_mask_for_iter(
        self,
        iter_depth: int,
        active_position_ids: torch.LongTensor,
        active_valid_mask: torch.LongTensor,
        cache: TaHCache,
        dtype: torch.dtype,
        use_cache: bool,
    ):
        # Build iteration-aware mask and decide cache update behaviour for this depth.
        # When Triton is enabled we emit a 4-D sentinel that the HF adapter
        # consumes; otherwise a standard additive SDPA mask.
        mask_impl = "triton" if self._triton_enabled else "sdpa"
        attention_mask = sparse_ops.create_tah_sdpa_attention_mask(
            iter_attention_mode=self.iter_attention_mode,
            active_position_ids=active_position_ids,
            active_valid_mask=active_valid_mask,
            cache=cache,
            iter_depth=iter_depth,
            dtype=dtype,
            iter_attention_impl=mask_impl,
        )
        iter_use_cache = (iter_depth < self.max_iter - 1) or use_cache
        return attention_mask, iter_use_cache

    @staticmethod
    def _stack_hidden_states(model_outputs, device: torch.device):
        # All consumers read only hidden_states[..., -1, :]; return just the last
        # layer with a singleton layer dim so [..., -1, :] is a no-op.  Avoids
        # ~(n_layers-1) x B x T x H bf16 of stack + permute memcpy per iteration.
        hidden_states = getattr(model_outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) == 0:
            return None
        last = hidden_states[-1]
        if last is None:
            return None
        return last.to(device=device).unsqueeze(-2)

    @staticmethod
    def _apply_iter_count_override(
        active_valid_continue_decision: Optional[torch.Tensor],
        active_iter_count: Optional[torch.LongTensor],
        active_valid_mask: torch.LongTensor,
        iter_depth: int,
        device: torch.device,
        training: bool,
    ):
        # If explicit iter_count is provided, override decider outputs for replay.
        # This allows training to replay rollout trajectories.
        if active_iter_count is None or active_valid_continue_decision is None:
            return active_valid_continue_decision
        flat_counts = active_iter_count[active_valid_mask == 1]
        if flat_counts.numel() == 0:
            return active_valid_continue_decision
        flat_counts = flat_counts.to(device=device)
        override_mask = flat_counts > 0
        forced_decision = flat_counts > iter_depth
        if override_mask.all():
            return forced_decision
        return torch.where(
            override_mask, forced_decision, active_valid_continue_decision
        )

    def _build_final_loss_kwargs(
        self,
        kwargs: Dict[str, Any],
        finalized_iter_labels: Optional[torch.Tensor],
        iter_count_labels: Optional[torch.Tensor],
    ):
        # Assemble common kwargs for final loss while preserving external hooks.
        loss_kwargs = dict(kwargs)
        if finalized_iter_labels is not None:
            loss_kwargs["iter_count_labels"] = finalized_iter_labels
        elif iter_count_labels is not None:
            loss_kwargs["iter_count_labels"] = iter_count_labels
        if hasattr(self, "logger_callback"):
            loss_kwargs["logger_callback"] = self.logger_callback
        loss_kwargs["dp_group"] = getattr(self, "dp_group", None)
        loss_kwargs["model"] = self
        vocab_parallel_ops = getattr(self, "vocab_parallel_ops", None)
        if vocab_parallel_ops is not None:
            loss_kwargs["vocab_parallel_ops"] = vocab_parallel_ops
            # Under log-prob mix methods outputs.logits already holds global
            # log-probabilities (see TaHConfig.weighted_hidden_method).
            loss_kwargs["logits_are_log_probs"] = True
        return loss_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor,
        iter_count: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TaHCache] = None,
        labels: Optional[torch.LongTensor] = None,
        iter_count_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        return_eval_metadata: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # 1) Validate unsupported HF options for this wrapper.
        if (output_attentions is not None) and output_attentions:
            raise AssertionError("TaH does not support output_attentions")

        # 2) Prepare base tensors and caches.
        labels_shifted, labels_all_shifted = self._shift_labels(labels, input_ids)
        max_iterations = self.max_iter
        batch_size, query_len = input_ids.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        input_embeds = self.embed_tokens(input_ids)
        dtype, device = input_embeds.dtype, input_embeds.device
        # Keep a reference to the iteration-0 embeddings; input_embeds is reassigned each depth.
        first_iter_embeds = input_embeds
        actual_iter_counts = torch.zeros_like(input_ids, dtype=torch.long)

        cache = (
            past_key_values
            if past_key_values is not None
            else TaHCache().to(device=device, dtype=dtype)
        )

        if attention_mask is not None:
            valid_mask = attention_mask.clone()[:, -query_len:].to(dtype=torch.long)
            assert valid_mask.shape == (
                batch_size,
                query_len,
            ), f"attention_mask shape must be (batch_size, seq_len), but got {attention_mask.shape}"
        else:
            valid_mask = torch.ones_like(input_ids, dtype=torch.long)

        position_ids = self._init_position_ids(
            batch_size, query_len, valid_mask, cache, position_ids, device
        )

        # 3) Initialize loss state for this forward.
        loss_func = self.train_loss if self.training else self.eval_loss
        loss_func.prepare_loss(batch_size, query_len, device, dtype)
        runtime = TaHForwardRuntime(
            model=self,
            loss_func=loss_func,
            labels_shifted=labels_shifted,
            valid_mask=valid_mask,
            position_ids=position_ids,
            first_iter_embeds=first_iter_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_eval_metadata=return_eval_metadata,
            kwargs=kwargs,
            dtype=dtype,
            device=device,
        )

        if runtime.use_iter_labeling:
            self.iter_label_generator.prepare(batch_size, query_len, device, dtype)

        current_iter_mask = torch.ones_like(input_ids, dtype=torch.bool)
        finished_mask = torch.zeros_like(current_iter_mask, dtype=torch.bool)
        iter_depth = 0

        cur_iter_decider = (
            self.iter_decider
            if self.training
            else (self.eval_iter_decider or self.iter_decider)
        )

        # Cross-rank iteration sync: under parameter-sharded backends (FSDP)
        # every rank must execute the same number of backbone passes, otherwise
        # per-layer allgathers desync and NCCL hangs. Routing is data-dependent,
        # so ranks agree globally on whether anyone still iterates; a rank with
        # no active tokens runs a tiny dummy pass instead of skipping.
        sync_iter_ranks = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        dummy_zero = None

        # 4) Main recurrent loop over iteration depth.
        while iter_depth < max_iterations:
            local_active = bool(current_iter_mask.any())
            if sync_iter_ranks:
                active_flag = torch.tensor(
                    int(local_active), device=device, dtype=torch.long
                )
                torch.distributed.all_reduce(
                    active_flag,
                    op=torch.distributed.ReduceOp.MAX,
                    group=getattr(self, "tah_sync_group", None),
                )
                if active_flag.item() == 0:
                    break
                if not local_active:
                    zero = self._run_dummy_backbone_pass(
                        first_iter_embeds, position_ids
                    )
                    dummy_zero = zero if dummy_zero is None else dummy_zero + zero
                    iter_depth += 1
                    continue
            elif not local_active:
                break

            (
                active_input_embeds,
                active_position_ids,
                active_valid_mask,
                active_iter_count,
                active_labels_shifted,
                active_iter_count_labels,
                active_labels_all_shifted,
                active_remaining_mass,
                active_first_iter_embeds,
            ) = self.to_active(
                current_iter_mask,
                input_embeds,
                position_ids,
                valid_mask,
                iter_count,
                labels_shifted,
                iter_count_labels,
                labels_all_shifted,
                runtime.remaining_mass,
                first_iter_embeds,
            )

            if active_valid_mask.shape[1] == 0:
                # Fall to the loop top: with sync enabled this becomes a dummy
                # pass while other ranks iterate; otherwise the loop exits.
                current_iter_mask = torch.zeros_like(current_iter_mask)
                continue

            # Forward only active tokens at this depth.
            attn_mask, iter_use_cache = self._build_attention_mask_for_iter(
                iter_depth=iter_depth,
                active_position_ids=active_position_ids,
                active_valid_mask=active_valid_mask,
                cache=cache,
                dtype=dtype,
                use_cache=use_cache,
            )

            active_outputs = self._process_sparse_iteration(
                sparse_input=active_input_embeds,
                position_ids=active_position_ids,
                valid_mask=active_valid_mask,
                cache_position=None,
                attention_mask=attn_mask,
                iter_depth=iter_depth,
                past_key_values=cache,
                use_cache=iter_use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                model=self.simple_base_model,
                **kwargs,
            )

            iter_depth += 1
            active_outputs.logits = active_outputs.logits.to(device=device)
            all_hidden = self._stack_hidden_states(active_outputs, device)

            # Generate or forward iteration labels for decider supervision.
            if runtime.use_iter_labeling:
                active_iter_count_labels = self.iter_label_generator.intra_iter_labels(
                    active_iter_count_labels=active_iter_count_labels,
                    active_logits=active_outputs.logits,
                    active_labels_shifted=active_labels_shifted,
                    iter_depth=iter_depth,
                    current_iter_mask=current_iter_mask,
                    active_valid_mask=active_valid_mask,
                    **kwargs,
                )

            active_valid_continue_decision, active_valid_continue_logits = (
                cur_iter_decider(
                    logits=active_outputs.logits,
                    iter_depth=iter_depth,
                    all_hidden_states=all_hidden,
                    active_valid_mask=active_valid_mask,
                    first_iter_embeds=active_first_iter_embeds,
                    labels_shifted=(
                        active_labels_all_shifted[active_valid_mask == 1]
                        if active_labels_all_shifted is not None
                        else None
                    ),
                    iter_count_labels=(
                        active_iter_count_labels[active_valid_mask == 1]
                        if active_iter_count_labels is not None
                        else None
                    ),
                )
            )
            runtime.capture_step(
                iter_depth=iter_depth,
                current_iter_mask=current_iter_mask,
                active_outputs=active_outputs,
                active_valid_continue_logits=active_valid_continue_logits,
            )
            # Convert "continue" decision into finished mask for this depth.
            active_valid_continue_decision = self._apply_iter_count_override(
                active_valid_continue_decision=active_valid_continue_decision,
                active_iter_count=active_iter_count,
                active_valid_mask=active_valid_mask,
                iter_depth=iter_depth,
                device=device,
                training=self.training,
            )

            # Causal attention: prompt positions (labels_shifted == -100) are
            # never the target of any supervised next-token prediction, and
            # under a causal mask response tokens cannot benefit from a "future"
            # second-iter KV at a prompt position via supervision-driven
            # gradient. Iter ≥ 2 on prompt is therefore wasted compute. Force
            # stop after the first iter for those positions. (No-op when labels
            # are absent at pure inference.)
            if (
                self.iter_attention_mode == "causal"
                and active_valid_continue_decision is not None
                and active_labels_shifted is not None
            ):
                flat_prompt = active_labels_shifted[active_valid_mask == 1] == -100
                if flat_prompt.any():
                    active_valid_continue_decision = active_valid_continue_decision & (
                        ~flat_prompt
                    )

            # Keep one token active for aligned graphs; explicit labels take precedence.
            if (
                self.training
                and getattr(self, "_tah_require_nonempty_recurrent_graph", False)
                and active_iter_count is None
                and iter_depth < max_iterations
                and active_valid_continue_decision is not None
                and active_valid_continue_decision.numel() > 0
                and not bool(active_valid_continue_decision.any())
            ):
                keep_idx = active_valid_continue_logits.detach().argmax()
                active_valid_continue_decision = active_valid_continue_decision.clone()
                active_valid_continue_decision[keep_idx] = True

            active_finished_mask = torch.ones_like(active_valid_mask, dtype=torch.bool)
            active_finished_mask[active_valid_mask == 1] = (
                ~active_valid_continue_decision
            )

            self.assign_active(
                current_iter_mask, src=active_finished_mask, dest=finished_mask
            )
            # Equivalent to ``actual_iter_counts[current_iter_mask] += 1`` but
            # avoids the nonzero-based sync of boolean-index in-place add.
            actual_iter_counts.add_(
                current_iter_mask.to(dtype=actual_iter_counts.dtype)
            )

            runtime.capture_posterior_inputs(
                iter_depth=iter_depth,
                all_hidden=all_hidden,
                active_first_iter_embeds=active_first_iter_embeds,
                current_iter_mask=current_iter_mask,
                active_valid_mask=active_valid_mask,
                active_finished_mask=active_finished_mask,
                cache=cache,
            )

            active_continue_prob = runtime.accumulate_iter(
                current_iter_mask=current_iter_mask,
                active_valid_mask=active_valid_mask,
                active_finished_mask=active_finished_mask,
                active_labels_shifted=active_labels_shifted,
                active_outputs=active_outputs,
                all_hidden=all_hidden,
                active_remaining_mass=active_remaining_mass,
                active_valid_continue_logits=active_valid_continue_logits,
                active_valid_continue_decision=active_valid_continue_decision,
            )

            next_iter_mask = (~finished_mask) & current_iter_mask & (valid_mask == 1)
            if next_iter_mask.any():
                # Update embeddings only for tokens continuing to next depth.
                # Compute nonzero indices ONCE and reuse: bool-mask indexing
                # (``t[mask]``) calls ``nonzero`` + alloc each time, forcing a
                # CPU-GPU sync; the prior form triggered this four times.
                active_next_iter_mask = (~active_finished_mask) & (
                    active_valid_mask == 1
                )
                ani_idx = torch.nonzero(active_next_iter_mask, as_tuple=True)
                active_input_embeds = active_input_embeds.clone()
                active_input_embeds[ani_idx] = self.input_updater(
                    prev_inputs=active_first_iter_embeds[ani_idx],
                    hidden_states=all_hidden[ani_idx] if all_hidden is not None else None,
                ).to(device=device, dtype=active_input_embeds.dtype)

                input_embeds = torch.zeros_like(input_embeds)
                self.assign_active_with_mask(
                    current_iter_mask=current_iter_mask,
                    assignment_mask=next_iter_mask,
                    src=active_input_embeds,
                    dest=input_embeds,
                )

                runtime.update_remaining_mass(
                    current_iter_mask=current_iter_mask,
                    next_iter_mask=next_iter_mask,
                    active_remaining_mass=active_remaining_mass,
                    active_continue_prob=active_continue_prob,
                )

            # Loop-top handles exit (and cross-rank alignment when syncing).
            current_iter_mask = next_iter_mask

        # Finalize the accumulated mixture for the configured loss.
        final_output_logits = runtime.finalize_logits(actual_iter_counts=actual_iter_counts)

        # 5) Final loss and logging.
        loss = None
        finalized_iter_labels = runtime.finalize()
        # NB: do NOT clear the final context here. With gradient checkpointing the
        # backbone is re-run during backward (TaHCache.disable_update), and that
        # recompute must see the same [ctx ++ live] KV layout as the forward. The
        # context lives on the per-forward cache and is released when it is GC'd.
        if labels_shifted is not None:
            loss_kwargs = self._build_final_loss_kwargs(
                kwargs=kwargs,
                finalized_iter_labels=finalized_iter_labels,
                iter_count_labels=iter_count_labels,
            )
            loss = loss_func.final_loss_func(
                logits=final_output_logits,
                labels_shifted=labels_shifted,
                iter_count=actual_iter_counts,
                training=self.training,
                **loss_kwargs,
            )
            if hasattr(self, "logger_callback"):
                self.logger_callback.log_iter_metrics(
                    labels_shifted=labels_shifted,
                    actual_iter_counts=actual_iter_counts,
                    finalized_iter_labels=finalized_iter_labels,
                    device=device,
                    training=self.training,
                )

        if self.training and loss is not None:
            # Keep updater params in every rank's autograd graph even when this
            # rank has no token continuing to iter>=2 in the current step.
            for p in self.input_updater.parameters():
                if p.requires_grad:
                    loss = loss + p.reshape(-1)[0].float() * 0.0
            # 0-valued dummy-pass terms keep backward-side collectives aligned
            # across ranks under parameter-sharded backends.
            if dummy_zero is not None:
                loss = loss + dummy_zero

        return TaHCausalLMOutputWithPast(
            loss=loss,
            logits=final_output_logits,
            past_key_values=cache if use_cache else None,
            hidden_states=runtime.final_output_hidden if output_hidden_states else None,
            attentions=None,
            iter_count=actual_iter_counts,
            iter_count_labels=finalized_iter_labels,
            **runtime.output_fields(),
        )

    def _run_dummy_backbone_pass(
        self, ref_embeds: torch.Tensor, position_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Minimal backbone forward used purely for cross-rank collective
        alignment (see the iteration-loop sync note in ``forward``). Two tokens
        are used so attention dispatch takes the standard prefill path, no
        cache is touched, and the returned scalar is exactly 0 but attached to
        the autograd graph so backward collectives stay aligned as well."""
        inner = getattr(self.simple_base_model, "model", self.simple_base_model)
        seq = min(2, ref_embeds.shape[1])
        dummy = ref_embeds[:1, :seq].detach()
        pos = position_ids[:1, :seq] if position_ids is not None else None
        out = inner(inputs_embeds=dummy, position_ids=pos, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden.float().sum() * 0.0

    _gather_active_2d = staticmethod(sparse_ops.gather_active_2d)
    _gather_active_3d = staticmethod(sparse_ops.gather_active_3d)
    to_active = staticmethod(sparse_ops.to_active)
    assign_active = staticmethod(sparse_ops.assign_active)
    assign_active_with_mask = staticmethod(sparse_ops.assign_active_with_mask)
    add_active_with_mask = staticmethod(sparse_ops.add_active_with_mask)
    create_TaH_sdpa_attention_mask = staticmethod(
        sparse_ops.create_tah_sdpa_attention_mask
    )

    def _process_sparse_iteration(
        self,
        sparse_input: torch.Tensor,
        position_ids: torch.Tensor,
        valid_mask: torch.LongTensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor,
        iter_depth: int,
        past_key_values: Optional[TaHCache],
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        model: Optional[PreTrainedModel] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Single sparse forward pass through base model with TaH cache metadata.
        if past_key_values is not None:
            past_key_values.current_iter_depth = iter_depth
            past_key_values.position_ids_to_cache = position_ids
            past_key_values.valid_mask_to_cache = valid_mask

        # Hybrid modes pick attn_implementation by iter depth:
        #   iter=0 (2-D / None mask) -> FA2
        #   iter>=1 (4-D mask)       -> "sdpa" (fa2_hybrid) or "tah_sdpa" (triton_hybrid)
        # Pure "sdpa" / "triton" paths never enter this branch.
        cfg = getattr(model, "config", None)
        saved_impl = None
        if self._fa2_hybrid_enabled and cfg is not None:
            is_iter0_shape = attention_mask is None or attention_mask.dim() == 2
            if is_iter0_shape:
                target = "flash_attention_2"
            else:
                target = "tah_sdpa" if self._triton_enabled else "sdpa"
            cur = getattr(cfg, "_attn_implementation", None)
            if cur != target:
                saved_impl = cur
                cfg._attn_implementation = target
        try:
            return model(
                inputs_embeds=sparse_input,
                position_ids=position_ids,
                cache_position=cache_position,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )
        finally:
            if saved_impl is not None:
                cfg._attn_implementation = saved_impl

    def save_pretrained(self, save_directory, **kwargs):
        # Save base model + TaH side modules/config. Sharded callers pass
        # gathered plain state dicts for the side modules.
        decider_sd = kwargs.pop("decider_state_dict", None)
        updater_sd = kwargs.pop("updater_state_dict", None)
        os.makedirs(save_directory, exist_ok=True)
        self.simple_base_model.save_pretrained(save_directory, **kwargs)
        save_iter_decider(self.iter_decider, save_directory, state_dict=decider_sd)
        save_input_updater(self.input_updater, save_directory, state_dict=updater_sd)
        config_dict = type_to_dict_string(asdict(self.tah_config))
        with open(
            os.path.join(save_directory, "tah_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_saved_tah_config(
        pretrained_model_name_or_path: str,
    ) -> Optional[TaHConfig]:
        # Load TaH config if checkpoint contains it.
        config_path = os.path.join(pretrained_model_name_or_path, "tah_config.json")
        if not os.path.exists(config_path):
            return None
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg_dict = dict_string_to_type(cfg_dict)
        if cfg_dict.get("iter_label_generator") in ("DynamicIterLabelGenerator", "dynamiclabel"):
            label_kwargs = dict(cfg_dict.get("iter_label_generator_kwargs") or {})
            strategy = label_kwargs.pop("strategy", "posterior_ce")
            if strategy != "posterior_ce":
                raise ValueError(f"Unsupported legacy iteration label strategy: {strategy}")
            cfg_dict["iter_label_generator_kwargs"] = label_kwargs
        valid_fields = {f.name for f in TaHConfig.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        return TaHConfig(**cfg_dict)

    @staticmethod
    def _merge_tah_config(
        saved: Optional[TaHConfig], provided: Optional[TaHConfig]
    ) -> TaHConfig:
        # User-provided non-empty fields override saved config.
        if provided is None and saved is None:
            return TaHConfig()
        if provided is None:
            return saved
        if saved is None:
            return provided
        merged = asdict(saved)
        for k, v in asdict(provided).items():
            if (v is not None) and (v != {}):
                merged[k] = v
        return TaHConfig(**merged)

    @staticmethod
    def _load_iter_decider_weights(
        tah_model,
        pretrained_model_name_or_path: str,
        final_config: TaHConfig,
        saved_config: Optional[TaHConfig],
    ):
        iter_decider_kwargs = dict(
            getattr(final_config, "iter_decider_kwargs", {}) or {}
        )
        # max_iter is injected by _build_iter_decider, not stored in
        # iter_decider_kwargs — without it the reloaded decider falls back to
        # the class default (2) and force-stops every token at iter_depth >= 2
        # (Qwen3MLPIterDecider.forward early-exit), silently crippling any
        # reloaded max_iter>2 checkpoint.
        iter_decider_kwargs.setdefault("max_iter", tah_model.max_iter)
        load_path = iter_decider_kwargs.pop("load_path", None)

        if load_path is None and saved_config is not None:
            if getattr(saved_config, "iter_decider", None) != final_config.iter_decider:
                logger.info(
                    "Detected different iter_decider class, skip old iter_decider weight loading"
                )
                return

        tah_model.iter_decider = load_iter_decider(
            load_path or pretrained_model_name_or_path,
            class_name=final_config.iter_decider,
            init_args=iter_decider_kwargs,
        )
        logger.info(
            "Loaded iter_decider from %s", load_path or pretrained_model_name_or_path
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        tah_config: Optional[TaHConfig] = None,
        **kwargs,
    ):
        # Rebuild wrapper from checkpoint: config -> base model -> side modules.
        device_map = kwargs.pop("device_map", None)
        saved_config = cls._load_saved_tah_config(pretrained_model_name_or_path)
        final_config = cls._merge_tah_config(saved_config, tah_config)

        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tah_model = cls(base_model, config=final_config)

        cls._load_iter_decider_weights(
            tah_model, pretrained_model_name_or_path, final_config, saved_config
        )
        tah_model.eval_iter_decider = tah_model._resolve_eval_iter_decider(
            getattr(final_config, "eval_iter_decider", None),
            getattr(final_config, "eval_iter_decider_kwargs", {}) or {},
        )

        input_updater_path = os.path.join(
            pretrained_model_name_or_path, "input_updater.bin"
        )
        if os.path.exists(input_updater_path):
            tah_model.input_updater = load_input_updater(
                pretrained_model_name_or_path,
                class_name=final_config.input_updater,
                init_args=final_config.input_updater_kwargs,
            )
            logger.info("Loaded input_updater from model checkpoint")

        if device_map is not None:
            tah_model._dispatch_with_device_map(device_map)
        return tah_model
