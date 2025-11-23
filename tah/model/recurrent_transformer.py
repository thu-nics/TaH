"""
TaH (Hierarchical Recurrent Reasoning) Transformer Wrapper.

This module wraps around standard transformer PreTrainedModel (e.g., Qwen3ForCausalLM)
to enable hierarchical recurrent processing with iteration-aware caching.
"""

import torch
import torch.nn.functional as F
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, Union, Tuple, Dict, Any, List, Union
from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from accelerate import dispatch_model

from tah.model.causal_cache import TaHCache
from tah.model.utils import (
    get_attr_recursive,
    dict_string_to_type,
    type_to_dict_string,
    get_device_map,
)
from tah.model.iter_decider import (
    save_iter_decider,
    load_iter_decider,
    get_iter_decider_class,
)
from tah.model.input_updater import get_input_updater_class
from tah.model.output_updater import get_output_updater_class
from tah.model.loss import get_loss_func_class
from tah.model.iter_label import get_iter_label_generator_class
from tah.model.tah_config import TaHConfig
from tah.model.adapter import (
    setup_adapter,
    save_adapter,
    load_adapter,
    configure_lora_for_iteration
)

logger = logging.get_logger(__name__)
# # Ensure INFO level logging is enabled for this module
# logging.set_verbosity_info()


@dataclass
class TaHCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Inherit from CausalLMOutputWithPast, add iter_count.
    
    Args:
        iter_count (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Number of iterations performed for each token in the sequence.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    iter_count: Optional[torch.LongTensor] = None
    iter_count_labels: Optional[torch.LongTensor] = None


class TaHForCausalLM(PreTrainedModel):
    """
    TaH wrapper for Causal Language Models that enables hierarchical recurrent processing.

    This wrapper takes a standard transformer model (e.g., Qwen3ForCausalLM) and adds
    support for iterative processing where:
    1. Each token can be processed multiple times (based on iter_count)
    2. The output of iteration i becomes the input of iteration i+1
    3. Deeper iterations can see cache from all previous iterations
    4. Previous iterations cannot see cache from future iterations
    """

    def __init__(
        self, base_model: PreTrainedModel, config: Optional[TaHConfig] = None, **kwargs
    ):
        """
        Initialize TaH wrapper.

        Args:
            base_model: The base transformer model to wrap (e.g., Qwen3ForCausalLM)
            max_iter: Maximum number of iterations in automatic mode
            iter_decider: Plug-in object that decides whether a token continues
            input_updater: Module that updates inputs between iterations
        """
        self.base_model._supports_sdpa = True
        super().__init__(base_model.config)
        self.config = base_model.config
        self.supports_gradient_checkpointing = True

        if config is None:
            config = TaHConfig()

        self.tah_config = config

        # Check the embedding key
        try:
            get_attr_recursive(base_model, config.embedding_key)
        except AttributeError:
            raise ValueError(
                f"Embedding_key {config.embedding_key} not found in base model"
            )
        self.embedding_key = config.embedding_key

        self.max_iter = config.max_iter

        # Build iter_decider from config
        # Create decider from class and kwargs
        decider_cls = get_iter_decider_class(config.iter_decider)
        config.iter_decider_kwargs["max_iter"] = self.max_iter
        self.iter_decider = decider_cls(**config.iter_decider_kwargs)

        # Optional: build a separate iter_decider for evaluation/inference
        # Falls back to training decider when not provided
        eval_iter_decider = getattr(config, "eval_iter_decider", None)
        if eval_iter_decider is not None:
            resolved = None
            if isinstance(eval_iter_decider, str):
                # Support hierarchical path referencing the built training iter_decider
                # Example: "iter_decider.primary_iter_decider.final_iter_decider"
                if eval_iter_decider.startswith("iter_decider"):
                    path = eval_iter_decider.split(".")
                    obj = self
                    for seg in path:
                        if not seg:
                            continue
                        if seg == "self":
                            obj = self
                        else:
                            obj = getattr(obj, seg)
                    resolved = obj
                # Class-name path
                else:
                    eval_decider_cls = get_iter_decider_class(eval_iter_decider)
                    resolved = eval_decider_cls(**getattr(config, 'eval_iter_decider_kwargs', {}))

            self.eval_iter_decider = resolved if resolved is not None else self.iter_decider
        else:
            self.eval_iter_decider = self.iter_decider

        # Build input_updater from config
        # Create updater from class and kwargs
        updater_cls = get_input_updater_class(config.input_updater)
        self.input_updater = updater_cls(**config.input_updater_kwargs)

        # Build output_updater from config
        # Create output updater from class and kwargs, 
        output_updater_cls = get_output_updater_class(config.output_updater or 'NoneUpdater')
        self.output_updater = output_updater_cls(**config.output_updater_kwargs)

        # Build loss func from config
        # Robustly pass max_iter when supported by the loss (wrapper-safe): try with max_iter, fallback without
        train_loss_func_cls = get_loss_func_class(config.train_loss)
        _train_kwargs = dict(getattr(config, 'train_loss_kwargs', {}) or {})
        try:
            self.train_loss = train_loss_func_cls(**{**_train_kwargs, "max_iter": self.max_iter})
        except TypeError as e:
            if "max_iter" in str(e):
                self.train_loss = train_loss_func_cls(**_train_kwargs)
            else:
                raise

        # Build eval loss func from config
        if config.eval_loss:
            eval_loss_func_cls = get_loss_func_class(config.eval_loss)
            _eval_kwargs = dict(getattr(config, 'eval_loss_kwargs', {}) or {})
            try:
                self.eval_loss = eval_loss_func_cls(**{**_eval_kwargs, "max_iter": self.max_iter})
            except TypeError as e:
                if "max_iter" in str(e):
                    self.eval_loss = eval_loss_func_cls(**_eval_kwargs)
                else:
                    raise
        else:
            self.eval_loss = self.train_loss

        # Build iter label generator from config (constructed here, prepared per forward)
        iter_label_generator_name = getattr(config, "iter_label_generator", None) or "FixedIterLabelGenerator"
        iter_label_generator_kwargs = getattr(config, "iter_label_generator_kwargs", None) or {}
        IterLabelGenCls = get_iter_label_generator_class(iter_label_generator_name)
        self.iter_label_generator = IterLabelGenCls(**iter_label_generator_kwargs)

        # Init base model
        self.simple_base_model = base_model

        # iter attention mode
        self.iter_attention_mode = config.iter_attention_mode


        # Setup adapter if enabled
        self._setup_adapter(config)

        # Tokens that require multiple iterations (iter_count > 1) are considered "hard".
        # Their loss will be multiplied by this factor during training. 1.0 means no reweighting.
        self.hard_token_relative_weight = 1.0
        self.avg_hard_token_ratio = None
        self.weight_hard = None
        self.weight_easy = None

        # TODO: Ensure input_updater is on the same device and dtype as the base model
        device_map = kwargs.pop("device_map", None)
        if device_map is not None:
            device_map = get_device_map(self, device_map, self.dtype)
            dispatch_model_kwargs = {
                "device_map": device_map,
                "offload_dir": None,
                "offload_index": None,
                "offload_buffers": False,
                # "skip_keys": ["past_key_values"],
                "skip_keys": self.simple_base_model._skip_keys_device_placement
            }
            self = dispatch_model(self, **dispatch_model_kwargs)

    def _setup_adapter(self, config: TaHConfig):
        """Setup adapter for the model (delegated)."""
        setup_adapter(self, config)
            
    def _configure_lora_for_iteration(self, iter_depth: int):
        """Configure LoRA adapters for the current iteration (delegated)."""
        configure_lora_for_iteration(self, iter_depth)
    
    
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # Since TaH is a wrapper without its own parameters, delegate to base model
        return self.simple_base_model.device

    def to(self, *args, **kwargs):
        """
        Move the model to the specified device/dtype. Delegates to the base model.
        """
        self.simple_base_model = self.simple_base_model.to(*args, **kwargs)
        if hasattr(self, "input_updater") and self.input_updater is not None:
            self.input_updater = self.input_updater.to(*args, **kwargs)
        if hasattr(self, "output_updater") and self.output_updater is not None:
            self.output_updater = self.output_updater.to(*args, **kwargs)
        if hasattr(self, "iter_decider") and self.iter_decider is not None:
            self.iter_decider = self.iter_decider.to(*args, **kwargs)
        if hasattr(self, "eval_iter_decider") and self.eval_iter_decider is not None:
            self.eval_iter_decider = self.eval_iter_decider.to(*args, **kwargs)
        if hasattr(self, "iter_label_generator") and self.iter_label_generator is not None:
            self.iter_label_generator = self.iter_label_generator.to(*args, **kwargs)  # type: ignore[attr-defined]
        return self

    def cuda(self, device=None):
        """
        Move the model to CUDA. Delegates to the base model.
        """
        self.simple_base_model = self.simple_base_model.cuda(device)
        if hasattr(self, "input_updater") and self.input_updater is not None:
            self.input_updater = self.input_updater.cuda(device)
        if hasattr(self, "output_updater") and self.output_updater is not None:
            self.output_updater = self.output_updater.cuda(device)
        if hasattr(self, "iter_decider") and self.iter_decider is not None:
            self.iter_decider = self.iter_decider.cuda(device)
        if hasattr(self, "eval_iter_decider") and self.eval_iter_decider is not None:
            self.eval_iter_decider = self.eval_iter_decider.cuda(device)
        if hasattr(self, "iter_label_generator") and self.iter_label_generator is not None:
            self.iter_label_generator = self.iter_label_generator.cuda(device)
        return self

    def cpu(self):
        """
        Move the model to CPU. Delegates to the base model.
        """
        self.simple_base_model = self.simple_base_model.cpu()
        if hasattr(self, "input_updater") and self.input_updater is not None:
            self.input_updater = self.input_updater.cpu()
        if hasattr(self, "output_updater") and self.output_updater is not None:
            self.output_updater = self.output_updater.cpu()
        if hasattr(self, "iter_decider") and self.iter_decider is not None:
            self.iter_decider = self.iter_decider.cpu()
        if hasattr(self, "eval_iter_decider") and self.eval_iter_decider is not None:
            self.eval_iter_decider = self.eval_iter_decider.cpu()
        if hasattr(self, "iter_label_generator") and self.iter_label_generator is not None:
            self.iter_label_generator = self.iter_label_generator.cpu()
        return self

    @property
    def embed_tokens(self):
        """Return the embedding layer from the base model."""
        if "lora" in self.adapter:
            return get_attr_recursive(self.simple_base_model.base_model.model, self.embedding_key)
        return get_attr_recursive(self.simple_base_model, self.embedding_key)

    def forward(
        self,
        input_ids: torch.LongTensor,
        iter_count: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TaHCache] = None,
        # input_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        iter_count_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False, # noqa
        new_sequence: Optional[bool] = False,         # used by oracle iter decider
        # cache_position: Optional[torch.LongTensor] = None,
        # logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with hierarchical recurrent processing.

        Warning: iter_count will be deprecated in future versions. Iteration will be fully controlled by the iter decider.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            iter_count: Number of iterations for each token of shape (batch_size, query_seq_len)
            attention_mask: Optional attention mask, with shape (batch_size, total_seq_len)
            position_ids: Optional position ids, with shape (batch_size, query_seq_len)
            labels: Optional labels for loss computation
            use_cache: Whether to use cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional arguments

        Returns:
            CausalLMOutputWithPast with results from final iteration
        """

        """
        Initializations
        """
        # TODO: support other functions of Transformers
        assert (output_attentions is None) or (output_attentions is False), "TaH does not support output_attentions"
        assert (output_hidden_states is None) or (output_hidden_states is False), "TaH does not support output_hidden_states"

        # shift labels
        if labels is not None:
            labels_shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            labels_all_shifted = F.pad(input_ids.clone(), (0, 1), value=-100)[..., 1:].contiguous()  # includes query tokens uniformly
        else:
            labels_shifted = None
            labels_all_shifted = None

        max_iterations = self.max_iter

        # Initialize scalars and tensors
        batch_size, query_len = input_ids.shape
        vocab_size = self.config.vocab_size
        hidden_size = self.config.hidden_size
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        input_embeds = self.embed_tokens(
            input_ids
        )  # (batch_size, query_len, hidden_size)
        dtype = input_embeds.dtype
        device = input_embeds.device
        final_output_logits = torch.zeros(
            batch_size, query_len, vocab_size, device=device, dtype=dtype
        )  # (batch_size, query_len, vocab_size)
        cumulative_logits = torch.zeros(
            batch_size, query_len, vocab_size, device=device, dtype=dtype
        )  # (batch_size, query_len, vocab_size) - for output updater accumulation
        actual_iter_counts = torch.zeros_like(
            input_ids, dtype=torch.long
        )  # (batch_size, query_len)

        # Initialize cache
        if past_key_values is not None:
            cache = past_key_values
        else:
            cache = TaHCache().to(device=device, dtype=dtype)  # noqa

        # Initialize token mask
        if attention_mask is not None:
            valid_mask = attention_mask.clone()[:, -query_len:].to(dtype=torch.long)
            assert valid_mask.shape == (
                batch_size,
                query_len,
            ), f"attention_mask shape must be (batch_size, seq_len), but got {attention_mask.shape}"
        else:
            valid_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Initialize position_ids
        if position_ids is None:
            # position_offset = cache.get_seq_length()  # Layer 0 Iter 0 cache length
            # TODO: design more efficient ways to get seq length of each batch
            position_ids = torch.clamp(
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
        else:
            position_ids = position_ids.clone()

        # Initialize loss func for the forward pass
        loss_func = self.train_loss if self.training else self.eval_loss
        loss_func.prepare_loss(
            batch_size, query_len, device, dtype,
            weight_hard=self.weight_hard,
            weight_easy=self.weight_easy,
            hard_token_relative_weight=self.hard_token_relative_weight
        )

        # Decide whether to use iter label generator this forward
        use_iter_labeling = (self.iter_label_generator is not None) and (labels_shifted is not None)
        # Prepare iter label generator buffers (train or eval) only when labeling is used
        if use_iter_labeling:
            self.iter_label_generator.prepare(batch_size, query_len, device, dtype)


        """
        Iterative processing
        """
        current_iter_mask = torch.ones_like(
            input_ids, dtype=torch.bool
        )  # (batch_size, query_len), 1 if the element is selected for current iter
        finished_mask = torch.zeros_like(
            current_iter_mask, dtype=torch.bool
        )  # default to unfinished
        iter_depth = 0

        while iter_depth < max_iterations and current_iter_mask.any():

            # Configure LoRA for current iteration if not already done
            self._configure_lora_for_iteration(iter_depth)
            
            # Extract sparse inputs for active tokens
            # TODO: modify the extraction logic to handle the token mask
            active_input_embeds, active_cumulative_logits, active_position_ids, active_valid_mask, active_iter_count, active_labels_shifted, active_iter_count_labels, active_labels_all_shifted = (
                self.to_active(
                    current_iter_mask, input_embeds, cumulative_logits, position_ids, valid_mask, iter_count, labels_shifted, iter_count_labels, labels_all_shifted
                )
            )


            # Break if no active tokens
            if active_valid_mask.shape[1] == 0:
                break

            # Create SDPA attention mask
            sdpa_attention_mask = self.create_TaH_sdpa_attention_mask(
                active_position_ids, active_valid_mask, cache, iter_depth, dtype=dtype
            )

            # Process this iteration with sparse inputs
            current_model = self._select_model_for_iteration(iter_depth)
                
            active_outputs = self._process_sparse_iteration(
                sparse_input=active_input_embeds,
                position_ids=active_position_ids,
                valid_mask=active_valid_mask,
                cache_position=None,  # cache position not used for now
                attention_mask=sdpa_attention_mask,
                iter_depth=iter_depth,
                past_key_values=cache,
                use_cache=True if iter_depth < max_iterations - 1 else use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True, # noqa: output_hidden_states must be True to get last hidden for iter decider
                model=current_model, # Pass the selected model to _process_sparse_iteration
                **kwargs,
            )

            # Update iter depth
            iter_depth += 1

            # noqa: Update output device, when device map is auto, output device may be different from the input one. Move back to input device
            active_outputs.logits = active_outputs.logits.to(device=device)
            
            # Update cumulative logits using output updater
            # For first iteration, active_prev_logits will be zeros, which output_updater handles correctly
            active_updated_cumulative_logits = self.output_updater(
                logits=active_outputs.logits,
                prev_logits=active_cumulative_logits,
                iter_depth=iter_depth - 1,  # iter_depth was incremented above, so use iter_depth - 1 for 0-indexed
            )
            
            # Write updated logits back to cumulative_logits
            # Use assign_active_no_inplace to avoid in-place modification that breaks autograd
            cumulative_logits = self.assign_active_no_inplace(
                current_iter_mask, 
                src=active_updated_cumulative_logits, 
                dest=cumulative_logits
            )

            all_hidden = None
            if hasattr(active_outputs, "hidden_states") and active_outputs.hidden_states is not None:
                hidden_states = active_outputs.hidden_states
                # hidden_states can be a tuple(list) of length = num_layers, each with shape (B, T, H)
                # Convert to a tensor shaped (B, T, L, H) so that boolean mask of shape (B, T) applies cleanly
                if isinstance(hidden_states, (tuple, list)):
                    layer_hidden_list = [h.to(device=device) for h in hidden_states]
                    if len(layer_hidden_list) > 0:
                        all_hidden = torch.stack(layer_hidden_list, dim=0).permute(1, 2, 0, 3)
                elif torch.is_tensor(hidden_states):
                    # If already a tensor (B, T, H), add layer dim to unify to (B, T, L=1, H)
                    all_hidden = hidden_states.to(device=device)
                    if all_hidden.dim() == 3:
                        all_hidden = all_hidden.unsqueeze(-2)

            # TODO: improve efficiency
            # We always call iter_decider, even if the count is given, to get the continue prob for all tokens
            # Choose decider based on train/eval mode
            cur_iter_decider = self.iter_decider if self.training else (self.eval_iter_decider or self.iter_decider)

            # Optionally compute per-iteration labels from generator and unify naming
            if use_iter_labeling:
                active_iter_count_labels = self.iter_label_generator.intra_iter_labels(
                    active_iter_count_labels=active_iter_count_labels,
                    active_logits=active_updated_cumulative_logits,
                    active_labels_shifted=active_labels_all_shifted,
                    iter_depth=iter_depth,
                    current_iter_mask=current_iter_mask,
                    active_valid_mask=active_valid_mask,
                )

            active_valid_continue_decision, active_valid_continue_logits = cur_iter_decider(
                logits=active_updated_cumulative_logits[active_valid_mask == 1],
                iter_depth=iter_depth,
                all_hidden_states=all_hidden[active_valid_mask == 1] if all_hidden is not None else None,
                labels_shifted=active_labels_all_shifted[active_valid_mask == 1] if active_labels_all_shifted is not None else None, # used by mismatch iter decider
                iter_count_labels=(active_iter_count_labels[active_valid_mask == 1] if active_iter_count_labels is not None else None),
            )
            
            
            # Ensure at least one labeled position continues when all labeled decisions are False
            if (
                (active_labels_shifted is not None)
                and (active_valid_continue_decision is not None)
                and (active_valid_continue_decision.numel() > 0)
                and iter_depth < self.max_iter
            ):
                label_mask_flat = (active_labels_shifted != -100)[active_valid_mask == 1]
                if label_mask_flat.any() and (not active_valid_continue_decision[label_mask_flat].any()):
                    candidate_indices = torch.nonzero(label_mask_flat, as_tuple=False).flatten()
                    chosen_idx = candidate_indices[torch.randint(low=0, high=candidate_indices.numel(), size=(1,), device=device)]
                    active_valid_continue_decision[chosen_idx] = True
            
            # Move tensors to correct device
            if active_valid_continue_logits is not None:
                active_valid_continue_logits = active_valid_continue_logits.to(device=device)

            # decide whether to finish current iteration
            active_finished_mask = torch.ones_like(active_valid_mask, dtype=torch.bool)
            # When explicit boolean decision is provided: finish where decision == False
            active_finished_mask[active_valid_mask == 1] = (~active_valid_continue_decision)
            self.assign_active(
                current_iter_mask, src=active_finished_mask, dest=finished_mask
            )

            actual_iter_counts[current_iter_mask] += 1

            # Calculate loss for all active tokens in current iteration
            if labels_shifted is not None and loss_func._is_intra_iter_loss:
                # Prepare kwargs for intra_iter_loss_func, including iter_depth and active_iter_count_labels
                intra_loss_kwargs = kwargs.copy()
                intra_loss_kwargs['iter_depth'] = iter_depth
                # pass iter_decider threshold for metric computation
                intra_loss_kwargs['iter_decider_threshold'] = cur_iter_decider.threshold
                # provide model handle for potential freeze control in loss
                intra_loss_kwargs['model'] = self
                # forward global_step if provided by caller
                if 'global_step' in kwargs:
                    intra_loss_kwargs['global_step'] = kwargs['global_step']
                # Use unified active_iter_count_labels for BCE targets if present
                if active_iter_count_labels is not None:
                    intra_loss_kwargs['active_iter_count_labels'] = active_iter_count_labels
                if all_hidden is not None:
                    intra_loss_kwargs['all_hidden_states'] = all_hidden
                
                # Compute loss for all currently active tokens
                loss_func.intra_iter_loss_func(
                    active_logits=active_updated_cumulative_logits,
                    current_iter_mask=current_iter_mask,
                    active_labels_shifted=active_labels_shifted,
                    active_valid_continue_logits=active_valid_continue_logits,
                    active_valid_mask=active_valid_mask,
                    **intra_loss_kwargs
                )
                
            if active_finished_mask.any():
                # Update actual iteration counts for tokens that finish
                # Copy accumulated logits from active_updated_cumulative_logits to final_output_logits for finished tokens
                self.assign_active_with_mask(
                    current_iter_mask,
                    assignment_mask=finished_mask,
                    src=active_updated_cumulative_logits,
                    dest=final_output_logits,
                )

            next_iter_mask = (
                (~finished_mask) & current_iter_mask & (valid_mask == 1)
            )  # iter mask can accept invalid inputs; currently, it filter to accept only valid for efficiency concideration
            if next_iter_mask.any():
                active_next_iter_mask = (~active_finished_mask) & (active_valid_mask == 1)
                # Always pass all_hidden_states to input_updater; selection handled inside updater
                active_input_embeds[active_next_iter_mask] = self.input_updater(
                    logits = active_updated_cumulative_logits[active_next_iter_mask],
                    prev_inputs = active_input_embeds[active_next_iter_mask],
                    embedding_weight = self.embed_tokens.weight, # should ignored by AlignDeviceHook and avoid weight moving
                    hidden_states = all_hidden[active_next_iter_mask] if all_hidden is not None else None,
                ).to(device=device)

                # Clone to prevent in-place modification on a tensor that autograd still needs for previous iterations
                input_embeds = torch.zeros_like(input_embeds)
                self.assign_active_with_mask(
                    current_iter_mask,
                    assignment_mask=next_iter_mask,
                    src=active_input_embeds,
                    dest=input_embeds,
                )

            # Create mask for tokens that need processing at this iteration
            current_iter_mask = next_iter_mask

            if not current_iter_mask.any():
                break

        # Compute loss if labels are provided
        loss = None
        if labels_shifted is not None:
            # Prepare kwargs for loss function, including iter_count_labels if available
            loss_kwargs = kwargs.copy()
            # Finalize generator-produced full labels for logging/loss if requested
            finalized_iter_labels = None
            if use_iter_labeling:
                finalized_iter_labels = self.iter_label_generator.finalize()
                # expose for analysis/logging
                # TODO: log final iter labels
                # if hasattr(self, 'logger_callback') and finalized_iter_labels is not None:
                #     self.logger_callback.last_iter_count_labels = finalized_iter_labels.detach().to('cpu')
            if finalized_iter_labels is not None:
                loss_kwargs['iter_count_labels'] = finalized_iter_labels
            elif iter_count_labels is not None:
                loss_kwargs['iter_count_labels'] = iter_count_labels
            # pass logger callback for metric logging if available
            if hasattr(self, 'logger_callback'):
                loss_kwargs['logger_callback'] = self.logger_callback
            # provide model handle for potential freeze control in loss
            loss_kwargs['model'] = self
            # forward global_step if provided by caller
            if 'global_step' in kwargs:
                loss_kwargs['global_step'] = kwargs['global_step']
            
            loss = loss_func.final_loss_func(
                logits=final_output_logits,
                labels_shifted=labels_shifted,
                iter_count=actual_iter_counts,
                training=self.training,
                **loss_kwargs
            )
            
            if hasattr(self, "logger_callback"):
                num_items_in_batch = kwargs.get("num_items_in_batch", None)
                if num_items_in_batch is not None:
                    valid_iter_mask = (labels_shifted.detach() != -100)
                    valid_iter_counts = actual_iter_counts.detach()[valid_iter_mask]
                    avg_valid_iter_count = torch.sum(valid_iter_counts).float()
                    self.logger_callback.avg_iter_count += float((avg_valid_iter_count / num_items_in_batch).item())
                else:
                    self.logger_callback.avg_iter_count = float((torch.mean(actual_iter_counts.detach().float())).item())


        # Create custom output that includes actual iteration counts
        output = TaHCausalLMOutputWithPast(
            loss=loss,
            logits=final_output_logits,
            past_key_values=cache if use_cache else None,
            hidden_states=None,
            attentions=None,
            iter_count=actual_iter_counts,
            iter_count_labels=finalized_iter_labels if 'finalized_iter_labels' in locals() else None,
        )

        return output

    @staticmethod
    def to_active(
        current_iter_mask: torch.BoolTensor,
        input_embeds: torch.Tensor,
        cumulative_logits: torch.Tensor,
        position_ids: torch.LongTensor,
        valid_mask: torch.LongTensor,
        iter_count: Optional[torch.LongTensor],
        labels_shifted: Optional[torch.LongTensor] = None,
        iter_count_labels: Optional[torch.LongTensor] = None,
        labels_all_shifted: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor, Union[torch.LongTensor, None], Union[torch.LongTensor, None], torch.BoolTensor, Union[torch.LongTensor, None]]:
        """
        Return the active tokens (padded to the batch-wise max length).

        * active_input_embeds : (B, max_active_len, H)
        * active_cumulative_logits : (B, max_active_len, V)
        * active_position_ids : (B, max_active_len)
        * active_valid_mask   : (B, max_active_len) – propagates `valid_mask`
        * active_iter_count   : (B, max_active_len) – propagates `iter_count`
        * active_labels_shifted       : (B, max_active_len) – propagates `labels`, None if labels is None
        * active_iter_count_labels    : (B, max_active_len) – propagates `iter_count_labels`, None if iter_count_labels is None
        """
        B, S, H = input_embeds.shape
        _, _, V = cumulative_logits.shape
        device = input_embeds.device

        active_per_seq = current_iter_mask.sum(1)  # (B,)
        max_len = int(active_per_seq.max())  # scalar
        if max_len == 0:  # nothing active
            empty_e = input_embeds.new_empty(B, 0, H)
            empty_i = position_ids.new_empty(B, 0)
            empty_mask = torch.empty(B, 0, dtype=torch.bool, device=device)
            return empty_e, empty_i, empty_i, None, None, None, None, None

        # ------------------------------------------------------------------
        # 1. Build gather_idx_clamped and pad_mask
        # ------------------------------------------------------------------
        SENTINEL = S  # out-of-range value
        base_idx = torch.arange(S, device=device).expand(B, S)
        base_idx = base_idx.masked_fill(~current_iter_mask, SENTINEL)  # (B, S)

        # Stable sort → [active … | SENTINEL …]
        sorted_idx, _ = torch.sort(base_idx, dim=1, stable=True)  # (B, S)
        gather_idx = sorted_idx[:, :max_len]  # (B, max_len)
        pad_mask = gather_idx.eq(SENTINEL)  # True → padded

        # Same index, but clamped so `gather` is always in-range
        gather_idx_clamped = gather_idx.clamp(max=S - 1)

        # ------------------------------------------------------------------
        # 2. Vectorised gather and zero-out
        # ------------------------------------------------------------------
        active_input_embeds = torch.gather(
            input_embeds, 1, gather_idx_clamped.unsqueeze(-1).expand(-1, -1, H)
        )  # (B, max_len, H)
        # Avoid in-place modification on tensors that autograd needs for the backward
        active_input_embeds = active_input_embeds.masked_fill(
            pad_mask.unsqueeze(-1), 0
        )
        
        active_cumulative_logits = torch.gather(
            cumulative_logits, 1, gather_idx_clamped.unsqueeze(-1).expand(-1, -1, V)
        )  # (B, max_len, V)
        active_cumulative_logits = active_cumulative_logits.masked_fill(pad_mask.unsqueeze(-1), 0)

        active_position_ids = torch.gather(position_ids, 1, gather_idx_clamped)
        active_position_ids = active_position_ids.masked_fill(pad_mask, 0)

        active_valid_mask = torch.gather(valid_mask, 1, gather_idx_clamped)
        active_valid_mask = active_valid_mask.masked_fill(pad_mask, 0)

        if iter_count is not None:
            active_iter_count = torch.gather(iter_count, 1, gather_idx_clamped)
            active_iter_count = active_iter_count.masked_fill(pad_mask, 0)
        else:
            active_iter_count = None
        
        if iter_count_labels is not None:
            active_iter_count_labels = torch.gather(iter_count_labels, 1, gather_idx_clamped)
            active_iter_count_labels = active_iter_count_labels.masked_fill(pad_mask, 0)
        else:
            active_iter_count_labels = None

        if labels_shifted is not None:
            active_labels_shifted = torch.gather(labels_shifted, 1, gather_idx_clamped)
            active_labels_shifted = active_labels_shifted.masked_fill(pad_mask, -100)
        else:
            active_labels_shifted = None
        
        if labels_all_shifted is not None:
            active_labels_all_shifted = torch.gather(labels_all_shifted, 1, gather_idx_clamped)
            active_labels_all_shifted = active_labels_all_shifted.masked_fill(pad_mask, -100)
        else:
            active_labels_all_shifted = None

        return active_input_embeds, active_cumulative_logits, active_position_ids, active_valid_mask, active_iter_count, active_labels_shifted, active_iter_count_labels, active_labels_all_shifted

    @staticmethod
    def assign_active(
        current_iter_mask: torch.BoolTensor,
        src: torch.Tensor,
        dest: torch.Tensor,
        pad_value: float | int = 0,
    ) -> torch.Tensor:
        """
        Scatter `src` (the output of `extract_active`) back into a dense tensor.

        Args:
            current_iter_mask : BoolTensor (B, S)
                True where a position should be filled from `src`.
            src : Tensor (B, max_active, ...)
                Active tokens, padded on the right inside the second dimension.
            dest : Tensor (B, S, ...)
                Tensor to be updated **in-place**.
            pad_value : scalar
                Value written to inactive (False) positions.

        Returns:
            dest : Tensor (B, S, ...)  — same object that was passed in
        """
        B, S = current_iter_mask.shape
        max_active = src.shape[1]

        active_counts = current_iter_mask.sum(1)  # (B,)

        for b in range(B):
            n = active_counts[b].item()
            if n:  # only copy when there is something to copy
                dest[b, current_iter_mask[b]] = src[b, :n]

        return dest

    @staticmethod
    def assign_active_no_inplace(
        current_iter_mask: torch.BoolTensor,
        src: torch.Tensor,
        dest: torch.Tensor,
        pad_value: float | int = 0,
    ) -> torch.Tensor:
        """
        Scatter `src` (the output of `extract_active`) back into a dense tensor without in-place modification.

        Args:
            current_iter_mask : BoolTensor (B, S)
                True where a position should be filled from `src`.
            src : Tensor (B, max_active, ...)
                Active tokens, padded on the right inside the second dimension.
            dest : Tensor (B, S, ...)
                Tensor to be updated (will be cloned, not modified in-place).
            pad_value : scalar
                Value written to inactive (False) positions.

        Returns:
            new_dest : Tensor (B, S, ...)  — new tensor with updates applied
        """
        B, S = current_iter_mask.shape
        max_active = src.shape[1]

        # Clone dest to avoid in-place modification
        new_dest = dest.clone()
        active_counts = current_iter_mask.sum(1)  # (B,)

        for b in range(B):
            n = active_counts[b].item()
            if n:  # only copy when there is something to copy
                new_dest[b, current_iter_mask[b]] = src[b, :n]

        return new_dest

    @staticmethod
    def assign_active_with_mask(
        current_iter_mask: torch.BoolTensor,
        assignment_mask: torch.BoolTensor,
        src: torch.Tensor,
        dest: torch.Tensor,
        pad_value: float | int = 0,
    ) -> torch.Tensor:
        """
        Scatter the masked `src` (the output of `extract_active`) back into a dense tensor.

        Args:
            current_iter_mask : BoolTensor (B, S)
                True where a position should be filled from `src`.
            assignment_mask : BoolTensor (B, S)
                True where a position should be filled from `src` to `dest`.
            src : Tensor (B, max_active, ...)
                Active tokens, padded on the right inside the second dimension.
            dest : Tensor (B, S, ...)
                Tensor to be updated **in-place**.
            pad_value : scalar
                Value written to inactive (False) positions.

        Returns:
            dest : Tensor (B, S, ...)  — same object that was passed in
        """
        B, S = current_iter_mask.shape

        # Only assign where both masks are True
        final_mask = current_iter_mask & assignment_mask
        active_counts = current_iter_mask.sum(1)  # (B,)

        for b in range(B):
            n_active = active_counts[b].item()
            if n_active == 0:
                continue

            # Get active positions and assignment positions for this batch
            active_pos = current_iter_mask[b].nonzero(as_tuple=False).flatten()
            assign_pos = final_mask[b].nonzero(as_tuple=False).flatten()

            # Find which src indices correspond to assignment positions
            src_indices = torch.searchsorted(active_pos, assign_pos)
            valid_mask = src_indices < n_active

            if valid_mask.any():
                dest[b, assign_pos[valid_mask]] = src[b, src_indices[valid_mask]]

        return dest

    def _compute_positions_for_iteration(
        self, active_position_ids: torch.Tensor, seq_length: int, cache_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute position_ids and cache_position for active tokens.

        Args:
            active_position_ids: Original positions of active tokens (batch_size, num_active)
            seq_length: Current sequence length (for new token position computation)
            cache_length: Current total length of KV cache

        Returns:
            position_ids: Adjusted positions for positional encoding (batch_size, num_active)
            cache_position: Sequential positions in growing cache (num_active,)
        """
        _, num_active = active_position_ids.shape

        # position_ids are the sequence positions (already correct from _extract_active_inputs)
        position_ids = active_position_ids

        # cache_position: sequential positions starting from cache_length
        cache_position = torch.arange(
            cache_length,
            cache_length + num_active,
            device=active_position_ids.device,
            dtype=torch.long,
        )

        return position_ids, cache_position

    def create_TaH_sdpa_attention_mask(
        self,
        active_position_ids: torch.Tensor,
        active_valid_mask: torch.LongTensor,
        cache: Optional[TaHCache],
        iter_depth: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Optional[torch.Tensor]:
        """
        Create SDPA attention mask where query at position p, iteration i
        The query can attend to cached KVs with position <= p AND iteration <= i.
        Note that the mask should have the same shape as the updated cache, which only contains the KVs with iteration <= i.
        The mask is added to the attention score, with min_dtype = torch.finfo(dtype).min being the masked part.

        Args:
            active_position_ids: Original positions of active tokens (batch_size, query_length)
            active_valid_mask: Mask indicating valid active tokens (batch_size, query_length)
            cache: Current cache object
            iter_depth: Current iteration depth
            dtype: Data type for the attention mask

        Returns:
            Attention mask of shape (batch_size, 1, query_length, filtered_cache_length + query_length) or None
        """
        batch_size, query_length = active_position_ids.shape
        device = active_position_ids.device

        # Get filtered cache positions (only iterations <= iter_depth)
        if (cache is not None) and (0 in cache._tah_position_id_cache):
            iter_index_cache = cache.get_cache_iter_index_upto_iter(
                layer_idx=0, upto_iter_idx=iter_depth
            )
            position_ids_cache_upto_iter = cache.get_position_id_upto_iter(
                layer_idx=0, upto_iter_idx=iter_depth, init_batch_size=batch_size
            )
            valid_mask_cache_upto_iter = cache.get_valid_mask_upto_iter(
                layer_idx=0, upto_iter_idx=iter_depth, init_batch_size=batch_size
            )  # TODO: implement
            kv_cache_length_upto_iter = iter_index_cache.shape[-1]
        else:
            iter_index_cache = torch.empty(size=(0,), device=device, dtype=torch.long)
            position_ids_cache_upto_iter = torch.empty(
                size=(batch_size, 0), device=device, dtype=torch.long
            )
            valid_mask_cache_upto_iter = torch.empty(
                size=(batch_size, 0), device=device, dtype=torch.long
            )
            kv_cache_length_upto_iter = 0

        # KV length for attention computation equals to the KV length from cache plus the current key/value length (=query length)
        kv_length_this_iter = kv_cache_length_upto_iter + query_length

        if kv_length_this_iter == 0:
            return None

        min_dtype = torch.finfo(dtype).min

        # Build complete KV position list: filtered cache + new positions being added
        kv_position_ids_upto_iter = torch.cat(
            (position_ids_cache_upto_iter, active_position_ids), dim=-1
        )  # shape: (batch_size, query_length + kv_cache_length_upto_iter)
        # Extract only valid positions based on active_valid_mask
        kv_valid_mask_upto_iter = torch.cat(
            (valid_mask_cache_upto_iter, active_valid_mask), dim=-1
        )  # shape: (batch_size, query_length + kv_cache_length_upto_iter)
        kv_iter_index = torch.cat(
            (
                iter_index_cache,
                torch.full(
                    (query_length,), iter_depth, dtype=torch.long, device=device
                ),
            ),
            dim=-1,
        )[
            None, :
        ]  # shape: (1-batch-size, query_length + kv_cache_length_upto_iter)

        # Expand query positions and iterations for broadcasting
        query_positions = active_position_ids[
            :, :, None
        ]  # (batch_size, query_length) -> (batch_size, query_length, 1)
        kv_position_ids_upto_iter = kv_position_ids_upto_iter[
            :, None, :
        ]  # (batch_size, total_kv_length) -> (batch_size, 1, total_kv_length)
        kv_valid_mask_upto_iter = kv_valid_mask_upto_iter[
            :, None, :
        ]  # (batch_size, total_kv_length) -> (batch_size, 1, total_kv_length)


        if self.iter_attention_mode == "duo":
            query_iter_index = torch.full_like(
                query_positions, iter_depth
            )  # (batch_size, query_length, 1)
        elif self.iter_attention_mode == "root":
            query_iter_index = torch.full_like(
                query_positions, 0
            )  # (batch_size, query_length, 1)
        elif self.iter_attention_mode == "same_iter":
            query_iter_index = torch.full_like(
                query_positions, iter_depth
            )  # (batch_size, query_length, 1)
        else:
            raise ValueError(f"Invalid iter attention mode: {self.iter_attention_mode}")
        
        kv_iter_index = kv_iter_index[:, None, :]  # (1, 1, total_kv_length)

        # Vectorized rule: query at (position=p, iter=i) can see cache entry at (position=cp, iter=ci)
        # if and only if cp <= p AND ci <= i
        position_mask = (
            kv_position_ids_upto_iter <= query_positions
        )  # (batch_size, query_length, total_kv_length)
        if self.iter_attention_mode == "same_iter":
            iteration_mask = (
                kv_iter_index == query_iter_index
            )  # (batch_size, query_length, total_kv_length)
        else:
            iteration_mask = (
                kv_iter_index <= query_iter_index
            )  # (batch_size, query_length, total_kv_length)
        valid_mask = (kv_valid_mask_upto_iter == 1)  # (batch_size, 1, total_kv_length)

        # Combine both conditions
        bool_attention_mask = (
            position_mask & iteration_mask & valid_mask
        )  # (batch_size, query_length, total_kv_length)

        # Create attention mask - start with all masked (min_dtype), then unmask where can_attend is True
        attention_mask = torch.full(
            (batch_size, query_length, kv_length_this_iter),
            min_dtype,
            device=device,
            dtype=dtype,
        )
        attention_mask[bool_attention_mask] = 0.0  # unmasked

        return attention_mask[:, None, :, :]

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
        """
        Process a single iteration through the base model with active/sparse inputs.

        Args:
            sparse_input: Active input embeddings (batch_size, num_active, hidden_size)
            position_ids: Active position ids (batch_size, num_active)
            valid_mask: Long tensor mask indicating the padding scenario of the original input tokens. 0 means masked.
            cache_position: Sequential positions in cache (num_active,)
            attention_mask: SDPA attention mask (batch_size, num_active, total_kv_length)
            iter_depth: Current iteration depth
            past_key_values: Cache object
            use_cache: Whether to use cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional arguments

        Returns:
            Model output for this iteration
        """
        
        # Set iteration depth and position metadata in cache
        if past_key_values is not None:
            past_key_values.current_iter_depth = iter_depth
            past_key_values.position_ids_to_cache = position_ids
            past_key_values.valid_mask_to_cache = valid_mask

        # Process through base model with active inputs
        outputs = model(
            inputs_embeds=sparse_input,
            position_ids=position_ids,
            cache_position=cache_position,  # noqa: not used for now
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        return outputs

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the TaH model by directly saving the base model to avoid wrapper prefixes.
        Also saves the TaHConfig for automatic loading.

        Args:
            save_directory: Directory where to save the model
            **kwargs: Additional arguments for saving
        """
        # Save adapter and base model
        save_adapter(self, save_directory, **kwargs)
        # Save iter_decider
        # If iter_decider is a noise wrapper, save its base (do not persist wrapper)
        try:
            from tah.model.iter_decider import NoisyWrapperIterDecider
            iter_to_save = self.iter_decider.base_iter_decider if isinstance(self.iter_decider, NoisyWrapperIterDecider) else self.iter_decider
        except Exception:
            iter_to_save = self.iter_decider
        save_iter_decider(iter_to_save, save_directory)
        
        
        # Save TaH config with special handling for type objects
        config_dict = asdict(self.tah_config)
        serializable_config = type_to_dict_string(config_dict)
        
        config_path = os.path.join(save_directory, "tah_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        tah_config: Optional[TaHConfig] = None,
        **kwargs,
    ):
        """
        Load a pretrained TaH model.

        Args:
            pretrained_model_name_or_path: Path to the saved TaH model directory
            tah_config: Optional TaHConfig to override specific saved config values
            *args, **kwargs: Arguments for model loading

        Returns:
            TaHForCausalLM instance
        """
        # Move to device after initializations are all done
        device_map = kwargs.pop("device_map", None)

        # Load saved config from checkpoint if it exists
        config_path = os.path.join(pretrained_model_name_or_path, "tah_config.json")
        saved_config = None
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            # Convert serialized type objects back to actual types
            config_dict = dict_string_to_type(config_dict)
            saved_config = TaHConfig(**config_dict)
            logger.info(f"Loaded TaH config from {config_path}")

        # Determine final config by selectively overriding saved config with provided config
        if tah_config is not None:
            if saved_config is not None:
                # Start with saved config and override specific fields from provided config
                final_config_dict = asdict(saved_config)
                provided_config_dict = asdict(tah_config)

                # Override only non-None values from provided config
                for key, value in provided_config_dict.items():
                    if (value is not None) and (value != {}):
                        final_config_dict[key] = value
                        logger.info(
                            f"Overriding config field '{key}' with provided value: {value}"
                        )

                final_config = TaHConfig(**final_config_dict)
            else:
                # No saved config, use provided config
                final_config = tah_config
                logger.info("No saved config found, using provided tah_config")
        else:
            if saved_config is not None:
                # Use saved config
                final_config = saved_config
            else:
                # No saved config and no provided config, use default
                logger.warning(
                    f"No tah_config.json found in {pretrained_model_name_or_path} and no tah_config provided. "
                    "Using default TaHConfig."
                )
                final_config = TaHConfig()

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        
        # Load tah model
        iter_decider_path = None
        if "load_path" in final_config.iter_decider_kwargs:
            iter_decider_path = final_config.iter_decider_kwargs.pop("load_path")
        
        # Create TaH model
        tah_model = cls(base_model, config=final_config)
        
        # Reload adapter specific weights/models (delegated)
        load_adapter(tah_model, pretrained_model_name_or_path, final_config, *args, **kwargs)
        
        # Decide whether to skip loading iter_decider weights based on class difference
        skip_iter_decider_loading = False
        load_base_iter_decider = False
        if 'saved_config' in locals() and (saved_config is not None):
            if getattr(saved_config, 'iter_decider', None) != final_config.iter_decider:
                skip_iter_decider_loading = True
                # If new base_iter_decider_cls equals the old model's iter_decider class, still load from old path
                old_iter_decider_cls_name = getattr(saved_config, 'iter_decider', None)
                final_kwargs = getattr(final_config, 'iter_decider_kwargs', None)
                if isinstance(final_kwargs, dict):
                    final_base = final_kwargs.get('base_iter_decider_cls')
                    if (final_base is not None) and (final_base == old_iter_decider_cls_name):
                        skip_iter_decider_loading = False
                        load_base_iter_decider = True
        
        # Load iter_decider
        if iter_decider_path is not None:
            loaded_iter_decider = load_iter_decider(iter_decider_path, class_name=final_config.iter_decider, init_args=final_config.iter_decider_kwargs)
            tah_model.iter_decider = loaded_iter_decider
            logger.info("Loaded iter_decider from newly provided load_path")
        else:
            if skip_iter_decider_loading:
                logger.info("Detected different iter_decider class; skipping old weight loading and using new iter_decider from final_config")
            else:
                
                if load_base_iter_decider:
                    loaded_iter_decider = load_iter_decider(
                        pretrained_model_name_or_path, 
                        class_name=final_config.iter_decider_kwargs.get('base_iter_decider_cls', None), 
                        init_args=final_config.iter_decider_kwargs.get('base_iter_decider_kwargs', {})
                    )
                    tah_model.iter_decider.base_iter_decider = loaded_iter_decider
                else:
                    loaded_iter_decider = load_iter_decider(pretrained_model_name_or_path, class_name=final_config.iter_decider, init_args=final_config.iter_decider_kwargs)
                    tah_model.iter_decider = loaded_iter_decider
                logger.info("Loaded iter_decider from model checkpoint")
        
        # Load eval_iter_decider
        eval_iter_decider = getattr(final_config, "eval_iter_decider", None)
        if eval_iter_decider is not None:
            resolved = None
            if isinstance(eval_iter_decider, str):
                # Support hierarchical path referencing the built training iter_decider
                # Example: "iter_decider.primary_iter_decider.final_iter_decider"
                if eval_iter_decider.startswith("iter_decider"):
                    path = eval_iter_decider.split(".")
                    obj = tah_model
                    for seg in path:
                        if not seg:
                            continue
                        if seg == "self":
                            obj = tah_model
                        else:
                            obj = getattr(obj, seg)
                    resolved = obj
                # Class-name path
                else:
                    eval_decider_cls = get_iter_decider_class(eval_iter_decider)
                    resolved = eval_decider_cls(**getattr(final_config, 'eval_iter_decider_kwargs', {}))

            tah_model.eval_iter_decider = resolved if resolved is not None else tah_model.iter_decider
        else:
            tah_model.eval_iter_decider = tah_model.iter_decider

        
        # Move to device if device map is provided
        if device_map is not None:
            device_map = get_device_map(tah_model, device_map, tah_model.dtype)
            dispatch_model_kwargs = {
                "device_map": device_map,
                "offload_dir": None,
                "offload_index": None,
                "offload_buffers": False,
                "skip_keys": tah_model.simple_base_model._skip_keys_device_placement
            }
            tah_model = dispatch_model(tah_model, **dispatch_model_kwargs)

        return tah_model
