"""
Adapter utilities for TaH wrapper.

This module centralizes all adapter-related logic (e.g., LoRA/multi-LoRA/cascade)
that previously lived inside `recurrent_transformer.py`, without changing any
existing runtime behavior.
"""


import os


from transformers.utils import logging


# PEFT imports (lazy optional)
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


logger = logging.get_logger(__name__)


def setup_adapter(wrapper, config) -> None:
    """Initialize adapter according to `config` and attach to wrapper.

    This function mutates `wrapper` to keep all public attributes identical
    to the previous implementation.
    """
    wrapper.adapter = config.adapter
    wrapper.adapter_config = None
    wrapper.lora_iter_to_adapter = {}

    if wrapper.adapter == "lora":
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library is required for LoRA support. Please install with: pip install peft"
            )
        base_grad = config.adapter_kwargs.pop("base_grad", True)
        adapter_grad = config.adapter_kwargs.pop("adapter_grad", True)
        
        wrapper.adapter_config = LoraConfig(**config.adapter_kwargs)  # type: ignore
        wrapper.simple_base_model = get_peft_model(wrapper.simple_base_model, wrapper.adapter_config)  # type: ignore
        # Reset adapter_kwargs
        config.adapter_kwargs["base_grad"] = base_grad
        config.adapter_kwargs["adapter_grad"] = adapter_grad
        
        if base_grad or not adapter_grad:
            for name, p in wrapper.simple_base_model.base_model.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad = adapter_grad
                else:
                    p.requires_grad = base_grad
            # Process LoRA layers if want to freeze them
            # for module in wrapper.simple_base_model.base_model.modules():
            #     if isinstance(module, LoraLayer):
            #         for p in module.parameters():
            #             p.requires_grad = True

        logger.info(f"LoRA enabled with config: {config.adapter_kwargs}")

    else:
        logger.info("Adapter disabled")


def configure_lora_for_iteration(wrapper, iter_depth: int) -> None:
    """Enable/disable and/or switch LoRA adapters per-iteration.

    Mirrors previous `_configure_lora_for_iteration` behavior.
    """
    if wrapper.adapter == "lora":
        if not hasattr(wrapper, "_lora_enabled_state"):
            wrapper.simple_base_model.base_model.disable_adapter_layers()
            wrapper._lora_enabled_state = False

        if iter_depth == 0:
            # Disable LoRA parameters only when transitioning from enabled -> disabled
            if wrapper._lora_enabled_state is not False:
                wrapper.simple_base_model.base_model.disable_adapter_layers()
                wrapper._lora_enabled_state = False
        elif iter_depth > 0:
            # Enable LoRA parameters only when transitioning from disabled -> enabled
            if wrapper._lora_enabled_state is not True:
                wrapper.simple_base_model.base_model.enable_adapter_layers()
                wrapper._lora_enabled_state = True
    


def save_adapter(wrapper, save_directory: str, **kwargs) -> None:
    """Save adapter-related weights and, where appropriate, the base model.

    Mirrors previous logic in `TaHForCausalLM.save_pretrained` for adapter branches.
    """
    if wrapper.adapter == "lora":
        # Save LoRA adapter(s)
        lora_dir = os.path.join(save_directory, "lora")
        os.makedirs(lora_dir, exist_ok=True)
        wrapper.simple_base_model.save_pretrained(lora_dir, **kwargs)

        # Directly save with cleaned keys by temporarily overriding state_dict method
        base_model = wrapper.simple_base_model.base_model.model
        original_state_dict = base_model.state_dict

        def cleaned_state_dict():
            """Return state_dict with cleaned keys (remove .base_layer)"""
            state_dict = original_state_dict()
            cleaned_dict = {}
            for key, value in state_dict.items():
                if 'lora' in key.lower():  # skip lora weights
                    continue
                cleaned_key = key.replace('.base_layer', '')
                cleaned_dict[cleaned_key] = value
            return cleaned_dict

        base_model.state_dict = cleaned_state_dict
        try:
            base_model.save_pretrained(save_directory, **kwargs)
        finally:
            base_model.state_dict = original_state_dict

        logger.info(f"Saving LoRA adapter and cleaned base model to {save_directory}")


    else:
        # Adapter disabled: directly save the base model
        wrapper.simple_base_model.save_pretrained(save_directory, **kwargs)
        logger.info(f"Saving base model to {save_directory}")


def load_adapter(wrapper, pretrained_model_name_or_path: str, final_config, *args, **kwargs) -> None:
    """Reload adapter-specific weights during `from_pretrained`.

    Mirrors previous logic for LoRA reload and cascade secondary model.
    """

    # Reload LoRA weights if needed
    if wrapper.adapter == "lora":
        logger.info("Reloading LoRA adapters from checkpoint after initialization")
        adapter_path = os.path.join(pretrained_model_name_or_path, "lora")
        base_grad = final_config.adapter_kwargs.pop("base_grad", True)
        adapter_grad = final_config.adapter_kwargs.pop("adapter_grad", True)
        wrapper.simple_base_model.load_adapter(adapter_path, adapter_name="default")
        logger.info(f"Reloaded LoRA adapter from {adapter_path}")
        # Set gradients based on parameter names: LoRA params get adapter_grad, others get base_grad
        for name, p in wrapper.simple_base_model.named_parameters():
            if "lora" in name.lower():
                p.requires_grad = adapter_grad
            else:
                p.requires_grad = base_grad



