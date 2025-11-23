"""
Utility functions for TaH model components.
"""
import importlib
from typing import Optional, List, Type, TYPE_CHECKING, Union
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
import transformers
import random
import numpy as np
import os

from accelerate import infer_auto_device_map
from accelerate.utils import get_balanced_memory


if TYPE_CHECKING:
    from tah.model.recurrent_transformer import TaHForCausalLM


def get_attr_by_path(root_obj, attr_path: str):
    """Resolve dotted attribute path on an object; returns None if any hop is missing."""
    current_obj = root_obj
    for name in attr_path.split('.'):
        if not hasattr(current_obj, name):
            return None
        current_obj = getattr(current_obj, name)
    return current_obj

def freeze_components(model, component_paths, accelerator):
    """Freeze parameters of components specified by dotted paths (e.g., 'model.cascade_model')."""
    if not component_paths:
        return
    for raw_path in component_paths:
        # Accept paths starting with 'model.' or direct module names under model
        path = raw_path
        if path.startswith('model.'):
            path = path[len('model.'):]

        target = get_attr_by_path(model, path)
        if target is None:
            accelerator.print(f"Warning: freeze_component '{raw_path}' not found on model.")
            continue

        params = list(target.parameters()) if hasattr(target, 'parameters') else []
        if not params:
            accelerator.print(f"Warning: freeze_component '{raw_path}' has no parameters to freeze.")
            continue

        for p in params:
            p.requires_grad = False
        num_params = sum(p.numel() for p in params)
        accelerator.print(f"Froze component '{raw_path}' ({num_params:,} params).")

def compute_trainable_param_size_gb(model) -> float:
    total_bytes = 0
    for p in model.parameters():
        if p.requires_grad:
            total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 3)


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    # Python built-in random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # Transformers
    transformers.set_seed(seed)
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds set to {seed}")


def get_attr_recursive(obj, attr_path):
    """
    Recursively get attribute from object using dot notation.

    Args:
        obj: Object to get attribute from
        attr_path: Dot-separated attribute path (e.g., "model.embed_tokens")

    Returns:
        The requested attribute

    Raises:
        AttributeError: If attribute doesn't exist
    """
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def class_string_to_type(cls_str: str) -> Type:
    """
    Convert a string of class to a class
    """
    module_name, class_name = cls_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def type_to_dict_string(obj):
    """
    Convert type objects to serializable strings
    """

    if isinstance(obj, dict):
        return {k: type_to_dict_string(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [type_to_dict_string(item) for item in obj]
    elif isinstance(obj, type):
        return {
            "__type__": True,
            "__module__": obj.__module__,
            "__name__": obj.__name__
        }
    elif isinstance(obj, torch.dtype):
        return {
            "__dtype__": True,
            "__str__": str(obj)
        }
    else:
        return obj

def dict_string_to_type(obj):
    """
    Convert serialized strings to type objects
    """

    if isinstance(obj, dict):
        if obj.get("__type__") is True:
            # This is a serialized type object
            module = importlib.import_module(obj["__module__"])
            return getattr(module, obj["__name__"])
        elif obj.get("__dtype__") is True:
            # This is a serialized torch.dtype object
            dtype_str = obj["__str__"]
            # Map string representations back to torch.dtype objects
            dtype_map = {
                "torch.float32": torch.float32,
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16
            }
            return dtype_map.get(dtype_str, torch.float32)  # Default to float32 if not found
        else:
            return {k: dict_string_to_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [dict_string_to_type(item) for item in obj]
    else:
        return obj


def sample_next_token(
    logits, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, do_sample=True
):
    """
    Sample next token from logits with various sampling strategies.

    Args:
        logits: Tensor of shape (batch_size, vocab_size) - logits for next token prediction
        temperature: Float > 0. Controls randomness. Lower = more deterministic. Default: 1.0
        top_p: Float between 0 and 1. Nucleus sampling - keep tokens with cumulative probability <= top_p. Default: 1.0
        top_k: Int >= 0. Keep only top k tokens. 0 means no filtering. Default: 0
        min_p: Float between 0 and 1. Remove tokens with probability < min_p * max_probability. Default: 0.0
        do_sample: Bool. If False, use greedy sampling (argmax). Default: True

    Returns:
        token_ids: Sampled token IDs as a tensor of shape (batch_size,)
    """
    # Handle greedy sampling cases
    if not do_sample:
        return torch.argmax(logits, dim=-1)

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Apply min_p filtering
    if min_p > 0.0:
        max_probs = torch.max(probs, dim=-1, keepdim=True)[0]  # (batch_size, 1)
        min_prob_threshold = min_p * max_probs
        probs = torch.where(probs >= min_prob_threshold, probs, torch.zeros_like(probs))
        # Renormalize
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    # Apply top_k filtering
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))  # Safety check
        top_k_probs, _ = torch.topk(probs, top_k, dim=-1)  # (batch_size, top_k)
        threshold = top_k_probs[..., -1:]  # (batch_size, 1) - the k-th largest value
        indices_to_remove = probs < threshold
        probs = torch.where(indices_to_remove, torch.zeros_like(probs), probs)
        # Renormalize
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    # Apply top_p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create a mask for indices to remove
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        probs = torch.where(indices_to_remove, torch.zeros_like(probs), probs)
        # Renormalize
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    # Sample from the filtered distribution
    return torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch_size,)


def TaHForCasualLM_generate(
    tah_model: "TaHForCausalLM",
    tokenizer: AutoTokenizer,
    model_inputs: dict,
    iter_count: Optional[torch.Tensor] = None,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    verbose: bool = True,
    **kwargs,
) -> tuple[list[list[int]], list[str]]:
    """
    Generation function for TaH model with sampling support for batched inputs.

    Args:
        tah_model: TaHForCausalLM model instance
        tokenizer: tokenizer instance
        model_inputs: dict containing 'input_ids', 'attention_mask', and other model inputs
        iter_count: torch.Tensor of shape (batch_size, seq_len) or None
        max_new_tokens: maximum number of new tokens to generate
        do_sample: whether to use sampling or greedy decoding
        temperature: sampling temperature (> 0.0)
        top_p: nucleus sampling probability threshold
        top_k: top-k sampling parameter (0 = disabled)
        min_p: minimum probability threshold relative to the most likely token
        verbose: whether to print debug output during generation
        **kwargs: additional keyword arguments to pass to the model
    Returns:
        generated_tokens: list of lists, each containing generated token IDs for each batch item
        generated_texts: list of decoded texts for each batch item
    """
    device = model_inputs["input_ids"].device
    batch_size = model_inputs["input_ids"].shape[0]
    tah_model.eval()

    # Initialize generation state
    cache = None
    output_tokens = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Keep track of current attention mask for extension
    current_attention_mask = model_inputs.get("attention_mask", None)

    if verbose:
        print("Input tokens with iteration counts:")

    with torch.no_grad():
        # Phase 1: Prefill - process initial input sequence
        outputs = _forward_and_display(
            tah_model,
            tokenizer,
            model_inputs,
            iter_count,
            cache,
            is_prefill=True,
            verbose=verbose,
            new_sequence=True,
            **kwargs,
        )
        cache = outputs.past_key_values

        if verbose:
            print("\n\nGenerating new tokens:")

        # Phase 2: Decoding - generate new tokens one by one
        for step in range(max_new_tokens):
            # Sample next token from current outputs for all batch items
            last_token_logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            next_token_ids = sample_next_token(
                logits=last_token_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                do_sample=do_sample,
            )  # (batch_size,)

            # Check for EOS token and update finished status
            if tokenizer.eos_token_id is not None:
                eos_mask = next_token_ids == tokenizer.eos_token_id
                finished = finished | eos_mask

            # Add tokens to output for non-finished sequences
            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    output_tokens[batch_idx].append(next_token_ids[batch_idx].item())

            # Check if all sequences are finished
            if finished.all():
                break

            # Prepare inputs for next token
            next_input_ids = next_token_ids.unsqueeze(1)
            next_model_inputs = {"input_ids": next_input_ids}

            # Extend attention mask
            if current_attention_mask is not None:
                new_token_mask = torch.ones(
                    batch_size, 1, dtype=current_attention_mask.dtype, device=device
                )
                current_attention_mask = torch.cat(
                    [current_attention_mask, new_token_mask], dim=1
                )
                next_model_inputs["attention_mask"] = current_attention_mask

            # Forward pass for next token
            outputs = _forward_and_display(
                tah_model,
                tokenizer,
                next_model_inputs,
                iter_count=None,  # Use automatic iteration from iter_decider
                cache=cache,
                is_prefill=False,
                verbose=verbose,
                new_sequence=False,
                **kwargs,
            )
            cache = outputs.past_key_values

    if verbose:
        print("\033[0m")  # Reset color

    # Decode generated texts
    generated_texts = [
        tokenizer.decode(tokens) if tokens else ""
        for tokens in output_tokens
    ]

    return output_tokens, generated_texts

def get_device_map(model: "TaHForCausalLM", device_map: Union[str, torch.device, int], dtype: torch.dtype):
    """
    Get the device map for the model. Input device map should choose from: 
        - a string: "auto", "balanced", "balanced_low_0", "sequential", or a device name like "cpu", "cuda:0"
        - a torch.device object
        - an int (device index)
        - a dict mapping module names to devices
    This function normalizes the device_map to a dict, or infers it if using auto-mapping.
    """
    # change device_map into a map if we passed an int, a str or a torch.device
    if isinstance(device_map, torch.device):
        device_map = {"": device_map}
    elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        try:
            device_map = {"": torch.device(device_map)}
        except RuntimeError:
            raise ValueError(
                "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
            )
    elif isinstance(device_map, int):
        if device_map < 0:
            raise ValueError(
                "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
            )
        else:
            device_map = {"": device_map}
    else:
        no_split_modules = model.simple_base_model._get_no_split_modules(device_map)
        no_split_modules.append(model.iter_decider.__class__.__name__)
        no_split_modules.append(model.input_updater.__class__.__name__)

        device_map_kwargs = {
            "no_split_module_classes": no_split_modules,
        }

        max_mem = get_balanced_memory(
            model,
            dtype=dtype,
            **device_map_kwargs,
        )
        device_map = infer_auto_device_map(
            model, 
            max_memory=max_mem, 
            dtype=dtype, 
            **device_map_kwargs
        )
    return device_map




def _forward_and_display(
    tah_model: "TaHForCausalLM",
    tokenizer: AutoTokenizer,
    model_inputs: dict,
    iter_count: Optional[torch.Tensor],
    cache: Optional[object],
    is_prefill: bool = False,
    new_sequence: bool = False,
    verbose: bool = True,
    **kwargs,
) -> object:
    """
    Uniform function for forward pass and token display for both prefill and decoding.

    Args:
        tah_model: TaH model instance
        tokenizer: tokenizer instance
        model_inputs: dict containing 'input_ids', 'attention_mask', and other model inputs
        iter_count: iteration counts for tokens
        cache: past key values cache
        is_prefill: whether this is the prefill phase or decoding phase
        new_sequence: whether this is a new sequence
        verbose: whether to display token colors and debug output

    Returns:
        Model outputs
    """
    # Extract input_ids and prepare forward pass arguments
    input_ids = model_inputs["input_ids"]

    # Prepare forward pass arguments with all available model inputs
    forward_kwargs = {
        "input_ids": input_ids,
        "iter_count": iter_count,
        "past_key_values": cache,
        "use_cache": True,
        "new_sequence": new_sequence,
        **kwargs,
    }

    # Add attention mask if available
    if "attention_mask" in model_inputs and model_inputs["attention_mask"] is not None:
        forward_kwargs["attention_mask"] = model_inputs["attention_mask"]

    # Add any other inputs that might be present
    for key, value in model_inputs.items():
        if key not in ["input_ids", "attention_mask"] and value is not None:
            forward_kwargs[key] = value

    # Forward pass
    # TODO: Add position ids
    outputs = tah_model(**forward_kwargs)

    # Display tokens with actual iteration counts, respecting attention mask
    if verbose:
        tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]

        # Get attention mask to avoid printing padding tokens
        attention_mask = model_inputs.get("attention_mask", None)
        if attention_mask is not None:
            # Only print tokens where attention mask is 1 (non-padding)
            valid_positions = attention_mask[0, -outputs.iter_count[0].shape[0]:] == 1
            tokens = [token for i, token in enumerate(tokens) if valid_positions[i]]

        if hasattr(outputs, "iter_count") and outputs.iter_count is not None:
            actual_counts = outputs.iter_count[0]
            if attention_mask is not None:
                actual_counts = actual_counts[valid_positions]
            for token, actual_count in zip(tokens, actual_counts):
                IterCountColors.print_token(token, actual_count.item())
        elif iter_count is not None:
            # Fallback to input iter_count
            iter_counts_to_use = iter_count[0]
            if attention_mask is not None:
                iter_counts_to_use = iter_counts_to_use[valid_positions]
            for token, count in zip(tokens, iter_counts_to_use):
                IterCountColors.print_token(token, count.item())
        else:
            # Default to 1 iteration
            for token in tokens:
                IterCountColors.print_token(token, 1)

    return outputs


class IterCountColors:
    """Utility class for handling iteration count based coloring."""

    @staticmethod
    def get_color(iter_count_val):
        """
        Get ANSI color code for given iteration count.

        Args:
            iter_count_val: The iteration count value

        Returns:
            ANSI color code string
        """
        colors = {
            1: "\033[0m",  # Default/reset (white)
            2: "\033[92m",  # Green
            3: "\033[94m",  # Blue
            4: "\033[91m",  # Red
            5: "\033[95m",  # Magenta
            6: "\033[93m",  # Yellow
        }
        return colors.get(iter_count_val, "\033[96m")  # Cyan for counts > 6

    @staticmethod
    def print_token(token_text, iter_count_val):
        """
        Print token with color based on iteration count.

        Args:
            token_text: The token text to print
            iter_count_val: The iteration count value for coloring
        """
        color = IterCountColors.get_color(iter_count_val)
        reset = "\033[0m"
        print(f"{color}{token_text}{reset}", end="", flush=True)

    @staticmethod
    def get_legend():
        """
        Get color legend string for iteration counts.

        Returns:
            String describing the color mapping with colors applied
        """
        reset = "\033[0m"
        legend_parts = [
            f"{IterCountColors.get_color(1)}Default=1 iter{reset}",
            f"{IterCountColors.get_color(2)}Green=2 iter{reset}",
            f"{IterCountColors.get_color(3)}Blue=3 iter{reset}",
            f"{IterCountColors.get_color(4)}Red=4 iter{reset}",
            f"{IterCountColors.get_color(5)}Magenta=5 iter{reset}",
            f"{IterCountColors.get_color(6)}Yellow=6 iter{reset}",
            f"{IterCountColors.get_color(7)}Cyan=7+ iter{reset}",
        ]
        return "Color legend: " + ", ".join(legend_parts)
