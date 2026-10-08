"""
Utility functions for TaH-adjacent modeling code.
"""

import importlib
import os
import random
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import infer_auto_device_map
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from tah2.model.tah_model import TaHForCausalLM


def get_attr_by_path(root_obj, attr_path: str):
    current_obj = root_obj
    for name in attr_path.split("."):
        if not hasattr(current_obj, name):
            return None
        current_obj = getattr(current_obj, name)
    return current_obj


def freeze_components(model, component_paths, accelerator):
    if not component_paths:
        return
    for raw_path in component_paths:
        path = raw_path[len("model.") :] if raw_path.startswith("model.") else raw_path
        target = get_attr_by_path(model, path)
        if target is None:
            accelerator.print(
                f"Warning: freeze_component '{raw_path}' not found on model."
            )
            continue
        params = list(target.parameters()) if hasattr(target, "parameters") else []
        if not params:
            accelerator.print(
                f"Warning: freeze_component '{raw_path}' has no parameters to freeze."
            )
            continue
        for p in params:
            p.requires_grad = False
        accelerator.print(
            f"Froze component '{raw_path}' ({sum(p.numel() for p in params):,} params)."
        )


def compute_trainable_param_size_gb(model) -> float:
    return sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    ) / (1024**3)


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"All random seeds set to {seed}")


def get_attr_recursive(obj, attr_path):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def type_to_dict_string(obj):
    if isinstance(obj, dict):
        return {k: type_to_dict_string(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [type_to_dict_string(item) for item in obj]
    if isinstance(obj, type):
        return {
            "__type__": True,
            "__module__": obj.__module__,
            "__name__": obj.__name__,
        }
    if isinstance(obj, torch.dtype):
        return {"__dtype__": True, "__str__": str(obj)}
    return obj


def dict_string_to_type(obj):
    if isinstance(obj, dict):
        if obj.get("__type__") is True:
            module_name = obj["__module__"]
            # Checkpoints saved before the package rename may retain the old namespace.
            if module_name == "tah" or module_name.startswith("tah."):
                module_name = "tah2" + module_name[3:]
            return getattr(importlib.import_module(module_name), obj["__name__"])
        if obj.get("__dtype__") is True:
            return {
                "torch.float32": torch.float32,
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
            }.get(obj["__str__"], torch.float32)
        return {k: dict_string_to_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dict_string_to_type(item) for item in obj]
    return obj


def sample_next_token(
    logits, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, do_sample=True
):
    if temperature == 0.0 or not do_sample:
        return torch.argmax(logits, dim=-1)
    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)
    if min_p > 0.0:
        probs = torch.where(
            probs >= min_p * torch.max(probs, dim=-1, keepdim=True)[0],
            probs,
            torch.zeros_like(probs),
        )
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        threshold = torch.topk(probs, top_k, dim=-1)[0][..., -1:]
        probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        sorted_indices_to_remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        probs = torch.where(indices_to_remove, torch.zeros_like(probs), probs)
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


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
    device = model_inputs["input_ids"].device
    batch_size = model_inputs["input_ids"].shape[0]
    tah_model.eval()
    cache = None
    output_tokens = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    current_attention_mask = model_inputs.get("attention_mask", None)

    if verbose:
        print("Input tokens with iteration counts:")

    with torch.no_grad():
        outputs = _forward_and_display(
            tah_model,
            tokenizer,
            model_inputs,
            iter_count,
            cache,
            verbose=verbose,
            **kwargs,
        )
        cache = outputs.past_key_values

        if verbose:
            print("\n\nGenerating new tokens:")

        for token_index in range(max_new_tokens):
            next_token_ids = sample_next_token(
                logits=outputs.logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                do_sample=do_sample,
            )
            if tokenizer.eos_token_id is not None:
                finished = finished | (next_token_ids == tokenizer.eos_token_id)
            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    output_tokens[batch_idx].append(next_token_ids[batch_idx].item())
            if finished.all() or token_index + 1 == max_new_tokens:
                break

            next_model_inputs = {"input_ids": next_token_ids.unsqueeze(1)}
            if current_attention_mask is not None:
                current_attention_mask = torch.cat(
                    [
                        current_attention_mask,
                        torch.ones(
                            batch_size,
                            1,
                            dtype=current_attention_mask.dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
                next_model_inputs["attention_mask"] = current_attention_mask
            outputs = _forward_and_display(
                tah_model,
                tokenizer,
                next_model_inputs,
                None,
                cache,
                verbose=verbose,
                **kwargs,
            )
            cache = outputs.past_key_values

    if verbose:
        print("\033[0m")

    return output_tokens, [
        tokenizer.decode(tokens) if tokens else "" for tokens in output_tokens
    ]


def get_device_map(
    model: "TaHForCausalLM",
    device_map: Union[str, torch.device, int],
    dtype: torch.dtype,
):
    if isinstance(device_map, torch.device):
        return {"": device_map}
    if isinstance(device_map, str) and device_map not in [
        "auto",
        "balanced",
        "balanced_low_0",
        "sequential",
    ]:
        try:
            return {"": torch.device(device_map)}
        except RuntimeError as e:
            raise ValueError(
                "When passing device_map as a string, the value needs to be a device name "
                f"or auto-mapping mode, but found {device_map}."
            ) from e
    if isinstance(device_map, int):
        if device_map < 0:
            raise ValueError("You can't pass device_map as a negative int.")
        return {"": device_map}
    if isinstance(device_map, dict):
        return device_map

    no_split_modules = model.simple_base_model._get_no_split_modules(device_map)
    no_split_modules.append(model.iter_decider.__class__.__name__)
    if getattr(model, "eval_iter_decider", None) is not None:
        no_split_modules.append(model.eval_iter_decider.__class__.__name__)
    no_split_modules.append(model.input_updater.__class__.__name__)

    kwargs = {"no_split_module_classes": no_split_modules}
    max_mem = get_balanced_memory(model, dtype=dtype, **kwargs)
    return infer_auto_device_map(model, max_memory=max_mem, dtype=dtype, **kwargs)


def _forward_and_display(
    tah_model: "TaHForCausalLM",
    tokenizer: AutoTokenizer,
    model_inputs: dict,
    iter_count: Optional[torch.Tensor],
    cache: Optional[object],
    verbose: bool = True,
    **kwargs,
) -> object:
    input_ids = model_inputs["input_ids"]
    forward_kwargs = {
        "input_ids": input_ids,
        "iter_count": iter_count,
        "past_key_values": cache,
        "use_cache": True,
        **kwargs,
    }
    if model_inputs.get("attention_mask", None) is not None:
        forward_kwargs["attention_mask"] = model_inputs["attention_mask"]
    for key, value in model_inputs.items():
        if key not in ["input_ids", "attention_mask"] and value is not None:
            forward_kwargs[key] = value

    outputs = tah_model(**forward_kwargs)
    if not verbose:
        return outputs

    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
    attention_mask = model_inputs.get("attention_mask", None)
    if attention_mask is not None:
        valid_positions = attention_mask[0, -outputs.iter_count[0].shape[0] :] == 1
        tokens = [token for i, token in enumerate(tokens) if valid_positions[i]]

    if hasattr(outputs, "iter_count") and outputs.iter_count is not None:
        actual_counts = outputs.iter_count[0]
        if attention_mask is not None:
            actual_counts = actual_counts[valid_positions]
        for token, actual_count in zip(tokens, actual_counts):
            IterCountColors.print_token(token, actual_count.item())
    elif iter_count is not None:
        iter_counts_to_use = iter_count[0]
        if attention_mask is not None:
            iter_counts_to_use = iter_counts_to_use[valid_positions]
        for token, count in zip(tokens, iter_counts_to_use):
            IterCountColors.print_token(token, count.item())
    else:
        for token in tokens:
            IterCountColors.print_token(token, 1)
    return outputs


class IterCountColors:
    @staticmethod
    def get_color(iter_count_val):
        return {
            1: "\033[0m",
            2: "\033[92m",
            3: "\033[94m",
            4: "\033[91m",
            5: "\033[95m",
            6: "\033[93m",
        }.get(iter_count_val, "\033[96m")

    @staticmethod
    def print_token(token_text, iter_count_val):
        print(
            f"{IterCountColors.get_color(iter_count_val)}{token_text}\033[0m",
            end="",
            flush=True,
        )

    @staticmethod
    def get_legend():
        reset = "\033[0m"
        return "Color legend: " + ", ".join(
            [
                f"{IterCountColors.get_color(1)}Default=1 iter{reset}",
                f"{IterCountColors.get_color(2)}Green=2 iter{reset}",
                f"{IterCountColors.get_color(3)}Blue=3 iter{reset}",
                f"{IterCountColors.get_color(4)}Red=4 iter{reset}",
                f"{IterCountColors.get_color(5)}Magenta=5 iter{reset}",
                f"{IterCountColors.get_color(6)}Yellow=6 iter{reset}",
                f"{IterCountColors.get_color(7)}Cyan=7+ iter{reset}",
            ]
        )
