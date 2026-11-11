"""Local TaH v1 loader: safetensors/shards and restricted torch checkpoints.

Base weights stay [out, in], adapters A=[rank, in], B=[out, rank].
No model construction or execution uses Transformers or PEFT.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .config import qwen_config, resolve_tah_config
from .decider import build_decider
from .layers import Embedding, Linear
from .model import EagerTaHModel
from .rope import RotaryEmbedding


def tensor_files(root, stem):
    for suffix in ("safetensors", "bin"):
        name = stem if suffix == "safetensors" else ("pytorch_model" if stem == "model" else stem)
        index = root / f"{name}.{suffix}.index.json"
        file = root / f"{name}.{suffix}"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            for filename in sorted(set(mapping.values())):
                path = (root / filename).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("checkpoint shard path escapes model directory")
                yield path, {k for k, v in mapping.items() if v == filename}
            return
        if file.exists():
            yield file, None
            return
    raise FileNotFoundError(f"no {stem} checkpoint weights in {root}")


def tensors(root, stem):
    seen = set()
    for path, expected in tensor_files(root, stem):
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if expected is not None and keys != expected:
                    raise ValueError(f"shard index does not match tensors in {path}")
                for key in handle.keys():
                    if key in seen:
                        raise ValueError(f"duplicate checkpoint tensor {key}")
                    seen.add(key)
                    yield key, handle.get_tensor(key)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if expected is not None and set(state) != expected:
                raise ValueError(f"shard index does not match tensors in {path}")
            for key, value in state.items():
                if key in seen:
                    raise ValueError(f"duplicate checkpoint tensor {key}")
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"non-tensor checkpoint entry {key}")
                seen.add(key)
                yield key, value


def install(module, key, tensor, device, dtype):
    parent_name, _, name = key.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    expected = getattr(parent, name)
    if expected is not None and expected.shape != tensor.shape:
        raise ValueError(f"shape mismatch for {key}: expected {tuple(expected.shape)}, got {tuple(tensor.shape)}")
    tensor = tensor.to(device=device, dtype=dtype).contiguous()
    if name in parent._parameters:
        setattr(parent, name, nn.Parameter(tensor, requires_grad=False))
    elif name in parent._buffers:
        setattr(parent, name, tensor)
    else:
        raise ValueError(f"unknown weight {key}")


def load_base(model, root, device, dtype):
    expected = set(model.state_dict())
    seen = set()
    tied = model.config["tie_word_embeddings"]
    for key, tensor in tensors(root, "model"):
        if key not in expected:
            raise ValueError(f"unexpected Qwen3 weight {key}")
        install(model, key, tensor, device, dtype)
        seen.add(key)
    if tied:
        embed_key, head_key = "model.embed_tokens.weight", "lm_head.weight"
        if embed_key not in seen and head_key in seen:
            model.model.embed_tokens.weight = model.lm_head.weight
        elif embed_key in seen and head_key in seen:
            if not torch.equal(model.model.embed_tokens.weight, model.lm_head.weight):
                raise ValueError("tied embedding and lm_head weights disagree")
        model.lm_head.weight = model.model.embed_tokens.weight
        if embed_key in seen or head_key in seen:
            seen.update((embed_key, head_key))
    if expected - seen:
        raise ValueError(f"missing Qwen3 weights: {sorted(expected - seen)}")


def validate_adapter(config):
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError("only LoRA adapters are supported")
    for key in ("use_dora", "fan_in_fan_out", "modules_to_save", "layer_replication",
                "trainable_token_indices", "megatron_config"):
        if config.get(key):
            raise ValueError(f"unsupported LoRA variant: {key}")
    if config.get("bias", "none") not in ("none", "all", "lora_only"):
        raise ValueError("unsupported LoRA bias mode")
    if config.get("r", 8) < 1:
        raise ValueError("LoRA rank must be positive")


def pattern_value(pattern, name, default):
    # PEFT selects the first matching suffix regex, preserving JSON order.
    for key, value in pattern.items():
        if re.match(rf"(.*\.)?({key})$", name):
            return value
    return default


def load_adapter(model, root, device, dtype):
    config = json.loads((root / "adapter_config.json").read_text())
    validate_adapter(config)
    grouped = {}
    bias_weights = {}
    for key, value in tensors(root, "adapter_model"):
        if not key.startswith("base_model.model."):
            raise ValueError(f"unsupported adapter key {key}")
        key = key[len("base_model.model."):]
        match = re.fullmatch(r"(.+)\.lora_(A|B)(?:\.default)?\.(weight|bias)", key)
        embedding_match = re.fullmatch(r"(.+)\.lora_embedding_(A|B)(?:\.default)?", key)
        if match:
            name, side, leaf = match.groups()
            part = side if leaf == "weight" else "bias"
            if leaf == "bias" and side != "B":
                raise ValueError(f"unsupported adapter key {key}")
            grouped.setdefault(name, {})[part] = value
        elif embedding_match:
            name, side = embedding_match.groups()
            grouped.setdefault(name, {})[side] = value
        elif config.get("bias", "none") != "none" and key.endswith(".bias"):
            bias_weights[key.replace(".base_layer", "")] = value
        elif key.replace(".base_layer", "") in ("model.embed_tokens.weight", "lm_head.weight"):
            # PEFT sometimes saves vocabulary weights along with adapters.
            if not torch.equal(value.to(device=device, dtype=dtype), model.get_parameter(key.replace(".base_layer", ""))):
                raise ValueError(f"adapter vocabulary weight differs from base: {key}")
        else:
            raise ValueError(f"unsupported adapter key {key}")
    modules = dict(model.named_modules())
    targets = config.get("target_modules")
    def targeted(name):
        if config.get("exclude_modules"):
            exclusions = config["exclude_modules"]
            if (isinstance(exclusions, str) and re.fullmatch(exclusions, name)) or (
                isinstance(exclusions, list) and any(name == x or name.endswith("." + x) for x in exclusions)):
                return False
        if isinstance(targets, str):
            found = name != "lm_head" if targets == "all-linear" else bool(re.fullmatch(targets, name))
        else:
            found = any(name == x or name.endswith("." + x) for x in (targets or []))
        indices = config.get("layers_to_transform")
        if found and indices is not None:
            indices = [indices] if isinstance(indices, int) else indices
            patterns = config.get("layers_pattern") or ["layers"]
            patterns = [patterns] if isinstance(patterns, str) else patterns
            layer = next((re.search(rf"\.{p}\.(\d+)\.", "." + name) for p in patterns
                          if re.search(rf"\.{p}\.(\d+)\.", "." + name)), None)
            found = layer is not None and int(layer[1]) in indices
        return found
    required = {name for name, module in modules.items() if isinstance(module, (Linear, Embedding)) and targeted(name)}
    if set(grouped) != required:
        raise ValueError(f"adapter target/tensor mismatch: missing={sorted(required - set(grouped))}, unexpected={sorted(set(grouped) - required)}")
    for name, parts in grouped.items():
        module = modules[name]
        if isinstance(module, Embedding):
            raise ValueError("embedding-target LoRA is out of scope: adapters must target internal linear modules")
        if name == "lm_head":
            raise ValueError("lm_head LoRA is out of scope: adapters must target internal linear modules")
        rank = pattern_value(config.get("rank_pattern", {}), name, config.get("r", 8))
        alpha = pattern_value(config.get("alpha_pattern", {}), name, config.get("lora_alpha", 8))
        if not isinstance(rank, int) or rank < 1 or not math.isfinite(alpha):
            raise ValueError(f"invalid LoRA scaling for {name}")
        in_size = module.in_features if isinstance(module, Linear) else module.num_embeddings
        out_size = module.out_features if isinstance(module, Linear) else module.embedding_dim
        if set(parts) - {"A", "B", "bias"} or not {"A", "B"}.issubset(parts):
            raise ValueError(f"incomplete LoRA weights for {name}")
        if tuple(parts["A"].shape) != (rank, in_size) or tuple(parts["B"].shape) != (out_size, rank):
            raise ValueError(f"LoRA shape mismatch for {name}")
        module.lora_A = nn.Parameter(parts["A"].to(device=device, dtype=dtype).contiguous(), requires_grad=False)
        module.lora_B = nn.Parameter(parts["B"].to(device=device, dtype=dtype).contiguous(), requires_grad=False)
        if "bias" in parts:
            if not config.get("lora_bias") or not isinstance(module, Linear) or parts["bias"].shape != (out_size,):
                raise ValueError(f"unexpected LoRA B bias for {name}")
            module.lora_bias = nn.Parameter(parts["bias"].to(device=device, dtype=dtype), requires_grad=False)
        elif config.get("lora_bias"):
            raise ValueError(f"missing LoRA B bias for {name}")
        module.scaling = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)
    for key, value in bias_weights.items():
        install(model, key, value, device, dtype)
    # TaH constructs an adapter from its config, then load_adapter replaces
    # it using the saved PEFT config. The latter is authoritative at runtime.
    model.adapter_config = config


def load_decider(config, root, base_config, device, dtype):
    kwargs = dict(config.iter_decider_kwargs)
    path = Path(kwargs.pop("load_path", root)) / "iter_decider.bin"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    name = config.iter_decider or payload.get("class")
    args = kwargs  # reference passes the resolved kwargs even if empty
    with torch.device("meta"):
        decider = build_decider(name, args, config.max_iter)
    expected = set(decider.state_dict())
    state = {k: v for k, v in payload.get("state_dict", {}).items() if k not in args}
    extra = set(state) - expected
    missing = expected - set(state) - {"threshold"}
    if extra or missing:
        raise ValueError(f"incomplete/unsupported decider weights: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for key, value in state.items():
        install(decider, key, value, device, dtype)
    if "threshold" not in state:
        initial_dtype = args.get("dtype", torch.bfloat16) if name == "MLPIterDecider" else torch.float32
        threshold = torch.tensor(args.get("threshold", .5), dtype=initial_dtype)
        install(decider, "threshold", threshold, device, dtype)
    if name == "MLPIterDecider":
        if decider.hidden_states_size != base_config["hidden_size"] or decider.topk > base_config["vocab_size"]:
            raise ValueError("decider dimensions are incompatible with Qwen3")
        if any(i < 0 or i > base_config["num_hidden_layers"] for i in decider.hidden_states_layer_nums):
            raise ValueError("decider hidden-state index outside Qwen3 layers")
    return decider


def load_model(model_path, config_override=None, *, device="cpu", dtype=torch.bfloat16):
    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError("engine requires a local downloaded TaH checkpoint directory")
    if dtype not in (torch.bfloat16, torch.float32, torch.float16):
        raise ValueError("unsupported engine model dtype")
    cfg = qwen_config(json.loads((root / "config.json").read_text()))
    tah = resolve_tah_config(root, config_override)
    # No duplicate initialized base allocation, including for 1.7B models.
    with torch.device("meta"):
        model = EagerTaHModel(cfg, tah)
    load_base(model, root, device, dtype)
    model.model.rotary_emb = RotaryEmbedding(cfg).to(device=device, dtype=dtype)
    validate_adapter(tah.adapter_kwargs)
    load_adapter(model, root / "lora", device, dtype)
    model.iter_decider = load_decider(tah, root, cfg, device, dtype)
    eval_name = tah.eval_iter_decider
    if not eval_name or eval_name in ("iter_decider", "self.iter_decider"):
        model.eval_iter_decider = model.iter_decider
    elif eval_name == "IterLabelDecider":
        model.eval_iter_decider = build_decider(eval_name, tah.eval_iter_decider_kwargs, tah.max_iter).to(device=device, dtype=dtype)
    else:
        raise ValueError("eval decider must alias iter_decider or be a deterministic IterLabelDecider; fresh random MLPs have no checkpoint weights")
    return model.eval()
