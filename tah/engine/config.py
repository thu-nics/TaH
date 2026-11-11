"""Engine-owned configuration parsing; no HF/PEFT imports."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import torch


@dataclass
class TaHConfig:
    embedding_key: str = "model.embed_tokens"
    max_iter: int | None = None
    iter_decider: str | None = None
    iter_decider_kwargs: dict = field(default_factory=dict)
    eval_iter_decider: str | None = None
    eval_iter_decider_kwargs: dict = field(default_factory=dict)
    input_updater_kwargs: dict = field(default_factory=dict)
    adapter: str = "lora"
    adapter_kwargs: dict = field(default_factory=dict)
    train_loss: str | None = None
    train_loss_kwargs: dict = field(default_factory=dict)
    eval_loss: str | None = None
    eval_loss_kwargs: dict = field(default_factory=dict)


def decode_config(value):
    if isinstance(value, list):
        return [decode_config(v) for v in value]
    if isinstance(value, dict):
        if value.get("__dtype__"):
            dtypes = {str(d): d for d in (torch.float32, torch.float16, torch.bfloat16)}
            if value["__str__"] not in dtypes:
                raise ValueError(f"unsupported dtype: {value}")
            return dtypes[value["__str__"]]
        if value.get("__type__"):
            raise ValueError("engine configs cannot import arbitrary serialized Python types")
        return {k: decode_config(v) for k, v in value.items()}
    return value


def resolve_tah_config(path: Path, override=None) -> TaHConfig:
    defaults = asdict(TaHConfig())
    saved = path / "tah_config.json"
    raw = decode_config(json.loads(saved.read_text())) if saved.exists() else {}
    defaults.update({k: v for k, v in raw.items() if k in defaults})
    if override is not None:
        override = asdict(override) if is_dataclass(override) else dict(override)
        # The eval backend constructs TaHConfig from known fields. Its scalar
        # defaults participate in the overlay; None and {} mean "use saved".
        fields = asdict(TaHConfig(**{k: decode_config(v) for k, v in override.items() if k in defaults}))
        defaults.update({k: v for k, v in fields.items() if v is not None and v != {}})
    config = TaHConfig(**defaults)
    if not isinstance(config.max_iter, int) or isinstance(config.max_iter, bool) or config.max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")
    if config.embedding_key != "model.embed_tokens" or config.adapter != "lora":
        raise ValueError("only Qwen3 model.embed_tokens and LoRA are supported")
    if int(config.input_updater_kwargs.get("topk", 100)) < 1:
        raise ValueError("input updater topk must be positive")
    return config


def qwen_config(raw: dict) -> dict:
    if raw.get("model_type") != "qwen3" or raw.get("quantization_config"):
        raise ValueError("engine requires an unquantized dense Qwen3 checkpoint")
    if raw.get("use_sliding_window") or raw.get("pruned_heads") or raw.get("partial_rotary_factor", 1) != 1:
        raise ValueError("sliding attention, pruned heads and partial RoPE are unsupported")
    cfg = dict(raw)
    defaults = dict(num_key_value_heads=cfg["num_attention_heads"],
                    head_dim=cfg["hidden_size"] // cfg["num_attention_heads"],
                    attention_bias=False, rms_norm_eps=1e-6, rope_theta=10000.,
                    rope_scaling=None, max_position_embeddings=32768,
                    hidden_act="silu", tie_word_embeddings=False, pad_token_id=None)
    for k, v in defaults.items():
        if cfg.get(k) is None and k not in ("pad_token_id", "rope_scaling"):
            cfg[k] = v
        else:
            cfg.setdefault(k, v)
    for k in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "head_dim"):
        if not isinstance(cfg[k], int) or cfg[k] < 1:
            raise ValueError(f"invalid Qwen3 {k}: {cfg[k]}")
    if cfg["num_attention_heads"] % cfg["num_key_value_heads"] or cfg["head_dim"] % 2:
        raise ValueError("Qwen3 requires divisible GQA heads and an even head_dim")
    if cfg["hidden_act"] not in ("silu", "relu", "gelu", "gelu_new", "gelu_pytorch_tanh"):
        raise ValueError(f"unsupported hidden_act {cfg['hidden_act']}")
    return cfg
