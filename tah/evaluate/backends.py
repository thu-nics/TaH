"""Per-backend model loaders + inference adapters for the eval driver.

Each ``setup_*`` function returns ``(model, inference_function)`` where
``inference_function(prompts: list[str]) -> list[(text, seconds)]``. This
shape is what the per-job runner consumes; it doesn't care which backend
produced the strings.

Backends:
  * ``setup_sglang`` — ``sgl.Engine`` for production-throughput inference.
  * ``setup_hf``     — vanilla ``AutoModelForCausalLM.generate``.
  * ``setup_tah``    — :class:`tah.model.tah_model.TaHForCausalLM` with
    its own iter-aware generation helper.
"""
from __future__ import annotations

import os
import time
from dataclasses import fields
from typing import Callable, Dict, Iterable, List, Tuple

import torch


InferenceFn = Callable[[List[str]], List[Tuple[str, float]]]


def time_inference(fn: Callable):
    """Run ``fn``, returning ``(result, elapsed_seconds)``. Uses CUDA events
    for tighter timing on GPU; falls back to wall clock otherwise."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        return out, start.elapsed_time(end) / 1000.0
    t0 = time.time()
    out = fn()
    return out, time.time() - t0


def warmup(model, tokenizer, backend: str) -> None:
    """One throwaway forward to JIT-compile / page caches per backend."""
    print(f"Warming up {backend} model…")
    if backend == "sglang":
        model.generate(["who are you?"], {
            "temperature": 0.6, "max_new_tokens": 100,
            "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        })
        return
    inputs = tokenizer("who are you?", return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        print(model.generate(**inputs, max_new_tokens=100, do_sample=True))


def cleanup(model, backend: str) -> None:
    if model is None:
        return
    if backend == "sglang":
        model.shutdown()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ────────────────────────────────────────────────────────────────────────────
# Backend setup
# ────────────────────────────────────────────────────────────────────────────


def setup_sglang(config: Dict, model_path: str, tokenizer, tp_size: int) -> Tuple[object, InferenceFn]:
    import sglang as sgl

    sampling = {
        "temperature": config["temperature"],
        "max_new_tokens": config["max_new_tokens"],
        "top_p": config["top_p"],
    }
    for opt in ("top_k", "min_p"):
        if config.get(opt) is not None:
            sampling[opt] = config[opt]

    print(f"Loading SGLang engine from: {model_path}")
    model = sgl.Engine(
        model_path=model_path,
        dtype=config.get("dtype", "bfloat16"),
        tp_size=tp_size,
        mem_fraction_static=config.get("mem_fraction_static", 0.90),
        host="127.0.0.1",
        port=int(os.getenv("SGLANG_NCCL_PORT", "30000")),
        attention_backend=config.get("attention_backend", "triton"),
    )
    warmup(model, tokenizer, "sglang")

    def infer(prompts: List[str]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        bs = config["batch_size"]
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            outputs, elapsed = time_inference(lambda: model.generate(batch, sampling))
            out.extend((o["text"], elapsed) for o in outputs)
        return out

    return model, infer


def setup_hf(config: Dict, model_path: str, tokenizer, tp_size: int = 1) -> Tuple[object, InferenceFn]:
    del tp_size  # accepted for dispatcher signature uniformity; HF uses device_map="auto"
    from transformers import AutoModelForCausalLM

    print(f"Loading Hugging Face model from: {model_path}  (visible CUDA: {torch.cuda.device_count()})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, config.get("dtype", "bfloat16")),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.get("use_flash_attention") else None,
        low_cpu_mem_usage=True,
    )
    gen_cfg = {
        "temperature": config["temperature"],
        "max_new_tokens": config["max_new_tokens"],
        "top_p": config["top_p"],
        "do_sample": config["temperature"] > 0.0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    for opt in ("top_k", "min_p"):
        if config.get(opt) is not None:
            gen_cfg[opt] = config[opt]
    warmup(model, tokenizer, "hf")

    def infer(prompts: List[str]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        bs = config["batch_size"]
        device = next(model.parameters()).device
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs, elapsed = time_inference(lambda: model.generate(**inputs, **gen_cfg))
            for j, gen in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                text = tokenizer.decode(gen[input_len:], skip_special_tokens=True)
                out.append((text, elapsed / len(outputs)))
        return out

    return model, infer


def setup_tah(config: Dict, model_path: str, tokenizer, tp_size: int = 1) -> Tuple[object, InferenceFn]:
    del tp_size  # accepted for dispatcher signature uniformity; TaH uses device_map="auto"
    from tah.model.tah_config import TaHConfig
    from tah.model.tah_model import TaHForCausalLM
    from tah.model.utils import TaHForCasualLM_generate

    print(f"Loading TaH model from: {model_path}  (visible CUDA: {torch.cuda.device_count()})")
    valid = {f.name for f in fields(TaHConfig)}
    override = TaHConfig(**{k: v for k, v in config.items() if k in valid})
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    model = TaHForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto",
        attn_implementation="sdpa", tah_config=override,
    ).to(dtype=dtype)

    def infer(prompts: List[str]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        bs = config["batch_size"]
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            (_, texts), elapsed = time_inference(lambda: TaHForCasualLM_generate(
                tah_model=model, tokenizer=tokenizer, model_inputs=inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=config["temperature"] > 0.0,
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config.get("top_k", 0),
                min_p=config.get("min_p", 0.0),
                verbose=False,
            ))
            per = elapsed / max(len(texts), 1)
            out.extend((t, per) for t in texts)
        return out

    return model, infer


_SETUP_BY_NAME = {"sglang": setup_sglang, "hf": setup_hf, "tah": setup_tah}


def setup_backend(
    backend: str, config: Dict, model_path: str, tokenizer, tp_size: int = 1,
) -> Tuple[object, InferenceFn]:
    """Dispatch to the named ``setup_*`` and forward all args."""
    try:
        setup = _SETUP_BY_NAME[backend]
    except KeyError:
        raise ValueError(f"unsupported backend {backend!r}; have {sorted(_SETUP_BY_NAME)}") from None
    return setup(config, model_path, tokenizer, tp_size)
