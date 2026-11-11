"""Per-backend model loaders + inference adapters for the eval driver.

Each ``setup_*`` function returns ``(model, inference_function)``. Legacy
inference functions accept ``prompts`` and return ``[(text, seconds)]``.
Callback backends may set ``infer.supports_callback = True`` and accept
``infer(prompts, on_result=callback)`` instead. Such a function receives the
complete prompt list once and calls ``callback(index, result)`` as results
arrive, where a result is ``(text, seconds)`` or
``(text, seconds, pass_counts)``. ``pass_counts`` is an optional sequence of
per-generated-token pass counts. The callback path lets the harness persist
each result immediately; existing backends need no changes.

Backends:
  * ``setup_sglang`` — ``sgl.Engine`` for production-throughput inference.
  * ``setup_hf``     — vanilla ``AutoModelForCausalLM.generate``.
  * ``setup_tah``    — :class:`tah.model.tah_model.TaHForCausalLM` with
    its own iter-aware generation helper.
  * ``setup_tah_engine`` — engine-owned generation with indexed callbacks.
"""
from __future__ import annotations

import inspect
import os
import sys
import threading
import time
from pathlib import Path
from dataclasses import fields
from typing import Callable, Dict, List, Optional, Tuple

import torch


InferenceFn = Callable[..., Optional[List[Tuple]]]


class _EngineTokenCounter:
    """Read CPU request state without hooks, engine locks or tensor access.

    The frozen engine keeps its complete, fixed ``states`` list local to generate.
    Locate the generating thread's frame once (also through inference_mode's
    wrapper), retain only the states, and drop all frame references. CPython's GIL
    makes Python int reads / list lengths safe. Read wasted *before* the length:
    restart clears generated before adding wasted, so a racing read can only lag,
    never double-count. A high-water mark hides that short accounting gap.
    """

    def __init__(self, engine):
        self.engine = engine
        self.code = inspect.unwrap(engine.generate).__code__
        self.thread_id = threading.get_ident()
        self.started = time.perf_counter()
        self.states = None
        self.total = 0
        self.final = None

    def snapshot(self):
        if self.final is not None:
            return self.final
        states = self.states
        if states is None:
            frame = sys._current_frames().get(self.thread_id)
            try:
                while frame is not None:
                    if frame.f_code is self.code and frame.f_locals.get('self') is self.engine:
                        states = frame.f_locals.get('states')
                        self.states = states
                        break
                    frame = frame.f_back
            finally:
                del frame
        if states is None:
            return None
        total = 0
        for state in states:
            wasted = state.wasted_tokens
            total += wasted + len(state.generated)
        self.total = max(self.total, total)
        return self.final if self.final is not None else (time.perf_counter() - self.started, self.total)

    def finish(self, total=None):
        snapshot = self.snapshot()
        elapsed = time.perf_counter() - self.started
        self.final = (elapsed, total if total is not None else snapshot[1] if snapshot else 0)
        self.states = None


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
            (_, texts, pass_counts), elapsed = time_inference(lambda: TaHForCasualLM_generate(
                tah_model=model, tokenizer=tokenizer, model_inputs=inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=config["temperature"] > 0.0,
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config.get("top_k", 0),
                min_p=config.get("min_p", 0.0),
                verbose=False,
                return_pass_counts=True,
            ))
            per = elapsed / max(len(texts), 1)
            out.extend((t, per, counts) for t, counts in zip(texts, pass_counts))
        return out

    return model, infer


def setup_tah_engine(config: Dict, model_path: str, tokenizer, tp_size: int = 1) -> Tuple[object, InferenceFn]:
    """One engine for all job prompts, with indexed completion callbacks."""
    if tp_size != 1:
        raise ValueError("tah_engine requires --tp_size_per_job 1 (one GPU per engine)")
    try:
        from tah.engine import TaHEngine, Request, SamplingParams, check_environment
        check_environment()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(f"tah_engine environment check failed: {exc}\n"
                           "Use --backend tah for the correctness reference backend.") from exc
    if config.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("tah_engine uses bfloat16; set dtype: bfloat16")
    path = model_path
    if not Path(path).is_dir():
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_path)
    options = dict(config.get("engine") or {})
    options.setdefault("max_batch_size", 64)
    count = config.get("_prompt_count")
    if count is not None:
        options["max_batch_size"] = min(options["max_batch_size"], max(1, count))
        if options.get("graph_batch_sizes") is not None:
            options["graph_batch_sizes"] = [b for b in options["graph_batch_sizes"]
                                             if b <= options["max_batch_size"]]
    options.setdefault("max_new_tokens", config["max_new_tokens"])
    options.setdefault("seed", config.get("_random_seed", 42))
    model = TaHEngine(path, config_override=config, **options)
    sampling = SamplingParams(temperature=config["temperature"], top_p=config["top_p"],
                              top_k=config.get("top_k", 0), min_p=config.get("min_p", 0.0),
                              do_sample=config["temperature"] > 0)
    warmed = False
    token_counter = None

    def infer(prompts, on_result=None):
        nonlocal warmed, token_counter
        token_counter = None
        # Exercise the real maximum resident shape outside measured inference.
        ids = [tokenizer.encode(p) for p in prompts]
        if ids and not warmed:
            model.generate([Request(ids[i % len(ids)], 4, i)
                            for i in range(model.max_batch_size)], sampling)
            warmed = True
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        callback_seconds = 0.0

        def completed(result):
            nonlocal callback_seconds
            started = time.perf_counter()
            if on_result is not None:
                on_result(result.request_id, dict(text=result.text,
                    seconds=result.timing["total_seconds"],
                    pass_counts=result.generated_pass_counts, token_ids=result.token_ids,
                    prompt_pass_counts=result.prompt_pass_counts,
                    finish_reason=result.finish_reason, timing=result.timing))
            callback_seconds += time.perf_counter() - started

        requests = [Request(p, config["max_new_tokens"], model.seed + i)
                    for i, p in enumerate(ids)]
        token_counter = _EngineTokenCounter(model)
        try:
            results = model.generate(requests, sampling, completed)
        except BaseException:
            token_counter.finish()
            raise
        token_counter.finish(model.last_run['generated_tokens'] + model.last_run.get('wasted_tokens', 0))
        infer.telemetry = dict(engine_run=model.last_run.copy(), memory_budget=model.memory_budget,
            engine_options=options, callback_seconds=callback_seconds,
            generation_seconds=model.last_run["wall_seconds"] - callback_seconds,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(model.device),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(model.device))
        return results

    infer.supports_callback = True
    infer.telemetry = {}
    infer.progress_snapshot = lambda: token_counter.snapshot() if token_counter is not None else None
    return model, infer


_SETUP_BY_NAME = {"sglang": setup_sglang, "hf": setup_hf, "tah": setup_tah,
                  "tah_engine": setup_tah_engine}


def setup_backend(
    backend: str, config: Dict, model_path: str, tokenizer, tp_size: int = 1,
) -> Tuple[object, InferenceFn]:
    """Dispatch to the named ``setup_*`` and forward all args."""
    try:
        setup = _SETUP_BY_NAME[backend]
    except KeyError:
        raise ValueError(f"unsupported backend {backend!r}; have {sorted(_SETUP_BY_NAME)}") from None
    return setup(config, model_path, tokenizer, tp_size)
