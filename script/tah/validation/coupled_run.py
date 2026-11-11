"""R1-O2 generation with the production engine and the real reference loop.

prepare freezes harness prompts, grading answers and request seeds; run executes
one implementation. No global seed can substitute for the per-request uniforms.
Completed blocks are resumable; interrupted blocks are replayed in their entirety.

Host RAM: ``run engine`` builds one engine with the evaluation's default
``host_swap_bytes`` (8 GiB), pinned in full at startup, so its process needs the
engine baseline RSS + 8 GiB. ``run reference``/``run oracle`` pin no swap arena.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .common import T, TOP_K, TOP_P, load_reference, model_sha256, write_json
from .coupling import multinomial_patch, uniforms
from .fp64_attention import POLICY as FP64_ATTENTION_POLICY, fp64_attention_guard

SEED = 42
THRESHOLD = 0.8984375
MAX_TOKENS = 4096
RECIPE = "script/tah/recipes/qwen3_1.7/eval_tah.yaml"
ORACLE_CONSTANTS = "bf16 weights, buffers (including inv_freq), threshold; cast arithmetic up; TF32 off"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hash():
    """Freeze production and validation code across implementations and resumes."""
    files = [p for directory in ("tah/engine", "tah/model", "tah/evaluate", "script/tah/validation")
             for p in Path(directory).glob("*.py")]
    return digest({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})


def prepare(model, output):
    """Tokenizer only; standardised row order and prompts come from the harness."""
    from transformers import AutoTokenizer
    from tah.evaluate.datasets import load_combined_dataset
    from tah.evaluate.jobs import _make_prompt
    tokenizer = AutoTokenizer.from_pretrained(model)
    rows, _ = load_combined_dataset("math500,gpqa")
    math = [r for r in rows if r["_source_dataset"] == "math500"][:200]
    gpqa = [r for r in rows if r["_source_dataset"] == "gpqa"]
    if len(math) != 200 or len(gpqa) != 198:
        raise ValueError("R1-O2 requires MATH500 [0,200) and GPQA-Diamond 198")
    problems = []
    for i, r in enumerate(math + gpqa):
        rendered = _make_prompt(r["question"], tokenizer, is_code=False)
        problems.append(dict(id=r["id"], dataset=r["_source_dataset"], index=r["_original_index"],
                             seed=SEED + i, prompt=tokenizer.encode(rendered), prompt_text=rendered,
                             answer=r["answer"]))
    manifest = dict(version=1, protocol="R1-O2", uniform_seed=SEED,
                    seed_mapping="42 + index in MATH500[0:200] followed by GPQA[0:198]",
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    problems=problems, oracle_ids=[p["id"] for p in problems if p["index"] < 50])
    validate_manifest(manifest)
    if Path(output).exists() and json.loads(Path(output).read_text()) != manifest:
        raise ValueError("refusing to replace a different manifest")
    write_json(output, manifest)


def validate_manifest(manifest):
    if manifest.get("protocol") == "R1-O2-C1-short":
        from .short_c1 import validate
        validate(manifest)
        return
    problems = manifest["problems"]
    ids = [p["id"] for p in problems]
    if not problems or len(set(ids)) != len(ids) or any(not p["prompt"] for p in problems):
        raise ValueError("empty/duplicate problems or empty prompts")
    if not set(manifest["oracle_ids"]) <= set(ids):
        raise ValueError("unknown oracle problem")
    if manifest.get("protocol") == "fixture":
        return
    expected = [(ds, j) for ds, n in (("math500", 200), ("gpqa", 198)) for j in range(n)]
    if [(p["dataset"], p["index"]) for p in problems] != expected:
        raise ValueError("manifest differs from the registered workload/order")
    if [p["seed"] for p in problems] != list(range(SEED, SEED + 398)):
        raise ValueError("manifest differs from the registered request seed mapping")
    if manifest["oracle_ids"] != [p["id"] for p in problems if p["index"] < 50]:
        raise ValueError("oracle must cover the first 50 problems of each dataset")
    if manifest.get("protocol") != "R1-O2":
        raise ValueError("unknown protocol")


def grade(problem, text):
    from tah.evaluate.jobs import _score_one
    predicted, has_answer, correct = _score_one(text, problem["answer"], problem["dataset"], is_code=False)
    return dict(predicted_answer=predicted, has_answer=has_answer, is_correct=correct)


def sample_path(root, pid):
    # A hash permits arbitrary harness ids without interpreting them as paths.
    return Path(root) / "samples" / (hashlib.sha256(pid.encode()).hexdigest() + ".json")


def trace_path(root, pid):
    return sample_path(root, pid).with_suffix(".npz")


def cdf_details(probs, u, token):
    """Oracle distances to BOTH adjacent CDF boundaries, in unnormalised units.

    Sort/order/arithmetic match inverse_cdf_draw. Normalised margin divides by
    total; no tolerance or near-tie gate is inferred from these diagnostics.
    """
    ranked, order = probs.double().sort(dim=-1, descending=True, stable=True)
    cdf = ranked.cumsum(-1)
    slot = (order == token[:, None]).long().argmax(-1)
    high = cdf.gather(-1, slot[:, None]).squeeze(-1)
    low = torch.where(slot > 0, cdf.gather(-1, (slot - 1).clamp_min(0)[:, None]).squeeze(-1), 0.)
    total = cdf[:, -1]
    target = u * total
    values = torch.stack((low, high, total, target,
                          torch.minimum((target - low).abs(), (high - target).abs()),
                          ranked[:, 0] - ranked[:, 1]), -1).cpu().numpy()
    return values


def reference_batch(model, tokenizer, problems, pad_id, max_tokens=MAX_TOKENS, record_oracle=False):
    """Real evaluation loop, left padding, fixed row identities even after EOS."""
    from tah.engine.reference_capture import pack
    from tah.model.utils import TaHForCasualLM_generate
    inputs, _ = pack([p["prompt"] for p in problems], pad_id, model.device, side="left")
    seeds = [p["seed"] for p in problems]
    step, ended = 0, [False] * len(problems)
    traces = [[] for _ in problems]
    sampled = [[] for _ in problems]
    last_counts, prompt_counts = [], []

    def forward_hook(module, args, outputs):
        nonlocal last_counts
        last_counts = outputs.iter_count[:, -1].tolist()
        if not prompt_counts:
            counts = outputs.iter_count.tolist()
            prompt_counts.extend([c[-len(p["prompt"]):] for c, p in zip(counts, problems)])

    def provider(b):
        nonlocal step
        if b != len(problems):
            raise AssertionError("reference changed row identities")
        result = uniforms(seeds, [step] * b, device=model.device)
        step += 1
        return result

    def observe(probs, u, token):
        details = cdf_details(probs, u, token) if record_oracle else None
        for r, t in enumerate(token.tolist()):
            if ended[r]:
                continue
            sampled[r].append(t)
            if details is not None:
                traces[r].append([*details[r].tolist(), last_counts[r]])
            ended[r] = tokenizer.eos_token_id is not None and t == tokenizer.eos_token_id

    hook = model.register_forward_hook(forward_hook)
    try:
        with multinomial_patch(provider, observe):
            tokens, texts, counts = TaHForCasualLM_generate(
                model, tokenizer, inputs, max_new_tokens=max_tokens, temperature=T,
                top_p=TOP_P, top_k=TOP_K, min_p=0., verbose=False, return_pass_counts=True)
    finally:
        hook.remove()
    results = []
    for r, p in enumerate(problems):
        results.append(dict(id=p["id"], seed=p["seed"], token_ids=tokens[r], text=texts[r],
                            sampled_token_ids=sampled[r], pass_counts=counts[r],
                            prompt_pass_counts=prompt_counts[r], finish_reason="eos" if ended[r] else "length"))
    return results, traces


def read_recipe(path, fixture):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text()) if path else {}
    if not fixture:
        for k, v in dict(temperature=T, top_p=TOP_P, top_k=TOP_K, min_p=0., max_new_tokens=MAX_TOKENS).items():
            if cfg.get(k) != v:
                raise ValueError(f"eval config must have {k}={v}")
        threshold = float(torch.tensor(cfg["iter_decider_kwargs"]["threshold"], dtype=torch.bfloat16))
        if threshold != THRESHOLD:
            raise ValueError("wrong effective decider threshold")
    return cfg


def run(args):
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    short = manifest.get("protocol") == "R1-O2-C1-short"
    fixture = manifest.get("protocol") == "fixture" or (short and manifest.get("fixture", False))
    if args.device == "cpu":
        cfg = json.loads((args.model / "config.json").read_text())
        if not fixture or cfg["hidden_size"] > 128 or cfg["vocab_size"] > 1024:
            raise ValueError("CPU execution is restricted to tiny fixture models")
    if short and (args.max_new_tokens != 512 or args.pilot):
        raise ValueError("short C1 requires cap512 and no pilot override")
    if args.max_new_tokens != MAX_TOKENS and not (args.pilot or fixture or short):
        raise ValueError("a token-cap override is allowed only for pilots/fixtures")
    if args.block < 32 and args.implementation == "engine" and not (args.pilot or fixture):
        raise ValueError("engine blocks must have at least 32 requests")
    cfg = read_recipe(args.eval_config, fixture)
    problems = manifest["problems"]
    if args.implementation == "oracle":
        problems = [p for p in problems if p["id"] in manifest["oracle_ids"]]
    if args.pilot:
        problems = [p for p in problems if p["index"] < args.pilot]
    subset = getattr(args, "subset", "all")
    if short:
        if subset not in manifest["subsets"]:
            raise ValueError("short C1 requires an explicit frozen stage subset")
        wanted = set(manifest["subsets"][subset])
        problems = [p for p in problems if p["id"] in wanted]
    elif subset != "all":
        raise ValueError("subset applies only to the short C1 protocol")
    batch_size = args.oracle_batch if args.implementation == "oracle" else 8
    block_size = args.block if args.implementation == "engine" else batch_size
    config = dict(version=1, implementation=args.implementation, manifest_sha256=digest(manifest),
                  model_sha256=model_sha256(args.model), eval_config=cfg, device=args.device,
                  source_sha256=source_hash(),
                  eval_config_sha256=hashlib.sha256(args.eval_config.read_bytes()).hexdigest()
                  if args.eval_config else None,
                  dtype=args.oracle_dtype if args.implementation == "oracle" else "bf16",
                  oracle_constants=ORACLE_CONSTANTS, block_size=block_size, batch=batch_size,
                  oracle_batch=batch_size if args.implementation == "oracle" else None,
                  max_new_tokens=args.max_new_tokens, pilot=args.pilot, fixture=fixture,
                  sampling=dict(temperature=T, top_p=TOP_P, top_k=TOP_K, min_p=0.),
                  seeds={p["id"]: p["seed"] for p in problems},
                  problems=[p["id"] for p in problems], oracle_ids=manifest["oracle_ids"],
                  engine_options=dict(cuda_graphs=True, overlap=True, compile=False,
                                      max_batch_size=min(64, args.block)))
    if short:
        config["subset"] = subset
    if config["dtype"] == "fp64":
        config["fp64_attention"] = FP64_ATTENTION_POLICY
    from .capture import environment
    config["environment"] = environment()
    out = args.out
    path = out / "run.json"
    if path.exists():
        info = json.loads(path.read_text())
        if info["config"] != config:
            raise ValueError("run directory belongs to a different configuration")
    else:
        info = dict(config=config, completed_blocks=[], telemetry={}, status="running")
        write_json(path, info)
    blocks = [problems[i:i + block_size] for i in range(0, len(problems), block_size)]
    # Validate saved completed blocks before skipping; partial samples are replaced.
    for b in info["completed_blocks"]:
        for p in blocks[b]:
            saved = json.loads(sample_path(out, p["id"]).read_text())
            if saved["id"] != p["id"] or saved["seed"] != p["seed"] or "is_correct" not in saved:
                raise ValueError("corrupt saved sample")
            if args.implementation == "oracle" and not trace_path(out, p["id"]).exists():
                raise ValueError("missing oracle trace")
    if len(info["completed_blocks"]) == len(blocks):
        return
    info["status"] = "running"
    write_json(path, info)
    if fixture:
        from types import SimpleNamespace
        tokenizer = SimpleNamespace(eos_token_id=manifest["eos_token_id"],
                                    decode=lambda ids: " ".join(map(str, ids)))
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.implementation == "engine":
        from tah.engine import TaHEngine, Request, SamplingParams
        model = TaHEngine(args.model, config_override=cfg, device=args.device, cuda_graphs=True,
                          overlap=True, compile=False, max_batch_size=min(64, block_size),
                          eos_token_id=manifest["eos_token_id"],
                          **(dict(kv_capacity=10000, prefill_chunk_size=4) if fixture else {}))
        if args.device != "cpu" and not (model.cuda_graphs and model.overlap):
            raise AssertionError("production graphs and overlap must be enabled")
        if not fixture and float(model.decider.threshold) != THRESHOLD:
            raise AssertionError("engine effective threshold differs from protocol")
    else:
        model = load_reference(args.model, args.eval_config,
                               args.oracle_dtype if args.implementation == "oracle" else "bf16", args.device)
        if not fixture and float(model._active_iter_decider.threshold) != THRESHOLD:
            raise AssertionError("reference effective threshold differs from protocol")

    for b, block in enumerate(blocks):
        if b in info["completed_blocks"]:
            continue
        started = time.monotonic()

        def persist(result):
            p = next(p for p in block if p["id"] == result["id"])
            result.update(dataset=p["dataset"], index=p["index"], **grade(p, result["text"]))
            write_json(sample_path(out, p["id"]), result)

        if args.implementation == "engine":
            # Warm maximum shape exactly as the harness does; no sampler patches,
            # observers or scheduling hooks on the production measurement path.
            model.generate([Request(block[i % len(block)]["prompt"], 4, i)
                            for i in range(model.max_batch_size)], SamplingParams(temperature=T, top_p=TOP_P,
                                                                                top_k=TOP_K))

            def completed(r):
                p = block[r.request_id]
                sampled = r.token_ids + ([manifest["eos_token_id"]] if r.finish_reason == "eos" else [])
                persist(dict(id=p["id"], seed=p["seed"], token_ids=r.token_ids,
                             sampled_token_ids=sampled, text=tokenizer.decode(r.token_ids),
                             pass_counts=r.generated_pass_counts, prompt_pass_counts=r.prompt_pass_counts,
                             finish_reason=r.finish_reason, timing=r.timing))

            model.generate([Request(p["prompt"], args.max_new_tokens, p["seed"]) for p in block],
                           SamplingParams(temperature=T, top_p=TOP_P, top_k=TOP_K), completed)
            info["telemetry"][str(b)] = model.last_run
            info["telemetry"][str(b)].update(memory_budget=model.memory_budget,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(model.device)
                if model.device.type == "cuda" else 0,
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(model.device)
                if model.device.type == "cuda" else 0)
        else:
            with fp64_attention_guard() if model.dtype == torch.float64 else nullcontext():
                results, traces = reference_batch(model, tokenizer, block, manifest["pad_token_id"],
                                                  args.max_new_tokens, args.implementation == "oracle")
            for r, trace in zip(results, traces):
                if args.implementation == "oracle":
                    target = trace_path(out, r["id"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_suffix(".tmp.npz")
                    np.savez_compressed(tmp, details=np.asarray(trace, dtype=np.float64).reshape(-1, 7))
                    os.replace(tmp, target)
                persist(r)
            info["telemetry"][str(b)] = dict(wall_seconds=time.monotonic() - started,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(model.device)
                if model.device.type == "cuda" else 0,
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(model.device)
                if model.device.type == "cuda" else 0)
        info["completed_blocks"].append(b)
        info["status"] = "complete" if len(info["completed_blocks"]) == len(blocks) else "running"
        write_json(path, info)
        print(f"{args.implementation}: block {b + 1}/{len(blocks)} complete", flush=True)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="freeze the full workload (tokenizer only)")
    prep.add_argument("--model", required=True)
    prep.add_argument("--output", type=Path, required=True)
    gen = sub.add_parser("run", help="resumable fixed-block coupled generation")
    gen.add_argument("implementation", choices=("engine", "reference", "oracle"))
    gen.add_argument("--manifest", type=Path, required=True)
    gen.add_argument("--model", type=Path, required=True)
    gen.add_argument("--eval-config", type=Path, help=f"required for protocol runs: {RECIPE}")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    gen.add_argument("--subset", default="all", help="short C1 only: frozen stage1 or stage2 subset")
    gen.add_argument("--pilot", type=int, default=0, help="first N per dataset, diagnostic only")
    gen.add_argument("--max-new-tokens", type=int, default=MAX_TOKENS, help="override for pilot/fixture only")
    gen.add_argument("--block", type=int, default=128, help="engine requests per resumable block")
    gen.add_argument("--oracle-dtype", choices=("fp32", "fp64"), default="fp32")
    gen.add_argument("--oracle-batch", type=int, default=8,
                     help="oracle rows per batch (default 8); reference stays fixed at 8")
    args = parser.parse_args(argv)
    if args.command == "run" and (args.pilot < 0 or args.max_new_tokens < 1 or args.block < 1):
        parser.error("pilot must be nonnegative; block and token cap must be positive")
    if args.command == "run":
        if args.oracle_batch < 1:
            parser.error("oracle batch must be positive")
        if args.implementation != "oracle" and args.oracle_batch != 8:
            parser.error("--oracle-batch applies only to the oracle; reference stays fixed at 8")
    return args


def main(argv=None):
    args = parse(argv)
    if args.command == "prepare":
        prepare(args.model, args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
