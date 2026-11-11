"""WP2 reference-only fixture generator.

CPU validation:
  CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python -m tah.engine.generate_fixtures \
      --profile tiny --device cpu --output data/engine/fixtures/cpu-tiny
CPU preparation of the production suite:
  CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python -m tah.engine.generate_fixtures \
      --profile full --prepare-only --output data/engine/fixtures/reference
After the GPU is free (explicit opt-in):
  CUDA_VISIBLE_DEVICES=0 .venv-tah/bin/python -m tah.engine.generate_fixtures \
      --profile full --device cuda --output data/engine/fixtures/reference

No GPU is selected automatically. Existing prepared inputs/checkpoints are
reused, and individual checkpoint capture can be selected with --checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch
from safetensors.torch import save_file

from tah.engine.fixtures import SCHEMA_VERSION
from tah.engine.reference_capture import capture, compare, divergence, pack, summarize


DATASETS = ("math500", "gpqa", "humaneval")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def checkpoint_metadata(path):
    # Hash weights as well as configs: random synthetic checkpoints must be
    # identifiable independently of their name and generator seed.
    return {str(file.relative_to(path)): {"bytes": file.stat().st_size, "sha256": digest(file)}
            for file in sorted(Path(path).rglob("*"))
            if file.is_file() and file.suffix in (".json", ".bin", ".safetensors")}


def prepare_inputs(args, tokenizer):
    path = args.output / "inputs.json"
    if path.exists():
        data = json.loads(path.read_text())
        if (data["profile"] != args.profile or data["seed"] != args.seed or
                data["tokenizer"] != str(args.real_model)):
            raise ValueError("existing inputs have a different profile/seed; choose a new output")
        return data["sequences"]
    from tah.evaluate.datasets import load_combined_dataset
    from tah.evaluate.codeeval import make_raw_chat_prompt_for_code_evaluation

    rows, _ = load_combined_dataset(list(DATASETS))
    lengths = (24, 47, 80) if args.profile == "tiny" else (256, 1024, 4096)
    sequences = []
    for dataset in DATASETS:
        selected = [row for row in rows if row["_source_dataset"] == dataset][:3]
        for index, (row, length) in enumerate(zip(selected, lengths)):
            if dataset == "humaneval":
                prompt = make_raw_chat_prompt_for_code_evaluation(
                    task_prompt=row["question"], reasoning=False, tokenizer=tokenizer)
                continuation = row.get("_original_canonical_solution", row["answer"])
            else:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["question"]}], tokenize=False,
                    add_generation_prompt=True)
                continuation = row.get("_original_solution", "") or (
                    "Let's reason through the problem step by step. "
                    f"The answer is \\boxed{{{row['answer']}}}.\n")
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
            if args.profile == "tiny":
                prompt_ids = [3 + token % 94 for token in prompt_ids[:length // 2]]
                continuation_ids = [3 + token % 94 for token in continuation_ids]
            # Keep the complete real prompt; long continuations intentionally
            # repeat a documented answer/solution to exercise ~4k context.
            length = max(length, len(prompt_ids) + 16)
            needed = length - len(prompt_ids)
            continuation_ids = (continuation_ids * (needed // len(continuation_ids) + 1))[:needed]
            sequences.append({
                "id": f"{dataset}-{index}", "dataset": dataset, "source_id": row["id"],
                "source_index": row["_original_index"], "question": row["question"],
                "rendered_prompt": prompt, "continuation_text": continuation,
                "continuation_kind": "repeated_dataset_solution_or_answer_scaffold",
                "prompt_ids": prompt_ids, "input_ids": prompt_ids + continuation_ids,
                "prompt_length": len(prompt_ids), "length": length,
                "tiny_token_remapping": args.profile == "tiny",
            })
    write_json(path, {"profile": args.profile, "seed": args.seed,
                      "tokenizer": str(args.real_model), "sequences": sequences})
    return sequences


def synthetic_config(args, *, max_iter, label=False):
    from tah.model.tah_config import TaHConfig
    tiny = args.profile == "tiny"
    config = TaHConfig(
        max_iter=max_iter, iter_decider="IterLabelDecider" if label else "MLPIterDecider",
        iter_decider_kwargs={"max_iter": max_iter} if label else {
            "topk": 8 if tiny else 100, "hidden_states_size": 32 if tiny else 1024,
            "hidden_states_layer_nums": [0, 1, 2] if tiny else [0, 8, 20, 28],
            "hidden_dims": [16, 16] if tiny else [64, 64], "expansion_factor": 2,
            "dropout_rate": 0.0, "threshold": 0.5, "max_iter": max_iter,
            "normalize_input": False, "dtype": torch.bfloat16,
        },
        input_updater_kwargs={"topk": 8 if tiny else 100},
        adapter_kwargs={"r": 4 if tiny else 8, "lora_alpha": 12 if tiny else 24,
                        "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj", "down_proj"],
                        "use_rslora": max_iter == 1, "task_type": "CAUSAL_LM"},
        train_loss="NextTokenPredLoss", eval_loss="NextTokenPredLoss",
    )
    return config


def prepare_checkpoints(args, tokenizer):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config
    from tah.model.tah_model import TaHForCausalLM

    paths = {}
    if args.profile == "full":
        paths["real"] = args.real_model.resolve()
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    for name, max_iter, label in (("mlp3", 3, False), ("mlp1", 1, False), ("label3", 3, True)):
        destination = args.model_output / args.profile / name
        paths[name] = destination.resolve()
        if (destination / "tah_config.json").exists():
            meta = json.loads((destination / "fixture_build.json").read_text())
            if meta["seed"] != args.seed or meta["base_source"] != args.base_model:
                raise ValueError("existing synthetic build has different seed/source")
            if args.profile == "full" and meta.get("tokenizer_source") != args.base_model:
                base_tokenizer.save_pretrained(str(destination))
                meta["tokenizer_source"] = args.base_model
                write_json(destination / "fixture_build.json", meta)
            continue
        torch.manual_seed(args.seed)
        if args.profile == "tiny":
            base_config = Qwen3Config(
                vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                tie_word_embeddings=max_iter != 1, bos_token_id=1, eos_token_id=2,
                max_position_embeddings=8192, attention_dropout=0.0,
            )
            base = AutoModelForCausalLM.from_config(base_config, attn_implementation="sdpa",
                                                   torch_dtype=torch.bfloat16)
        else:
            # Pretrained 0.6B-Base, with random adapter and random decider.
            base = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="cpu",
                                                       torch_dtype=torch.bfloat16,
                                                       attn_implementation="sdpa")
        model = TaHForCausalLM(base, synthetic_config(args, max_iter=max_iter, label=label))
        model.to(device="cpu", dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                if "lora_A" in key or "lora_B" in key:
                    parameter.normal_(0, 0.03)
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(destination))
        if args.profile == "full":
            base_tokenizer.save_pretrained(str(destination))
        write_json(destination / "fixture_build.json", {
            "seed": args.seed, "base_source": args.base_model,
            "base_random": args.profile == "tiny", "max_iter": max_iter,
            "random_lora_std": 0.03, "decider": model.iter_decider.__class__.__name__,
            "tokenizer_source": args.base_model if args.profile == "full" else None,
        })
        print(f"Prepared {name}: {destination}", flush=True)
        del model, base
        gc.collect()
    return paths


@torch.no_grad()
def calibrate(model, sequences, seed, chunk_size=256):
    """Tune ONE bf16 threshold for conditional continuation at both depths.

    Try quantiles of real reference probabilities, then independent random
    decider initializations if necessary. Never intercept/override decisions.
    Save the accepted decider and threshold before reload-and-capture.
    """
    from tah.model.iter_decider import MLPIterDecider
    probe = [item["input_ids"][:512] for item in sequences]
    attempts = []
    for trial in range(12):
        torch.manual_seed(seed + trial)
        model.iter_decider = MLPIterDecider(**model.tah_config.iter_decider_kwargs).to(
            device=model.device, dtype=model.dtype).eval()
        model.eval_iter_decider = model.iter_decider
        decider = model.iter_decider
        decider.threshold.fill_(-1)
        all_traces, _ = capture(model, probe, hidden_positions=0, chunk_size=chunk_size)
        probabilities = torch.cat([trace["continue_probability"][0] for trace in all_traces]).float()
        candidates = probabilities.quantile(torch.linspace(0.60, 0.90, 13)).to(torch.bfloat16).unique()
        for threshold in candidates.tolist():
            decider.threshold.fill_(threshold)
            traces, _ = capture(model, probe, hidden_positions=0, chunk_size=chunk_size)
            rates = []
            for depth in (0, 1):
                visited = sum(int(trace["visited"][depth].sum()) for trace in traces)
                continued = sum(int(trace["decision"][depth].sum()) for trace in traces)
                rates.append(continued / max(visited, 1))
            attempts.append({"trial": trial, "threshold": threshold, "rates": rates})
            if all(0.10 <= rate <= 0.40 for rate in rates):
                # Short prefixes cheaply reject poor candidates. Before
                # accepting one, verify the ENTIRE singleton fixture corpus
                # in bounded chunks, one sequence at a time. Otherwise a
                # threshold valid on prompts can fail on long continuations.
                full_rates = rates
                if any(len(item["input_ids"]) > 512 for item in sequences):
                    counts = [[0, 0], [0, 0]]
                    for item in sequences:
                        full, full_logits = capture(model, [item["input_ids"]],
                                                    hidden_positions=0, chunk_size=chunk_size)
                        for depth in (0, 1):
                            counts[depth][0] += int(full[0]["visited"][depth].sum())
                            counts[depth][1] += int(full[0]["decision"][depth].sum())
                        del full, full_logits
                    full_rates = [continued / max(visited, 1) for visited, continued in counts]
                attempts[-1]["full_corpus_rates"] = full_rates
                if not all(0.10 <= rate <= 0.40 for rate in full_rates):
                    continue
                model.tah_config.iter_decider_kwargs["threshold"] = float(threshold)
                decider._init_args["threshold"] = float(threshold)
                return {"status": "PASS", "threshold": threshold, "conditional_rates": rates,
                        "decider_seed": seed + trial, "probe_lengths": list(map(len, probe)),
                        "full_corpus_rates": full_rates, "chunk_size": chunk_size,
                        "method_version": 3,
                        "attempts": attempts}
    raise RuntimeError(f"MLP threshold calibration failed: {attempts}")


def load_reference(args, name, path):
    import yaml
    from tah.model.tah_config import TaHConfig
    from tah.model.tah_model import TaHForCausalLM

    override = None
    if name == "real":
        raw = yaml.safe_load(args.eval_config.read_text())
        valid = {field.name for field in fields(TaHConfig)}
        override = TaHConfig(**{key: value for key, value in raw.items() if key in valid})
    return TaHForCausalLM.from_pretrained(str(path), tah_config=override,
                                         torch_dtype=torch.bfloat16, device_map=args.device,
                                         attn_implementation="sdpa").to(dtype=torch.bfloat16).eval()


def labels_for(sequence):
    # All legal oracle depths plus ignore, with ~25% continuing at each depth.
    return [3 if position % 16 == 0 else 2 if position % 4 == 0 else
            -100 if position % 11 == 0 else 1 for position in range(len(sequence))]


def save_record(args, manifest, tensors, *, checkpoint, kind, sequence, variant, **metadata):
    file = Path(checkpoint) / kind / f"{sequence['id']}-{variant}.safetensors"
    path = args.output / file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path))
    record = {"file": str(file), "sha256": digest(path), "bytes": path.stat().st_size,
              "checkpoint": checkpoint, "kind": kind, "sequence_id": sequence["id"],
              "dataset": sequence["dataset"], "source_id": sequence["source_id"],
              "prompt_length": sequence["prompt_length"], "variant": variant, **metadata}
    manifest["records"].append(record)


def run_checkpoint(args, name, path, sequences, tokenizer, manifest):
    from tah.model.utils import TaHForCasualLM_generate
    from tah.model.iter_decider import save_iter_decider

    model = load_reference(args, name, path)
    if args.profile == "full" and name != "real":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side="left")
    calibration_path = path / "fixture_calibration.json"
    calibration = None
    if name == "mlp3":
        if (calibration_path.exists() and
                json.loads(calibration_path.read_text()).get("method_version") == 3 and
                json.loads(calibration_path.read_text()).get("chunk_size") == args.chunk_size):
            calibration = json.loads(calibration_path.read_text())
        else:
            calibration = calibrate(model, sequences, args.seed, args.chunk_size)
            # Persist only decider/config; base + random LoRA already saved by
            # the reference constructor/save_pretrained preparation step.
            from tah.model.tah_model import _config_to_serialisable
            save_iter_decider(model.iter_decider, str(path))
            write_json(path / "tah_config.json", _config_to_serialisable(asdict(model.tah_config)))
            write_json(calibration_path, calibration)
        del model
        gc.collect()
        model = load_reference(args, name, path)
        if float(model.iter_decider.threshold) != calibration["threshold"]:
            raise AssertionError("calibrated threshold did not survive reference reload")
    label = name == "label3"
    threshold = float(model._active_iter_decider.threshold.detach().cpu())
    metadata = {"path": str(path), "files": checkpoint_metadata(path),
                "effective_tah_config": asdict(model.tah_config), "threshold": threshold,
                "calibration": calibration, "dtype": str(model.dtype),
                "base_config": model.config.to_dict(), "device": str(model.device)}
    manifest["checkpoints"][name] = metadata
    pad_id = 0 if args.profile == "tiny" else tokenizer.eos_token_id
    rows = []
    depth_totals = [{"visited": 0, "continued": 0} for _ in range(model.max_iter)]
    for item in sequences:
        seq = item["input_ids"]
        oracle = [labels_for(seq)] if label else None
        baseline, base_logits = capture(model, [seq], pad_id, labels=oracle, topk=args.topk,
                                        chunk_size=args.chunk_size)
        base = baseline[0]
        for depth, totals in enumerate(depth_totals):
            totals["visited"] += int(base["visited"][depth].sum())
            totals["continued"] += int(base["decision"][depth].sum())
        save_record(args, manifest, base, checkpoint=name, kind="teacher_forced",
                    sequence=item, variant="single", chunk_size=args.chunk_size,
                    probability_kind="deterministic_label" if label else "bf16_sigmoid_with_cap_zero")
        # Same target in 4 different layouts; companions have different
        # lengths and content, and the target also changes row index.
        companion = sequences[(sequences.index(item) + 4) % len(sequences)]["input_ids"][:113]
        variants = (("left-extra", [seq], 0, "left", 7),
                    ("mixed-left", [seq, companion], 0, "left", 0),
                    ("mixed-reordered", [companion, seq], 1, "left", 11),
                    ("mixed-right", [companion, seq], 1, "right", 7))
        for variant, batch, target, side, extra in variants:
            batch_labels = [labels_for(tokens) for tokens in batch] if label else None
            candidate, logits = capture(model, batch, pad_id, side=side, extra=extra,
                                        labels=batch_labels, topk=args.topk,
                                        chunk_size=args.chunk_size)
            stats = compare(base, candidate[target], base_logits[0], logits[target], threshold)
            rows.append({"sequence_id": item["id"], "variant": variant, **stats})
            save_record(args, manifest, candidate[target], checkpoint=name,
                        kind="teacher_forced", sequence=item, variant=variant,
                        batch_sequence_lengths=list(map(len, batch)), target_row=target,
                        padding_side=side, extra_padding=extra, chunk_size=args.chunk_size)
            del candidate, logits
        del baseline, base_logits, base
        print(f"Captured {name}/{item['id']} ({len(seq)} tokens)", flush=True)
    generations = []
    for item in [item for item in sequences if item["id"].endswith(("-0", "-1"))]:
        prompt = item["prompt_ids"]
        other = sequences[(sequences.index(item) + 3) % len(sequences)]["prompt_ids"]
        # Reference generation reads logits[:, -1]; only LEFT padding is
        # meaningful here. Right padding is exercised by teacher forcing.
        baseline_tokens = None
        for variant, batch, target, extra in (
                ("single", [prompt], 0, 0), ("left-extra", [prompt], 0, 7),
                ("mixed-left", [prompt, other], 0, 0),
                ("mixed-reordered", [other, prompt], 1, 11)):
            inputs, _ = pack(batch, 0 if args.profile == "tiny" else tokenizer.eos_token_id,
                             model.device, extra=extra)
            prompt_pass_counts = []

            def observe_prompt(module, hook_args, hook_kwargs, output):
                if not prompt_pass_counts:
                    prompt_pass_counts.extend(
                        output.iter_count[row][inputs["attention_mask"][row].bool()].detach().cpu()
                        for row in range(len(batch)))

            hook = model.register_forward_hook(observe_prompt, with_kwargs=True)
            try:
                tokens, _, counts = TaHForCasualLM_generate(
                    model, tokenizer, inputs, do_sample=False, temperature=0.0,
                    max_new_tokens=args.generation_tokens, verbose=False, return_pass_counts=True)
            finally:
                hook.remove()
            if baseline_tokens is None:
                baseline_tokens = tokens[target]
            else:
                generations.append({"sequence_id": item["id"], "variant": variant,
                                    "reference_length": len(baseline_tokens),
                                    "candidate_length": len(tokens[target]),
                                    "first_divergence": divergence(baseline_tokens, tokens[target])})
            save_record(args, manifest, {
                "prompt_ids": torch.tensor(prompt, dtype=torch.long),
                "generated_ids": torch.tensor(tokens[target], dtype=torch.long),
                "generation_pass_count": torch.tensor(counts[target], dtype=torch.long),
                "prompt_pass_count": prompt_pass_counts[target],
            }, checkpoint=name, kind="greedy", sequence=item, variant=variant,
                        max_new_tokens=args.generation_tokens, labels_supplied=False,
                        finish_reason="length" if len(tokens[target]) == args.generation_tokens else "eos",
                        eos_token_id=tokenizer.eos_token_id, batch_prompt_lengths=list(map(len, batch)),
                        target_row=target, extra_padding=extra)
    rates = [totals["continued"] / max(totals["visited"], 1) for totals in depth_totals]
    stats = {"teacher_forced": {"summary": summarize(rows), "comparisons": rows},
             "greedy": {"comparisons": generations,
                        "divergence_positions": [row["first_divergence"] for row in generations],
                        "divergent_pairs": sum(row["first_divergence"] is not None for row in generations),
                        "total_pairs": len(generations)},
             "depth_totals": depth_totals, "conditional_continue_rates": rates}
    # Do not silently bless poor synthetic depth coverage on the full set.
    if name == "mlp3":
        stats["full_fixture_depth_coverage_pass"] = all(0.10 <= rate <= 0.40 for rate in rates[:-1])
    write_json(args.output / name / "self_noise.json", stats)
    manifest["self_noise"][name] = stats
    print(f"Self-noise {name}: {stats['teacher_forced']['summary']}", flush=True)
    del model
    gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/engine/fixtures/cpu-tiny"))
    parser.add_argument("--model-output", type=Path, default=Path("data/engine/fixture-models"))
    parser.add_argument("--real-model", type=Path, default=Path("models/TaH-plus-1.7B"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--eval-config", type=Path, default=Path("script/tah/recipes/qwen3_1.7/eval_tah.yaml"))
    parser.add_argument("--checkpoint", action="append", choices=("real", "mlp3", "mlp1", "label3"))
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256, help="0 = unchunked reference")
    parser.add_argument("--generation-tokens", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        parser.error("CUDA is explicitly hidden; GPU work is forbidden in this session")
    if args.profile == "full" and args.device == "cpu" and not args.prepare_only:
        parser.error("full reference capture requires an explicit --device cuda; use --prepare-only on CPU")
    if args.chunk_size < 0 or args.topk < 1 or args.threads < 1:
        parser.error("chunk-size must be >=0, topk/threads >=1")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.generation_tokens = args.generation_tokens or (8 if args.profile == "tiny" else 256)
    if args.generation_tokens < 1:
        parser.error("generation-tokens must be >=1")
    if args.profile == "tiny" and args.checkpoint and "real" in args.checkpoint:
        parser.error("the tiny profile has no real checkpoint")
    from transformers import AutoTokenizer
    import transformers
    import peft

    tokenizer = AutoTokenizer.from_pretrained(str(args.real_model), padding_side="left")
    if args.profile == "tiny":
        # Reference generator only needs eos_token_id/decode. Keep EOS inside
        # tiny vocab without pretending these tokens are real Qwen text.
        class TinyTokenizer:
            eos_token_id = 2

            def decode(self, tokens):
                return " ".join(map(str, tokens))

        generation_tokenizer = TinyTokenizer()
    else:
        generation_tokenizer = tokenizer
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = prepare_inputs(args, tokenizer)
    paths = prepare_checkpoints(args, tokenizer)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": SCHEMA_VERSION, "profile": args.profile,
        "production_gates_eligible": False, "seed": args.seed,
        "records": [], "checkpoints": {}, "self_noise": {},
    }
    manifest["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                               "transformers": transformers.__version__, "peft": peft.__version__,
                               "device": args.device, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                               "threads": args.threads}
    manifest["reference_source_sha256"] = {
        file: digest(file) for file in ("tah/model/tah_model.py", "tah/model/iter_decider.py",
                                       "tah/model/causal_cache.py", "tah/model/utils.py")}
    manifest["tool_source_sha256"] = {
        file: digest(file) for file in ("tah/engine/generate_fixtures.py",
                                       "tah/engine/reference_capture.py", "tah/engine/fixtures.py")}
    manifest["eval_config"] = {"path": str(args.eval_config), "sha256": digest(args.eval_config)}
    manifest["prepared_checkpoints"] = {name: str(path) for name, path in paths.items()}
    settings = {"topk": args.topk, "chunk_size": args.chunk_size,
                                    "generation_tokens": args.generation_tokens,
                                    "hidden_positions_per_sequence": 12,
                                    "greedy_divergence_index": "zero-based, null means identical including EOS"}
    if manifest["records"] and manifest.get("capture_settings") != settings:
        parser.error("existing fixture capture settings differ; choose a new output")
    manifest["capture_settings"] = settings
    manifest["status"] = "prepared" if args.prepare_only else "in_progress"
    manifest["production_gates_eligible"] = False
    write_json(manifest_path, manifest)
    if args.prepare_only:
        print(f"Prepared {len(sequences)} sequences and {len(paths)} checkpoint paths; no capture.")
        return
    started = time.monotonic()
    for name, path in paths.items():
        if args.checkpoint and name not in args.checkpoint:
            continue
        manifest["records"] = [record for record in manifest["records"] if record["checkpoint"] != name]
        run_checkpoint(args, name, path, sequences, generation_tokenizer, manifest)
        write_json(manifest_path, manifest)
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["status"] = "complete" if set(paths) == set(manifest["self_noise"]) else "partial"
    manifest["fixture_tensor_bytes"] = sum(record["bytes"] for record in manifest["records"])
    write_json(manifest_path, manifest)
    failed_coverage = [name for name, stats in manifest["self_noise"].items()
                       if stats.get("full_fixture_depth_coverage_pass") is False]
    if failed_coverage:
        manifest["status"] = "failed_depth_coverage"
    manifest["production_gates_eligible"] = args.profile == "full" and manifest["status"] == "complete"
    # Include manifest itself in total bytes; iterate to stable digit width.
    for _ in range(4):
        manifest["total_fixture_bytes"] = sum(file.stat().st_size for file in args.output.rglob("*") if file.is_file())
        write_json(manifest_path, manifest)
    print(f"Wrote {len(manifest['records'])} records to {args.output}; "
          f"{manifest['fixture_tensor_bytes']} tensor bytes", flush=True)
    if failed_coverage:
        raise RuntimeError(f"10–40% depth coverage failed for {failed_coverage}; see self_noise.json")


if __name__ == "__main__":
    main()
