"""Resumable collection with one atomic shard per complete reference problem."""
import argparse
import json
from pathlib import Path
import time

import torch

from .core import RoutingDecider, digest, override_decider, paired_step_fast
from .run import load_model, load_references


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="models/TaH-plus-0.6B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imitation-threshold", type=float, default=0.9)
    p.add_argument("--max-reference-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    records = load_references(args.references)
    settings = {"model": args.model, "imitation_threshold": args.imitation_threshold,
                "reference_sha256": digest(records), "seed": args.seed,
                "max_reference_tokens": args.max_reference_tokens,
                "core_sha256": digest(Path(__file__).with_name("core.py").read_text()),
                "collector_sha256": digest(Path(__file__).read_text()),
                "history_policy": "frozen_imitation", "loss2": "native_additive_logits",
                "feature_storage": "float16", "truncated": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "collection.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("Collection settings changed; use a new output directory")
    manifest.write_text(json.dumps(settings, indent=2))
    shards = args.output_dir / "shards"
    shards.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args)
    n = model.config.num_hidden_layers
    layers = (1, max(1, n // 2), n)
    metadata = {**settings, "hidden_layers": list(layers), "hidden_size": model.config.hidden_size,
                "feature_order": "shallow,middle,final,entropy,max_probability",
                "torch_version": str(torch.__version__)}
    decider = RoutingDecider(model.iter_decider, layers).eval()
    start = time.perf_counter()
    total_rows = 0
    with torch.inference_mode(), override_decider(model, decider):
        for number, record in enumerate(records, 1):
            path = shards / (digest(record["id"])[:20] + ".pt")
            if path.exists():
                shard = torch.load(path, map_location="cpu", weights_only=True)
                if shard["problem"]["id"] != record["id"] or shard["metadata"] != metadata:
                    raise ValueError("Mismatched cached problem")
                total_rows += len(shard["rows"])
                continue
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": record["question"]}],
                                                    tokenize=True, add_generation_prompt=True)
            reference = tokenizer.encode(record["reference"], add_special_tokens=False)
            if not 1 <= len(reference) <= args.max_reference_tokens:
                raise ValueError(f"Reference length invalid for {record['id']}; refusing truncation")
            cache, history = None, []
            if len(prompt) > 1:
                prefill = model(input_ids=torch.tensor([prompt[:-1]], device=model.device), use_cache=True)
                cache, history = prefill.past_key_values, prefill.iter_count[0].tolist()
                del prefill
            prefix, current = prompt[:-1].copy(), prompt[-1]
            states, losses, rows = [], [], []
            for index, target in enumerate(reference):
                history_hash = digest({"tokens": prefix, "depths": history})
                state, pair, depth, cache = paired_step_fast(
                    model, decider, cache, torch.tensor([[current]], device=model.device),
                    torch.tensor(target, device=model.device),
                )
                states.append(state.half())
                losses.append(pair)
                rows.append({"problem_id": record["id"], "reference_index": index,
                             "position": len(prefix), "input_id": current, "target_id": target,
                             "history_sha256": history_hash, "imitation_depth": depth})
                prefix.append(current)
                history.append(depth)
                current = target
            problem = {**record, "prompt_token_ids": prompt, "reference_token_ids": reference,
                       "imitation_history_depths": history}
            atomic_save({"features": torch.stack(states), "action_losses": torch.stack(losses),
                         "rows": rows, "problem": problem, "metadata": metadata}, path)
            total_rows += len(rows)
            progress = {"stage": "collecting", "completed_problems": number, "total_problems": len(records),
                        "rows": total_rows, "seconds_this_run": time.perf_counter() - start}
            (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
            del cache
    # Merge once, preserving problem boundaries and deterministic reference order.
    states, losses, rows, problems = [], [], [], []
    for record in records:
        shard = torch.load(shards / (digest(record["id"])[:20] + ".pt"), map_location="cpu", weights_only=True)
        states.append(shard["features"])
        losses.append(shard["action_losses"])
        first = len(rows)
        rows.extend(shard["rows"])
        problems.append({**shard["problem"], "row_start": first, "row_end": len(rows)})
    atomic_save({"schema_version": 1, "features": torch.cat(states), "action_losses": torch.cat(losses),
                 "rows": rows, "problems": problems, "metadata": metadata}, args.output_dir / "cache.pt")
    (args.output_dir / "progress.json").write_text(json.dumps({"stage": "cache_complete", "problems": len(problems), "rows": len(rows)}, indent=2))
    print(f"CACHE COMPLETE: {len(problems)} problems, {len(rows)} paired tokens", flush=True)


if __name__ == "__main__":
    main()
