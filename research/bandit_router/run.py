"""Run from the repo root: python -m research.bandit_router.run --help."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .core import (BanditRouter, RoutingDecider, bandit_loss, digest, override_decider,
                   paired_step, problem_split)


def load_model(args):
    from transformers import AutoTokenizer
    from tah.model.tah_model import TaHForCausalLM
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = TaHForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device, attn_implementation="sdpa"
    ).eval().requires_grad_(False)
    if model.max_iter != 2:
        raise ValueError("This prototype requires a depth-2 TaH checkpoint")
    model.iter_decider.threshold.data.fill_(args.imitation_threshold)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return model, tokenizer


def load_references(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Reference problem IDs must be unique")
    if any(not r["question"].strip() or not r["reference"].strip() for r in records):
        raise ValueError("Each problem needs nonempty question and reference text")
    return records


def collect(args):
    if args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    records = load_references(args.references)
    model, tokenizer = load_model(args)
    n = model.config.num_hidden_layers
    layers = (1, max(1, n // 2), n)
    decider = RoutingDecider(model.iter_decider, layers).eval()
    states, losses, rows, problems = [], [], [], []
    started = time.perf_counter()
    with torch.inference_mode(), override_decider(model, decider):
        for record in records:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": record["question"]}],
                tokenize=True, add_generation_prompt=True,
            )
            reference = tokenizer.encode(record["reference"], add_special_tokens=False)
            if not prompt or not reference:
                raise ValueError("Empty tokenized prompt or reference")
            reference = reference[:args.max_tokens]
            cache = None
            # Leave the final prompt token to predict the first reference token.
            if len(prompt) > 1:
                prefill = model(input_ids=torch.tensor([prompt[:-1]], device=model.device), use_cache=True)
                cache = prefill.past_key_values
                history = prefill.iter_count[0].tolist()
            else:
                history = []
            prefix = prompt[:-1].copy()
            current = prompt[-1]
            first = len(rows)
            for index, target in enumerate(reference):
                # Hash the exact consumed tokens and their imitation depths.
                history_hash = digest({"tokens": prefix, "depths": history})
                state, pair, depth, cache = paired_step(
                    model, decider, cache,
                    torch.tensor([[current]], device=model.device),
                    torch.tensor(target, device=model.device),
                )
                states.append(state)
                losses.append(pair)
                rows.append({"problem_id": record["id"], "reference_index": index,
                             "position": len(prefix), "input_id": current, "target_id": target,
                             "history_sha256": history_hash, "imitation_depth": depth})
                prefix.append(current)
                history.append(depth)
                current = target
            problems.append({**record, "row_start": first, "row_end": len(rows),
                             "prompt_token_ids": prompt, "reference_token_ids": reference,
                             "imitation_history_depths": history})
            print(f"Captured {record['id']}: {len(reference)} paired states", flush=True)
    artifact = {
        "schema_version": 1, "features": torch.stack(states), "action_losses": torch.stack(losses),
        "rows": rows, "problems": problems,
        "metadata": {"model": args.model, "imitation_threshold": args.imitation_threshold,
                     "hidden_layers": list(layers), "hidden_size": model.config.hidden_size,
                     "feature_order": "shallow,middle,final,entropy,max_probability",
                     "reference_sha256": digest(records), "seed": args.seed,
                     "torch_version": str(torch.__version__), "seconds": time.perf_counter() - started,
                     "collector_sha256": digest(Path(__file__).read_text()),
                     "core_sha256": digest(Path(__file__).with_name("core.py").read_text()),
                     "history_policy": "frozen_imitation", "loss2": "native_additive_logits"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {args.output}")
    torch.save(artifact, args.output)
    pair = artifact["action_losses"]
    print(json.dumps({"cache": str(args.output), "rows": len(rows),
                      "feature_dim": artifact["features"].shape[1],
                      "mean_loss_gain": float((pair[:, 0] - pair[:, 1]).mean())}, indent=2))


def metrics(probabilities, losses, mask, penalty):
    reward = losses[:, 0] - losses[:, 1] - penalty
    return {"continue_fraction": float(mask.float().mean()), "average_depth": 1 + float(mask.float().mean()),
            "token_nll": float(torch.where(mask, losses[:, 1], losses[:, 0]).mean()),
            "local_reward": float((mask * reward).mean()),
            "expected_reward": float((probabilities * reward).mean())}


def train(args):
    if args.epochs < 1 or args.penalty < 0:
        raise ValueError("Need positive epochs and nonnegative penalty")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    states, losses = cache["features"], cache["action_losses"]
    if not torch.isfinite(states).all() or not torch.isfinite(losses).all():
        raise ValueError("Cache contains nonfinite features or losses")
    train_problems, valid_problems = problem_split(cache["problems"], args.seed, args.validation_fraction)
    train_ids = {p["id"] for p, selected in zip(cache["problems"], train_problems) if selected}
    train_mask = torch.tensor([r["problem_id"] in train_ids for r in cache["rows"]])
    valid_mask = ~train_mask
    torch.manual_seed(args.seed)
    router = BanditRouter(states.shape[1], args.width)
    router.mean.copy_(states[train_mask].mean(0))
    router.scale.copy_(states[train_mask].std(0, unbiased=False).clamp_min(1e-5))
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    initial = float(bandit_loss(router(states[train_mask]), losses[train_mask], args.penalty).detach())
    router.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = bandit_loss(router(states[train_mask]), losses[train_mask], args.penalty)
        loss.backward()
        optimizer.step()
    router.eval()
    with torch.no_grad():
        probabilities = router(states[valid_mask]).sigmoid()
        pair = losses[valid_mask]
        imitation = torch.tensor([r["imitation_depth"] == 2 for r in cache["rows"]])[valid_mask]
        target = float(imitation.float().mean())
        # Calibrate only on held-out problems. Include boundary choices.
        candidates = torch.cat((torch.tensor([0., 1.]), probabilities.unique()))
        threshold = min(candidates.tolist(), key=lambda t: abs(float((probabilities > t).float().mean()) - target))
        entropy = states[valid_mask, -2]
        entropy_candidates = torch.cat((torch.tensor([float("-inf"), float("inf")]), entropy.unique()))
        entropy_threshold = min(entropy_candidates.tolist(), key=lambda t: abs(float((entropy > t).float().mean()) - target))
        masks = {"always_1": torch.zeros_like(imitation), "always_2": torch.ones_like(imitation),
                 "imitation": imitation, "entropy": entropy > entropy_threshold,
                 "bandit": probabilities > threshold, "reference_oracle": pair[:, 0] - pair[:, 1] > args.penalty}
        report = {"initial_train_objective": initial,
                  "final_train_objective": float(bandit_loss(router(states[train_mask]), losses[train_mask], args.penalty)),
                  "penalty": args.penalty, "threshold": threshold,
                  "entropy_threshold": str(entropy_threshold),
                  "train_problem_ids": sorted(train_ids),
                  "validation_problem_ids": [p["id"] for p, v in zip(cache["problems"], valid_problems) if v],
                  "train_rows": int(train_mask.sum()), "validation_rows": int(valid_mask.sum()),
                  "validation": {k: metrics(probabilities if k == "bandit" else m.float(), pair, m, args.penalty) for k, m in masks.items()},
                  "note": "Local reference-token diagnostic only; tiny pilot is not answer-accuracy evidence. Threshold ties can prevent exact depth matching."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "state_dict": router.state_dict(), "feature_dim": states.shape[1],
                "width": args.width, "threshold": threshold, "penalty": args.penalty,
                "cache_metadata": cache["metadata"], "report": report}, args.output)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def generate(args):
    if args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    model, tokenizer = load_model(args)
    checkpoint = torch.load(args.router, map_location="cpu", weights_only=True)
    if checkpoint["cache_metadata"]["model"] != args.model:
        raise ValueError("Router checkpoint was trained for a different model path")
    router = BanditRouter(checkpoint["feature_dim"], checkpoint["width"])
    router.load_state_dict(checkpoint["state_dict"])
    router.to(model.device).eval().requires_grad_(False)
    threshold = args.threshold if args.threshold is not None else checkpoint["threshold"]
    if args.policy == "entropy" and args.threshold is None:
        threshold = float(checkpoint["report"]["entropy_threshold"])
    decider = RoutingDecider(model.iter_decider, checkpoint["cache_metadata"]["hidden_layers"],
                             policy=args.policy, router=router, threshold=threshold,
                             logit_threshold=checkpoint.get("logit_threshold") if args.threshold is None else None).eval()
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                                           tokenize=True, add_generation_prompt=True)
    ids, depths, cache = [], [], None
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    with torch.inference_mode(), override_decider(model, decider):
        prefill_passes = 0
        # Prompt context follows the same frozen imitation policy as collection.
        if len(prompt) > 1:
            decider.policy = "imitation"
            output = model(input_ids=torch.tensor([prompt[:-1]], device=model.device), use_cache=True)
            cache = output.past_key_values
            prefill_passes = int(output.iter_count.sum())
        decider.policy = args.policy
        current = torch.tensor([[prompt[-1]]], device=model.device)
        for step in range(args.max_tokens):
            output = model(input_ids=current, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(output.logits[0, -1].argmax())
            depths.append(int(output.iter_count[0, -1]))
            ids.append(token)
            if token == tokenizer.eos_token_id:
                break
            current = torch.tensor([[token]], device=model.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    result = {"policy": args.policy, "prompt": args.prompt, "output": tokenizer.decode(ids, skip_special_tokens=True),
              "token_ids": ids, "depths": depths, "average_depth": sum(depths) / len(depths),
              "continue_fraction": sum(d == 2 for d in depths) / len(depths),
              "total_latent_passes": prefill_passes + sum(depths), "prefill_passes": prefill_passes,
              "seconds": time.perf_counter() - started,
              "termination": "eos" if ids[-1] == tokenizer.eos_token_id else "token_limit",
              "decoding": "greedy", "threshold": threshold,
              "logit_threshold": str(decider.logit_threshold)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--references", type=Path, default=Path(__file__).with_name("references.jsonl"))
    c.add_argument("--max-tokens", type=int, default=32)
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--penalty", type=float, default=0.05)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--width", type=int, default=64)
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--validation-fraction", type=float, default=0.1)
    g = sub.add_parser("generate")
    g.add_argument("--router", type=Path, required=True)
    g.add_argument("--prompt", default="What is 17 + 25?")
    g.add_argument("--policy", choices=["stop", "continue", "imitation", "entropy", "bandit"], default="bandit")
    g.add_argument("--threshold", type=float)
    g.add_argument("--max-tokens", type=int, default=32)
    for command in (c, t, g):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=42)
    for command in (c, g):
        command.add_argument("--model", default="models/TaH-plus-0.6B")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--imitation-threshold", type=float, default=0.9)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    {"collect": collect, "train": train, "generate": generate}[args.command](args)


if __name__ == "__main__":
    main()
