"""Mini-batch training with early stopping at a matched validation budget."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from .core import BanditRouter, bandit_loss, calibrate_budget, problem_split
from .run import metrics


@torch.no_grad()
def scores_for(router, states, batch_size):
    return torch.cat([router(chunk).cpu() for chunk in states.split(batch_size)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--penalties", nargs="+", type=float, default=[0., 0.01, 0.05, 0.1, 0.2])
    p.add_argument("--budgets", nargs="+", type=float, default=[0., 0.05, 0.1, 0.2, 0.3, 0.5, 1.])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if min(args.penalties) < 0 or args.epochs < 1 or args.patience < 1:
        raise ValueError("Invalid training settings")
    torch.set_num_threads(args.threads)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    states, losses = cache["features"].float(), cache["action_losses"].float()
    if not torch.isfinite(states).all() or not torch.isfinite(losses).all():
        raise ValueError("Nonfinite cache values")
    train_problems, val_problems = problem_split(cache["problems"], args.seed, args.validation_fraction)
    train_ids = {p["id"] for p, selected in zip(cache["problems"], train_problems) if selected}
    train_mask = torch.tensor([r["problem_id"] in train_ids for r in cache["rows"]])
    train_states, val_states = states[train_mask], states[~train_mask]
    train_losses, val_losses = losses[train_mask], losses[~train_mask]
    imitation = torch.tensor([r["imitation_depth"] == 2 for r in cache["rows"]])[~train_mask]
    target = float(imitation.float().mean())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"cache_metadata": cache["metadata"], "seed": args.seed,
              "train_problem_ids": sorted(train_ids),
              "validation_problem_ids": [p["id"] for p, v in zip(cache["problems"], val_problems) if v],
              "train_rows": len(train_states), "validation_rows": len(val_states),
              "imitation_continue_fraction": target,
              "training": {k: v for k, v in vars(args).items() if k not in ("cache", "output_dir")}}
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Sweep settings changed; choose a new directory")
    config_path.write_text(json.dumps(config, indent=2))
    sweep_results = []
    start = time.perf_counter()
    for penalty in args.penalties:
        name = f"lambda_{penalty:g}"
        report_path = args.output_dir / f"{name}.report.json"
        if report_path.exists():
            sweep_results.append(json.loads(report_path.read_text()))
            continue
        torch.manual_seed(args.seed)
        router = BanditRouter(states.shape[1], args.width)
        router.mean.copy_(train_states.mean(0))
        router.scale.copy_(train_states.std(0, unbiased=False).clamp_min(1e-5))
        optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        curve, best_nll, best_epoch, best_state, stale = [], float("inf"), 0, None, 0
        for epoch in range(args.epochs + 1):
            if epoch:
                router.train()
                indices = torch.randperm(len(train_states))
                for batch in indices.split(args.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss = bandit_loss(router(train_states[batch]), train_losses[batch], penalty)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(router.parameters(), 1.)
                    optimizer.step()
            router.eval()
            val_scores = scores_for(router, val_states, args.batch_size)
            train_scores = scores_for(router, train_states, args.batch_size)
            threshold, chosen = calibrate_budget(val_scores, val_losses, target, penalty)
            deployed = metrics(val_scores.sigmoid(), val_losses, chosen, penalty)
            point = {"epoch": epoch,
                     "train_objective": float(bandit_loss(train_scores, train_losses, penalty)),
                     "validation_objective": float(bandit_loss(val_scores, val_losses, penalty)),
                     "matched_validation": deployed, "logit_threshold": str(threshold)}
            curve.append(point)
            if deployed["token_nll"] < best_nll - 1e-6:
                best_nll, best_epoch = deployed["token_nll"], epoch
                best_state, stale = copy.deepcopy(router.state_dict()), 0
            else:
                stale += 1
            (args.output_dir / f"{name}.curve.json").write_text(json.dumps(curve, indent=2))
            print(json.dumps({"penalty": penalty, "epoch": epoch, "train_objective": point["train_objective"],
                              "validation_nll": deployed["token_nll"], "best_epoch": best_epoch}), flush=True)
            if stale >= args.patience:
                break
        router.load_state_dict(best_state)
        router.eval()
        val_scores = scores_for(router, val_states, args.batch_size)
        threshold, chosen = calibrate_budget(val_scores, val_losses, target, penalty)
        entropy_threshold, _ = calibrate_budget(val_states[:, -2], val_losses, target, penalty)
        budget_curve = []
        for budget in sorted(set(args.budgets + [target])):
            bt, bm = calibrate_budget(val_scores, val_losses, budget, penalty)
            et, em = calibrate_budget(val_states[:, -2], val_losses, budget, penalty)
            budget_curve.append({"target_continue_fraction": budget, "logit_threshold": str(bt),
                                 "entropy_threshold": str(et),
                                 "bandit": metrics(val_scores.sigmoid(), val_losses, bm, penalty),
                                 "entropy": metrics(em.float(), val_losses, em, penalty)})
        penalty_baselines = {
            "always_1": metrics(torch.zeros(len(val_states)), val_losses, torch.zeros_like(imitation), penalty),
            "always_2": metrics(torch.ones(len(val_states)), val_losses, torch.ones_like(imitation), penalty),
            "imitation": metrics(imitation.float(), val_losses, imitation, penalty),
        }
        report = {"penalty": penalty, "best_epoch": best_epoch, "epochs_run": len(curve) - 1,
                  "threshold": float(torch.tensor(threshold).sigmoid()), "logit_threshold": str(threshold),
                  "entropy_threshold": str(entropy_threshold),
                  "validation": {**penalty_baselines, "bandit": metrics(val_scores.sigmoid(), val_losses, chosen, penalty)},
                  "budget_curve": budget_curve, "curve_file": f"{name}.curve.json",
                  "checkpoint_selection": "lowest validation token NLL at imitation's continuation budget",
                  "note": "Validation is used for checkpoint and threshold selection; these are local surrogate diagnostics, not unbiased test scores or answer accuracy."}
        checkpoint = {"schema_version": 2, "state_dict": router.state_dict(), "feature_dim": states.shape[1],
                      "width": args.width, "threshold": report["threshold"], "logit_threshold": threshold,
                      "penalty": penalty, "cache_metadata": cache["metadata"], "report": report}
        torch.save(checkpoint, args.output_dir / f"{name}.pt")
        report_path.write_text(json.dumps(report, indent=2))
        sweep_results.append(report)
    best = min(sweep_results, key=lambda r: r["validation"]["bandit"]["token_nll"])
    selected_path = args.output_dir / f"lambda_{best['penalty']:g}.pt"
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    torch.save(selected, args.output_dir / "selected_router.pt")
    summary = {"status": "complete", "problems": len(cache["problems"]), "paired_tokens": len(states),
               "train_problems": len(train_ids), "validation_problems": len(cache["problems"]) - len(train_ids),
               "selected_penalty": best["penalty"], "selected_epoch": best["best_epoch"],
               "matched_validation": best["validation"], "sweep": sweep_results,
               "seconds": time.perf_counter() - start, "selection": "validation matched-budget NLL"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "sweep"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
