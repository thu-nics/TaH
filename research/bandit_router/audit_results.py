"""Reproduce the selected policy and summarize paired validation problems."""
import argparse
import json
from pathlib import Path

import torch

from .core import BanditRouter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    config = json.loads((root / "training/config.json").read_text())
    # Keep batch geometry and CPU reduction settings identical to training:
    # a threshold exactly on an observed score is sensitive to rounding.
    torch.set_num_threads(config["training"]["threads"])
    cache = torch.load(root / "cache.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(root / "training/selected_router.pt", map_location="cpu", weights_only=True)
    ids = set(config["validation_problem_ids"])
    selected_rows = [r["problem_id"] in ids for r in cache["rows"]]
    valid = torch.tensor(selected_rows)
    states, losses = cache["features"][valid].float(), cache["action_losses"][valid]
    router = BanditRouter(checkpoint["feature_dim"], checkpoint["width"])
    router.load_state_dict(checkpoint["state_dict"])
    router.eval()
    with torch.inference_mode():
        scores = torch.cat([router(x) for x in states.split(config["training"]["batch_size"])])
    bandit = scores > checkpoint["logit_threshold"]
    imitation = torch.tensor([r["imitation_depth"] == 2 for r in cache["rows"]])[valid]
    a = torch.where(imitation, losses[:, 1], losses[:, 0])
    b = torch.where(bandit, losses[:, 1], losses[:, 0])
    expected = checkpoint["report"]["validation"]["bandit"]
    torch.testing.assert_close(b.mean(), torch.tensor(expected["token_nll"]), rtol=0, atol=1e-6)
    torch.testing.assert_close(bandit.float().mean(), torch.tensor(expected["continue_fraction"]), rtol=0, atol=1e-8)
    rows = [r for r in cache["rows"] if r["problem_id"] in ids]
    per_problem = []
    for key in sorted(ids):
        mask = torch.tensor([r["problem_id"] == key for r in rows])
        per_problem.append({"problem_id": key, "tokens": int(mask.sum()),
                            "imitation_nll": float(a[mask].mean()), "bandit_nll": float(b[mask].mean()),
                            "nll_gain_sum": float((a[mask] - b[mask]).sum()),
                            "imitation_continues": int(imitation[mask].sum()),
                            "bandit_continues": int(bandit[mask].sum())})
    generator = torch.Generator().manual_seed(42)
    samples = torch.randint(len(per_problem), (10000, len(per_problem)), generator=generator)
    numerators = torch.tensor([r["nll_gain_sum"] for r in per_problem])
    denominators = torch.tensor([r["tokens"] for r in per_problem])
    bootstrap = numerators[samples].sum(1) / denominators[samples].sum(1)
    curve = json.loads((root / "training" / checkpoint["report"]["curve_file"]).read_text())
    report = {"validation_tokens": len(a), "continued_imitation": int(imitation.sum()),
              "continued_bandit": int(bandit.sum()), "changed_decisions": int((imitation != bandit).sum()),
              "imitation_nll": float(a.mean()), "bandit_nll": float(b.mean()),
              "nll_gain": float((a - b).mean()),
              "relative_nll_reduction_percent": float((a.mean() - b.mean()) / a.mean() * 100),
              "problems_improved": sum(r["nll_gain_sum"] > 1e-6 for r in per_problem),
              "problems_worsened": sum(r["nll_gain_sum"] < -1e-6 for r in per_problem),
              "problems_tied": sum(abs(r["nll_gain_sum"]) <= 1e-6 for r in per_problem),
              "descriptive_problem_bootstrap_nll_gain_95_interval": torch.quantile(bootstrap, torch.tensor([0.025, 0.975])).tolist(),
              "bootstrap_note": "Conditional on model and threshold selected on the same validation set; not an independent test confidence interval.",
              "saved_report_reproduced": True, "per_problem": per_problem,
              "selected_curve_initial": curve[0],
              "selected_curve_best": curve[checkpoint["report"]["best_epoch"]],
              "selected_curve_last": curve[-1]}
    (root / "training/audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_problem"}, indent=2))


if __name__ == "__main__":
    main()
