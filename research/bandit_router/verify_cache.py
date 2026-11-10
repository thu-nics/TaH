"""Replay imitation without branching; verify each cached state and chosen loss."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .core import RoutingDecider, digest, override_decider
from .run import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-problems", type=int, default=0, help="0 verifies every problem")
    args = parser.parse_args()
    saved = torch.load(args.cache, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    model, _ = load_model(SimpleNamespace(model=metadata["model"], device=args.device,
                                          imitation_threshold=metadata["imitation_threshold"]))
    decider = RoutingDecider(model.iter_decider, metadata["hidden_layers"]).eval()
    checked = 0
    with torch.inference_mode(), override_decider(model, decider):
        for problem in saved["problems"][:args.max_problems or None]:
            prompt = problem["prompt_token_ids"]
            cache, history = None, []
            if len(prompt) > 1:
                output = model(input_ids=torch.tensor([prompt[:-1]], device=model.device), use_cache=True)
                cache, history = output.past_key_values, output.iter_count[0].tolist()
            prefix, current = prompt[:-1].copy(), prompt[-1]
            for index, target in enumerate(problem["reference_token_ids"]):
                row_index = problem["row_start"] + index
                row = saved["rows"][row_index]
                assert row["history_sha256"] == digest({"tokens": prefix, "depths": history})
                output = model(input_ids=torch.tensor([[current]], device=model.device),
                               past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                depth = int(output.iter_count.item())
                assert depth == row["imitation_depth"]
                expected = saved["features"][row_index]
                torch.testing.assert_close(decider.state[0].cpu().to(expected.dtype), expected, rtol=0, atol=0)
                loss = F.cross_entropy(output.logits[:, -1].float(), torch.tensor([target], device=model.device))
                torch.testing.assert_close(loss.cpu(), saved["action_losses"][row_index, depth - 1], rtol=1e-6, atol=1e-6)
                prefix.append(current)
                history.append(depth)
                current = target
                checked += 1
            assert history == problem["imitation_history_depths"]
    report = {"verified_rows": checked, "verified_problems": min(args.max_problems or len(saved["problems"]), len(saved["problems"])),
              "fixed_history_replay": "passed", "depth1_features": "bit_exact_at_storage_dtype",
              "imitation_action_losses": "passed"}
    args.cache.with_suffix(".verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
