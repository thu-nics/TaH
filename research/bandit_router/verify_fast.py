"""Check optimized collection against two separate action forwards on CUDA."""
import argparse
from types import SimpleNamespace

import torch

from .core import RoutingDecider, override_decider, paired_step, paired_step_fast
from .run import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    model, tokenizer = load_model(SimpleNamespace(model="models/TaH-plus-0.6B", device=args.device,
                                                  imitation_threshold=0.9))
    n = model.config.num_hidden_layers
    decider = RoutingDecider(model.iter_decider, (1, n // 2, n)).eval()
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What is 17 + 25?"}],
                                          tokenize=True, add_generation_prompt=True)
    reference = tokenizer.encode("<think>\n17 + 25 = 42.\n</think>\nThe answer is \\boxed{42}.", add_special_tokens=False)
    with torch.inference_mode(), override_decider(model, decider):
        cache = model(input_ids=torch.tensor([prompt[:-1]], device=model.device), use_cache=True).past_key_values
        current = prompt[-1]
        for target in reference:
            inputs = torch.tensor([[current]], device=model.device)
            label = torch.tensor(target, device=model.device)
            slow = paired_step(model, decider, cache, inputs, label)
            fast = paired_step_fast(model, decider, cache, inputs, label)
            torch.testing.assert_close(slow[0], fast[0], rtol=0, atol=0)
            torch.testing.assert_close(slow[1], fast[1], rtol=0, atol=0)
            assert slow[2] == fast[2]
            for name in ("_k", "_v", "_pos", "_valid"):
                a, b = getattr(slow[3], name), getattr(fast[3], name)
                assert a.keys() == b.keys()
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            cache, current = fast[3], target
    print(f"PASS: {len(reference)} tokens; both losses, states, committed KV and metadata bit-exact", flush=True)


if __name__ == "__main__":
    main()
