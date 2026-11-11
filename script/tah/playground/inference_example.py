"""End-to-end demo: load TaH-plus-1.7B from HF Hub and run sampling generation
with per-token iter-count colouring.

Run:
    python script/tah/playground/inference_example.py                    # quick demo (512 tokens)
    python script/tah/playground/inference_example.py --max-new-tokens 16384  # full reasoning chain
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

_ITER_COLORS = {1: "\033[0m", 2: "\033[92m", 3: "\033[94m", 4: "\033[91m",
                5: "\033[95m", 6: "\033[93m"}


def _color(count):
    return _ITER_COLORS.get(count, "\033[96m")


def _print_coloured(token_ids, pass_counts, tokenizer):
    for token, count in zip(token_ids, pass_counts):
        print(f"{_color(count)}{tokenizer.decode([token])}\033[0m", end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nics-efc/TaH-plus-1.7B", help="HF id or local path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("tah_engine", "tah"), default="tah_engine",
                        help="Generation backend (default engine; tah is the correctness reference)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engine-options", type=json.loads, default=None,
                        help="JSON object with TaHEngine constructor options")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Cap on generated tokens (default 512; "
                             "raise to 16384+ to see a full reasoning chain).")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "Six points $A, B, C, D, E$ and $F$ lie in a straight line in that order. "
        "Suppose that $G$ is a point not on the line and that $AC = 26$, $BD = 22$, "
        "$CE = 31$, $DF = 33$, $AF = 73$, $CG = 40$, and $DG = 30$. "
        "Find the area of $\\triangle BGE$.",
    ]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        for p in prompts
    ]
    model_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, padding_side="left",
    )

    print("\nInitial input:")
    for i, p in enumerate(prompts):
        print(f"Sample {i+1}: {p[:100]}{'…' if len(p) > 100 else ''}")
    print(f"Input IDs shape: {tuple(model_inputs.input_ids.shape)}")
    print("Color legend: " + ", ".join(f"{_color(n)}={n} iter\033[0m" for n in range(1, 8)))

    if args.backend == "tah_engine":
        try:
            from tah.engine import TaHEngine, Request, SamplingParams, check_environment
            check_environment()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(f"tah_engine environment check failed: {exc}\n"
                               "Use --backend tah for the correctness reference backend.") from exc
        path = args.model
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download
            path = snapshot_download(path)
        if args.engine_options is not None and not isinstance(args.engine_options, dict):
            parser.error("--engine-options must be a JSON object")
        options = dict(args.engine_options or {})
        options.setdefault("max_batch_size", len(prompts))
        options.setdefault("graph_batch_sizes", list(range(1, len(prompts) + 1)))
        options.setdefault("device", args.device)
        engine = TaHEngine(path, **options)
        results = engine.generate([Request(tokenizer.encode(t), args.max_new_tokens, args.seed + i)
                                   for i, t in enumerate(texts)],
                                  SamplingParams(temperature=.6, top_p=.95, top_k=20))
        output_tokens = [r.token_ids for r in results]
        final_texts = [r.text for r in results]
        for result in results:
            print(f"\nSample {result.request_id + 1} coloured generation:")
            # Counts belong to the context position that sampled each token.
            _print_coloured(result.token_ids, result.generated_pass_counts, tokenizer)
    else:
        from tah.model.tah_model import TaHForCausalLM
        from tah.model.utils import TaHForCasualLM_generate, set_all_seeds
        set_all_seeds(args.seed)
        tah_model = TaHForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=args.device,
            attn_implementation="sdpa",
        )
        output_tokens, final_texts = TaHForCasualLM_generate(
            tah_model=tah_model, tokenizer=tokenizer,
            model_inputs=dict(model_inputs.to(device=tah_model.device)),
            max_new_tokens=args.max_new_tokens, do_sample=True, temperature=.6,
            top_p=.95, top_k=20, min_p=0., verbose=True,
        )

    print("\n" + "=" * 50)
    print("FINAL GENERATED TEXTS")
    print("=" * 50)
    for i, text in enumerate(final_texts):
        print(f"\nSample {i+1} ({len(output_tokens[i])} tokens):")
        print("-" * 30)
        print(text)


if __name__ == "__main__":
    main()
