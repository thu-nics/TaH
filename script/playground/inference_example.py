"""End-to-end demo: load TaH-plus-1.7B from HF Hub and run sampling generation
with per-token iter-count colouring.

Run:
    python script/playground/inference_example.py
"""
import torch
from transformers import AutoTokenizer

from tah.model.tah_model import TaHForCausalLM
from tah.model.utils import IterCountColors, TaHForCasualLM_generate


def main():
    model_name = "nics-efc/TaH-plus-1.7B"
    device_map = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tah_model = TaHForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    print(f"Device: {tah_model.device}, Dtype: {tah_model.dtype}")

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
    ).to(device=tah_model.device)

    print("\nInitial input:")
    for i, p in enumerate(prompts):
        print(f"Sample {i+1}: {p[:100]}{'…' if len(p) > 100 else ''}")
    print(f"Input IDs shape: {tuple(model_inputs.input_ids.shape)}")
    print(IterCountColors.get_legend())

    output_tokens, final_texts = TaHForCasualLM_generate(
        tah_model=tah_model,
        tokenizer=tokenizer,
        model_inputs=dict(model_inputs),
        max_new_tokens=16384,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        verbose=True,
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
