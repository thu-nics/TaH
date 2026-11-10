"""Check batching/padding/cache behavior against the released generation helper."""
import importlib.util
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tah.model.tah_model import TaHForCausalLM
from tah.model.utils import TaHForCasualLM_generate
import tah.model.utils as utils

spec = importlib.util.spec_from_file_location("paper_run", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    original_sample = utils.sample_next_token
    utils.sample_next_token = lambda logits, **kwargs: logits.argmax(dim=-1)
    try:
        tokenizer = AutoTokenizer.from_pretrained("models/TaH-plus-0.6B")
        tokenizer.pad_token = tokenizer.eos_token
        model = TaHForCausalLM.from_pretrained("models/TaH-plus-0.6B", torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
        model.iter_decider.threshold.data.fill_(0.9)
        prompts = [tokenizer.apply_chat_template([{"role":"user", "content": q}], tokenize=False, add_generation_prompt=True) for q in ("What is 1+1?", "Alice has 3 apples and buys 7 more. How many apples does she have?")]
        result = runner.generate(model, tokenizer, prompts, tah=True, cap=24, seed=42)
        enc = tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt").to("cuda:0")
        _, official = TaHForCasualLM_generate(model, tokenizer, dict(enc), max_new_tokens=24, do_sample=False, verbose=False)
        assert [r["output"] for r in result] == official, "Batched runner differs from released generation helper"
        for i, prompt in enumerate(prompts):
            single = runner.generate(model, tokenizer, [prompt], tah=True, cap=24, seed=42)[0]
            assert single["output"] == result[i]["output"], "Padding changes greedy output"
            assert single["prediction_depth_histogram"] == result[i]["prediction_depth_histogram"]
        calls = [0]
        def uneven_eos(logits, **kwargs):
            calls[0] += 1
            tokens = logits.argmax(dim=-1)
            tokens[0] = tokenizer.eos_token_id
            if calls[0] == 3:
                tokens[1] = tokenizer.eos_token_id
            return tokens
        utils.sample_next_token = uneven_eos
        uneven = runner.generate(model, tokenizer, prompts, tah=True, cap=8, seed=42)
        assert [r["output_tokens"] for r in uneven] == [0, 2]
        assert all(r["ended_with_eos"] and not r["truncated"] for r in uneven)
        assert uneven[0]["total_valid_token_passes"] == uneven[0]["prefill_passes"]
        assert runner.answer_text("<think>Options are A, B, C, D") == ""
        assert runner.answer_text("<think>maybe A</think>\\boxed{B}") == "\\boxed{B}"
        from tah.evaluate.matheval import evaluator_map
        assert evaluator_map["gsm8k"].rule_judge("\\boxed{42}", "42")[0]
        assert not evaluator_map["gsm8k"].rule_judge("\\boxed{41}", "42")[0]
        assert runner.grade_final("mmlu_stem", "\\boxed{B}", "B")[0]
        assert not runner.grade_final("mmlu_stem", "\\boxed{A}", "B")[0]
        assert not runner.grade_final("mmlu_stem", "<think>The answer is B", "B")[0]
        assert runner.grade_final("mmlu_stem", "<think>Maybe A</think>The answer is B.", "B")[0]
        print("PASS: released generation parity, left padding, routing accounting, final-answer filtering and grading", flush=True)
    finally:
        utils.sample_next_token = original_sample


if __name__ == "__main__":
    main()
