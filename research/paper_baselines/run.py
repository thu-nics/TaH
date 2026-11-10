"""Resumable Standard/TaH+ evaluation using the paper's decoding budgets.

All generated text, termination reasons and actual TaH iteration counts are saved.
Repeated samples estimate pass@1 by their mean, never by best-of-eight success.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATASETS = ["gsm8k", "math500", "amc23", "aime25", "olympiadbench", "mmlu_stem", "gpqa", "humaneval", "mbpp"]


def answer_text(output):
    """Do not grade option letters or intermediate numbers inside unfinished CoT."""
    if "<think>" in output:
        return output.split("</think>", 1)[1] if "</think>" in output else ""
    return output


def grade_final(name, output, gold):
    from tah.evaluate.matheval import evaluator_map
    text = answer_text(output)
    if name in ("mmlu_stem", "gpqa"):
        # The released string extractor does not recognize \\boxed{B}.
        boxed = re.findall(r"\\boxed\s*\{\s*(?:\\(?:text|mathrm)\s*\{\s*)?([A-D])\s*\}", text)
        anchored = re.findall(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([A-D])\b", text, re.I)
        bare = re.fullmatch(r"\s*\(?([A-D])\)?[.\s]*", text)
        prediction = boxed[-1] if boxed else (anchored[-1].upper() if anchored else (bare.group(1) if bare else "No extracted answer"))
        return prediction == gold.strip().upper(), prediction
    return evaluator_map[name].rule_judge(text, gold)


@torch.inference_mode()
def generate(model, tokenizer, prompts, *, tah, cap, seed):
    from tah.model.utils import sample_next_token
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda:0")
    attn = inputs["attention_mask"]
    batch = len(prompts)
    ids = [[] for _ in prompts]
    depths = [[] for _ in prompts]
    finished = torch.zeros(batch, dtype=torch.bool, device="cuda:0")
    eos = [False] * batch
    actual_passes = [0] * batch
    prefill_passes = [0] * batch
    eos_depths = [0] * batch
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    kwargs = {"new_sequence": True} if tah else {
        "position_ids": (attn.cumsum(dim=1) - 1).clamp_min(0)
    }
    out = model(**inputs, use_cache=True, **kwargs)
    counts = out.iter_count if tah else attn
    for i in range(batch):
        prefill_passes[i] = int(counts[i][attn[i].bool()].sum())
        actual_passes[i] += prefill_passes[i]
    cache = out.past_key_values
    for step in range(cap):
        next_ids = sample_next_token(out.logits[:, -1], do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
        current_depth = out.iter_count[:, -1] if tah else torch.ones(batch, device="cuda:0", dtype=torch.long)
        for i in range(batch):
            if finished[i]:
                continue
            token = int(next_ids[i])
            if token == tokenizer.eos_token_id:
                eos[i] = True
                eos_depths[i] = int(current_depth[i])
                finished[i] = True
            else:
                ids[i].append(token)
                depths[i].append(int(current_depth[i]))
        if bool(finished.all()) or step + 1 == cap:
            break
        # Finished rows stay in the batch but contribute no valid cache positions.
        next_ids = next_ids.masked_fill(finished, tokenizer.pad_token_id)
        valid = (~finished).long().unsqueeze(1)
        attn = torch.cat((attn, valid), dim=1)
        kwargs = {"new_sequence": False} if tah else {
            "position_ids": (attn.sum(dim=1) - 1).clamp_min(0).unsqueeze(1)
        }
        out = model(input_ids=next_ids.unsqueeze(1), attention_mask=attn, past_key_values=cache, use_cache=True, **kwargs)
        cache = out.past_key_values
        count = out.iter_count[:, -1] if tah else valid[:, 0]
        for i in range(batch):
            if not finished[i]:
                actual_passes[i] += int(count[i])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3
    records = []
    for i in range(batch):
        records.append({
            "output": tokenizer.decode(ids[i], skip_special_tokens=False),
            "output_tokens": len(ids[i]), "ended_with_eos": eos[i],
            "truncated": not eos[i] and len(ids[i]) == cap,
            "prediction_depth_histogram": dict(Counter(depths[i])),
            "prediction_passes": sum(depths[i]), "eos_prediction_depth": eos_depths[i],
            "prefill_passes": prefill_passes[i], "total_valid_token_passes": actual_passes[i],
            "batch_seconds": seconds, "batch_size": batch,
            "amortized_seconds": seconds / batch, "peak_allocated_gib": peak,
        })
    del out, cache, inputs, attn
    return records


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n = len(rows)
    tokens = sum(r["output_tokens"] for r in rows)
    value = {
        "completed_samples": n, "completed_problems": len({r["problem_id"] for r in rows}),
        "pass_at_1": sum(r.get("is_correct", False) for r in rows) / n if n and "is_correct" in rows[0] else None,
        "truncation_rate": sum(r["truncated"] for r in rows) / n if n else None,
        "mean_prediction_depth": sum(r["prediction_passes"] for r in rows) / tokens if tokens else None,
        "output_tokens": tokens,
        "total_valid_token_passes": sum(r["total_valid_token_passes"] for r in rows),
        "generation_seconds": sum(r["amortized_seconds"] for r in rows),
        "peak_allocated_gib": max((r["peak_allocated_gib"] for r in rows), default=0),
        "accuracy_status": "pending_evalplus" if rows and "is_correct" not in rows[0] else "graded",
    }
    path.with_suffix(".summary.json").write_text(json.dumps(value, indent=2))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["standard", "tah"], default=["standard", "tah"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means full benchmarks")
    parser.add_argument("--pilot-cap", type=int, default=0, help="0 uses paper length budgets")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "models": args.models, "datasets": args.datasets, "batch_size": args.batch_size,
        "limit": args.limit, "pilot_cap": args.pilot_cap,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "threshold": 0.9, "max_iter": 2, "seed": 42,
        "manifest": manifest, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes": "Exact problem text/gold/repeat counts/length budgets recovered from authors' public checkpoint logs. Released repository chat and code-completion templates used. Author generated answers are never supplied as model inputs. Code correctness comes only from EvalPlus base+plus tests. Pilot results are not full benchmark scores.",
        "grading": "Primary score extracts only text after </think> when a thinking block is present. Multiple-choice scoring recognizes boxed/anchored final letters; raw repository score also retained for comparison.",
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Refusing to mix results from different run settings")
    config_path.write_text(json.dumps(config, indent=2))
    for mode in args.models:
        model_path = "models/Standard-0.6B" if mode == "standard" else "models/TaH-plus-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        if mode == "tah":
            from tah.model.tah_model import TaHForCausalLM
            model = TaHForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
            model.iter_decider.threshold.data.fill_(0.9)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        model.eval().requires_grad_(False)
        generate(model, tokenizer, [tokenizer.apply_chat_template([{"role":"user", "content":"What is 1+1?"}], tokenize=False, add_generation_prompt=True)], tah=mode=="tah", cap=8, seed=42)
        for name in args.datasets:
            info = manifest["datasets"].get(name, {})
            if info.get("status") != "ready":
                print(f"SKIP {mode}/{name}: {info.get('reason', 'not prepared')}", flush=True)
                continue
            source = Path(info.get("data_file", ROOT / "data" / f"{name}.jsonl"))
            if info.get("data_sha256") and hashlib.sha256(source.read_bytes()).hexdigest() != info["data_sha256"]:
                raise ValueError(f"Dataset {name} differs from manifest")
            problems = [json.loads(line) for line in source.read_text().splitlines()]
            if args.limit:
                problems = problems[:args.limit]
            repeats = info.get("repeat_size", 8 if name in ("amc23", "aime25") else 1)
            cap = args.pilot_cap or info.get("max_new_tokens", 4096 if name in ("gsm8k", "mmlu_stem") else 8192)
            target = args.output / f"{mode}_{name}.jsonl"
            completed = set()
            if target.exists():
                completed = {(r["problem_id"], r["sample_idx"]) for r in map(json.loads, target.read_text().splitlines())}
            jobs = [(p, sample) for p in problems for sample in range(repeats) if (p["id"], sample) not in completed]
            is_code = name in ("humaneval", "mbpp")
            print(f"START {mode}/{name}: remaining={len(jobs)}, cap={cap}, repeats={repeats}", flush=True)
            for offset in range(0, len(jobs), args.batch_size):
                batch = jobs[offset:offset + args.batch_size]
                prompts = []
                for p, sample in batch:
                    if is_code:
                        from tah.evaluate.codeeval import make_raw_chat_prompt_for_code_evaluation
                        prompt = make_raw_chat_prompt_for_code_evaluation(p["question"], False, tokenizer)
                    else:
                        prompt = tokenizer.apply_chat_template([{"role":"user", "content":p["question"]}], tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                # Stable seeds across resumed batches and models.
                seed = 42 + int(hashlib.sha256(f"{name}:{batch[0][0]['id']}:{batch[0][1]}".encode()).hexdigest()[:8], 16) % 1000000
                records = generate(model, tokenizer, prompts, tah=mode=="tah", cap=cap, seed=seed)
                with target.open("a") as handle:
                    for (p, sample), record in zip(batch, records):
                        record.update(problem_id=p["id"], sample_idx=sample, dataset=name, model=mode, seed=seed, question=p["question"], correct_answer=p["answer"])
                        if is_code:
                            from tah.evaluate.codeeval import sanitize
                            record["solution"] = sanitize(record["output"], p["entry_point"])
                            record["task_id"] = p["id"]
                        else:
                            from tah.evaluate.matheval import evaluator_map
                            raw_correct, raw_predicted = evaluator_map[name].rule_judge(record["output"], p["answer"])
                            correct, predicted = grade_final(name, record["output"], p["answer"])
                            record.update(is_correct=correct, predicted_answer=predicted, repository_raw_is_correct=raw_correct, repository_raw_predicted=raw_predicted)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary = summarize(target)
                print(f"PROGRESS {mode}/{name}: {summary}", flush=True)
            if is_code and not args.limit:
                from tah.evaluate.codeeval import evaluate
                solutions = target.with_name(target.stem + "_solutions.jsonl")
                records = [json.loads(line) for line in target.read_text().splitlines()]
                solutions.write_text("".join(json.dumps({"task_id":r["task_id"], "solution":r["solution"]}) + "\n" for r in records))
                evaluate(name, str(solutions), parallel=4)
                eval_path = solutions.with_suffix(".eval_results.json")
                scores = json.loads(eval_path.read_text())
                summary = summarize(target)
                summary.update(pass_at_1=scores["pass_at_k"]["plus"]["pass@1"], accuracy_status="evalplus_base_and_plus_graded")
                target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
            print(f"DONE {mode}/{name}", flush=True)
        del model
        torch.cuda.empty_cache()
    (args.output / "COMPLETE").write_text("All accessible requested benchmark jobs finished; see manifest for blocked inputs.\n")


if __name__ == "__main__":
    main()
