"""Select complete MoT math solutions; exclude normalized benchmark overlap."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .core import digest


def normalized(text):
    text = re.split(r"Please reason step.by.step", text, flags=re.I)[0]
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def shingles(text):
    tokens = normalized(text).split()
    return {tuple(tokens[i:i + 5]) for i in range(max(0, len(tokens) - 4))}


class OverlapScreen:
    def __init__(self, questions):
        self.exact = {normalized(q) for q in questions}
        self.shingles = [shingles(q) for q in questions]
        self.index = defaultdict(set)
        for i, terms in enumerate(self.shingles):
            for term in terms:
                self.index[term].add(i)

    def overlaps(self, question):
        if normalized(question) in self.exact:
            return True
        terms = shingles(question)
        candidates = set().union(*(self.index[t] for t in terms)) if terms else set()
        for i in candidates:
            other = self.shingles[i]
            union = terms | other
            if union and len(terms & other) / len(union) >= 0.8:
                return True
        return False


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--max-reference-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--scan-limit", type=int, default=93733)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="models/TaH-plus-0.6B")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    benchmark_files = [Path(f"research/paper_baselines/author_data/{name}.jsonl") for name in ("gsm8k", "math500")]
    questions = [r["question"] for file in benchmark_files for r in map(json.loads, file.read_text().splitlines())]
    screen = OverlapScreen(questions)
    repo = "open-r1/Mixture-of-Thoughts"
    revision = HfApi().dataset_info(repo).sha
    source = load_dataset(repo, "math", split="train", revision=revision, streaming=True)
    source = source.shuffle(seed=args.seed, buffer_size=2048)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    counts, selected, seen = Counter(), [], set()
    for scanned, row in enumerate(source):
        if scanned >= args.scan_limit:
            break
        counts["scanned"] += 1
        if counts["scanned"] % 1000 == 0:
            print(f"Scan progress: selected {len(selected)}, counts={dict(counts)}", flush=True)
        messages = row["messages"]
        if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            counts["unsupported_messages"] += 1
            continue
        question, reference = [m["content"] for m in messages]
        key = normalized(question)
        if key in seen:
            counts["duplicate"] += 1
            continue
        if screen.overlaps(question):
            counts["benchmark_overlap"] += 1
            continue
        length = len(tokenizer.encode(reference, add_special_tokens=False))
        if not 16 <= length <= args.max_reference_tokens:
            counts["reference_length"] += 1
            continue
        if "</think>" not in reference or not reference.split("</think>", 1)[1].strip():
            counts["missing_final_answer"] += 1
            continue
        prompt_length = len(tokenizer.apply_chat_template([messages[0]], tokenize=True, add_generation_prompt=True))
        if prompt_length > args.max_prompt_tokens:
            counts["prompt_length"] += 1
            continue
        seen.add(key)
        selected.append({"id": "mot_" + digest(key)[:16], "question": question,
                         "reference": reference, "reference_tokens": length,
                         "source": row["source"]})
        if len(selected) % 25 == 0:
            print(f"Selected {len(selected)}/{args.count}; scanned {counts['scanned']}", flush=True)
        if len(selected) >= args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"Only selected {len(selected)} complete solutions: {dict(counts)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    manifest = {"dataset": repo, "config": "math", "split": "train", "revision": revision,
                "seed": args.seed, "problems": len(selected), "counts": dict(counts),
                "reference_tokens": sum(r["reference_tokens"] for r in selected),
                "max_reference_tokens": args.max_reference_tokens, "truncated": False,
                "selection": "Seeded streaming shuffle; complete short solutions only (length-biased pilot)",
                "overlap_screen": "normalized exact match and 5-word-shingle Jaccard >= 0.8",
                "benchmark_files": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in benchmark_files},
                "reference_sha256": digest(selected)}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
