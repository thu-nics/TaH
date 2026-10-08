"""Regenerate the released 1.7B prompts and tokenize assistant supervision.

AM prompts are already formatted with their domain instructions. Tool prompts
contain reference conversations and tool schemas. Serve Qwen3-8B and Qwen3-32B
at the supplied mini-sglang URLs; the latter simulates unmatched tool calls.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from datasets import Dataset, Features, Sequence, Value
from tool_rollout import Engine, Servers
from tqdm import tqdm
from transformers import AutoTokenizer


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def assistant_spans(text):
    marker, ending = "<|im_start|>assistant\n", "<|im_end|>\n"
    spans, pos = [], 0
    while (start := text.find(marker, pos)) >= 0:
        end = text.find(ending, start + len(marker))
        if end < 0:
            return []
        pos = end + len(ending)
        spans.append((start + len(marker), pos))
    return spans


def tokenize_row(tokenizer, text, spans, truncate=False):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, mask = encoded["input_ids"], []
    cursor = 0
    for start, end in encoded["offset_mapping"]:
        while cursor < len(spans) and spans[cursor][1] <= start:
            cursor += 1
        if end <= start or cursor == len(spans):
            mask.append(0)
            continue
        lo, hi = spans[cursor]
        if start < hi and end > lo and not (lo <= start and end <= hi):
            return None
        mask.append(int(lo <= start and end <= hi))
    if len(ids) > 16384:
        if not truncate:
            return None
        ids, mask = ids[:16384], mask[:16384]
        text = tokenizer.decode(ids)
    if not any(mask):
        return None
    return {"real_text": text, "real_token": ids, "mask": mask}


def tool_rows(tokenizer, record):
    # Keep only complete, well-formed episodes; later episodes cannot use a
    # malformed earlier episode as history.
    messages, turns = record["messages"], iter(record["turns"])
    safe, episode, cut = [], -1, None
    for message in messages:
        role = message["role"]
        if role == "user":
            episode += 1
        if role == "assistant":
            generation = next(turns)["gen"]
            invalid = (
                generation.get("truncated")
                or generation.get("bad_call_json")
                or generation.get("unclosed_call")
                or generation.get("nested_think")
                or (
                    generation.get("unbalanced_think")
                    and not generation.get("repaired_unclosed_think")
                )
                or any(
                    error in {"unknown_tool", "arguments_not_object"}
                    for error in generation.get("call_schema_errors", [])
                )
            )
            invalid |= any(
                tag in (message.get("content") or "")
                for tag in (
                    "<tool_call>",
                    "</tool_call>",
                    "<tool_response>",
                    "</tool_response>",
                )
            )
            if invalid:
                cut = episode
                break
        elif role in {"user", "tool"}:
            tags = ("<think>", "</think>")
            if role == "user":
                tags += (
                    "<tool_call>",
                    "</tool_call>",
                    "<tool_response>",
                    "</tool_response>",
                )
            if any(tag in str(message.get("content") or "") for tag in tags):
                cut = episode
                break
        safe.append(message)
    if cut is not None:
        starts = [i for i, message in enumerate(safe) if message["role"] == "user"]
        if cut < len(starts):
            safe = safe[: starts[cut]]
    starts = [i for i, message in enumerate(safe) if message["role"] == "user"]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(safe)
        current = safe[start:end]
        count = sum(message["role"] == "assistant" for message in current)
        if not count:
            continue
        text = tokenizer.apply_chat_template(
            safe[:end],
            tools=record["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        spans = assistant_spans(text)[-count:]
        if len(spans) != count:
            continue
        row = tokenize_row(tokenizer, text, spans)
        if row:
            yield row


def generate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    client = Servers(args.urls, args.model)
    engine = None
    if args.kind == "tool_calling":
        if not args.sim_urls:
            raise ValueError("Tool regeneration requires --sim-urls serving Qwen3-32B")
        simulator_tokenizer = AutoTokenizer.from_pretrained(args.sim_model)
        cfg = SimpleNamespace(
            max_new_tokens=3072,
            max_steps=8,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            sim_max_tokens=512,
            sim_temperature=0.7,
            eos_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
        engine = Engine(
            client,
            Servers(args.sim_urls, args.sim_model),
            tokenizer,
            simulator_tokenizer,
            cfg,
        )

    def one(prompt):
        if engine:
            record = engine.run_sequential(prompt)
            record.update(uid=prompt["uid"], tools=prompt["tools"])
            return record
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["input"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        budget = 16384 - len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        if budget <= 0:
            raise ValueError(f"Prompt {prompt['id']} exceeds the context limit")
        result = client.complete(prefix, budget, 0.6)
        return {
            "id": prompt["id"],
            "prefix": prefix,
            "output": result["text"],
            "truncated": not result["token_ids"]
            or result["token_ids"][-1] != tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with (
        ThreadPoolExecutor(args.workers) as pool,
        Path(args.output).open("w") as stream,
    ):
        for record in tqdm(pool.map(one, read_jsonl(args.prompts))):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


def build(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    seen = set()
    if args.exclude:
        from datasets import load_from_disk

        for text in load_from_disk(args.exclude)["real_text"]:
            seen.add(hashlib.blake2b(text.encode(), digest_size=16).digest())

    def rows():
        for path in args.inputs:
            for record in read_jsonl(path):
                if "messages" in record:
                    candidates = tool_rows(tokenizer, record)
                    prefix = "tool_calling"
                else:
                    text = record["prefix"] + record["output"]
                    if not record["truncated"]:
                        text += "<|im_end|>\n"
                    row = tokenize_row(
                        tokenizer, text, [(len(record["prefix"]), len(text))], True
                    )
                    candidates = [row] if row else []
                    prefix = record["id"]
                for row in candidates:
                    digest = hashlib.blake2b(
                        row["real_text"].encode(), digest_size=16
                    ).digest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    row["data_id"] = f"{prefix}-{digest.hex()}"
                    yield row

    features = Features(
        {
            "real_text": Value("string"),
            "real_token": Sequence(Value("int32")),
            "mask": Sequence(Value("int8")),
            "data_id": Value("string"),
        }
    )
    # A new cache directory ensures that rerunning generation at the same input
    # paths never reuses an older tokenized dataset.
    with TemporaryDirectory(prefix="tah2-data-") as cache:
        Dataset.from_generator(rows, features=features, cache_dir=cache).save_to_disk(
            args.output
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--kind", choices=["am", "tool_calling"], required=True)
    gen.add_argument("--prompts", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--urls", nargs="+", required=True)
    gen.add_argument("--sim-urls", nargs="+", default=[])
    gen.add_argument("--model", default="Qwen/Qwen3-8B")
    gen.add_argument("--sim-model", default="Qwen/Qwen3-32B")
    gen.add_argument("--workers", type=int, default=16)
    prep = commands.add_parser("build")
    prep.add_argument("--inputs", nargs="+", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    prep.add_argument(
        "--exclude", help="Prepared eval split to exclude when building train"
    )
    args = parser.parse_args()
    (generate if args.command == "generate" else build)(args)


if __name__ == "__main__":
    main()
