"""Select model-generated sequences for R1-O1 and write a token-level manifest.

    python -m script.tah.validation.select_sequences --output data/engine/oracle/manifest.json

Sources are evaluation ``samples.jsonl`` files: reference runs store text only (the
output is re-tokenized and kept only when the token count equals the saved
``pass_stat_tokens`` and decoding reproduces the text exactly); engine runs store
``token_ids`` (kept when the rebuilt prompt matches the saved ``prompt_pass_counts``
length). Prompts are rebuilt with the harness chat template. Selection is
deterministic: candidates are ordered by a seeded hash per (backend, dataset) and
taken round-robin until both minimums hold with equal counts per backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

DEFAULT_SOURCES = [
    "reference:math500=data/engine/wacc_reference_recovery/pending/combined/math500_tah/20261011_182527/TASK_0_200/samples.jsonl",
    "reference:gpqa=data/engine/wacc_reference_recovery/pending/combined/gpqa_tah/20261011_182527/samples.jsonl",
    "engine:math500=data/engine/wacc_engine_protocol_seed42/canonical/math500_tah_engine/20261013T135820+0000/TASK_0_200/samples.jsonl",
    "engine:gpqa=data/engine/wacc_engine_protocol_seed42/canonical/gpqa_tah_engine/20261013T135820+0000/samples.jsonl",
    "engine:math500=data/engine/wp10/wacc-seed2024/math500_tah_engine/20261013_133615/TASK_0_200/samples.jsonl",
    "engine:gpqa=data/engine/wp10/wacc-seed2024/gpqa_tah_engine/20261013_133615/samples.jsonl",
]
CODE_DATASETS = ("humaneval", "mbpp")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_source(text):
    kind, rest = text.split(":", 1)
    dataset, path = rest.split("=", 1)
    if kind not in ("reference", "engine"):
        raise ValueError(f"source backend must be reference or engine: {text}")
    return kind, dataset, Path(path)


def candidates(kind, dataset, path, tokenizer, max_new_tokens, max_length):
    from tah.evaluate.jobs import _make_prompt
    eos = tokenizer.eos_token_id
    kept, skipped = [], {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1
    for line in path.open():
        r = json.loads(line)
        # As the harness backends tokenize prompts (default special-token handling).
        prompt = tokenizer.encode(_make_prompt(r["problem"], tokenizer, is_code=dataset in CODE_DATASETS))
        if kind == "engine":
            tokens = list(r["token_ids"])
            if len(prompt) != len(r["prompt_pass_counts"]):
                skip("prompt length differs from saved prompt_pass_counts")
                continue
            if len(tokens) != len(r["pass_counts"]):
                skip("token count differs from saved pass_counts")
                continue
            finish = r["finish_reason"]
        else:
            tokens = tokenizer.encode(r["output"], add_special_tokens=False)
            if len(tokens) != r.get("pass_stat_tokens"):
                skip("re-tokenized length differs from pass_stat_tokens")
                continue
            if tokenizer.decode(tokens) != r["output"]:
                skip("re-tokenized text does not round-trip")
                continue
            finish = "length" if len(tokens) >= max_new_tokens else "eos"
        if not tokens and finish != "eos":
            skip("empty output")
            continue
        forced = tokens + ([eos] if finish == "eos" else [])
        if len(prompt) + len(forced) - 1 > max_length:
            skip("longer than max length")
            continue
        sid = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{kind}-{dataset}-{r['id']}-s{r.get('sample', 0)}-{sha256_text(str(path))[:6]}")
        kept.append(dict(id=sid, backend=kind, dataset=dataset, source=str(path), source_id=r["id"],
                         sample=r.get("sample", 0), finish=finish, correct=r.get("is_correct"),
                         prompt=prompt, forced=forced))
    return kept, skipped


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def choose(groups, seed, min_sequences, min_positions):
    """Per backend, interleave its datasets in seeded-hash order; take one per backend per round."""
    streams = {}
    for backend in sorted({k[0] for k in groups}):
        lists = [sorted(groups[k], key=lambda s: sha256_text(f"{seed}:{s['id']}"))
                 for k in sorted(groups) if k[0] == backend]
        streams[backend] = [s for i in range(max(map(len, lists))) for items in lists if i < len(items)
                            for s in [items[i]]]
    chosen, positions = [], 0
    for i in range(min(map(len, streams.values()))):
        if len(chosen) >= min_sequences and positions >= min_positions:
            return chosen
        for backend in sorted(streams):
            chosen.append(streams[backend][i])
            positions += len(streams[backend][i]["forced"])
    if len(chosen) >= min_sequences and positions >= min_positions:
        return chosen
    raise SystemExit("not enough candidates to meet the minimums with balanced backends")


def validity_subset(sequences, batch, min_sequences=20, min_positions=20_000):
    chosen, positions = [], 0
    for seq in sequences:
        if len(chosen) >= min_sequences and positions >= min_positions and len(chosen) % batch == 0:
            break
        chosen.append(seq["id"])
        positions += len(seq["forced"])
    return chosen


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=None,
                   help="backend:dataset=path/samples.jsonl (repeatable; default: the R1 runs)")
    p.add_argument("--tokenizer", type=Path, default=Path("models/TaH-plus-1.7B"))
    p.add_argument("--max-new-tokens", type=int, default=4096, help="generation cap of the reference sources")
    p.add_argument("--max-length", type=int, default=8192, help="teacher-forced positions per sequence")
    p.add_argument("--min-sequences", type=int, default=200)
    p.add_argument("--min-positions", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--validity-batch", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    groups, sources = {}, []
    for text in args.source or DEFAULT_SOURCES:
        kind, dataset, path = parse_source(text)
        if not path.exists():
            print(f"skip missing source {path}")
            continue
        kept, skipped = candidates(kind, dataset, path, tokenizer, args.max_new_tokens, args.max_length)
        groups.setdefault((kind, dataset), []).extend(kept)
        sources.append(dict(backend=kind, dataset=dataset, path=str(path), sha256=sha256(path),
                            kept=len(kept), skipped=skipped))
        print(json.dumps(sources[-1]))
    chosen = choose(groups, args.seed, args.min_sequences, args.min_positions)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    manifest = dict(version=1, tokenizer=str(args.tokenizer), eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad, selection=dict(seed=args.seed, min_sequences=args.min_sequences,
                                                     min_positions=args.min_positions,
                                                     max_length=args.max_length,
                                                     max_new_tokens=args.max_new_tokens),
                    sources=sources, sequences=chosen,
                    subsets=dict(validity=validity_subset(chosen, args.validity_batch)),
                    totals=dict(sequences=len(chosen), generated_positions=sum(len(s["forced"]) for s in chosen),
                                by_backend={b: sum(s["backend"] == b for s in chosen)
                                            for b in sorted({s["backend"] for s in chosen})},
                                by_dataset={d: sum(s["dataset"] == d for s in chosen)
                                            for d in sorted({s["dataset"] for s in chosen})}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    print(json.dumps(manifest["totals"]))


if __name__ == "__main__":
    main()
