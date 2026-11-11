"""Frozen two-stage, 512-token C1-only workload and paired analysis (C2 withdrawn)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze import BOOTSTRAP_SEED, RESAMPLES, run_info
from .common import write_json
from .coupled_analyze import load_run, prefix
from .coupled_run import digest

PROTOCOL = "R1-O2-C1-short"
CAP = 512
PROMPT_LIMIT = 675
EXTRA = 512
INITIAL = 692


def validate(manifest):
    ps = manifest["problems"]
    ids = [p["id"] for p in ps]
    stages = manifest["subsets"]
    if manifest.get("protocol") != PROTOCOL or manifest.get("max_new_tokens") != CAP:
        raise ValueError("wrong short C1 protocol/cap")
    if manifest.get("prompt_limit") != PROMPT_LIMIT or any(len(p["prompt"]) > PROMPT_LIMIT for p in ps):
        raise ValueError("short C1 prompt exceeds the frozen memory envelope")
    if not ids or len(set(ids)) != len(ids) or len(set(p["seed"] for p in ps)) != len(ps):
        raise ValueError("empty/duplicate C1 identities or seeds")
    if not all(p["prompt"] for p in ps) or manifest["oracle_ids"] != ids:
        raise ValueError("every C1 problem needs a prompt and oracle")
    if stages["stage1"] + stages["stage2"] != ids or set(stages["stage1"]) & set(stages["stage2"]):
        raise ValueError("stages must be disjoint ordered portions of the master manifest")
    if not manifest.get("fixture"):
        if len(stages["stage1"]) != INITIAL:
            raise ValueError("short C1 stage1 must have exactly 692 frozen problems")
        if len(stages["stage2"]) != EXTRA:
            raise ValueError("C1 stage 2 must add exactly 512 frozen problems")
        if {p["dataset"] for p in ps[:len(stages["stage1"]) ]} != {"math500", "gpqa"}:
            raise ValueError("stage 1 requires Math500 and GPQA")
        if any(p["dataset"] != "gsm8k" for p in ps[len(stages["stage1"]):]):
            raise ValueError("stage 2 must be the frozen GSM8K increment")
        if manifest.get("selection_rule") != "all eligible Math500/GPQA; SHA256 order first 512 eligible GSM8K":
            raise ValueError("unknown workload selection rule")


def prepare(model, output, source_root=Path("data/engine/paper_protocol_seed42/canonical"), *, fixture=False):
    """Read only question/answer/identity fields from complete R2 files; tokenizer only."""
    from transformers import AutoTokenizer
    from tah.evaluate.jobs import _make_prompt
    tok = AutoTokenizer.from_pretrained(str(model))
    groups, excluded, sources, ordinal = {}, [], [], 0
    for ds in ("math500", "gpqa", "gsm8k"):
        files = sorted(Path(source_root).glob(f"{ds}_tah_engine/**/samples.jsonl"))
        if len(files) != 1:
            raise ValueError(f"need exactly one canonical source for {ds}")
        path = files[0]
        rows = [json.loads(line) for line in path.open()]
        rows.sort(key=lambda r: int(r["id"].rsplit("_", 1)[1]))
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("source contains repeated problems")
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        kept = []
        for index, row in enumerate(rows):
            prompt_text = _make_prompt(row["problem"], tok, is_code=False)
            prompt = tok.encode(prompt_text)
            p = dict(id=row["id"], dataset=ds, index=index, seed=42 + ordinal,
                     prompt=prompt, prompt_text=prompt_text, answer=row["correct_answer"])
            ordinal += 1  # assign before exclusions; never renumber later
            if len(prompt) <= PROMPT_LIMIT:
                kept.append(p)
            else:
                excluded.append(dict(id=p["id"], dataset=ds, prompt_tokens=len(prompt)))
        groups[ds] = kept
    def key(p):
        return len(p["prompt"]), p["id"]
    first = sorted(groups["math500"] + groups["gpqa"], key=key)
    candidates = sorted(groups["gsm8k"], key=lambda p: hashlib.sha256(f"20261013:{p['id']}".encode()).hexdigest())
    if len(candidates) < EXTRA:
        raise ValueError("not enough eligible independent GSM8K problems")
    second = sorted(candidates[:EXTRA], key=key)
    problems = first + second
    manifest = dict(version=1, protocol=PROTOCOL, max_new_tokens=CAP, prompt_limit=PROMPT_LIMIT,
                    uniform_seed=42, seed_mapping="42 + pre-exclusion ordinal in Math500, GPQA, GSM8K",
                    selection_rule="all eligible Math500/GPQA; SHA256 order first 512 eligible GSM8K",
                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                    problems=problems, oracle_ids=[p["id"] for p in problems], sources=sources,
                    excluded=excluded, subsets=dict(stage1=[p["id"] for p in first],
                                                   stage2=[p["id"] for p in second]))
    if fixture:
        manifest["fixture"] = True
    validate(manifest)
    output = Path(output)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise ValueError("refusing to replace a different frozen C1 manifest")
    write_json(output, manifest)
    print(json.dumps(dict(stage1=len(first), stage2=len(second), excluded=excluded)))


def bound(engine, reference, alpha=.025):
    e, r = np.asarray(engine, float), np.asarray(reference, float)
    if e.shape != r.shape or e.ndim != 1 or len(e) < 2 or not np.isfinite(e+r).all():
        raise ValueError("C1 needs aligned finite problem vectors")
    if ((e < 0) | (r < 0) | (e > CAP) | (r > CAP)).any():
        raise ValueError("short C1 prefixes must be in [0,512]")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Chunk bootstrap indices to bound host scratch at maximum workload.
    boots = []
    d = e-r
    for start in range(0, RESAMPLES, 256):
        idx = rng.integers(0, len(d), size=(min(256, RESAMPLES-start), len(d)))
        boots.extend(d[idx].mean(1).tolist())
    lower = min(float(np.quantile(boots, alpha)), float(d.mean()))
    limit = -.1*float(r.mean())
    return dict(problems=len(d), engine_mean=float(e.mean()), reference_mean=float(r.mean()),
                difference=float(d.mean()), difference_lower=lower, alpha=alpha, limit=limit,
                status="PASS" if lower >= limit else "not shown")


def analyze(manifest_path, root, look, validity_report):
    manifest = json.loads(Path(manifest_path).read_text())
    validate(manifest)
    root = Path(root)
    if look == 2:
        previous = json.loads((root/"look1.json").read_text())
        if previous["next_action"] != "continue_fixed_512":
            raise ValueError("C1 stage 2 was not called for by the interim rule")
    es, rs, observations, provenance = [], [], [], None
    for stage in range(1, look+1):
        stage_root = root/f"stage{stage}"
        oracle = Path((stage_root/"oracle.path").read_text().strip())
        loaded = {}
        for impl, directory in (("engine", stage_root/"engine"), ("reference", stage_root/"reference"),
                                ("oracle", oracle)):
            cfg, loaded[impl] = load_run(directory, manifest, impl)
            if cfg["pilot"] or cfg["fixture"] or cfg["max_new_tokens"] != CAP:
                raise ValueError("pilot/fixture/wrong-cap C1 cannot establish a confirmatory result")
            if cfg.get("subset") != f"stage{stage}" or cfg["problems"] != manifest["subsets"][f"stage{stage}"]:
                raise ValueError("wrong stage coverage/order")
            if any(len(sample["token_ids"]) > CAP for sample in loaded[impl].values()):
                raise ValueError("sample exceeds the frozen cap")
            if cfg["dtype"] != ("fp32" if impl == "oracle" else "bf16"):
                raise ValueError("wrong implementation precision")
            if impl == "engine" and cfg["engine_options"] != dict(cuda_graphs=True, overlap=True, compile=False,
                                                                  max_batch_size=min(64, cfg["block_size"])):
                raise ValueError("wrong production engine options")
            if impl == "reference" and cfg["batch"] != 8:
                raise ValueError("reference comparator must stay b8")
            frozen = {k: cfg[k] for k in ("model_sha256", "source_sha256", "eval_config_sha256",
                                           "environment", "oracle_constants", "sampling")}
            if provenance is not None and frozen != provenance:
                raise ValueError("C1 input/source/environment changed across implementations or stages")
            provenance = frozen
        for pid in manifest["subsets"][f"stage{stage}"]:
            e, r, o = (loaded[k][pid] for k in ("engine", "reference", "oracle"))
            ep, rp = prefix(e["token_ids"], o["token_ids"]), prefix(r["token_ids"], o["token_ids"])
            es.append(ep)
            rs.append(rp)
            observations.append(dict(id=pid, dataset=o["dataset"], engine=ep, reference=rp,
                engine_cap_censored=ep == CAP, reference_cap_censored=rp == CAP,
                engine_no_observed_difference=e["sampled_token_ids"] == o["sampled_token_ids"],
                reference_no_observed_difference=r["sampled_token_ids"] == o["sampled_token_ids"]))
    evidence = json.loads(Path(validity_report).read_text())
    validity = evidence.get("validity", {})
    vi = run_info(evidence["oracle"]["run"])
    valid = (validity.get("size_requirement_met", False) and validity.get("status", "").startswith("PASS")
             and vi["model_sha256"] == provenance["model_sha256"]
             and vi["config"]["eval_config_sha256"] == provenance["eval_config_sha256"]
             and evidence["oracle"]["threshold"] == .8984375
             and vi["config"].get("oracle_constants") ==
             "bf16-rounded buffers including inv_freq (R1-O amendment 2026-10-13)")
    result = bound(es, rs)
    best_remaining_margin_sum = float(np.sum(es)-.9*np.sum(rs) + (EXTRA*CAP if look == 1 else 0))
    impossible = best_remaining_margin_sum < 0
    action = ("stop_for_futility" if not valid or impossible else "stop_for_success" if result["status"] == "PASS"
              else "continue_fixed_512" if look == 1 else "stop_at_maximum_not_shown")
    report = dict(protocol=PROTOCOL, manifest_sha256=digest(manifest), look=look, C1=result,
                  C2="withdrawn before confirmatory data", validity_met=bool(valid), next_action=action,
                  best_possible_final_margin_sum=best_remaining_margin_sum,
                  conclusion="PASS" if valid and result["status"] == "PASS" else "not shown",
                  per_problem=observations,
                  censoring={k: sum(p[k] for p in observations)/len(observations)
                             for k in ("engine_cap_censored", "reference_cap_censored",
                                       "engine_no_observed_difference", "reference_no_observed_difference")})
    write_json(root/f"look{look}.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--source-root", type=Path, default=Path("data/engine/paper_protocol_seed42/canonical"))
    ana = sub.add_parser("analyze")
    ana.add_argument("--manifest", type=Path, required=True)
    ana.add_argument("--root", type=Path, required=True)
    ana.add_argument("--look", type=int, choices=(1, 2), required=True)
    ana.add_argument("--validity-report", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == "prepare":
        prepare(args.model, args.output, args.source_root)
    else:
        print(json.dumps(analyze(args.manifest, args.root, args.look, args.validity_report), indent=2))


if __name__ == "__main__":
    main()
