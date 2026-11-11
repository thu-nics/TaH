"""Approved O1 paired two-look bounds; immutable stage records, no C2 criterion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import analyze as a
from . import near_tie


def bounds(engine, reference):
    idx = a.bootstrap_indices(len(engine))
    out = {}
    for name, key in (("M1", "m1_sum"), ("M2", "m2_sum"), ("M3", "m3_sum")):
        e = np.array([r[key] for r in engine])
        r = np.array([r[key] for r in reference])
        if not np.isfinite(e+r).all() or r.sum() <= 0:
            raise ValueError("invalid forced-error data")
        upper = float(np.quantile(e[idx].sum(1)/r[idx].sum(1), .975))
        out[name] = dict(estimate=float(e.sum()/r.sum()), upper975=upper,
                         status="PASS" if upper <= 1.1 else "not shown")
    n = np.array([r["m2_n"] for r in reference], float)
    if not np.array_equal(n, [r["m2_n"] for r in engine]):
        raise ValueError("different paired pass counts")
    eb = np.array([r["bias_sum"] for r in engine])
    rb = np.array([r["bias_sum"] for r in reference])
    rm2 = np.array([r["m2_sum"] for r in reference])
    d = (np.abs(eb[idx].sum(1)/n[idx].sum(1))-1.1*np.abs(rb[idx].sum(1)/n[idx].sum(1))
         -.1*rm2[idx].sum(1)/n[idx].sum(1))
    upper = float(np.quantile(d, .975))
    out["bias"] = dict(upper975=upper, status="PASS" if upper <= 0 else "not shown")
    return out


def retain_successes(current, previous):
    """A criterion's first-look success is final, regardless of other criteria."""
    for name, row in current.items():
        if previous[name]["status"] == "PASS":
            row["observed_status"] = row["status"]
            row["status"] = "PASS"
            row["stopped_for_success_at_look"] = 1
    return current


def next_action(criteria, look, valid):
    if not valid:
        return "stop_for_futility"
    if all(row["status"] == "PASS" for row in criteria.values()):
        return "stop_for_success"
    return "continue_fixed_136" if look == 1 else "stop_at_maximum_not_shown"


def analyze(root, manifest_path, look):
    root, manifest_path = Path(root), Path(manifest_path)
    if look not in (1, 2):
        raise ValueError("O1 needs look 1 or 2")
    m = json.loads(manifest_path.read_text())
    stages = [m["subsets"][f"stage{s}"] for s in (1, 2)]
    if len(stages[0]) != 64 or len(stages[1]) != 136 or len(set(stages[0]+stages[1])) != 200:
        raise ValueError("O1 needs fixed disjoint 64/+136 IDs")
    datasets = {s["id"]: s["dataset"] for s in m["sequences"]}
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    previous = None
    extend = dict(forced=False, free=False)
    if look == 2:
        previous = json.loads((root/"o1/look1.json").read_text())
        if previous["manifest_sha256"] != manifest_hash:
            raise ValueError("O1 manifest changed between looks")
        extend = {arm: previous[f"{arm}_next_action"] == "continue_fixed_136" for arm in extend}
        if not previous["validity_met"] or not any(extend.values()):
            raise ValueError("O1 stage2 not called for by interim rule")
    forced_e, forced_r, free_e, free_r = [], [], [], []
    for stage in range(1, look+1):
        directory = root/"o1"/f"stage{stage}"
        oracle = Path((directory/"oracle-fp32.path").read_text().strip())
        info = a.run_info(oracle)
        expected = stages[stage-1]
        if not info["complete"] or info["manifest_sha256"] != manifest_hash or info["sequences"] != expected:
            raise ValueError("incomplete/wrong O1 stage oracle")
        vocab = json.loads((Path(info["config"]["model"])/"config.json").read_text())["vocab_size"]
        for arm, engine, reference in (("forced", forced_e, forced_r), ("free", free_e, free_r)):
            if stage == 2 and not extend[arm]:
                continue
            for implementation, rows in (("engine", engine), ("ref", reference)):
                run = directory/f"{implementation}-{arm}"
                ri = a.run_info(run)
                if not (ri["complete"] and ri["sequences"] == expected and ri["manifest_sha256"] == manifest_hash
                        and ri["model_sha256"] == info["model_sha256"]):
                    raise ValueError("incomplete/wrong O1 implementation stage")
                cfg = ri["config"]
                if cfg["mode"] != arm or cfg["dtype"] != "bf16":
                    raise ValueError("wrong implementation precision/mode")
                if implementation == "ref" and (cfg["batch"] != 8 or cfg["layout"] != "generate"):
                    raise ValueError("O1 primary reference must use the evaluation b8 layout")
                if implementation == "engine" and cfg["max_batch_size"] < 32:
                    raise ValueError("O1 engine must allow at least 32 residents")
                if arm == "forced":
                    rows.extend(a.collect(run, oracle, expected, vocab, forced=True,
                                          threshold=info["threshold"], datasets=datasets))
                else:
                    rows.extend(near_tie.extract(run, oracle))
    for arm, rows in (("forced", forced_e), ("free", free_e)):
        expanded = look == 2 and extend[arm]
        positions = sum(r["n"] if arm == "forced" else len(r["top1"]["eligible_events"]) for r in rows)
        if len(rows) != (200 if expanded else 64) or positions < (500_000 if expanded else 160_000):
            raise ValueError(f"O1 approved {arm} stage size not met")
    result = bounds(forced_e, forced_r)
    tail = near_tie.compare(free_e, free_r, alpha=.025)
    if previous is not None:
        retain_successes(result, previous["criteria"])
        retain_successes(tail["criteria"], previous["tail"]["criteria"])
    tail["conclusion"] = "PASS" if all(v["status"] == "PASS" for v in tail["criteria"].values()) else "not shown"
    legacy = json.loads((root/"o1/stage1/analysis/r1o1.json").read_text())
    validity = legacy.get("validity", {})
    valid = validity.get("size_requirement_met", False) and validity.get("status", "").startswith("PASS")
    forced_action = next_action(result, look, valid)
    free_action = next_action(tail["criteria"], look, valid)
    actions = (forced_action, free_action)
    action = ("stop_for_futility" if not valid else "stop_for_success" if set(actions) == {"stop_for_success"}
              else "continue_fixed_136" if "continue_fixed_136" in actions else "stop_at_maximum_not_shown")
    original_max = {k: v["original_max"] for k, v in tail["criteria"].items()}
    report = dict(look=look, alpha_per_look=.025, resamples=10_000, seed=a.BOOTSTRAP_SEED,
                  manifest_sha256=manifest_hash, criteria=result, tail=tail, validity_met=bool(valid),
                  forced_sequences=len(forced_e), free_sequences=len(free_e),
                  calibrated_near_tie=tail["conclusion"], original_maximum=dict(binding=False, criteria=original_max),
                  forced_next_action=forced_action, free_next_action=free_action,
                  next_action=action, conclusion="PASS" if valid and action == "stop_for_success" else "not shown")
    a.write_json(root/"o1"/f"look{look}.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--look", type=int, choices=(1, 2), required=True)
    args = p.parse_args(argv)
    print(json.dumps(analyze(args.root, args.manifest, args.look), indent=2))


if __name__ == "__main__":
    main()
