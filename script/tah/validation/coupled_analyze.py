"""Deterministic paired R1-O2 analysis; fixed margins from report/PROTOCOL.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from script.tah.evaluation.compare_runs import t_critical
from .analyze import BOOTSTRAP_SEED, DELTA, RESAMPLES
from .common import write_json
from .coupled_run import digest, sample_path, trace_path, validate_manifest
from .coupling import uniform

ACCURACY_MARGIN = .01


def prefix(a, b):
    """Number of returned tokens before first difference, capped at both lengths."""
    return next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def c1(engine, reference):
    e, r = np.asarray(engine, float), np.asarray(reference, float)
    if e.shape != r.shape or e.ndim != 1 or not len(e):
        raise ValueError("C1 needs aligned nonempty problem vectors")
    d = e - r
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(d), size=(RESAMPLES, len(d)))
    lower = float(np.quantile(d[idx].mean(1), .05, method="linear"))
    limit = -DELTA * float(r.mean())
    return dict(problems=len(d), engine_mean=float(e.mean()), reference_mean=float(r.mean()),
                difference=float(d.mean()), difference_lower95=lower, limit=limit,
                status="PASS" if lower >= limit else "not shown")


def c2(engine, reference):
    e, r = np.asarray(engine, float), np.asarray(reference, float)
    if e.shape != r.shape or e.ndim != 1 or not len(e):
        raise ValueError("C2 needs aligned nonempty problem vectors")
    d = e - r
    mean = float(d.mean())
    if len(d) < 2:
        return dict(problems=len(d), difference=mean, ci90=None, status="not shown")
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    critical = t_critical(.95, len(d) - 1)
    ci = [mean - critical * se, mean + critical * se]
    return dict(problems=len(d), engine_accuracy=float(e.mean()), reference_accuracy=float(r.mean()),
                difference=mean, paired_se=se, ci90=ci, margin=ACCURACY_MARGIN,
                method="paired Student t TOST, df=n-1",
                discordant_engine_only=int(((e == 1) & (r == 0)).sum()),
                discordant_reference_only=int(((e == 0) & (r == 1)).sum()),
                status="PASS" if ci[0] > -ACCURACY_MARGIN and ci[1] < ACCURACY_MARGIN else "not shown")


def load_run(root, manifest, implementation):
    info = json.loads((Path(root) / "run.json").read_text())
    cfg = info["config"]
    if info["status"] != "complete" or cfg["implementation"] != implementation:
        raise ValueError(f"{root}: incomplete or wrong implementation")
    if cfg["manifest_sha256"] != digest(manifest):
        raise ValueError(f"{root}: different workload manifest")
    samples = {}
    lookup = {p["id"]: p for p in manifest["problems"]}
    expected_blocks = (len(cfg["problems"]) + cfg["block_size"] - 1) // cfg["block_size"]
    if sorted(info["completed_blocks"]) != list(range(expected_blocks)):
        raise ValueError(f"{root}: missing/duplicate block markers")
    for pid in cfg["problems"]:
        s = json.loads(sample_path(root, pid).read_text())
        if s["id"] != pid or s["seed"] != lookup[pid]["seed"] or type(s["is_correct"]) is not bool:
            raise ValueError(f"{root}: invalid sample identity/grade")
        if len(s["token_ids"]) != len(s["pass_counts"]):
            raise ValueError(f"{root}: mismatched pass counts")
        expected = s["token_ids"] + ([manifest["eos_token_id"]] if s["finish_reason"] == "eos" else [])
        if s["sampled_token_ids"] != expected:
            raise ValueError(f"{root}: inconsistent sampled trajectory")
        if implementation == "oracle":
            with np.load(trace_path(root, pid)) as data:
                details = data["details"]
            if details.shape != (len(expected), 7) or not np.isfinite(details).all():
                raise ValueError(f"{root}: incomplete/nonfinite oracle trace")
            low, high, total, target, distance = details[:, :5].T
            if (total <= 0).any() or (low > target).any() or (target > high).any():
                raise ValueError(f"{root}: inconsistent oracle CDF boundaries")
            if not np.allclose(target / total, [uniform(s["seed"], j) for j in range(len(expected))],
                               rtol=0, atol=2e-15):
                raise ValueError(f"{root}: oracle uniforms are not aligned")
            if not np.allclose(distance, np.minimum(target - low, high - target), rtol=0, atol=2e-15):
                raise ValueError(f"{root}: inconsistent oracle margin")
        samples[pid] = s
    return cfg, samples


def oracle_margin(root, oracle, candidate, position):
    """At the first differing draw, both have the same prefix/state input.

    A candidate which hits its length cap has no differing draw; label censoring
    explicitly. Oracle EOS draws have trace rows even though EOS is not returned.
    """
    oa, ca = oracle["sampled_token_ids"], candidate["sampled_token_ids"]
    j = prefix(oa, ca)
    if oa == ca:
        return None
    if j >= len(oa) or j >= len(ca):
        return dict(position=position, sampled_position=j, kind="length_censored", oracle_margin=None)
    with np.load(trace_path(root, oracle["id"])) as data:
        details = data["details"]
    if len(details) != len(oa) or not np.isfinite(details).all():
        raise ValueError("oracle trace missing or nonfinite")
    low, high, total, target, distance, gap, passes = details[j]
    if total <= 0:
        raise ValueError("oracle sampling total must be positive")
    return dict(position=position, sampled_position=j, kind="token_or_eos", oracle_token=oa[j],
                candidate_token=ca[j], cdf_lower=float(low), cdf_upper=float(high), total=float(total),
                uniform=float(target / total), uniform_times_total=float(target),
                oracle_margin=float(distance), normalized_margin=float(distance / total),
                top_probability_gap=float(gap / total), oracle_passes=int(passes))


def comparison(a, b):
    ids = list(a)
    if set(ids) != set(b):
        raise ValueError("comparison requires the same problems")
    rows = []
    for pid in ids:
        x, y = a[pid], b[pid]
        rows.append(dict(id=pid, prefix=prefix(x["token_ids"], y["token_ids"]),
                         identical=x["sampled_token_ids"] == y["sampled_token_ids"],
                         first_draw_difference=None if x["sampled_token_ids"] == y["sampled_token_ids"]
                         else prefix(x["sampled_token_ids"], y["sampled_token_ids"])))
    divergent = [r["first_draw_difference"] for r in rows if not r["identical"]]
    hist = {str(j): divergent.count(j) for j in sorted(set(divergent))}
    return dict(problems=len(rows), identical_fraction=sum(r["identical"] for r in rows) / len(rows),
                divergence_positions=dict(histogram=hist, quantiles=None if not divergent else
                                          np.quantile(divergent, [0, .25, .5, .75, 1]).tolist()), rows=rows)


def analyze(manifest, engine_dir, reference_dir, oracle_dir, validity_report=None):
    validate_manifest(manifest)
    ec, engine = load_run(engine_dir, manifest, "engine")
    rc, reference = load_run(reference_dir, manifest, "reference")
    oc, oracle = load_run(oracle_dir, manifest, "oracle")
    for key in ("model_sha256", "source_sha256", "environment", "eval_config", "eval_config_sha256",
                "max_new_tokens", "pilot", "fixture",
                "oracle_constants", "sampling"):
        if ec[key] != rc[key] or rc[key] != oc[key]:
            raise ValueError(f"implementations differ in {key}")
    if ec["problems"] != rc["problems"] or not set(oracle) <= set(engine):
        raise ValueError("engine/reference coverage differs, or oracle is outside it")
    allowed_oracle = [pid for pid in manifest["oracle_ids"] if pid in engine]
    if oc["problems"] != allowed_oracle:
        raise ValueError("oracle coverage differs from first 50 per dataset")
    if ec["dtype"] != "bf16" or rc["dtype"] != "bf16" or oc["dtype"] not in ("fp32", "fp64"):
        raise ValueError("wrong implementation precisions")
    if ec["engine_options"] != dict(cuda_graphs=True, overlap=True, compile=False,
                                   max_batch_size=min(64, ec["block_size"])):
        raise ValueError("engine production settings differ")
    ids = list(oracle)
    epre = [prefix(engine[p]["token_ids"], oracle[p]["token_ids"]) for p in ids]
    rpre = [prefix(reference[p]["token_ids"], oracle[p]["token_ids"]) for p in ids]
    report = dict(protocol="R1-O2", delta=DELTA, bootstrap_resamples=RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
                  oracle_dtype=oc["dtype"], C1=c1(epre, rpre),
                  C2=c2([engine[p]["is_correct"] for p in engine], [reference[p]["is_correct"] for p in engine]),
                  comparisons={"engine_reference": comparison(engine, reference),
                               "engine_oracle": comparison({p: engine[p] for p in ids}, oracle),
                               "reference_oracle": comparison({p: reference[p] for p in ids}, oracle)},
                  per_problem_C1=[dict(id=p, dataset=oracle[p]["dataset"], engine=epre[j], reference=rpre[j])
                                  for j, p in enumerate(ids)], oracle_divergences=[], per_dataset={})
    for pid in ids:
        for impl, samples in (("engine", engine), ("reference", reference)):
            m = oracle_margin(oracle_dir, oracle[pid], samples[pid],
                              prefix(samples[pid]["token_ids"], oracle[pid]["token_ids"]))
            if m is not None:
                report["oracle_divergences"].append(dict(id=pid, implementation=impl, **m))
    margins = [r["normalized_margin"] for r in report["oracle_divergences"] if "normalized_margin" in r]
    report["oracle_margin_distribution"] = dict(count=len(margins), quantiles=None if not margins else
                                                np.quantile(margins, [0, .25, .5, .75, .95, 1]).tolist())
    for ds in sorted({p["dataset"] for p in manifest["problems"]}):
        all_ids = [p for p in engine if engine[p]["dataset"] == ds]
        subset = [j for j, p in enumerate(ids) if oracle[p]["dataset"] == ds]
        if all_ids and subset:
            report["per_dataset"][ds] = dict(C1=c1([epre[j] for j in subset], [rpre[j] for j in subset]),
                C2=c2([engine[p]["is_correct"] for p in all_ids], [reference[p]["is_correct"] for p in all_ids]))
    complete = not (ec["pilot"] or ec["fixture"]) and len(engine) == 398 and len(oracle) == 100
    validity = None
    valid = False
    if validity_report:
        evidence = json.loads(Path(validity_report).read_text())
        validity = evidence.get("validity")
        # The R1-O1 report supplies the mandatory arithmetic validity check.
        if evidence["oracle"]["run"]:
            from .analyze import run_info
            vi = run_info(evidence["oracle"]["run"])
            if vi["model_sha256"] != ec["model_sha256"]:
                raise ValueError("oracle validity is for a different checkpoint")
            if vi["config"].get("eval_config_sha256") != ec["eval_config_sha256"]:
                raise ValueError("oracle validity is for a different evaluation config")
            if vi["config"].get("oracle_constants") != (
                    "bf16-rounded buffers including inv_freq (R1-O amendment 2026-10-13)"):
                raise ValueError("oracle validity predates the constants amendment")
            if evidence["oracle"]["threshold"] != 0.8984375:
                raise ValueError("oracle validity uses a different threshold")
        valid = bool(validity and validity.get("size_requirement_met") and
                     (validity["status"].startswith("PASS") or oc["dtype"] == "fp64"))
    report.update(full_workload=complete, oracle_validity=validity,
                  oracle_validity_met=valid,
                  conclusion="PASS" if complete and valid and report["C1"]["status"] == "PASS" and
                  report["C2"]["status"] == "PASS" else "not shown")
    return report


def markdown(report):
    a, b = report["C1"], report["C2"]
    lines = ["# R1-O2 — coupled trajectories", "",
             f"**R1-O2: {report['conclusion']}**; full workload: {report['full_workload']}; "
             f"oracle validity met: {report['oracle_validity_met']}.", "",
             f"C1: {a['problems']} problems; engine prefix mean {a['engine_mean']:.6g}, "
             f"reference {a['reference_mean']:.6g}; paired difference {a['difference']:.6g}, "
             f"95% lower bound {a['difference_lower95']:.6g} ≥ {a['limit']:.6g}: **{a['status']}**.", "",
             f"Problem bootstrap: {RESAMPLES} resamples, seed {BOOTSTRAP_SEED}, δ={DELTA}.", "",
             f"C2: {b['problems']} problems; paired accuracy difference {100*b['difference']:.4g} pp, "
             f"90% CI {None if b['ci90'] is None else [100*x for x in b['ci90']]} pp, "
             f"inside ±1 pp: **{b['status']}** (paired Student t TOST).", "",
             "| Trajectories | Problems | Identical fraction | Divergence position min/Q1/median/Q3/max |",
             "|---|---:|---:|---|"]
    for name, c in report["comparisons"].items():
        lines.append(f"| {name} | {c['problems']} | {c['identical_fraction']:.6g} | "
                     f"{c['divergence_positions']['quantiles']} |")
    lines += ["", "Returned token IDs exclude EOS; full identity includes the EOS draw. "
              "Length censoring is labelled; prefixes are capped at the shorter returned length.", "",
              f"Oracle normalised CDF margins min/Q1/median/Q3/P95/max: "
              f"{report['oracle_margin_distribution']['quantiles']}.", "",
              "| Problem | Implementation | Draw position | Oracle token | Candidate token | Oracle CDF margin |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report["oracle_divergences"]:
        lines.append(f"| {row['id']} | {row['implementation']} | {row['sampled_position']} | "
                     f"{row.get('oracle_token', 'censored')} | {row.get('candidate_token', 'censored')} | "
                     f"{row.get('normalized_margin')} |")
    lines += ["", "CDF distances and probability gaps are diagnostics; no extra near-tie margin is imposed. "
              "Per-problem values, histograms and dataset statistics are in r1o2.json.", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--validity-report", type=Path, help="R1-O1 r1o1.json (required for a protocol PASS)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    report = analyze(json.loads(args.manifest.read_text()), args.engine, args.reference, args.oracle,
                     args.validity_report)
    write_json(args.output / "r1o2.json", report)
    (args.output / "r1o2.md").write_text(markdown(report))
    print(f"R1-O2: {report['conclusion']}")


if __name__ == "__main__":
    main()
