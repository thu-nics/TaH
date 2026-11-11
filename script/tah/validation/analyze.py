"""R1-O1 analysis: oracle errors of engine vs reference, criteria as pre-registered.

    python -m script.tah.validation.analyze --oracle <oracle run> \
        --reference <ref b8 forced> --engine <engine forced> \
        --reference-free <ref b8 free> --engine-free <engine free> \
        [--reference-b1 <ref b1 forced> --reference-b1-free <ref b1 free>] \
        [--validity-oracle <fp64 oracle run> --validity-fp32 <fp32 forced vs fp64> \
         --validity-reference <ref b8 forced vs fp64>] --output <dir>

All metrics are over generated positions (positions whose output sampled a token).
Definitions (forced mode, per position, mean over positions):
  M1 total-variation distance between the implementation's and the oracle's sampling
     distribution (each its own sampler; T 0.6, top-k 20, top-p 0.95);
  M2 |p_impl - p_oracle| of the decider continue-probability, per (position, pass) for
     every visited pass below the pass cap;
  M3 full-vocabulary logit RMSE: sqrt(mean_v (x_v - o_v)^2) per position.
Ratios engine/reference are ratios of sums over the same positions; their one-sided
95% upper bound is the 0.95 quantile of a cluster (sequence) bootstrap.
Writes r1o1.json and r1o1.md (deterministic).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import load_record, write_json

DELTA = .1
RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261013
VALIDITY_FACTOR = 1 / 20
BUCKETS = [(0, 1024, "0-1k"), (1024, 2048, "1-2k"), (2048, 4096, "2-4k"), (4096, 1 << 30, "4k+")]


def run_info(directory):
    return json.loads((Path(directory) / "run.json").read_text())


def tv(rec, oracle):
    """Per generated position TV distance between sparse distributions."""
    out = np.empty(len(rec["top1"]))
    for j in range(len(out)):
        a = dict(zip(rec["dist_ids"][rec["ptr"][j]:rec["ptr"][j + 1]].tolist(),
                     rec["dist_p"][rec["ptr"][j]:rec["ptr"][j + 1]].tolist()))
        b = dict(zip(oracle["dist_ids"][oracle["ptr"][j]:oracle["ptr"][j + 1]].tolist(),
                     oracle["dist_p"][oracle["ptr"][j]:oracle["ptr"][j + 1]].tolist()))
        out[j] = .5 * sum(abs(a.get(k, 0.) - b.get(k, 0.)) for k in sorted(set(a) | set(b)))
    return out


def per_sequence(rec, oracle, vocab, *, forced, dataset, threshold):
    """Sums per sequence (and per secondary group) for one implementation."""
    first = int(rec["first"])
    n = len(rec["top1"])
    positions = np.arange(first, first + n)
    out = dict(dataset=dataset, n=n)
    # Pass-level fields at generated positions; the cap pass has no probability.
    gen = slice(first, first + n)
    both = rec["visited"][:, gen] & oracle["visited"][:, gen]
    if forced and not np.array_equal(rec["visited"], oracle["visited"]):
        raise AssertionError("forced run visited different passes than the oracle")
    if forced and not np.array_equal(rec["decision"], oracle["decision"]):
        raise AssertionError("forced run made different decisions than the oracle")
    prob_ok = both & ~np.isnan(rec["prob"][:, gen]) & ~np.isnan(oracle["prob"][:, gen])
    diff = np.where(prob_ok, rec["prob"][:, gen] - oracle["prob"][:, gen], 0.)
    out["m2_sum"], out["bias_sum"], out["m2_n"] = float(np.abs(diff).sum()), float(diff.sum()), int(prob_ok.sum())
    out["m2_depth"] = [[float(np.abs(diff[d]).sum()), int(prob_ok[d].sum())] for d in range(diff.shape[0])]
    tvs = tv(rec, oracle)
    out["m1_sum"] = float(tvs.sum())
    top1_diff = rec["top1"] != oracle["top1"]
    out["top1_disagree"] = int(top1_diff.sum())
    out["passes"] = int(rec["visited"][:, gen].sum())
    own_flip = both & (rec["own_decision"][:, gen] != oracle["decision"][:, gen])
    out["own_decision_disagree"], out["decisions"] = int(own_flip.sum()), int(both.sum())
    if forced:
        rmse = np.sqrt(rec["sq"] / vocab)
        if np.isnan(rmse).any():
            out["m3_sum"] = None
            out["maxabs"] = None
        else:
            out["m3_sum"] = float(rmse.sum())
            out["maxabs"] = float(rec["maxabs"].max())
    else:
        rmse = None
        # Free mode: oracle margins at every disagreement.
        flip = both & (rec["decision"][:, gen] != oracle["decision"][:, gen])
        margins = np.abs(oracle["prob"][:, gen] - threshold)[flip]
        out["decision_flip_margins"] = sorted(float(m) for m in margins if not np.isnan(m))
        out["top1_flip_margins"] = sorted(float(m) for m in oracle["gap"][top1_diff])
    groups = {}
    for lo, hi, name in BUCKETS:
        mask = (positions >= lo) & (positions < hi)
        if mask.any():
            g = dict(n=int(mask.sum()), m1_sum=float(tvs[mask].sum()),
                     m2_sum=float(np.abs(diff[:, mask]).sum()), m2_n=int(prob_ok[:, mask].sum()),
                     top1_disagree=int(top1_diff[mask].sum()))
            if rmse is not None and not np.isnan(rmse).any():
                g["m3_sum"] = float(rmse[mask].sum())
            groups[name] = g
    out["buckets"] = groups
    return out


def collect(run_dir, oracle_dir, sequences, vocab, *, forced, threshold, datasets):
    rows = []
    for sid in sequences:
        rec, oracle = load_record(run_dir, sid), load_record(oracle_dir, sid)
        rows.append(per_sequence(rec, oracle, vocab, forced=forced, dataset=datasets[sid], threshold=threshold))
    return rows


def bootstrap_indices(count):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, count, size=(RESAMPLES, count))


def ratio_bound(num, den, idx):
    """Point ratio sum(num)/sum(den) and the bootstrap one-sided 95% upper bound."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    point = num.sum() / den.sum() if den.sum() > 0 else float("inf")
    boot = num[idx].sum(1) / den[idx].sum(1)
    return float(point), float(np.quantile(boot, .95, method="linear"))


def mean_of(rows, key, count):
    total = sum(r[key] for r in rows)
    n = sum(r[count] for r in rows)
    return total / n if n else float("nan")


def metric_table(engine, reference, idx):
    out = {}
    for name, key, count in (("M1", "m1_sum", "n"), ("M2", "m2_sum", "m2_n"), ("M3", "m3_sum", "n")):
        if any(r.get(key) is None for r in engine + reference):
            out[name] = dict(status="missing")
            continue
        for a, b in zip(engine, reference):
            if a[count] != b[count]:
                raise AssertionError(f"{name}: engine and reference measured different positions")
        point, upper = ratio_bound([r[key] for r in engine], [r[key] for r in reference], idx)
        out[name] = dict(engine=mean_of(engine, key, count), reference=mean_of(reference, key, count),
                         ratio=point, ratio_upper95=upper, margin=1 + DELTA,
                         status="PASS" if upper <= 1 + DELTA else "not shown")
    return out


def bias_check(engine, reference, idx):
    n = np.array([r["m2_n"] for r in engine], float)
    eb = np.array([r["bias_sum"] for r in engine], float)
    engine_bias = eb.sum() / n.sum()
    ref_bias = mean_of(reference, "bias_sum", "m2_n")
    ref_m2 = mean_of(reference, "m2_sum", "m2_n")
    limit = (1 + DELTA) * abs(ref_bias) + .1 * ref_m2
    rn = np.array([r["m2_n"] for r in reference], float)
    rb = np.array([r["bias_sum"] for r in reference], float)
    if not np.array_equal(n, rn):
        raise AssertionError("bias: engine and reference measured different passes")
    rm2 = np.array([r["m2_sum"] for r in reference], float)
    boot = (np.abs(eb[idx].sum(1) / n[idx].sum(1))
            - (1 + DELTA) * np.abs(rb[idx].sum(1) / rn[idx].sum(1))
            - .1 * rm2[idx].sum(1) / rn[idx].sum(1))
    upper = float(np.quantile(boot, .95, method="linear"))
    return dict(engine_bias=float(engine_bias), reference_bias=float(ref_bias), reference_M2=float(ref_m2),
                D=float(abs(engine_bias) - limit), D_upper95=upper,
                status="PASS" if upper <= 0 else "not shown")


def margin_check(engine_free, reference_free):
    out = {}
    for kind in ("decision_flip_margins", "top1_flip_margins"):
        e = [m for r in engine_free for m in r[kind]]
        f = [m for r in reference_free for m in r[kind]]
        emax, fmax = (max(e) if e else None), (max(f) if f else None)
        if not e:
            status = "PASS"
        elif not f:
            status = "not shown"
        else:
            status = "PASS" if emax <= (1 + DELTA) * fmax else "not shown"
        out[kind.replace("_margins", "")] = dict(engine_count=len(e), reference_count=len(f),
                                                  engine_max_margin=emax, reference_max_margin=fmax,
                                                  limit=None if fmax is None else (1 + DELTA) * fmax,
                                                  status=status)
    return out


def secondary(rows_by_impl, datasets_order):
    out = {}
    for impl, rows in rows_by_impl.items():
        if rows is None:
            continue
        entry = dict(positions=sum(r["n"] for r in rows),
                     top1_disagreement_rate=sum(r["top1_disagree"] for r in rows) / sum(r["n"] for r in rows),
                     own_decision_disagreement_rate=sum(r["own_decision_disagree"] for r in rows) /
                     max(sum(r["decisions"] for r in rows), 1),
                     mean_passes_per_token=sum(r["passes"] for r in rows) / sum(r["n"] for r in rows),
                     M1=mean_of(rows, "m1_sum", "n"), M2=mean_of(rows, "m2_sum", "m2_n"))
        if all(r.get("m3_sum") is not None for r in rows) and "m3_sum" in rows[0]:
            entry["M3"] = mean_of(rows, "m3_sum", "n")
            entry["max_abs_logit_error"] = max(r["maxabs"] for r in rows)
        entry["M2_per_pass"] = []
        for d in range(len(rows[0]["m2_depth"])):
            s = sum(r["m2_depth"][d][0] for r in rows)
            c = sum(r["m2_depth"][d][1] for r in rows)
            if c:
                entry["M2_per_pass"].append(dict(pass_depth=d + 1, M2=s / c, passes=c))
        entry["per_dataset"] = {}
        for ds in datasets_order:
            sub = [r for r in rows if r["dataset"] == ds]
            if sub:
                e = dict(sequences=len(sub), positions=sum(r["n"] for r in sub),
                         M1=mean_of(sub, "m1_sum", "n"), M2=mean_of(sub, "m2_sum", "m2_n"),
                         top1_disagreement_rate=sum(r["top1_disagree"] for r in sub) / sum(r["n"] for r in sub))
                if "M3" in entry:
                    e["M3"] = mean_of(sub, "m3_sum", "n")
                entry["per_dataset"][ds] = e
        entry["per_position_bucket"] = {}
        for _, _, name in BUCKETS:
            sub = [r["buckets"][name] for r in rows if name in r["buckets"]]
            if sub:
                e = dict(positions=sum(g["n"] for g in sub), M1=mean_of(sub, "m1_sum", "n"),
                         M2=mean_of(sub, "m2_sum", "m2_n"),
                         top1_disagreement_rate=sum(g["top1_disagree"] for g in sub) / sum(g["n"] for g in sub))
                if all("m3_sum" in g for g in sub):
                    e["M3"] = mean_of(sub, "m3_sum", "n")
                entry["per_position_bucket"][name] = e
        out[impl] = entry
    return out


def validity(args, vocab, datasets):
    oracle = args.validity_oracle
    sequences = run_info(args.validity_fp32)["sequences"]
    threshold = run_info(oracle).get("threshold")
    fp32 = collect(args.validity_fp32, oracle, sequences, vocab, forced=True, threshold=threshold, datasets=datasets)
    ref = collect(args.validity_reference, oracle, sequences, vocab, forced=True, threshold=threshold,
                  datasets=datasets)
    out = dict(sequences=len(sequences), positions=sum(r["n"] for r in fp32), metrics={})
    ok = len(sequences) >= 20 and out["positions"] >= 20_000
    for name, key, count in (("M1", "m1_sum", "n"), ("M2", "m2_sum", "m2_n"), ("M3", "m3_sum", "n")):
        a, b = mean_of(fp32, key, count), mean_of(ref, key, count)
        passed = a <= VALIDITY_FACTOR * b
        ok &= passed
        out["metrics"][name] = dict(fp32_vs_fp64=a, reference_vs_fp64=b, limit=VALIDITY_FACTOR * b,
                                    status="PASS" if passed else "FAIL")
    out["status"] = "PASS: float32 oracle valid" if ok else "FAIL: use the float64 oracle"
    out["size_requirement_met"] = len(sequences) >= 20 and out["positions"] >= 20_000
    return out


def markdown(report):
    lines = ["# R1-O1 — per-step error against the oracle", ""]
    lines += [f"Oracle: `{report['oracle']['dtype']}` ({report['oracle']['run']}); sequences "
              f"{report['sequences']}, generated positions {report['positions']}; δ = {DELTA}; "
              f"bootstrap {RESAMPLES} resamples, seed {BOOTSTRAP_SEED}.", ""]
    lines += ["| Metric | Engine | Reference b8 | Ratio | Upper 95% | Margin | Result |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name, m in report["primary"].items():
        if m.get("status") == "missing":
            lines.append(f"| {name} | – | – | – | – | – | missing |")
            continue
        lines.append(f"| {name} | {m['engine']:.6g} | {m['reference']:.6g} | {m['ratio']:.4f} | "
                     f"{m['ratio_upper95']:.4f} | {m['margin']:.2f} | {m['status']} |")
    b = report["bias"]
    lines += ["", f"Bias: engine {b['engine_bias']:.3g}, reference {b['reference_bias']:.3g}; "
              f"paired D {b['D']:.3g}, upper 95% {b['D_upper95']:.3g} ≤ 0: **{b['status']}**", ""]
    if report.get("no_large_errors"):
        lines += ["| Free-mode disagreement | Engine count | Engine max margin | Reference count | "
                  "Reference max margin | Limit | Result |", "|---|---:|---:|---:|---:|---:|---|"]
        for kind, m in report["no_large_errors"].items():
            fmt = lambda v: "–" if v is None else f"{v:.4g}"  # noqa: E731
            lines.append(f"| {kind} | {m['engine_count']} | {fmt(m['engine_max_margin'])} | "
                         f"{m['reference_count']} | {fmt(m['reference_max_margin'])} | {fmt(m['limit'])} | "
                         f"{m['status']} |")
        lines.append("")
    if report.get("validity"):
        v = report["validity"]
        lines += [f"Oracle validity ({v['sequences']} sequences, {v['positions']} positions): **{v['status']}**", "",
                  "| Metric | fp32 vs fp64 | reference vs fp64 | limit (1/20) | Result |", "|---|---:|---:|---:|---|"]
        for name, m in v["metrics"].items():
            lines.append(f"| {name} | {m['fp32_vs_fp64']:.4g} | {m['reference_vs_fp64']:.4g} | "
                         f"{m['limit']:.4g} | {m['status']} |")
        lines.append("")
    if report.get("reference_b1"):
        lines += ["Reference batch 1 as comparator (secondary): " + ", ".join(
            f"{k} ratio {m['ratio']:.4f} (upper {m['ratio_upper95']:.4f})"
            for k, m in report["reference_b1"].items() if m.get("status") != "missing"), ""]
    lines += [f"**R1-O1: {report['conclusion']}**", ""]
    return "\n".join(lines)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="reference b8, forced decisions")
    p.add_argument("--engine", type=Path, required=True, help="engine, forced decisions")
    p.add_argument("--reference-free", type=Path)
    p.add_argument("--engine-free", type=Path)
    p.add_argument("--reference-b1", type=Path)
    p.add_argument("--reference-b1-free", type=Path)
    p.add_argument("--validity-oracle", type=Path)
    p.add_argument("--validity-fp32", type=Path)
    p.add_argument("--validity-reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def qualified_conclusion(report, complete, statuses):
    """A metric PASS cannot override missing coverage or arithmetic validity."""
    req = report["requirements"]
    validity = report.get("validity", {})
    valid = bool(validity.get("size_requirement_met") and
                 (validity.get("status", "").startswith("PASS") or report["oracle"]["dtype"] == "fp64"))
    covered = req["sequences_at_least_200"] and req["positions_at_least_500k"] and req["free_mode_present"]
    if not complete:
        return "incomplete"
    return "PASS" if covered and valid and all(s == "PASS" for s in statuses) else "not shown"


def main(argv=None):
    args = parse(argv)
    manifest = json.loads(args.manifest.read_text())
    datasets = {s["id"]: s.get("dataset", "unknown") for s in manifest["sequences"]}
    order = sorted(set(datasets.values()))
    oracle_info = run_info(args.oracle)
    sequences = oracle_info["sequences"]
    for run in (args.reference, args.engine):
        if run_info(run)["sequences"] != sequences:
            raise SystemExit(f"{run} covers different sequences than the oracle")
    vocab = json.loads((Path(oracle_info["config"]["model"]) / "config.json").read_text())["vocab_size"]
    threshold = oracle_info.get("threshold")
    get = lambda run, forced: collect(run, args.oracle, sequences, vocab, forced=forced,  # noqa: E731
                                      threshold=threshold, datasets=datasets) if run else None
    engine, reference = get(args.engine, True), get(args.reference, True)
    engine_free, reference_free = get(args.engine_free, False), get(args.reference_free, False)
    b1_ids = run_info(args.reference_b1)["sequences"] if args.reference_b1 else None
    get_b1 = lambda run, forced: collect(run, args.oracle, b1_ids, vocab, forced=forced,  # noqa: E731
                                         threshold=threshold, datasets=datasets) if run else None
    b1, b1_free = get_b1(args.reference_b1, True), get_b1(args.reference_b1_free, False)
    idx = bootstrap_indices(len(sequences))
    report = dict(oracle=dict(run=str(args.oracle), dtype=oracle_info["config"]["dtype"], threshold=threshold),
                  sequences=len(sequences), positions=sum(r["n"] for r in engine),
                  delta=DELTA, resamples=RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
                  primary=metric_table(engine, reference, idx), bias=bias_check(engine, reference, idx))
    if engine_free and reference_free:
        report["no_large_errors"] = margin_check(engine_free, reference_free)
    if b1:
        # Secondary comparator, possibly on a subset of the sequences.
        engine_b1 = [engine[sequences.index(sid)] for sid in b1_ids]
        report["reference_b1"] = metric_table(engine_b1, b1, bootstrap_indices(len(b1_ids)))
        report["reference_b1_sequences"] = len(b1_ids)
    report["secondary"] = secondary(dict(engine_forced=engine, reference_b8_forced=reference,
                                         reference_b1_forced=b1, engine_free=engine_free,
                                         reference_b8_free=reference_free, reference_b1_free=b1_free), order)
    if args.validity_oracle and args.validity_fp32 and args.validity_reference:
        report["validity"] = validity(args, vocab, datasets)
    statuses = [m["status"] for m in report["primary"].values()] + [report["bias"]["status"]]
    statuses += [m["status"] for m in report.get("no_large_errors", {}).values()]
    complete = "no_large_errors" in report and all(s != "missing" for s in statuses)
    report["requirements"] = dict(sequences_at_least_200=len(sequences) >= 200,
                                  positions_at_least_500k=report["positions"] >= 500_000,
                                  free_mode_present="no_large_errors" in report,
                                  oracle_validity=report.get("validity", {}).get("status"))
    report["conclusion"] = qualified_conclusion(report, complete, statuses)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "r1o1.json", report)
    (args.output / "r1o1.md").write_text(markdown(report))
    print((args.output / "r1o1.md").read_text())


if __name__ == "__main__":
    main()
