"""Tail-margin non-inferiority and implementation-self diagnostics, added after pilots.

The original maximum rule is always reported. The added criterion compares the
mean within-sequence 90th percentile of oracle margins at disagreements,
with a studentized paired sequence bootstrap and unchanged 10% relative allowance. A separate exact permutation
check diagnoses false alarms under exchangeability; it is NOT non-inferiority.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from math import sqrt
from pathlib import Path

import numpy as np

from .analyze import BOOTSTRAP_SEED, RESAMPLES, run_info
from .common import load_record, write_json
from script.tah.evaluation.compare_runs import t_critical

KINDS = ("decision", "top1")
QUANTILE = .9


def extract(run, oracle):
    info, oi = run_info(run), run_info(oracle)
    if not info.get("complete") or not oi.get("complete") or set(info["sequences"]) != set(oi["sequences"]):
        raise ValueError("near-tie needs complete, matching sequence sets")
    for key in ("manifest_sha256", "model_sha256"):
        if info[key] != oi[key]:
            raise ValueError("near-tie provenance differs")
    if info["config"]["mode"] != "free":
        raise ValueError("near-tie requires free decisions")
    threshold = oi["threshold"]
    rows = []
    for sid in oi["sequences"]:
        r, o = load_record(run, sid), load_record(oracle, sid)
        first, n = int(o["first"]), len(o["top1"])
        if int(r["first"]) != first or len(r["top1"]) != n:
            raise ValueError("unaligned near-tie positions")
        gen = slice(first, first+n)
        flip = r["visited"][:, gen] & o["visited"][:, gen] & (r["decision"][:, gen] != o["decision"][:, gen])
        depth, position = np.nonzero(flip)
        margins = np.abs(o["prob"][:, gen] - threshold)[flip]
        top = np.flatnonzero(r["top1"] != o["top1"])
        eligible_depth, eligible_position = np.nonzero(
            r["visited"][:, gen] & o["visited"][:, gen] & np.isfinite(o["prob"][:, gen]))
        row = dict(id=sid, decision=dict(margins=margins, events=depth*n+position,
                                        eligible_events=eligible_depth*n+eligible_position),
                   top1=dict(margins=o["gap"][top], events=top, eligible_events=np.arange(n)))
        if any(not np.isfinite(row[k]["margins"]).all() for k in KINDS):
            raise ValueError("nonfinite disagreement margin")
        rows.append(row)
    return rows


def pooled_quantile(arrays, q=QUANTILE):
    values = np.concatenate(arrays)
    return float(np.quantile(values, q, method="inverted_cdf")) if len(values) else 0.


def resampled_quantiles(arrays, indices, q=QUANTILE):
    """Exact quantile of a cluster-resampled event pool, without copying events."""
    values = np.concatenate(arrays)
    if not len(values):
        return np.zeros(len(indices))
    owners = np.repeat(np.arange(len(arrays)), [len(a) for a in arrays])
    order = np.argsort(values, kind="stable")
    values, owners = values[order], owners[order]
    result = np.empty(len(indices))
    for start in range(0, len(indices), 128):
        idx = indices[start:start+128]
        counts = np.array([np.bincount(x, minlength=len(arrays)) for x in idx])
        cumulative = counts[:, owners].cumsum(1)
        targets = np.ceil(q*cumulative[:, -1])
        slots = (cumulative >= targets[:, None]).argmax(1)
        result[start:start+len(idx)] = np.where(targets > 0, values[slots], 0.)
    return result


def compare(engine, reference, *, paired=True, resamples=RESAMPLES, seed=BOOTSTRAP_SEED, alpha=.05):
    if not 0 < alpha < .5:
        raise ValueError("near-tie alpha must be between 0 and .5")
    if len(engine) < 2 or len(reference) < 2 or (paired and [r["id"] for r in engine] != [r["id"] for r in reference]):
        raise ValueError("paired near-tie analysis needs the same ordered sequence IDs")
    rng = np.random.default_rng(seed)
    ie = rng.integers(0, len(engine), size=(resamples, len(engine)))
    ir = ie if paired else rng.integers(0, len(reference), size=(resamples, len(reference)))
    result = {}
    for kind in KINDS:
        e, r = [x[kind]["margins"] for x in engine], [x[kind]["margins"] for x in reference]
        x = np.array([pooled_quantile([v]) for v in e])
        y = np.array([pooled_quantile([v]) for v in r])
        qe, qr = float(x.mean()), float(y.mean())
        em, rm = max((max(v, default=0.) for v in e)), max((max(v, default=0.) for v in r))
        point = qe-1.1*qr
        if paired:
            d = x-1.1*y
            boot = d[ie]
            se = d.std(ddof=1)/np.sqrt(len(d))
            bse = boot.std(1, ddof=1)/np.sqrt(len(d))
            bmean = boot.mean(1)
        else:
            xe, yr = x[ie], y[ir]
            se = np.sqrt(x.var(ddof=1)/len(x) + 1.21*y.var(ddof=1)/len(y))
            bse = np.sqrt(xe.var(1, ddof=1)/len(x) + 1.21*yr.var(1, ddof=1)/len(y))
            bmean = xe.mean(1)-1.1*yr.mean(1)
        centered = bmean-point
        tboot = np.divide(centered, bse, out=np.zeros_like(centered), where=bse > 0)
        tboot[(bse == 0) & (np.abs(centered) > 1e-15)] = np.sign(centered[(bse == 0) & (np.abs(centered) > 1e-15)])*np.inf
        critical = max(t_critical(1-alpha, min(len(x), len(y))-1),
                       -float(np.quantile(tboot, alpha, method="inverted_cdf")))
        upper = float(point+critical*se) if np.isfinite(critical) else None
        ne, nr = sum(map(len, e)), sum(map(len, r))
        enough = ne >= 20 and nr >= 20
        status = "PASS" if enough and upper is not None and upper <= 0 else "not shown"
        result[kind] = dict(engine_events=sum(map(len, e)), reference_events=sum(map(len, r)),
            engine_mean_q90=qe, reference_mean_q90=qr, D=qe-1.1*qr, upper_bound=upper,
            status=status, original_max=dict(engine=em if ne else None, reference=rm if nr else None,
                engine_events=ne, reference_events=nr, limit=1.1*rm if nr else None,
                status="PASS" if not ne or (nr and em <= 1.1*rm) else "not shown"),
            standard_error=float(se), critical=float(critical) if np.isfinite(critical) else None)
        if alpha in (.05, .025):
            result[kind]["upper95" if alpha == .05 else "upper975"] = upper
    return dict(quantile=QUANTILE, estimator="mean within-sequence q90 (zero if no disagreements)",
                method="studentized cluster bootstrap with Student-t critical floor", delta=.1,
                alpha=alpha, confidence=1-alpha, resamples=resamples, paired=paired,
                original_max_binding=False,
                criteria=result, conclusion="PASS" if all(r["status"] == "PASS" for r in result.values()) else "not shown")


def correlations(a, b):
    if [r["id"] for r in a] != [r["id"] for r in b]:
        raise ValueError("different sequences in composition demonstration")
    out = {}
    for kind in KINDS:
        am = np.array([max(x[kind]["margins"], default=0.) for x in a])
        bm = np.array([max(x[kind]["margins"], default=0.) for x in b])
        corr = float(np.corrcoef(am, bm)[0, 1]) if am.std() and bm.std() else None
        intersections = unions = 0
        possible = ax = by = shared = 0
        for x, y in zip(a, b):
            sx, sy = set(x[kind]["events"].tolist()), set(y[kind]["events"].tolist())
            intersections += len(sx & sy)
            unions += len(sx | sy)
            ux = set(x[kind].get("eligible_events", x[kind]["events"]).tolist())
            uy = set(y[kind].get("eligible_events", y[kind]["events"]).tolist())
            eligible = ux & uy
            possible += len(eligible)
            ax += len(sx & eligible)
            by += len(sy & eligible)
            shared += len(sx & sy & eligible)
        denominator = sqrt(ax*(possible-ax)*by*(possible-by))
        phi = (possible*shared-ax*by)/denominator if denominator else None
        out[kind] = dict(per_sequence_maxima_pearson=corr, disagreement_position_phi=phi,
                         jointly_eligible_positions=possible, shared_disagreement_positions=intersections,
                         union_disagreement_positions=unions, position_jaccard=intersections/unions if unions else 1.)
    return out


def self_split(rows, splits=1000, bootstrap=1000):
    """Diagnostic only: random halves, plus an exact conditional equality test.

    Report NI success and failure-to-prove separately from false equality alarms.
    Random halves are unpaired groups, so their bootstrap draws are independent.
    """
    n = len(rows)
    if n % 2 or n > 20 or n < 4:
        raise ValueError("self-split diagnostic needs 4–20 sequences, even count")
    half = n//2
    partitions = list(combinations(range(n), half))
    universe = set(range(n))
    null = {k: [] for k in KINDS}
    for chosen in partitions:
        left, right = list(chosen), sorted(universe-set(chosen))
        for kind in KINDS:
            null[kind].append(float(np.mean([pooled_quantile([rows[i][kind]["margins"]]) for i in left]))
                              - float(np.mean([pooled_quantile([rows[i][kind]["margins"]]) for i in right])))
    null = {k: np.sort(v) for k, v in null.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    counters = {k: dict(max_not_shown=0, noninferiority_not_shown=0, equality_false_alarm=0, boundary_false_noninferiority=0) for k in KINDS}
    for iteration in range(splits):
        order = rng.permutation(n)
        left, right = [rows[i] for i in order[:half]], [rows[i] for i in order[half:]]
        result = compare(left, right, paired=False, resamples=bootstrap, seed=BOOTSTRAP_SEED+iteration)
        boundary_left = [dict(id=x["id"], **{k: dict(margins=1.1*x[k]["margins"],
            events=x[k]["events"]) for k in KINDS}) for x in left]
        boundary = compare(boundary_left, right, paired=False, resamples=bootstrap,
                           seed=BOOTSTRAP_SEED+iteration)
        for kind in KINDS:
            counters[kind]["boundary_false_noninferiority"] += boundary["criteria"][kind]["status"] == "PASS"
            observed = result["criteria"][kind]
            statistic = observed["engine_mean_q90"]-observed["reference_mean_q90"]
            p = (len(null[kind])-np.searchsorted(null[kind], statistic-1e-15, side="left"))/len(null[kind])
            counters[kind]["max_not_shown"] += observed["original_max"]["status"] != "PASS"
            counters[kind]["noninferiority_not_shown"] += observed["status"] != "PASS"
            counters[kind]["equality_false_alarm"] += p <= .05
    return dict(diagnostic_only=True, sequences=n, per_half=half, splits=splits,
                bootstrap_per_split=bootstrap, exact_permutations=len(partitions),
                rates={k: {s: v/splits for s, v in c.items()} for k, c in counters.items()},
                interpretation="Equality false-alarm calibration is not NI power; not shown does not establish inferiority.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--self-splits", type=int, default=0)
    p.add_argument("--bootstrap-per-split", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=.05,
                   help="one-sided comparison alpha; .025 for each confirmatory look, .05 for diagnostics")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    e = extract(args.engine, args.oracle)
    if args.self_splits:
        report = self_split(e, args.self_splits, args.bootstrap_per_split)
    else:
        if args.reference is None:
            p.error("comparison needs --reference")
        r = extract(args.reference, args.oracle)
        report = compare(e, r, alpha=args.alpha)
        report["composition_correlations"] = correlations(e, r)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
