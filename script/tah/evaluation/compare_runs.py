#!/usr/bin/env python3
"""Protocol-aware paired equivalence and multi-seed aggregation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    from .run_protocol import load_protocol
except ImportError:  # direct execution: python script/tah/evaluation/compare_runs.py
    from run_protocol import load_protocol


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 10001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _ibeta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees: int) -> float:
    if degrees <= 0:
        raise ValueError("Student-t confidence intervals require at least two paired problems")
    if value == 0:
        return 0.5
    x = degrees / (degrees + value * value)
    tail = 0.5 * _ibeta(x, degrees / 2.0, 0.5)
    return 1.0 - tail if value > 0 else tail


def t_critical(probability: float, degrees: int) -> float:
    if degrees <= 0:
        raise ValueError("degrees of freedom must be positive")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -t_critical(1.0 - probability, degrees)
    low, high = 0.0, 1.0
    while _student_t_cdf(high, degrees) < probability:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2
        if _student_t_cdf(middle, degrees) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1", "yes"}:
        return True
    if str(value).strip().lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid correctness value: {value!r}")


def _dataset_for(problem_id: str, expected: dict[str, dict[str, set[int]]]) -> str | None:
    return next((name for name in sorted(expected, key=len, reverse=True)
                 if problem_id.startswith(name + "_")), None)


def _detail_files(run_dir: Path) -> list[tuple[Path, Path | None]]:
    """Return one combined CSV per eval leaf, or worker CSVs for partial leaves."""
    found: list[tuple[Path, Path | None]] = []
    covered: list[Path] = []
    for status_path in sorted(run_dir.rglob("run_status.json")):
        leaf = status_path.parent
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ValueError(f"incomplete run status in {status_path}: {status.get('status')!r}")
        combined = leaf / "detailed_results.csv"
        files = [combined] if combined.exists() else sorted(leaf.glob("job_*/detailed_results.csv"))
        if not files:
            raise ValueError(f"completed run has no detailed results CSV: {leaf}")
        found.extend((path, status_path) for path in files)
        covered.append(leaf)
    if not found:
        files = [p for p in sorted(run_dir.rglob("detailed_results.csv")) if "job_" not in p.parts]
        found.extend((p, None) for p in files)
    return found


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = []
    for row in rows:
        if not row.get("problem_id"):
            continue
        if "sample_idx" not in row:
            raise ValueError(f"missing sample_idx column in {path}")
        row["problem_id"] = str(row["problem_id"])
        row["sample_idx"] = int(row["sample_idx"])
        row["is_correct"] = _bool(row.get("is_correct"))
        result.append(row)
    return result


def _code_passes_for_run(run_dir: Path, dataset: str, pids: set[str]) -> dict[tuple[str, int], bool]:
    """Read EvalPlus plus-status rows for pass@1 benchmarks."""
    result: dict[tuple[str, int], bool] = {}
    prefix = dataset + "_"
    for path in run_dir.rglob("code_solutions.eval_results.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        eval_rows = data.get("eval", {})
        for task_id, completions in eval_rows.items():
            pid = prefix + str(task_id)
            if pid not in pids:
                continue
            for sample_idx, completion in enumerate(completions):
                base_pass = str(completion.get("base_status", "")).lower() == "pass"
                plus_status = completion.get("plus_status")
                plus_pass = base_pass and (plus_status is None or str(plus_status).lower() == "pass")
                key = (pid, sample_idx)
                if key in result:
                    raise ValueError(f"duplicate EvalPlus result {key} in {run_dir}")
                result[key] = plus_pass
    return result


def read_run(run_dir: Path, expected: dict[str, dict[str, set[int]]]) -> dict:
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    detail_files = _detail_files(run_dir)
    if not detail_files:
        raise ValueError(f"no detailed_results.csv found under {run_dir}")
    rows: list[dict] = []
    status_counts: dict[Path, int] = {}
    for path, status_path in detail_files:
        part = _read_csv(path)
        rows.extend(part)
        if status_path is not None:
            status_counts[status_path] = status_counts.get(status_path, 0) + len(part)
    seen: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["problem_id"], row["sample_idx"])
        if key in seen:
            raise ValueError(f"duplicate (problem, sample) {key} in {run_dir}")
        seen[key] = row

    for status_path, actual_count in status_counts.items():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("expected_samples") is not None and int(status["expected_samples"]) != actual_count:
            raise ValueError(f"run status expected_samples mismatch in {status_path}: {status['expected_samples']} vs {actual_count}")
        if status.get("completed_samples") is not None and int(status["completed_samples"]) != actual_count:
            raise ValueError(f"run status completed_samples mismatch in {status_path}: {status['completed_samples']} vs {actual_count}")

    by_dataset: dict[str, dict[tuple[str, int], dict]] = {name: {} for name in expected}
    for key, row in seen.items():
        dataset = _dataset_for(key[0], expected)
        if dataset is None:
            raise ValueError(f"unexpected problem ID {key[0]!r} in {run_dir}")
        by_dataset[dataset][key] = row
    for dataset, expected_problems in expected.items():
        wanted = {(pid, sample) for pid, sample_ids in expected_problems.items() for sample in sample_ids}
        actual = set(by_dataset[dataset])
        if actual != wanted:
            missing = sorted(wanted - actual)
            extra = sorted(actual - wanted)
            raise ValueError(f"incomplete {dataset} run {run_dir}: expected {len(wanted)} pairs, got {len(actual)}; missing={missing[:8]}, extra={extra[:8]}")

    for dataset in ("humaneval", "mbpp"):
        if dataset not in expected:
            continue
        pids = set(expected[dataset])
        pass_results = _code_passes_for_run(run_dir, dataset, pids)
        needed = {(pid, sample) for pid, samples in expected[dataset].items() for sample in samples}
        if set(pass_results) != needed:
            raise ValueError(f"missing EvalPlus plus-status pairs in {run_dir} for {dataset}: expected {len(needed)}, got {len(pass_results)}")
        for key, value in pass_results.items():
            by_dataset[dataset][key]["is_correct"] = value

    for dataset, rows_for_dataset in by_dataset.items():
        if not rows_for_dataset:
            raise ValueError(f"run {run_dir} has no rows for protocol dataset {dataset}")
    return {"path": str(run_dir), "by_dataset": by_dataset,
            "rows": [row for items in by_dataset.values() for row in items.values()]}


def expected_from_protocol(protocol_path: Path) -> tuple[dict[str, dict[str, set[int]]], dict, dict]:
    protocol, _, groups = load_protocol(protocol_path.resolve())
    expected: dict[str, dict[str, set[int]]] = {}
    repeat_by_dataset: dict[str, set[int]] = {}
    for group in groups:
        for dataset, keys in group["expected_keys"].items():
            for pid, sample in keys:
                expected.setdefault(dataset, {}).setdefault(str(pid), set()).add(int(sample))
            repeat_by_dataset.setdefault(dataset, set()).add(group["repeat_size"])
    inconsistent = {dataset: sizes for dataset, sizes in repeat_by_dataset.items() if len(sizes) > 1}
    if inconsistent:
        raise ValueError(f"protocol uses different repeat_size values within a dataset: {inconsistent}")
    return expected, protocol, {k: next(iter(v)) for k, v in repeat_by_dataset.items()}


def _pass_stats(rows: list[dict]) -> dict:
    token_count = 0
    pass_sum = 0.0
    two_plus_sum = 0.0
    for row in rows:
        count = int(float(row.get("pass_stat_tokens") or 0))
        if count <= 0:
            continue
        mean = row.get("mean_passes_per_token")
        frac = row.get("fraction_tokens_at_least_2_passes")
        if mean in (None, "") or frac in (None, ""):
            continue
        token_count += count
        pass_sum += float(mean) * count
        two_plus_sum += float(frac) * count
    return {
        "pass_stat_tokens": token_count,
        "mean_passes_per_token": pass_sum / token_count if token_count else None,
        "fraction_tokens_at_least_2_passes": two_plus_sum / token_count if token_count else None,
    }


def _run_group(paths: list[Path], expected: dict[str, dict[str, set[int]]]) -> dict:
    if not paths:
        raise ValueError("at least one run directory is required for each group")
    runs = [read_run(path, expected) for path in paths]
    by_dataset: dict[str, dict[str, list[bool]]] = {name: {pid: [] for pid in pids} for name, pids in expected.items()}
    rows_by_dataset: dict[str, list[dict]] = {name: [] for name in expected}
    for run in runs:
        for dataset, pairs in run["by_dataset"].items():
            for (pid, _sample), row in pairs.items():
                by_dataset[dataset][pid].append(bool(row["is_correct"]))
                rows_by_dataset[dataset].append(row)
    return {"runs": runs, "correct_by_dataset": by_dataset, "rows_by_dataset": rows_by_dataset}


def _mean_se(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def paired_stats(reference: dict[str, list[bool]], candidate: dict[str, list[bool]], margin: float) -> dict:
    diffs = [statistics.fmean(candidate[pid]) - statistics.fmean(reference[pid])
             for pid in sorted(reference)]
    mean, se = _mean_se(diffs)
    n = len(diffs)
    if n > 1:
        t95 = t_critical(0.975, n - 1)
        t90 = t_critical(0.95, n - 1)
        ci95 = [mean - t95 * se, mean + t95 * se]
        ci90 = [mean - t90 * se, mean + t90 * se]
    else:
        ci95 = ci90 = [None, None]
    z = mean / se if se else (0.0 if mean == 0 else None)
    verdict = bool(n > 1 and ci90[0] > -margin and ci90[1] < margin)
    return {"n_problems": n, "mean_difference": mean, "paired_se": se,
            "paired_z": z, "ci95": ci95, "ci90": ci90,
            "margin": margin, "tost_equivalent": verdict,
            "tost_verdict": "equivalent" if verdict else "equivalence_not_shown"}


def _pair_analysis(a: dict, b: dict, label_a: str, label_b: str,
                   expected: dict[str, dict[str, set[int]]], margin: float) -> dict:
    datasets: dict[str, dict] = {}
    all_ref: dict[str, list[bool]] = {}
    all_candidate: dict[str, list[bool]] = {}
    for dataset in expected:
        ref = a["correct_by_dataset"][dataset]
        cand = b["correct_by_dataset"][dataset]
        datasets[dataset] = paired_stats(ref, cand, margin)
        all_ref.update({f"{dataset}::{pid}": val for pid, val in ref.items()})
        all_candidate.update({f"{dataset}::{pid}": val for pid, val in cand.items()})
        ref_rows = a["rows_by_dataset"][dataset]
        cand_rows = b["rows_by_dataset"][dataset]
        rstats = _pass_stats(ref_rows)
        cstats = _pass_stats(cand_rows)
        datasets[dataset]["reference_pass_stats"] = rstats
        datasets[dataset]["candidate_pass_stats"] = cstats
        datasets[dataset]["passes_per_token_ratio"] = (
            cstats["mean_passes_per_token"] / rstats["mean_passes_per_token"]
            if rstats["mean_passes_per_token"] not in (None, 0)
            and cstats["mean_passes_per_token"] is not None else None
        )
    pooled = paired_stats(all_ref, all_candidate, margin)
    pooled["reference_pass_stats"] = _pass_stats([r for rows in a["rows_by_dataset"].values() for r in rows])
    pooled["candidate_pass_stats"] = _pass_stats([r for rows in b["rows_by_dataset"].values() for r in rows])
    rmean = pooled["reference_pass_stats"]["mean_passes_per_token"]
    cmean = pooled["candidate_pass_stats"]["mean_passes_per_token"]
    pooled["passes_per_token_ratio"] = cmean / rmean if rmean not in (None, 0) and cmean is not None else None
    return {"comparison": f"{label_a} vs {label_b}", "reference_group": label_a,
            "candidate_group": label_b, "datasets": datasets, "pooled": pooled}


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _fixed(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_outputs(prefix: Path, document: dict, markdown: str) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    print(f"Markdown: {md_path}\nJSON: {json_path}")


def run_equivalence(args: argparse.Namespace) -> int:
    expected, protocol, _ = expected_from_protocol(args.protocol_config)
    group_a = _run_group([Path(x) for x in args.a], expected)
    group_b = _run_group([Path(x) for x in args.b], expected)
    comparisons = [_pair_analysis(group_a, group_b, args.label_a, args.label_b, expected, args.margin)]
    if args.c:
        group_c = _run_group([Path(x) for x in args.c], expected)
        comparisons.append(_pair_analysis(group_a, group_c, args.label_a, args.label_c, expected, args.margin))
    document = {
        "analysis": "R1 paired per-problem equivalence",
        "protocol_config": str(args.protocol_config.resolve()),
        "margin": args.margin,
        "comparisons": comparisons,
    }
    lines = ["# Equivalence analysis", "", f"Protocol: `{protocol.get('name', args.protocol_config.stem)}`",
             f"Equivalence margin: ±{args.margin * 100:.2f} percentage points.", ""]
    for comparison in comparisons:
        lines.extend([f"## {comparison['comparison']}", "",
                      "Difference is candidate minus reference. SE and confidence intervals use paired problem scores.", "",
                      "| Dataset | Problems | Difference | SE | 95% CI | 90% CI | z | TOST | Passes/token ratio |",
                      "|---|---:|---:|---:|---:|---:|---:|---|---:|"])
        rows = [(dataset, stats) for dataset, stats in comparison["datasets"].items()]
        rows.append(("Pooled", comparison["pooled"]))
        for dataset, stats in rows:
            ci95 = stats["ci95"]
            ci90 = stats["ci90"]
            ci95s = "n/a" if ci95[0] is None else f"[{_percent(ci95[0])}, {_percent(ci95[1])}]"
            ci90s = "n/a" if ci90[0] is None else f"[{_percent(ci90[0])}, {_percent(ci90[1])}]"
            z = "n/a" if stats["paired_z"] is None else f"{stats['paired_z']:.2f}"
            ratio = stats["passes_per_token_ratio"]
            lines.append(f"| {dataset} | {stats['n_problems']} | {_percent(stats['mean_difference'])} | {_percent(stats['paired_se'])} | {ci95s} | {ci90s} | {z} | {stats['tost_verdict']} | {'n/a' if ratio is None else f'{ratio:.4f}'} |")
        lines.append("")
    _write_outputs(args.output_prefix, document, "\n".join(lines))
    return 0


def _summary(values_by_problem: dict[str, list[bool]]) -> dict:
    per_problem = {pid: statistics.fmean(values) for pid, values in values_by_problem.items()}
    values = list(per_problem.values())
    mean, se = _mean_se(values)
    n = len(values)
    if n > 1:
        critical = t_critical(0.975, n - 1)
        ci = [mean - critical * se, mean + critical * se]
    else:
        ci = [None, None]
    return {"n_problems": n, "accuracy": mean, "se": se, "ci95": ci,
            "correct_sample_fraction": mean}


def _benchmark_pass_stats(runs: list[dict], dataset: str) -> dict:
    return _pass_stats([row for run in runs for row in run["by_dataset"][dataset].values()])


def run_aggregate(args: argparse.Namespace) -> int:
    expected, protocol, _ = expected_from_protocol(args.protocol_config)
    runs = [read_run(Path(path), expected) for path in args.run]
    combined: dict[str, dict[str, list[bool]]] = {name: {pid: [] for pid in ids} for name, ids in expected.items()}
    for run in runs:
        for dataset, pairs in run["by_dataset"].items():
            for (pid, _sample), row in pairs.items():
                combined[dataset][pid].append(bool(row["is_correct"]))

    paper_values = dict(protocol.get("paper_values", {})) if not args.no_paper_values else {}
    results: dict[str, dict] = {}
    for dataset in expected:
        result = _summary(combined[dataset])
        result["paper_value"] = paper_values.get(dataset)
        result["inside_95_ci"] = (
            bool(result["ci95"][0] <= result["paper_value"] <= result["ci95"][1])
            if result["paper_value"] is not None and result["ci95"][0] is not None else None
        )
        result["pass_stats"] = _benchmark_pass_stats(runs, dataset)
        results[dataset] = result

    benchmark_means = [value["accuracy"] for value in results.values()]
    average = statistics.fmean(benchmark_means)
    variance_components = [results[name]["se"] ** 2 / (len(results) ** 2) for name in results]
    average_se = math.sqrt(sum(variance_components))
    if len(results) > 1 and average_se > 0:
        numerator = sum(variance_components) ** 2
        denominator = sum((v * v) / max(1, results[name]["n_problems"] - 1)
                          for name, v in zip(results, variance_components))
        df = numerator / denominator if denominator else max(1, sum(x["n_problems"] for x in results.values()) - len(results))
        crit = t_critical(0.975, max(1, int(df)))
        average_ci = [average - crit * average_se, average + crit * average_se]
    else:
        average_ci = [None, None]
    paper_average = protocol.get("paper_average")
    average_inside = (bool(average_ci[0] <= paper_average <= average_ci[1])
                      if paper_average is not None and average_ci[0] is not None else None)
    all_rows = [row for run in runs for row in run["rows"]]
    average_record = {
        "n_benchmarks": len(results), "accuracy": average, "se": average_se,
        "ci95": average_ci, "paper_value": paper_average, "inside_95_ci": average_inside,
        "pass_stats": _pass_stats(all_rows),
    }
    document = {
        "analysis": "R2 multi-seed aggregate",
        "protocol_config": str(args.protocol_config.resolve()),
        "run_directories": [str(Path(path).resolve()) for path in args.run],
        "n_seed_runs": len(runs), "benchmarks": results, "benchmark_average": average_record,
    }
    include_paper = any(x["paper_value"] is not None for x in results.values()) or paper_average is not None
    lines = ["# Paper protocol aggregate", "",
             f"Protocol: `{protocol.get('name', args.protocol_config.stem)}`; seed runs: {len(runs)}.",
             "Accuracy is the mean problem score after pooling samples across seeds; SE is computed across problems.", "",
             "| Benchmark | Problems | Mean accuracy | SE | 95% CI | Passes/token | ≥2 passes/token |" + (" Paper | Inside 95% CI |" if include_paper else ""),
             "|---|---:|---:|---:|---:|---:|---:|" + ("---:|---|" if include_paper else "")]
    for dataset, result in results.items():
        ci = result["ci95"]
        ci_text = "n/a" if ci[0] is None else f"[{_percent(ci[0])}, {_percent(ci[1])}]"
        passes = result["pass_stats"]
        line = f"| {dataset} | {result['n_problems']} | {_percent(result['accuracy'])} | {_percent(result['se'])} | {ci_text} | {_fixed(passes['mean_passes_per_token'])} | {_fixed(passes['fraction_tokens_at_least_2_passes'])} |"
        if include_paper:
            paper = "n/a" if result["paper_value"] is None else _percent(result["paper_value"])
            inside = "n/a" if result["inside_95_ci"] is None else ("yes" if result["inside_95_ci"] else "no")
            line += f" {paper} | {inside} |"
        lines.append(line)
    avg_ci = average_record["ci95"]
    avg_ci_text = "n/a" if avg_ci[0] is None else f"[{_percent(avg_ci[0])}, {_percent(avg_ci[1])}]"
    ap = average_record["pass_stats"]
    avg_line = f"| {len(results)}-benchmark average | — | {_percent(average)} | {_percent(average_se)} | {avg_ci_text} | {_fixed(ap['mean_passes_per_token'])} | {_fixed(ap['fraction_tokens_at_least_2_passes'])} |"
    if include_paper:
        paper = "n/a" if paper_average is None else _percent(paper_average)
        inside = "n/a" if average_inside is None else ("yes" if average_inside else "no")
        avg_line += f" {paper} | {inside} |"
    lines.append(avg_line)
    lines.append("")
    _write_outputs(args.output_prefix, document, "\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    eq = sub.add_parser("equivalence", help="paired R1 equivalence/TOST analysis")
    eq.add_argument("--protocol-config", type=Path, required=True)
    eq.add_argument("--a", action="append", required=True, help="run directory for reference/group A; repeat for seeds")
    eq.add_argument("--b", action="append", required=True, help="run directory for candidate/group B; repeat for seeds")
    eq.add_argument("--c", action="append", help="optional second reference/group for A-vs-C self-noise")
    eq.add_argument("--label-a", default="Reference")
    eq.add_argument("--label-b", default="Candidate")
    eq.add_argument("--label-c", default="Reference run 2")
    eq.add_argument("--margin", type=float, default=0.03, help="equivalence margin in accuracy units (default 0.03 = 3 pp)")
    eq.add_argument("--output-prefix", type=Path, default=Path("equivalence"))
    eq.set_defaults(func=run_equivalence)

    agg = sub.add_parser("aggregate", help="aggregate the R2 protocol across seed runs")
    agg.add_argument("--protocol-config", type=Path, required=True)
    agg.add_argument("--run", action="append", required=True, help="canonical protocol run directory; repeat once per seed")
    agg.add_argument("--no-paper-values", action="store_true")
    agg.add_argument("--output-prefix", type=Path, default=Path("aggregate"))
    agg.set_defaults(func=run_aggregate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "equivalence" and not 0 < args.margin < 1:
            raise ValueError("margin must be strictly between 0 and 1")
        return args.func(args)
    except Exception as exc:
        print(f"compare_runs.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
