"""G-TF assertions and trace conversion, used by pytest/tooling only."""
from __future__ import annotations

import torch

from .reference_capture import compare, summarize


def decision_metrics(reference, candidate, threshold):
    """Count flips on common visits; later visitation cascades are diagnostic.

    Margins are symmetric: both measured probabilities must be near the
    threshold. The reference noise calibration uses this same convention.
    """
    both = reference["visited"] & candidate["visited"]
    flips = both & (reference["decision"] != candidate["decision"])
    margins = torch.maximum((reference["continue_probability"].float() - threshold).abs(),
                            (candidate["continue_probability"].float() - threshold).abs())
    assert torch.isfinite(margins[flips]).all(), "nonfinite probability at a decision flip"
    return [{"depth": depth + 1, "common_visited_compared": int(both[depth].sum()),
             "common_decision_mismatches": int(flips[depth].sum()),
             "max_flip_margin": float(margins[depth][flips[depth]].max()) if flips[depth].any() else 0.}
            for depth in range(len(both))]


def output_trace(output, template):
    trace = {key: getattr(output, key).detach().cpu() for key in (
        "visited", "decision", "decision_logits", "continue_probability", "decider_hidden")}
    trace["pass_count"] = output.iter_count.cpu()
    # Coordinate systems come from the oracle; top-k errors are measured at
    # reference indices, not by comparing independently sorted lists.
    for key in ("final_topk_values", "final_topk_indices", "logit_positions", "final_logits_sample"):
        trace[key] = template[key]
    return trace


def compare_output(template, output, reference_logits, threshold):
    candidate = output_trace(output, template)
    row = compare(template, candidate, reference_logits, output.logits.cpu(), threshold)
    for depth, metrics in zip(row["per_depth"], decision_metrics(template, candidate, threshold)):
        depth.update(metrics)
    return row


def require_production(suite):
    assert suite.manifest.get("production_gates_eligible") is True, "G-TF requires production fixtures; cpu-tiny is development only"
    assert suite.manifest.get("profile") == "full"
    assert set(suite.manifest["checkpoints"]) == {"real", "mlp3", "mlp1", "label3"}
    assert suite.manifest["self_noise"]["mlp3"]["full_fixture_depth_coverage_pass"] is True
    for name in suite.manifest["checkpoints"]:
        records = list(suite.records(kind="teacher_forced", checkpoint=name))
        assert len(records) == 45, f"missing WP2 TF layouts for {name}"
        assert len({r["sequence_id"] for r in records}) == 9


def derive_decision_limits(noise):
    """Revised SPEC §7 limits, using counts rather than rounded agreement."""
    per_depth = []
    floor = noise.get('mlp_count_floor', 0)
    learned_depths = noise.get('learned_depths', [])
    for depth in noise["per_depth"]:
        count = depth.get("common_visited_compared", depth["decision_compared"] - depth["visited_mismatches"])
        flips = depth["common_decision_mismatches"]
        assert 0 <= flips <= count
        per_depth.append({"depth": depth["depth"], "reference_common_visited": count,
                          "reference_flips": flips,
                          "flip_rate_limit": 1.5 * (max(floor, flips) if depth['depth'] in learned_depths else flips) / max(count, 1)})
        if floor:
            per_depth[-1]['effective_reference_flips'] = max(floor, flips) if depth['depth'] in learned_depths else flips
    max_margin = noise.get("reference_max_flip_margin")
    if max_margin is None:
        # Zero-flip development noise needs no margin measurement. A noisy
        # production baseline must provide the measured value, never a guess.
        assert not any(d["reference_flips"] for d in per_depth), "missing measured reference flip margin"
        max_margin = 0.
    return {"per_depth": per_depth, "reference_max_flip_margin": max_margin,
            "flip_margin_limit": max(.01, 1.5 * max_margin),
            "reference_pass_count_mismatches": noise["pass_count_mismatches"],
            "reference_token_comparisons": noise["top1_compared"],
            "pass_count_mismatch_rate_limit": 1.5 * (max(floor, noise["pass_count_mismatches"]) if learned_depths else noise["pass_count_mismatches"]) / max(noise["top1_compared"], 1)}


def production_noise(suite, name, tolerance_export):
    """Derive decision limits from immutable WP2 baseline/layout traces.

    The WP2 export predates revised decision rules; its logit limits remain
    authoritative. Reconstruct common counts and the largest flip margin
    from the 36 saved reference pairs, and verify WP2's recorded counts.
    """
    import hashlib
    assert tolerance_export["production_gates_eligible"] is True
    assert hashlib.sha256((suite.root / "manifest.json").read_bytes()).hexdigest() == tolerance_export["manifest_sha256"]
    noise = dict(suite.manifest["self_noise"][name]["teacher_forced"]["summary"])
    exported = tolerance_export["checkpoints"][name]
    assert noise == exported["reference_self_noise"], "WP2 manifest/export self-noise differs"
    for metric in ("logit_mae", "logit_rmse", "logit_max_abs"):
        assert noise[f"g_tf_{metric}_limit"] == exported["G-TF"][f"{metric}_limit"]
    assert noise["g_tf_top1_disagreement_limit"] == exported["G-TF"]["top1_disagreement_limit"]
    per_depth = [{"common_visited_compared": 0, "common_decision_mismatches": 0, "max_flip_margin": 0.}
                 for _ in noise["per_depth"]]
    records = list(suite.records(kind="teacher_forced", checkpoint=name))
    passes, tokens, pairs = 0, 0, 0
    for record in records:
        assert hashlib.sha256((suite.root / record["file"]).read_bytes()).hexdigest() == record["sha256"]
    for baseline_record in (r for r in records if r["variant"] == "single"):
        baseline = suite.load(baseline_record)
        for record in records:
            if record["sequence_id"] != baseline_record["sequence_id"] or record["variant"] == "single":
                continue
            candidate = suite.load(record)
            metrics = decision_metrics(baseline, candidate, suite.manifest["checkpoints"][name]["threshold"])
            for total, row in zip(per_depth, metrics):
                total["common_visited_compared"] += row["common_visited_compared"]
                total["common_decision_mismatches"] += row["common_decision_mismatches"]
                total["max_flip_margin"] = max(total["max_flip_margin"], row["max_flip_margin"])
            passes += int((baseline["pass_count"] != candidate["pass_count"]).sum())
            tokens += baseline["pass_count"].numel()
            pairs += 1
    assert pairs == 36
    assert passes == noise["pass_count_mismatches"] and tokens == noise["top1_compared"]
    for stored, measured in zip(noise["per_depth"], per_depth):
        assert measured["common_visited_compared"] == stored["decision_compared"] - stored["visited_mismatches"]
        assert measured["common_decision_mismatches"] == stored["common_decision_mismatches"]
    noise["reference_max_flip_margin"] = max(d["max_flip_margin"] for d in per_depth)
    return noise


def gtf_summary(rows):
    assert rows, "G-TF has no comparisons"
    stats = summarize(rows)
    for index, depth in enumerate(stats["per_depth"]):
        common = sum(r["per_depth"][index]["common_visited_compared"] for r in rows)
        maximum = max(r["per_depth"][index]["max_flip_margin"] for r in rows)
        depth.update(common_visited_compared=common, max_flip_margin=maximum,
                     common_flip_rate=depth["common_decision_mismatches"] / max(common, 1))
    stats["pass_count_mismatch_rate"] = stats["pass_count_mismatches"] / max(stats["top1_compared"], 1)
    return stats


def assert_gtf(rows, noise):
    stats = gtf_summary(rows)
    limits = derive_decision_limits(noise)
    assert len(stats["per_depth"]) == len(limits["per_depth"])
    for depth, limit in zip(stats["per_depth"], limits["per_depth"]):
        assert depth["common_flip_rate"] <= limit["flip_rate_limit"], (depth, limit)
        assert depth["max_flip_margin"] <= limits["flip_margin_limit"], (depth, limits)
    assert stats["pass_count_mismatch_rate"] <= limits["pass_count_mismatch_rate_limit"], (stats, limits)
    for metric in ("logit_mae", "logit_rmse", "logit_max_abs"):
        assert stats[metric] <= noise[f"g_tf_{metric}_limit"], (metric, stats[metric], noise)
    disagreement = stats["top1_mismatches"] / max(stats["top1_compared"], 1)
    assert disagreement <= noise["g_tf_top1_disagreement_limit"], (disagreement, noise)
    for coordinates in ("retained_topk_error", "retained_dense_error"):
        for metric in ("logit_mae", "logit_rmse", "logit_max_abs"):
            assert stats[coordinates][metric] <= noise[coordinates][f"g_tf_{metric}_limit"], (coordinates, metric, stats, noise)
    return stats
