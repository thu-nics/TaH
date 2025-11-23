#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_problem_stats(detailed_csv: Path) -> Tuple[Dict[str, float], Dict[str, bool], Dict[str, str]]:
    """Load detailed_results.csv and compute per-problem stats:
    - mean_correct_map: problem_id -> mean(is_correct) across samples (float in [0,1])
    - any_correct_map: problem_id -> any(is_correct) across samples (bool)
    - pred_map: problem_id -> representative predicted_answer (prefer a correct sample, else first non-empty)
    """
    df = pd.read_csv(detailed_csv)
    # print(df)
    # Normalize id types
    if "problem_id" in df.columns:
        df["problem_id"] = df["problem_id"].astype(str).str.strip()
    required_cols = {"problem_id", "is_correct"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{detailed_csv} 缺少必要列: {required_cols}，实际列: {df.columns.tolist()}")

    # Determine predicted_answer column if present
    pred_col = "predicted_answer" if "predicted_answer" in df.columns else None

    # mean correctness per problem
    mean_correct = df.groupby("problem_id")["is_correct"].mean().reset_index()
    mean_correct_map = dict(zip(mean_correct["problem_id"], mean_correct["is_correct"].astype(float)))

    # any correctness per problem
    any_correct = df.groupby("problem_id")["is_correct"].any().reset_index()
    any_correct_map = dict(zip(any_correct["problem_id"], any_correct["is_correct"].astype(bool)))

    # Representative predicted_answer per problem
    pred_map: Dict[str, str] = {}
    if pred_col is None:
        # No predicted column; leave empty strings
        for pid in df["problem_id"].unique():
            pred_map[pid] = ""
        return mean_correct_map, any_correct_map, pred_map

    # Prefer first correct sample's predicted_answer; else first non-empty
    for pid, grp in df.groupby("problem_id", sort=False):
        rep = ""
        # Prefer correct
        corr_rows = grp[grp["is_correct"] == True]
        if not corr_rows.empty:
            val = corr_rows.iloc[0][pred_col]
            rep = "" if pd.isna(val) else str(val)
        else:
            # First non-empty
            non_empty = grp[grp[pred_col].notna() & (grp[pred_col].astype(str).str.len() > 0)]
            if not non_empty.empty:
                rep = str(non_empty.iloc[0][pred_col])
            else:
                rep = ""
        pred_map[pid] = rep

    return mean_correct_map, any_correct_map, pred_map


def main():
    parser = argparse.ArgumentParser(description="Compute Routellm thresholded accuracy by aligning IDs and selecting models by score threshold.")
    parser.add_argument("--scores", required=True, type=str, help="RouteLLM scores CSV (columns: id, score)")
    parser.add_argument("--strong-detailed", required=True, type=str, help="Strong model detailed_results.csv")
    parser.add_argument("--weak-detailed", required=True, type=str, help="Weak model detailed_results.csv")
    parser.add_argument("--strong-size", default=4.0, type=float, help="Strong model params in B (default 4.0)")
    parser.add_argument("--weak-size", default=0.6, type=float, help="Weak model params in B (default 0.6)")
    parser.add_argument("--target-avg", default=1.7, type=float, help="Target average params in B (default 1.7)")
    parser.add_argument("--rounding", default="round", choices=["round", "floor", "ceil"], help="How to convert proportion to count (default round)")
    parser.add_argument("--save-csv", default=None, type=str, help="可选：保存详细路由与正误结果到该 CSV 文件")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    strong_path = Path(args.strong_detailed)
    weak_path = Path(args.weak_detailed)

    # Load scores
    scores_df = pd.read_csv(scores_path)
    if not {"id", "score"}.issubset(scores_df.columns):
        raise ValueError(f"{scores_path} 需要包含列 ['id','score']，实际列: {scores_df.columns.tolist()}")
    # Normalize ids
    scores_df["id"] = scores_df["id"].astype(str).str.strip()

    # Compute strong proportion p s.t. p*strong + (1-p)*weak = target
    denom = (args.strong_size - args.weak_size)
    if denom <= 0:
        raise ValueError("strong_size 必须大于 weak_size")
    p_strong = (args.target_avg - args.weak_size) / denom
    p_strong = max(0.0, min(1.0, p_strong))

    # Decide top-K by score to assign to strong
    n = len(scores_df)
    if args.rounding == "round":
        k = int(round(p_strong * n))
    elif args.rounding == "floor":
        k = int(math.floor(p_strong * n))
    else:
        k = int(math.ceil(p_strong * n))
    k = max(0, min(n, k))

    # Sort descending by score; tie-breaker: stable by original order
    scores_sorted = scores_df.sort_values(["score", "id"], ascending=[False, True]).reset_index(drop=True)
    # Determine threshold value for reporting (score at position k-1)
    threshold = float("nan")
    if k > 0:
        threshold = float(scores_sorted.iloc[k - 1]["score"]) if k - 1 < len(scores_sorted) else float("nan")

    # Build model assignment
    assign_series = pd.Series(["strong"] * k + ["weak"] * (n - k))
    assign_df = scores_sorted.copy()
    assign_df["assigned_model"] = assign_series.values

    # Load per-problem stats
    strong_mean_map, strong_any_map, strong_pred_map = load_problem_stats(strong_path)
    weak_mean_map, weak_any_map, weak_pred_map = load_problem_stats(weak_path)

    # Report missing ids coverage
    ids_set = set(scores_df["id"].tolist())
    miss_strong = [qid for qid in ids_set if qid not in strong_mean_map]
    # print(('olympiadbench_1606' in ids_set), ('olympiadbench_1606' in strong_mean_map))
    miss_weak = [qid for qid in ids_set if qid not in weak_mean_map]
    if miss_strong:
        print(f"[WARN] {len(miss_strong)} ids not found in strong detailed_results (treated as incorrect)")
    if miss_weak:
        print(f"[WARN] {len(miss_weak)} ids not found in weak detailed_results (treated as incorrect)")

    # Join mean correctness by assignment
    def pick_correct_mean(row):
        qid = row["id"]
        if row["assigned_model"] == "strong":
            return float(strong_mean_map.get(qid, 0.0))
        return float(weak_mean_map.get(qid, 0.0))

    assign_df["final_correct_mean"] = assign_df.apply(pick_correct_mean, axis=1)

    # Attach model-wise representative predicted answers and correctness
    assign_df["strong_predicted"] = assign_df["id"].map(lambda x: strong_pred_map.get(x, ""))
    assign_df["strong_correct_mean"] = assign_df["id"].map(lambda x: float(strong_mean_map.get(x, 0.0)))
    assign_df["strong_is_correct_any"] = assign_df["id"].map(lambda x: bool(strong_any_map.get(x, False)))
    assign_df["weak_predicted"] = assign_df["id"].map(lambda x: weak_pred_map.get(x, ""))
    assign_df["weak_correct_mean"] = assign_df["id"].map(lambda x: float(weak_mean_map.get(x, 0.0)))
    assign_df["weak_is_correct_any"] = assign_df["id"].map(lambda x: bool(weak_any_map.get(x, False)))

    # Final routed predicted
    # print(assign_df["strong_predicted"])
    def pick_pred(row):
        # if row['assigned_model'] == 'strong':
        #     print(row["strong_predicted"])
        return row["strong_predicted"] if row["assigned_model"] == "strong" else row["weak_predicted"]

    assign_df["final_predicted"] = assign_df.apply(pick_pred, axis=1)

    # Compute accuracy as average of per-problem mean correctness
    total = len(assign_df)
    sum_mean = float(assign_df["final_correct_mean"].sum())
    acc = sum_mean / total if total > 0 else 0.0

    # Report
    print("Routing with target average params:")
    print(f"  strong_size={args.strong_size}B, weak_size={args.weak_size}B, target_avg={args.target_avg}B")
    print(f"  proportion_strong={p_strong:.6f}  -> select_top_k={k}/{n}")
    print(f"  score_threshold≈{threshold}")
    print("")
    print("Final accuracy (mean across samples per problem):")
    print(f"  avg-correctness-sum={sum_mean:.2f} / total={total}  -> accuracy={acc:.4f}")

    # Also print a small breakdown
    strong_correct_mean = float(assign_df.loc[assign_df["assigned_model"] == "strong", "final_correct_mean"].sum())
    weak_correct_mean = float(assign_df.loc[assign_df["assigned_model"] == "weak", "final_correct_mean"].sum())
    print("")
    print("Breakdown:")
    print(f"  strong: {k} selected, mean-correct-sum={strong_correct_mean:.2f}")
    print(f"  weak:   {n - k} selected, mean-correct-sum={weak_correct_mean:.2f}")

    # Optional save CSV with detailed assignment and outputs
    if args.save_csv:
        out_cols = [
            "id", "score", "assigned_model",
            "strong_predicted", "strong_correct_mean", "strong_is_correct_any",
            "weak_predicted", "weak_correct_mean", "weak_is_correct_any",
            "final_predicted", "final_correct_mean",
        ]
        save_df = assign_df[out_cols].copy()
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_df.to_csv(out_path, index=False, encoding="utf-8")
        print("")
        print(f"Saved detailed routing CSV to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


