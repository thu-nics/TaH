#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RouteLLM Batch Scoring Script
- Read dataset (CSV/JSON/JSONL)
- Use RouteLLM router (default: mf) to calculate "strong win rate" score for each question
- Write CSV output in original order: id, score

Dependencies:
  pip install "routellm[serve,eval]" pandas tqdm

Usage examples:
  python baseline_routellm_getscore.py \
      --data /path/to/dataset.jsonl \
      --out /path/to/out.csv

Different column names:
  python baseline_routellm_getscore.py \
      --data /path/to/dataset.csv --id-col qid --text-col prompt --out scores.csv

Specify router and config:
  python baseline_routellm_getscore.py \
      --data ds.csv --out scores.csv \
      --router mf \
      --config /path/to/config.example.yaml

Load dataset from eval_unified (accuracy mode, corresponding to dataset_configs.json):
  python baseline_routellm_getscore.py \
      --dataset-name gsm8k \
      --out scores.csv
  # Multiple datasets combined:
  python baseline_routellm_getscore.py \
      --dataset-name "gsm8k,math500" \
      --out scores.csv

Note: Uses mf router by default. The mf router requires OPENAI_API_KEY for embedding calculation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

# RouteLLM imports
from routellm.controller import Controller

# Unified evaluation dataset loading (accuracy mode)
try:
    from tah.evaluate.eval_unified import load_datasets_with_config  # Returns (combined_data, field_mapping)
except Exception:
    load_datasets_with_config = None  # Allow --data branch when tah package is not installed

def read_dataset(path: Path, id_col: str, text_col: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append(obj)
        df = pd.DataFrame(rows)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            raise ValueError("JSON top level is an object, expected an array.")
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file type: {suffix} (only .csv/.jsonl/.json are supported)")

    # Auto-detect column names (if not explicitly provided and default columns don't exist)
    if id_col not in df.columns:
        cand = [c for c in ["id", "qid", "question_id", "sample_id"] if c in df.columns]
        if cand:
            id_col = cand[0]
        else:
            raise ValueError(f"ID column not found (tried default '{id_col}', also no common candidates in {df.columns.tolist()})")
    if text_col not in df.columns:
        cand = [c for c in ["question", "prompt", "input", "text"] if c in df.columns]
        if cand:
            text_col = cand[0]
        else:
            raise ValueError(f"Question column not found (tried default '{text_col}', also no common candidates in {df.columns.tolist()})")

    # Keep only these two columns and preserve order
    df = df[[id_col, text_col]].copy()
    df.columns = ["id", "question"]
    return df


def build_controller(router_name: str, config_path: str = None,
                     strong_model: str = None, weak_model: str = None) -> Controller:
    """
    Build RouteLLM Controller.
    - routers: Pass one or more router names (here we only use one)
    - Optional: Pass strong/weak model names (doesn't affect scoring, mainly for actual routing scenarios)
    - Optional: Pass config file (overrides default configuration)
    """
    kwargs = dict(routers=[router_name])
    if config_path:
        kwargs["config"] = config_path
    kwargs["strong_model"] = strong_model
    kwargs["weak_model"] = weak_model
    client = Controller(**kwargs)
    return client


def score_one(controller: Controller, router_name: str, question: str) -> float:
    """
    Calculate "strong win rate" score for a single question.
    Note:
    - RouteLLM documentation states that each router implements `calculate_strong_win_rate(prompt)->float`
    - Controller holds router instances internally. Attribute names may differ across versions, so we support two access methods.
    """
    # Method A: Prefer public/semi-public router table
    if hasattr(controller, "routers") and isinstance(controller.routers, dict):
        router = controller.routers.get(router_name)
        if router and hasattr(router, "calculate_strong_win_rate"):
            return float(router.calculate_strong_win_rate(question))

    # Method B: Some versions use _routers private attribute
    if hasattr(controller, "_routers") and isinstance(controller._routers, dict):
        router = controller._routers.get(router_name)
        if router and hasattr(router, "calculate_strong_win_rate"):
            return float(router.calculate_strong_win_rate(question))

    raise RuntimeError(
        "Failed to get router instance or its calculate_strong_win_rate method from Controller. "
        "Please upgrade routellm version or ensure router name is correct when building Controller."
    )


def main():
    parser = argparse.ArgumentParser(description="RouteLLM Batch Scoring")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", type=str, help="Dataset path (.csv/.jsonl/.json)")
    group.add_argument("--dataset-name", type=str, help="eval_unified dataset name (comma-separated for multiple)")
    parser.add_argument("--out", required=True, type=str, help="Output CSV path")
    parser.add_argument("--id-col", default="id", type=str, help="ID column name in dataset when using --data (default: id)")
    parser.add_argument("--text-col", default="question", type=str, help="Text column name in dataset when using --data (default: question)")
    parser.add_argument("--router", default="mf", type=str, help="Router name (default: mf)")
    parser.add_argument("--config", default=None, type=str, help="RouteLLM config YAML path (optional)")
    parser.add_argument("--strong-model", default=None, type=str, help="Strong model name (optional)")
    parser.add_argument("--weak-model", default=None, type=str, help="Weak model name (optional)")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size (this script scores one by one, batch size only for display pacing)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path = out_path/ f"{args.router}/{args.dataset_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read data
    if getattr(args, "dataset_name", None):
        if load_datasets_with_config is None:
            raise RuntimeError("Failed to import eval_unified dataset loading function. Please ensure tah package is in PYTHONPATH or use --data method.")
        # Align with eval_unified: support comma-separated or single dataset names
        dataset_names = [name.strip() for name in args.dataset_name.split(',') if name.strip()]
        combined_data, field_mapping = load_datasets_with_config(dataset_names)
        # Use standard fields (eval_unified already unified to id/question/answer/...)
        df = pd.DataFrame(combined_data)
        if not {"id", "question"}.issubset(df.columns):
            raise ValueError("Standard columns 'id' and 'question' not found after loading from eval_unified.")
        df = df[["id", "question"]].copy()
    else:
        if not args.data:
            raise ValueError("Must provide either --data or --dataset-name.")
        data_path = Path(args.data)
        df = read_dataset(data_path, id_col=args.id_col, text_col=args.text_col)

    # Build Controller (no actual chat generation, only use router scores)
    controller = build_controller(
        router_name=args.router,
        config_path=args.config,
        strong_model=args.strong_model,
        weak_model=args.weak_model,
    )

    # Calculate scores one by one (in original order)
    scores: List[Tuple] = []
    pbar = tqdm(total=len(df), desc=f"Scoring with RouteLLM[{args.router}]")
    for i, row in df.iterrows():
        qid = row["id"]
        qtext = row["question"]
        try:
            score = score_one(controller, args.router, qtext)
        except Exception as e:
            # Return NaN on failure; print error to STDERR
            print(f"[WARN] id={qid} scoring failed: {e}", file=sys.stderr)
            score = float("nan")
        scores.append((qid, score))
        pbar.update(1)
    pbar.close()

    # Write result CSV
    out_df = pd.DataFrame(scores, columns=["id", "score"])
    # Preserve order: pandas maintains insertion order; for safety, re-align with original df order:
    order = {k: idx for idx, k in enumerate(df["id"].tolist())}
    out_df["__order"] = out_df["id"].map(order)
    out_df = out_df.sort_values("__order").drop(columns="__order")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Write complete: {out_path}")

if __name__ == "__main__":
    main()