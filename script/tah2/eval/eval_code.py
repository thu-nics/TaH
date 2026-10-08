"""Evaluate code generations with EvalPlus."""

import argparse
import json
import os
import re
from pathlib import Path


_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_DEF_NAME_RE = re.compile(r"\bdef\s+\w+\s*\(")


def extract_code(text: str, entry_point: str = "") -> str:
    """Prefer a fenced block defining the requested function."""
    blocks = _BLOCK_RE.findall(text)
    if not blocks:
        return ""
    if entry_point:
        for b in blocks:
            if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", b):
                return b
    for b in blocks:
        if _DEF_NAME_RE.search(b):
            return b
    return blocks[-1]


def detect_dataset_from_path(samples_path: Path) -> str:
    """Detect a dataset from its evaluation directory."""
    grandparent = samples_path.parent.parent.name
    for name in ("mbpp", "humaneval"):
        if grandparent.startswith(name + "_") or grandparent == name:
            return name
    return None


def detect_dataset_from_samples(samples: list) -> str:
    """Fallback: auto-detect evalplus dataset from sample ids."""
    for s in samples:
        sid = s.get("id", "")
        for name in ("mbpp", "humaneval"):
            if sid.startswith(f"{name}_"):
                return name
    return None


def restore_task_id(standardized_id: str, dataset: str) -> str:
    """Restore an EvalPlus task ID from its filesystem-safe form."""
    prefix = dataset + "_"
    if not standardized_id.startswith(prefix):
        return standardized_id
    safe_id = standardized_id[len(prefix):]
    return safe_id.replace("_", "/", 1)


def build_solutions(samples: list, dataset: str) -> list:
    """Build EvalPlus entries, re-extracting code from each raw output."""
    solutions = []
    for s in samples:
        code = extract_code(s.get("output", ""), s.get("entry_point", ""))
        task_id = restore_task_id(s["id"], dataset)
        solutions.append({"task_id": task_id, "solution": code})
    return solutions


def run_evalplus(solutions_path: str, dataset: str):
    """Run evalplus evaluation and return (pass_at_k_dict, result_path)."""
    from tah2.evaluate.codeeval import evaluate as code_evaluate

    code_evaluate(
        dataset=dataset,
        samples=solutions_path,
        i_just_wanna_run=True,
    )

    result_path = solutions_path.replace(".jsonl", ".eval_results.json")
    if not os.path.isfile(result_path):
        result_path = solutions_path.replace(".jsonl", "_eval_results.json")

    if os.path.isfile(result_path):
        with open(result_path, "r") as f:
            results = json.load(f)
        return results.get("pass_at_k", {}), result_path
    return {}, result_path


def rename_dir_with_accuracy(samples_dir: Path, pass_at_k: dict):
    """Rename the directory to include both base and plus pass@1 accuracy."""
    base_pak = pass_at_k.get("base", {})
    plus_pak = pass_at_k.get("plus", {})
    base_acc = base_pak.get("pass@1")
    plus_acc = plus_pak.get("pass@1")

    if base_acc is None and plus_acc is None:
        print("Warning: no pass@1 found in results, skip renaming.")
        return samples_dir

    parts = []
    if base_acc is not None:
        parts.append(f"{base_acc:.3f}")
    if plus_acc is not None:
        parts.append(f"{plus_acc:.3f}")
    acc_suffix = "_".join(parts)

    current_name = samples_dir.name

    stripped_name = re.sub(r"_\d+\.\d{3}$", "", current_name)

    new_dir = samples_dir.parent / f"{stripped_name}_{acc_suffix}"
    if new_dir == samples_dir:
        print(f"Directory name unchanged: {samples_dir}")
        return samples_dir
    if new_dir.exists():
        print(f"Target directory already exists: {new_dir}, skip renaming.")
        return samples_dir

    samples_dir.rename(new_dir)
    print(f"Renamed: {samples_dir.name} -> {new_dir.name}")
    return new_dir


def evaluate_one(samples_path: str, args):
    """Evaluate a single samples.jsonl file."""
    samples_path = Path(samples_path)
    if not samples_path.is_file():
        print(f"Error: {samples_path} not found.")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating: {samples_path}")
    print(f"{'='*60}")

    from tah2.evaluate.common import load_jsonl, save_jsonl

    samples = load_jsonl(samples_path)

    if not samples:
        print("Warning: empty samples.jsonl, skipping.")
        return

    dataset = detect_dataset_from_path(samples_path) or detect_dataset_from_samples(samples)
    if dataset is None:
        print(f"Error: cannot detect dataset type for {samples_path}, skipping.")
        return
    print(f"Dataset: {dataset}  |  Samples: {len(samples)}")

    solutions = build_solutions(samples, dataset)

    solutions_file = samples_path.parent / "code_solutions.jsonl"
    save_jsonl(solutions_file, solutions)
    print(f"Wrote {len(solutions)} solutions to: {solutions_file}")

    pass_at_k, result_path = run_evalplus(str(solutions_file), dataset)
    print(f"Eval results: {result_path}")

    if pass_at_k:
        for split, metrics in pass_at_k.items():
            for k, v in metrics.items():
                print(f"  [{split}] {k}: {v:.4f}")

    samples_dir = samples_path.parent
    rename_dir_with_accuracy(samples_dir, pass_at_k)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone code evaluation: samples.jsonl -> evalplus -> rename folder with accuracy"
    )
    parser.add_argument(
        "--path", type=str, nargs="+", required=True,
        help="Path(s) to samples.jsonl file(s). Dataset type is auto-detected from directory name."
    )
    args = parser.parse_args()

    failed = False
    for sp in args.path:
        try:
            evaluate_one(sp, args)
        except Exception as e:
            failed = True
            import traceback
            print(f"Error evaluating {sp}: {e}")
            traceback.print_exc()

    print("\nAll evaluations complete.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
