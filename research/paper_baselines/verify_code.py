"""Smoke-check EvalPlus base+extra grading with a known canonical solution."""
import json
from pathlib import Path


def main():
    import tah.evaluate.codeeval as evaluator
    from evalplus.data import get_human_eval_plus
    problem = next(iter(get_human_eval_plus().values()))
    evaluator.get_human_eval_plus = lambda: {problem["task_id"]: problem}
    # A separate cache key prevents changing the real full-dataset oracle cache.
    evaluator.get_human_eval_plus_hash = lambda: "cs475_smoke_canonical_file_v1"
    root = Path(__file__).resolve().parent / "code_grader_check"
    root.mkdir(exist_ok=True)
    samples = root / "canonical.jsonl"
    samples.write_text(json.dumps({"task_id": problem["task_id"], "solution": problem["prompt"] + problem["canonical_solution"]}) + "\n")
    evaluator.evaluate("humaneval", str(samples), parallel=1)
    result = json.loads(samples.with_suffix(".eval_results.json").read_text())
    assert result["pass_at_k"]["plus"]["pass@1"] == 1.0
    print("PASS: EvalPlus canonical solution passes base and extra tests", flush=True)


if __name__ == "__main__":
    main()
