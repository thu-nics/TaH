"""Run the larger experiment unattended; resume saved collection shards."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("research/bandit_router/math200"))
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--max-reference-tokens", type=int, default=1024)
    args = p.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.time()

    def status(stage, **extra):
        report = {"stage": stage, "elapsed_seconds": time.time() - start, **extra}
        temporary = root / "status.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(root / "status.json")
        print(json.dumps(report), flush=True)

    def run(stage, module, *arguments):
        status(stage)
        subprocess.run([sys.executable, "-u", "-m", f"research.bandit_router.{module}", *map(str, arguments)], check=True)

    try:
        references = root / "references.jsonl"
        if not references.exists():
            run("preparing_references", "prepare_math", "--output", references,
                "--count", args.count, "--max-reference-tokens", args.max_reference_tokens)
        manifest = json.loads(references.with_suffix(".manifest.json").read_text())
        if manifest["problems"] != args.count or manifest["max_reference_tokens"] != args.max_reference_tokens:
            raise ValueError("Prepared reference settings differ from this run")
        if not (root / "cache.pt").exists():
            run("collecting", "collect_large", "--references", references,
                "--output-dir", root, "--max-reference-tokens", args.max_reference_tokens)
        run("verifying_history", "verify_cache", "--cache", root / "cache.pt", "--max-problems", 3)
        run("training_sweep", "train_sweep", "--cache", root / "cache.pt", "--output-dir", root / "training")
        for policy in ("imitation", "bandit"):
            run(f"generation_{policy}", "run", "generate", "--router", root / "training/selected_router.pt",
                "--policy", policy, "--max-tokens", 512, "--output", root / f"generation_{policy}.json")
        status("complete", summary=str(root / "training/summary.json"))
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
