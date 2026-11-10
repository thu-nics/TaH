"""Run the full accessible benchmark suite sequentially on the shared GPU."""
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = ["humaneval", "mbpp", "gsm8k", "math500", "amc23", "aime25", "olympiadbench", "mmlu_stem", "gpqa"]


def main():
    lock = (ROOT / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    status = {"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat(), "jobs": {}}
    for name in ORDER:
        info = manifest["datasets"].get(name, {})
        status["jobs"][name] = {"state": "queued" if info.get("status") == "ready" else "blocked", "reason": info.get("reason")}
    def save():
        temp = ROOT / "status.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(ROOT / "status.json")
    save()
    for name in ORDER:
        job = status["jobs"][name]
        if job["state"] == "blocked":
            continue
        output = ROOT / "full" / name
        command = [sys.executable, "-u", str(ROOT / "run.py"), "--datasets", name, "--batch-size", "8", "--output", str(output)]
        job.update(state="running", command=command, started=datetime.now(timezone.utc).isoformat())
        save()
        with (ROOT / f"full_{name}.log").open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        job.update(state="complete" if result.returncode == 0 else "failed", returncode=result.returncode, finished=datetime.now(timezone.utc).isoformat())
        save()
    status["finished"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
