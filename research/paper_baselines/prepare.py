"""Freeze benchmark inputs from the authors' public checkpoint evaluation logs."""
import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "author_data"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = {"datasets": {}, "models": {}}
    for name in ("Standard-0.6B", "TaH-plus-0.6B"):
        repo = f"nics-efc/{name}"
        sha = api.model_info(repo).sha
        path = snapshot_download(repo, revision=sha, local_dir=f"models/{name}")
        manifest["models"][name] = {"repo": repo, "revision": sha, "path": path}
    from evalplus.data import get_human_eval_plus, get_mbpp_plus, get_human_eval_plus_hash, get_mbpp_plus_hash
    code_problems = {"humaneval": get_human_eval_plus(), "mbpp": get_mbpp_plus()}
    code_hashes = {"humaneval": get_human_eval_plus_hash(), "mbpp": get_mbpp_plus_hash()}
    archive = Path("models/TaH-plus-0.6B/eval_results")
    for name in ("gsm8k", "math500", "amc23", "aime25", "olympiadbench", "gpqa", "mmlu_stem", "humaneval", "mbpp"):
        try:
            samples = next(archive.glob(f"{name}_tah/*/samples.jsonl"))
            recipe_path = next(samples.parent.glob("*.yaml"))
            recipe = yaml.safe_load(recipe_path.read_text())
            unique = {}
            for line in samples.read_text().splitlines():
                entry = json.loads(line)
                item = {"id": entry["id"], "question": entry["problem"], "answer": entry["correct_answer"]}
                if name in code_problems:
                    matches = [p for p in code_problems[name].values() if p["entry_point"] == entry["entry_point"] and p["prompt"].strip() == entry["problem"].strip()]
                    if len(matches) != 1:
                        raise ValueError(f"Cannot uniquely map {entry['id']} to EvalPlus")
                    item.update(id=matches[0]["task_id"], entry_point=entry["entry_point"])
                if item["id"] in unique and unique[item["id"]] != item:
                    raise ValueError(f"Inconsistent repeated input for {item['id']}")
                unique[item["id"]] = item
            rows = list(unique.values())
            path = DATA / f"{name}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            manifest["datasets"][name] = {
                "status": "ready", "source": "authors_public_evaluation_artifacts",
                "source_repo": "nics-efc/TaH-plus-0.6B",
                "revision": manifest["models"]["TaH-plus-0.6B"]["revision"],
                "source_samples": str(samples), "source_recipe": str(recipe_path),
                "source_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
                "data_file": str(path), "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "problems": len(rows), "repeat_size": recipe["repeat_size"],
                "max_new_tokens": recipe["max_new_tokens"],
                "temperature": recipe["temperature"], "top_p": recipe["top_p"], "top_k": recipe["top_k"],
            }
            if name in code_hashes:
                manifest["datasets"][name]["evalplus_hash"] = code_hashes[name]
        except Exception as exc:
            manifest["datasets"][name] = {"status": "blocked", "reason": str(exc)}
        print(name, manifest["datasets"][name], flush=True)
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
