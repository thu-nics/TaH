"""Shared utilities for eval_offline.py and eval_online.py."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple



CODE_ANSWER_TYPES = {"humaneval", "mbpp"}


def load_jsonl(path: str | Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str | Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rewrite_detailed_results(
    path: str | Path,
    updates: Dict[tuple[str, int], Dict],
    default: Optional[Dict] = None,
) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"No columns in {path}")
    for row in rows:
        key = str(row["problem_id"]), int(row["sample_idx"])
        if key not in updates and default is None:
            raise ValueError(f"Missing grade for {key}")
        row.update(updates.get(key, default))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_data_range(data_range_list, total_problems: int = None) -> Tuple[int, int]:
    """Parse [end] or [start, end]; return (start_idx, end_idx), exclusive end."""
    if not data_range_list:
        return 0, total_problems if total_problems is not None else 0
    if len(data_range_list) == 1:
        start_idx, end_idx = 0, data_range_list[0]
    elif len(data_range_list) == 2:
        start_idx, end_idx = data_range_list[0], data_range_list[1]
    else:
        raise ValueError(
            f"data_range expects 1 or 2 values, got {len(data_range_list)}"
        )
    if total_problems is not None:
        end_idx = min(end_idx, total_problems)
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError(f"Invalid data_range: start={start_idx}, end={end_idx}")
    return start_idx, end_idx


def load_datasets_with_config(dataset_names) -> Tuple[list, Dict]:
    """Load dataset(s) by name; return (data_list, standardized_field_mapping).

    All items are standardized to fields: id, question (template applied), answer.
    """
    from datasets import load_dataset

    if isinstance(dataset_names, str):
        names = [n.strip() for n in dataset_names.split(",") if n.strip()]
    else:
        names = [str(n).strip() for n in dataset_names if str(n).strip()]
    names = [n.lower() for n in names]
    if not names:
        raise ValueError("No dataset names provided")

    config_file = Path(__file__).parent / "eval_configs" / "dataset_configs.json"
    with open(config_file, "r", encoding="utf-8") as f:
        all_configs = json.load(f)

    combined_data: list = []
    answer_types: list = []
    print(f"Loading {len(names)} dataset(s): {names}")

    for dataset_name in names:
        if dataset_name not in all_configs:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: {list(all_configs)}"
            )
        dc = all_configs[dataset_name]

        path = dc["path"]
        if dc.get("data_files"):
            ds_obj = load_dataset(
                path,
                data_files=dc["data_files"],
            )
        elif path.endswith(".json") or path.endswith(".jsonl"):
            if not Path(path).exists():
                # Hub reference "<org>/<repo>/<file>.jsonl" (e.g.
                # TaH-plus/eval_science_code/mbpp.jsonl): fetch to the local
                # HF cache once and read it like a local file.
                from huggingface_hub import hf_hub_download

                parts = path.split("/")
                if len(parts) < 3:
                    raise FileNotFoundError(
                        f"Dataset file '{path}' not found locally and not a "
                        f"'<org>/<repo>/<file>' hub reference."
                    )
                path = hf_hub_download(
                    repo_id="/".join(parts[:2]),
                    repo_type="dataset",
                    filename="/".join(parts[2:]),
                )
            if dataset_name in ("mbpp", "humaneval"):
                ds_obj = {"train": load_jsonl(path)}
            else:
                ds_obj = load_dataset("json", data_files=path)
        elif dc.get("subset"):
            ds_obj = load_dataset(path, dc["subset"])
        elif dc.get("version_tag"):
            ds_obj = load_dataset(path, version_tag=dc["version_tag"])
        else:
            ds_obj = load_dataset(path)

        split = dc.get("split_name", "test")
        for candidate in (split, "train", "test"):
            if candidate in ds_obj:
                dataset = ds_obj[candidate]
                break
        else:
            raise ValueError(
                f"No suitable split among {list(ds_obj)} for '{dataset_name}'"
            )

        if "filter" in dc:
            fc = dc["filter"]
            if isinstance(dataset, list):
                dataset = [item for item in dataset if item.get(fc["key"]) in fc["value"]]
            else:
                dataset = dataset.filter(
                    lambda x: x.get(fc["key"]) in fc["value"]
                )

        fm = {
            "id_field": dc["id_field"],
            "question_field": dc["question_field"],
            "answer_field": dc["answer_field"],
            "answer_type": dc["answer_type"],
            "prompt_template": dc.get("prompt_template", "{question}"),
        }
        entry_point_field = dc.get("entry_point")
        assertion_field = dc.get("assertion")
        answer_types.append(fm["answer_type"])

        for idx, item in enumerate(list(dataset)):
            raw_id = item.get(fm["id_field"])
            original_id = str(raw_id) if raw_id is not None else f"{dataset_name}_{idx}"
            safe_id = original_id.replace("/", "_").replace("\\", "_")

            q = str(item.get(fm["question_field"], "")).strip()
            choices = item.get(dc.get("choices_field", ""), [])
            if choices:
                q += "\n" + "\n".join(
                    f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)
                )
            tmpl = fm["prompt_template"]
            if tmpl and "{question}" in tmpl:
                q = tmpl.replace("{question}", q)

            row: Dict = {
                "id": f"{dataset_name}_{safe_id}",
                "_original_id": original_id,
                "question": q,
                "answer": str(item.get(fm["answer_field"], "")).strip(),
                "_source_dataset": dataset_name,
            }
            if dc.get("version_tag"):
                row["_dataset_version"] = dc["version_tag"]
            if entry_point_field:
                row["entry_point"] = item.get(entry_point_field)
            if assertion_field:
                row["assertion"] = item.get(assertion_field)
            for k, v in item.items():
                if k not in (
                    fm["id_field"],
                    fm["question_field"],
                    fm["answer_field"],
                ) and not k.startswith("_"):
                    row[f"_original_{k}"] = v
            combined_data.append(row)

        print(f"  {dataset_name}: {len(list(dataset))} samples")

    print(f"Total: {len(combined_data)} samples")
    if len(set(answer_types)) > 1:
        print(f"Warning: mixed answer types {set(answer_types)}, using first")

    field_mapping = {
        "id_field": "id",
        "question_field": "question",
        "answer_field": "answer",
        "answer_type": answer_types[0] if answer_types else "string",
        "prompt_template": "{question}",
        "dataset_names": names,
    }
    return combined_data, field_mapping


def compute_metrics(detailed_results_path: str) -> Optional[Dict]:
    """Compute pass@n, cons@n, avg@n from detailed_results.csv."""
    groups: Dict[str, list] = defaultdict(list)
    with open(detailed_results_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["problem_id"]].append(row)
    if not groups:
        return None

    n = max(len(v) for v in groups.values())
    num_problems = len(groups)
    total_correct = total_samples = pass_count = cons_count = total_tokens = 0
    request_times = []

    for rows in groups.values():
        corrects = [r["is_correct"] == "True" for r in rows]
        c, k = sum(corrects), len(rows)
        total_correct += c
        total_samples += k
        if c > 0:
            pass_count += 1

        counts: Counter = Counter()
        correct_map: Dict[str, bool] = {}
        for r in rows:
            ans = (
                r["predicted_answer"]
                if r.get("has_answer") != "False" and r["predicted_answer"]
                else "__NO_ANSWER__"
            )
            counts[ans] += 1
            if r["is_correct"] == "True":
                correct_map[ans] = True
        if correct_map.get(counts.most_common(1)[0][0], False):
            cons_count += 1

        for r in rows:
            try:
                output_tokens = int(r["output_tokens"])
                total_tokens += output_tokens
            except (ValueError, KeyError):
                output_tokens = None
            try:
                request_time = float(r["processing_time"])
            except (ValueError, KeyError):
                request_time = None
            if request_time is not None:
                request_times.append(request_time)

    metrics = {
        f"pass@{n}": round(pass_count / num_problems, 4),
        f"cons@{n}": round(cons_count / num_problems, 4),
        f"avg@{n}": round(total_correct / total_samples, 4),
        "avg_output_length": (
            round(total_tokens / total_samples, 2) if total_samples else 0.0
        ),
        "num_problems": num_problems,
        "num_samples": total_samples,
        "repeat_size": n,
    }
    if request_times:
        total_request_time = sum(request_times)
        metrics.update(
            {
                "avg_request_time": round(total_request_time / len(request_times), 4),
                "end_2_end_time": round(total_request_time, 4),
            }
        )
        if total_request_time > 0:
            metrics["token_throughtput"] = round(total_tokens / total_request_time, 4)
            metrics["request_throught"] = round(
                len(request_times) / total_request_time, 4
            )
    return metrics


def update_iter_count_distribution(
    iter_dist: Dict[int, int], tracker_records: List[Dict]
) -> None:
    """Convert cumulative depth populations to exact iteration counts."""
    if not tracker_records:
        return
    depth_counts = Counter(
        int(row["iter_depth"]) for row in tracker_records if "iter_depth" in row
    )
    max_iter_count = max(depth_counts, default=-1) + 1
    for ic in range(1, max_iter_count + 1):
        delta = depth_counts.get(ic - 1, 0) - depth_counts.get(ic, 0)
        if delta > 0:
            iter_dist[ic] = iter_dist.get(ic, 0) + delta


def load_iter_count_distribution(iter_dist_path: Path) -> Dict[int, int]:
    iter_dist: Dict[int, int] = {}
    if not iter_dist_path.exists():
        return iter_dist

    with open(iter_dist_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("iter_count")
            if not key or key == "Total Tokens":
                continue
            try:
                iter_dist[int(key)] = int(float(row["token_count"]))
            except (TypeError, ValueError, KeyError):
                continue
    return iter_dist


def save_iter_count_distribution(
    iter_dist_path: Path, iter_dist: Dict[int, int]
) -> float:
    total_tokens = sum(iter_dist.values())
    if total_tokens <= 0:
        if iter_dist_path.exists():
            iter_dist_path.unlink()
        return 0.0

    weighted_total = sum(ic * cnt for ic, cnt in iter_dist.items())
    with open(iter_dist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iter_count", "token_count", "percentage"])
        for ic in sorted(iter_dist):
            cnt = iter_dist[ic]
            if cnt > 0:
                writer.writerow([ic, cnt, cnt / total_tokens * 100])
        writer.writerow([])
        writer.writerow(["Total Tokens", total_tokens, 100.0])
    return weighted_total / total_tokens


def save_results_json(
    results_path: Path,
    detailed_results_path: Path,
    avg_iter_count: float = 0.0,
    extra_metrics: Optional[Dict] = None,
) -> None:
    if not detailed_results_path.exists():
        return

    metrics = compute_metrics(str(detailed_results_path))
    if not metrics:
        return
    if avg_iter_count > 0:
        metrics["avg_iter_count"] = round(avg_iter_count, 4)
    if extra_metrics:
        metrics.update(extra_metrics)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def _save_job_stats(output_dir: Path, batch_result: Dict) -> None:
    """Write evaluation_stats.csv from a batch_result dict."""
    all_results = batch_result["all_results"]
    problem_stats = batch_result["problem_stats"]

    total = len(all_results)
    total_correct = sum(bool(r["is_correct"]) for r in all_results)
    avg_len = (
        sum(int(r["output_tokens"]) for r in all_results) / total if total else 0.0
    )

    with open(
        output_dir / "evaluation_stats.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "problem_id",
                "accuracy",
                "correct_count",
                "total_samples",
                "avg_output_length",
            ]
        )
        for pid, s in problem_stats.items():
            writer.writerow(
                [
                    pid,
                    s["accuracy"],
                    s["correct_count"],
                    s["total_samples"],
                    f"{s['avg_output_length']:.2f}",
                ]
            )
        writer.writerow([])
        writer.writerow(
            [
                "Total Accuracy",
                f"{total_correct / total:.3f}" if total else "0.000",
                total_correct,
                total,
                f"{avg_len:.2f}",
            ]
        )


def _save_detailed_stats(output_dir: Path, rows: List[Dict]) -> None:
    """Rebuild evaluation_stats.csv after external grading."""
    typed_rows = [
        {**row, "is_correct": str(row["is_correct"]) == "True"} for row in rows
    ]
    groups = defaultdict(list)
    for row in typed_rows:
        groups[row["problem_id"]].append(row)
    problem_stats = {}
    for problem_id, group in groups.items():
        correct = sum(row["is_correct"] for row in group)
        problem_stats[problem_id] = {
            "accuracy": f"{correct / len(group):.3f}",
            "correct_count": correct,
            "total_samples": len(group),
            "avg_output_length": sum(int(row["output_tokens"]) for row in group)
            / len(group),
        }
    _save_job_stats(
        output_dir,
        {"all_results": typed_rows, "problem_stats": problem_stats},
    )


def _maybe_rename_with_accuracy(combo_out_dir: Path, field_mapping: Dict) -> Path:
    """Append accuracy to output dir name; skip for code datasets."""
    if field_mapping.get("answer_type") in CODE_ANSWER_TYPES:
        return combo_out_dir
    accuracy_str = ""
    stats_csv = combo_out_dir / "evaluation_stats.csv"
    if stats_csv.exists():
        with open(stats_csv, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0] == "Total Accuracy":
                    accuracy_str = row[1]
                    break
    if accuracy_str and not combo_out_dir.name.endswith(f"_{accuracy_str}"):
        new_dir = combo_out_dir.parent / f"{combo_out_dir.name}_{accuracy_str}"
        combo_out_dir.rename(new_dir)
        return new_dir
    return combo_out_dir
