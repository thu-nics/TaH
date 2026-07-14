"""Dataset loading + per-row standardisation for the eval driver.

Each entry in ``eval_configs/dataset_configs.json`` describes how to fetch a
benchmark (HF id or local file), which split to use, optional row filter,
and the field names to map onto our standard ``(id, question, answer)``
schema. ``load_combined_dataset`` reads the entry, normalises every row,
and returns a flat ``list[dict]`` plus a small ``field_mapping`` dict that
downstream code uses to know the answer type (math / livecodebench /
humaneval / mbpp).

Why standardise: the per-job runner only knows the standard schema, so a
new benchmark only needs a config-file entry to be evaluable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


_CONFIG_PATH = Path(__file__).with_name("eval_configs") / "dataset_configs.json"


def _split_names(names: Union[str, List[str]]) -> List[str]:
    """Accept comma-separated string or list; return a clean ``list[str]``."""
    if isinstance(names, str):
        items = [n.strip() for n in names.split(",")]
    elif isinstance(names, list):
        items = [n.strip() if isinstance(n, str) else n for n in names]
    else:
        raise TypeError(f"dataset_names must be str or list, got {type(names).__name__}")
    items = [n for n in items if n]
    if not items:
        raise ValueError("no dataset names provided")
    return items


def _load_one(name: str, configs: Dict[str, Dict]) -> Tuple[List[dict], Dict[str, str]]:
    """Load one dataset (HF id or local file), apply optional split + filter,
    and standardise rows to ``{id, _original_id, question, answer, ...}``."""
    if name not in configs:
        raise ValueError(f"dataset {name!r} not in eval_configs/dataset_configs.json (have {sorted(configs)})")
    cfg = configs[name]
    path = cfg["path"]

    # Local JSON/JSONL takes a different code path because mbpp/humaneval files
    # contain test scaffolding that load_dataset can't infer schemas for.
    if path.endswith((".json", ".jsonl")):
        if name in ("mbpp", "humaneval"):
            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(ln) for ln in f if ln.strip()]
            split = "train"
            ds_obj: Dict[str, Any] = {"train": rows}
        else:
            from datasets import load_dataset
            ds_obj = load_dataset("json", data_files=path)
    else:
        from datasets import load_dataset
        if cfg.get("subset"):
            ds_obj = load_dataset(path, cfg["subset"])
        elif cfg.get("version_tag"):
            ds_obj = load_dataset(path, version_tag=cfg["version_tag"])
        else:
            ds_obj = load_dataset(path)

    split = cfg.get("split_name", "test")
    if split not in ds_obj:
        for fallback in ("train", "test"):
            if fallback in ds_obj:
                split = fallback
                break
        else:
            raise ValueError(f"split {cfg.get('split_name')!r} not in {list(ds_obj)}")
    ds = ds_obj[split]
    print(f"  {name}: split={split!r}, {len(ds)} rows")

    if "filter" in cfg:
        f = cfg["filter"]
        ds = ds.filter(lambda x, k=f["key"], v=f["value"]: x.get(k) in v)

    id_field = cfg["id_field"]
    q_field = cfg["question_field"]
    a_field = cfg["answer_field"]
    template = cfg.get("prompt_template", "{question}")
    entry_point_field = cfg.get("entry_point")

    standardised: List[dict] = []
    for idx, row in enumerate(ds):
        original_id = str(row[id_field]) if row.get(id_field) is not None else str(idx)
        question = template.replace("{question}", str(row.get(q_field, "")).strip())
        item = {
            "id": f"{name}_{original_id}",
            "_original_id": original_id,
            "question": question,
            "answer": str(row.get(a_field, "")).strip(),
            "_source_dataset": name,
        }
        if entry_point_field:
            item["entry_point"] = row.get(entry_point_field)
        # Carry remaining columns under _original_<col> for downstream debugging.
        for k, v in row.items():
            if k not in (id_field, q_field, a_field) and not k.startswith("_"):
                item[f"_original_{k}"] = v
        standardised.append(item)
    return standardised, {
        "id_field": "id",
        "question_field": "question",
        "answer_field": "answer",
        "answer_type": cfg["answer_type"],
        "prompt_template": "{question}",
    }


def load_combined_dataset(dataset_names: Union[str, List[str]]) -> Tuple[List[dict], Dict]:
    """Resolve one or more benchmark names → flat row list + field mapping.

    Field mapping is derived from the *first* dataset's ``answer_type`` and
    is stamped onto every row (combined runs of mixed answer types are not
    expected; we warn but proceed).
    """
    names = _split_names(dataset_names)
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        configs = json.load(f)

    print(f"Loading {len(names)} dataset(s): {names}")
    combined: List[dict] = []
    answer_types: List[str] = []
    field_mapping: Dict[str, Any] = {}
    for name in names:
        rows, fm = _load_one(name, configs)
        combined.extend(rows)
        answer_types.append(fm["answer_type"])
        if not field_mapping:
            field_mapping = dict(fm)

    if len(set(answer_types)) > 1:
        print(f"WARNING: multiple answer_types in combined run: {set(answer_types)} — using {answer_types[0]!r}")
    field_mapping["dataset_names"] = names
    print(f"Combined dataset: {len(combined)} rows total\n")
    return combined, field_mapping
