"""Inference engine and batch processing for eval_offline persistent workers.

eval_offline.py uses:
  - initialize_inference_engine  (load model, return inference_fn)
  - process_batch_items          (run inference + save per-job results)
  - combine_job_results          (merge per-job dirs into final output)
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .common import CODE_ANSWER_TYPES, compute_metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _time_inference(func, cuda_available=True):
    import torch
    if cuda_available and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func()
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end) / 1000.0
    t0 = time.time()
    result = func()
    return result, time.time() - t0


def _warmup_model(model, tokenizer):
    from tah2.minisgl.core import SamplingParams as SP
    model.generate([tokenizer.encode("who are you?")], SP(temperature=0.6, top_p=0.95, top_k=20, max_tokens=100))


def _cleanup_resources(model, backend):
    import torch
    if model is None:
        return
    if backend == "mini_sglang":
        if hasattr(model, "shutdown"):
            model.shutdown()
        del model
    elif backend == "hf":
        for attr in ("iter_decider", "eval_iter_decider"):
            dec = getattr(model, attr, None)
            if dec and hasattr(dec, "shutdown"):
                if attr == "eval_iter_decider" and dec is getattr(model, "iter_decider", None):
                    continue
                dec.shutdown()
        del model
    else:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"\bdef\s+\w+\s*\(")


def extract_code(text: str, entry_point: str = "") -> str:
    """Pick the python block most likely to be the solution.

    Distilled / verbose students often emit ``solution block`` followed
    by an ``Example Usage`` block of ``print(...)`` calls. Taking the
    last block — the legacy behavior — extracts the demo and fails. Prefer
    the block that defines the asked-for function.
    """
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        return ""
    if entry_point:
        for b in blocks:
            if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", b):
                return b
    for b in blocks:
        if _DEF_RE.search(b):
            return b
    return blocks[-1]


# ---------------------------------------------------------------------------
# Public: initialize_inference_engine
# ---------------------------------------------------------------------------

def initialize_inference_engine(config: Dict, model_path: str, tp_size: int, backend: str):
    """Load model once; return (inference_fn, tokenizer, cleanup_fn, tracker, set_threshold_fn, set_gen_params_fn)."""
    import torch
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if backend not in ("hf", "mini_sglang"):
        raise ValueError(f"Unsupported backend: {backend}")
    if tp_size != 1:
        raise ValueError("Offline evaluation supports tp_size=1; use online evaluation for TP serving.")
    if backend == "hf" and not Path(model_path).is_dir():
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(model_path, allow_patterns=[
            "*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken",
            "pytorch_model*.bin", "input_updater.bin", "iter_decider.bin",
        ])

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    tracker = None
    _is_tah = False

    if backend == "mini_sglang":
        from tah2.minisgl.core import SamplingParams as MiniSP
        from tah2.minisgl.llm import LLM as MiniLLM

        mini_tah_threshold = config.get("tah_iter_threshold")
        if mini_tah_threshold is None:
            mini_tah_threshold = (config.get("iter_decider_kwargs") or {}).get("threshold")

        model = MiniLLM(
            model_path,
            dtype=getattr(torch, config.get("dtype", "bfloat16")),
            max_seq_len_override=config.get("max_seq_len_override", 32768),
            cuda_graph_max_bs=config.get("cuda_graph_max_bs", 128),
            page_size=config.get("page_size", 1),
            tah_iter_threshold=mini_tah_threshold,
        )
        from tah2.minisgl.models.tah_qwen3 import TaHQwen3ForCausalLM
        _is_tah = isinstance(model.engine.model, TaHQwen3ForCausalLM)
        if _is_tah:
            dec = type(model.engine.model.tah_decider).__name__
            thr = getattr(model.engine.model.tah_decider, "_threshold", None)
            print(f"[mini_sglang] TaH model: decider={dec}, threshold={thr}")
        else:
            print("[mini_sglang] Standard model, TaH disabled.")
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            tokenizer = model.tokenizer

        _warmup_model(model, tokenizer)

        def _iter_log_to_records(iter_counts):
            return [{"iter_depth": d} for cnt in iter_counts for d in range(cnt)]

        def inference_function(inputs):
            mini_sp = MiniSP(temperature=config["temperature"], top_p=config["top_p"],
                             top_k=config.get("top_k") or -1, max_tokens=config["max_new_tokens"])
            outputs = []
            for i in range(0, len(inputs), config["batch_size"]):
                batch = inputs[i : i + config["batch_size"]]
                ids = [tokenizer.encode(t) for t in batch]
                if _is_tah:
                    model.engine.tah_iter_log.clear()
                res, elapsed = _time_inference(lambda: model.generate(ids, mini_sp), torch.cuda.is_available())
                per = elapsed / max(len(res), 1)
                for j, out in enumerate(res):
                    text = tokenizer.decode(out.get("token_ids", []), skip_special_tokens=True)
                    if _is_tah:
                        records = _iter_log_to_records(model.engine.tah_iter_log.get(j, []))
                        outputs.append((text, per, records))
                    else:
                        outputs.append((text, per))
            return outputs

    elif backend == "hf":
        from tah2.model.tah_model import TaHForCausalLM
        from tah2.utils.modeling import TaHForCasualLM_generate
        from tah2.utils.tracker import TaHTracker

        # Keep the checkpoint's attention mode, kernel and component shapes.
        override_cfg = TaHForCausalLM._load_saved_tah_config(model_path)
        _is_tah = override_cfg is not None and override_cfg.max_iter > 1
        load_kwargs = {
            "torch_dtype": getattr(torch, config.get("dtype", "bfloat16")),
            "device_map": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            "attn_implementation": "sdpa",
        }
        if _is_tah:
            for key in ("embedding_key", "max_iter", "iter_decider", "eval_iter_decider"):
                if key in config:
                    setattr(override_cfg, key, config[key])
            for key in ("iter_decider_kwargs", "eval_iter_decider_kwargs"):
                if config.get(key):
                    setattr(override_cfg, key, {**getattr(override_cfg, key), **config[key]})
            model = TaHForCausalLM.from_pretrained(model_path, tah_config=override_cfg, **load_kwargs)
            model = model.to(dtype=model.dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.eval()
        print(f"[hf] {'TaH2' if _is_tah else 'Standard'} model: {type(model).__name__}")

        if _is_tah and config.get("use_tracker"):
            tracker = TaHTracker(top_k=(config.get("tracker_kwargs") or {}).get("top_k", 5))
            tracker.attach(model)

        prompt_iter_count = config.get("prompt_iter_count")

        def inference_function(inputs):
            outputs = []
            for i in range(0, len(inputs), config["batch_size"]):
                batch = inputs[i : i + config["batch_size"]]
                enc = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left")
                dev = model.device
                enc = {k: v.to(dev) for k, v in enc.items()}
                iter_count = None
                if prompt_iter_count is not None:
                    bs, sl = enc["input_ids"].shape
                    iter_count = prompt_iter_count * torch.ones(bs, sl, dtype=torch.long, device=dev)
                prev = len(tracker.records) if tracker else 0

                def gen():
                    with torch.no_grad():
                        if not _is_tah:
                            generation_kwargs = {
                                "max_new_tokens": config["max_new_tokens"],
                                "do_sample": config["temperature"] > 0.0,
                                "eos_token_id": tokenizer.eos_token_id,
                                "pad_token_id": tokenizer.pad_token_id,
                            }
                            if generation_kwargs["do_sample"]:
                                generation_kwargs.update(
                                    temperature=config["temperature"], top_p=config["top_p"],
                                    top_k=max(config.get("top_k") or 0, 0),
                                    min_p=config.get("min_p", 0.0),
                                )
                            tokens = model.generate(**enc, **generation_kwargs)
                            return tokenizer.batch_decode(tokens[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                        return TaHForCasualLM_generate(
                            tah_model=model, tokenizer=tokenizer, model_inputs=enc,
                            iter_count=iter_count, max_new_tokens=config["max_new_tokens"],
                            do_sample=config["temperature"] > 0.0, temperature=config["temperature"],
                            top_p=config["top_p"], top_k=config.get("top_k", 0),
                            min_p=config.get("min_p", 0.0), verbose=False,
                        )[1]

                texts, elapsed = _time_inference(gen, torch.cuda.is_available())
                per = elapsed / max(len(texts), 1)
                if tracker:
                    by_batch = {}
                    for rec in tracker.records[prev:]:
                        by_batch.setdefault(rec.get("batch_idx", 0), []).append(rec)
                    for j, text in enumerate(texts):
                        outputs.append((text, per, by_batch.get(j, [])))
                else:
                    for text in texts:
                        outputs.append((text, per))
            return outputs

    def cleanup_function():
        nonlocal model
        if tracker is not None:
            tracker.detach()
        _cleanup_resources(model, backend)
        model = None

    def set_threshold_fn(threshold: float):
        if backend == "mini_sglang" and _is_tah:
            model.engine.model.tah_decider._threshold = threshold
        elif backend == "hf" and _is_tah:
            decider = model.eval_iter_decider if model.eval_iter_decider is not None else model.iter_decider
            if "threshold" in decider._buffers and decider._buffers["threshold"] is not None:
                decider._buffers["threshold"].fill_(threshold)
            base = getattr(decider, "base_iter_decider", None)
            if base is not None and "threshold" in base._buffers and base._buffers["threshold"] is not None:
                base._buffers["threshold"].fill_(threshold)

    def set_generation_params_fn(max_new_tokens=None, batch_size=None, repeat_size=None):
        for key, value in (("max_new_tokens", max_new_tokens), ("batch_size", batch_size), ("repeat_size", repeat_size)):
            if value is not None:
                config[key] = value

    return inference_function, tokenizer, cleanup_function, tracker, set_threshold_fn, set_generation_params_fn


# ---------------------------------------------------------------------------
# Public: process_batch_items
# ---------------------------------------------------------------------------

def process_batch_items(
    config: Dict,
    combined_dataset_name: str,
    output_dir: Path,
    job_id: int,
    problems_data: List,
    field_mapping: Dict,
    inference_function,
    tokenizer,
    tracker,
    unified_code_solutions_file=None,
) -> Dict:
    """Run inference on a batch of problems; save per-sample results; return batch stats."""
    import tah2.evaluate.matheval as matheval

    answer_type = field_mapping.get("answer_type", "boxed")
    is_code = answer_type in CODE_ANSWER_TYPES
    if is_code:
        import tah2.evaluate.codeeval as codeeval

    detail_dir = output_dir / "details"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "detailed_results.csv"
    fieldnames = ["problem_id", "sample_idx", "correct_answer", "predicted_answer",
                  "has_answer", "is_correct", "input_tokens", "output_tokens", "processing_time"]
    file_exists = results_file.exists() and results_file.stat().st_size > 0

    # Prepare problem metadata and inputs
    problem_data, all_inputs, input_map = [], [], []
    for idx, item in enumerate(problems_data):
        actual_idx = item.get("_original_index", idx)
        id_field = field_mapping["id_field"]
        problem_id = str(item[id_field]) if id_field in item and item[id_field] is not None else f"problem_{actual_idx}"
        problem_text = str(item.get(field_mapping["question_field"], "")).strip()
        tmpl = field_mapping.get("prompt_template", "{question}")
        if tmpl and "{question}" in tmpl:
            problem_text = tmpl.replace("{question}", problem_text)
        correct_answer = str(item.get(field_mapping["answer_field"], "")).strip()

        prob_dir = detail_dir / problem_id
        prob_dir.mkdir(parents=True, exist_ok=True)

        pd_entry = {
            "problem_id": problem_id,
            "original_problem_id": item.get("_original_id", problem_id),
            "problem_text": problem_text,
            "correct_answer": correct_answer,
            "problem_dir": prob_dir,
            "actual_idx": actual_idx,
        }
        if is_code:
            pd_entry["entry_point"] = item.get("entry_point", "")
            pd_entry["assertion"] = item.get("assertion", "")
            pd_entry["starter_code"] = item.get("_original_starter_code", "")
        problem_data.append(pd_entry)

        for sample_idx in range(config["repeat_size"]):
            if is_code:
                prompt = codeeval.make_raw_chat_prompt_for_code_evaluation(
                    task_prompt=problem_text, assertion=pd_entry["assertion"],
                    answer_type=answer_type, reasoning=True, tokenizer=tokenizer,
                    starter_code=pd_entry["starter_code"],
                )
            else:
                messages = [{"role": "user", "content": problem_text}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_inputs.append(prompt)
            input_map.append((len(problem_data) - 1, sample_idx))

    # Run inference
    batch_outputs = inference_function(all_inputs)

    all_results = []
    with open(results_file, "a" if file_exists else "w", newline="", encoding="utf-8") as f_res:
        writer = csv.DictWriter(f_res, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, out in enumerate(batch_outputs):
            prob_idx, sample_idx = input_map[i]
            pd_entry = problem_data[prob_idx]
            problem_id = pd_entry["problem_id"]
            problem_text = pd_entry["problem_text"]
            correct_answer = pd_entry["correct_answer"]
            prob_dir = pd_entry["problem_dir"]

            if isinstance(out, tuple) and len(out) == 3:
                output_text, proc_time, sample_tracker_records = out
            else:
                output_text, proc_time = out
                sample_tracker_records = None

            if is_code:
                predicted_answer, has_answer, is_correct = "pending_code_eval", False, False
            else:
                eval_result = matheval.evaluator_map[combined_dataset_name].rule_judge(output_text, correct_answer)
                is_correct = eval_result[0]
                if eval_result[1] == "No extracted answer":
                    predicted_answer, has_answer = "", False
                else:
                    predicted_answer, has_answer = eval_result[1], True

            input_tokens = len(tokenizer.encode(problem_text))
            output_tokens = len(tokenizer.encode(output_text))

            row = {
                "problem_id": problem_id,
                "sample_idx": sample_idx,
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "has_answer": has_answer,
                "is_correct": is_correct,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "processing_time": proc_time,
            }
            all_results.append(row)

            # Save per-sample detail
            detail = {"problem": problem_text, "output": output_text, "correct_answer": correct_answer,
                      "predicted_answer": predicted_answer, "is_correct": is_correct}
            if is_code:
                detail["extracted_code"] = extract_code(output_text)
                detail["entry_point"] = pd_entry["entry_point"]
                if unified_code_solutions_file:
                    import fcntl
                    entry = {"task_id": pd_entry["original_problem_id"], "solution": str(detail["extracted_code"])}
                    with open(unified_code_solutions_file, "a", encoding="utf-8") as fc:
                        fcntl.flock(fc.fileno(), fcntl.LOCK_EX)
                        try:
                            fc.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        finally:
                            fcntl.flock(fc.fileno(), fcntl.LOCK_UN)
            with open(prob_dir / f"sample_{sample_idx}.json", "w", encoding="utf-8") as fj:
                json.dump(detail, fj, ensure_ascii=False, indent=2)

            if sample_tracker_records:
                pd.DataFrame(sample_tracker_records).to_csv(prob_dir / f"sample_{sample_idx}_tracker.csv", index=False)

            writer.writerow(row)

    # Build problem_stats
    by_problem: Dict[str, list] = {}
    for r in all_results:
        by_problem.setdefault(r["problem_id"], []).append(r)
    problem_stats = {
        pid: {
            "accuracy": f"{sum(1 for r in rows if r['is_correct']) / len(rows):.3f}",
            "correct_count": sum(1 for r in rows if r["is_correct"]),
            "total_samples": len(rows),
            "avg_output_length": sum(r["output_tokens"] for r in rows) / len(rows),
        }
        for pid, rows in by_problem.items()
    }
    return {"all_results": all_results, "problem_stats": problem_stats}


# ---------------------------------------------------------------------------
# Public: combine_job_results
# ---------------------------------------------------------------------------

def combine_job_results(output_dir: Path, job_nums: int, del_job_dir: bool = False, save_all_trackers: bool = True):
    """Merge per-job subdirectories into a single combined result under output_dir."""
    import shutil

    all_results = []
    problem_stats: Dict = {}
    sample_output_len: Dict = {}
    iter_dist = {i: 0 for i in range(1, 6)}
    all_tracker_files = []

    samples_path = output_dir / "samples.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path.write_text("")  # truncate

    for job_id in range(job_nums):
        job_dir = output_dir / f"job_{job_id}"

        # detailed_results.csv
        res_file = job_dir / "detailed_results.csv"
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["is_correct"] = row["is_correct"] == "True"
                    if "has_boxed_answer" in row:
                        row["has_answer"] = row["has_boxed_answer"] == "True"
                        del row["has_boxed_answer"]
                    else:
                        row["has_answer"] = row.get("has_answer") == "True"
                    row["sample_idx"] = int(row["sample_idx"])
                    row["input_tokens"] = int(row["input_tokens"])
                    row["output_tokens"] = int(row["output_tokens"])
                    row["processing_time"] = float(row["processing_time"])
                    all_results.append(row)
                    sample_output_len[(row["problem_id"], row["sample_idx"])] = row["output_tokens"]

        # evaluation_stats.csv
        stats_file = job_dir / "evaluation_stats.csv"
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if not row or row[0] in ("", "Total Accuracy"):
                        continue
                    problem_stats[row[0]] = {
                        "accuracy": f"{float(row[1]):.3f}",
                        "correct_count": int(row[2]),
                        "total_samples": int(row[3]),
                        "avg_output_length": float(row[4]) if len(row) >= 5 else 0.0,
                    }

        # details/ subdirs: samples.jsonl + trackers
        details_dir = job_dir / "details"
        if not details_dir.exists():
            continue
        for prob_dir in details_dir.iterdir():
            if not prob_dir.is_dir():
                continue
            # Append sample JSONs to aggregated samples.jsonl
            for sj in sorted(prob_dir.glob("sample_*.json")):
                with open(sj, "r", encoding="utf-8") as fj:
                    obj = json.load(fj)
                try:
                    sidx = int(sj.stem.split("_")[-1])
                except Exception:
                    sidx = -1
                obj["id"] = prob_dir.name
                obj["sample"] = sidx
                with open(samples_path, "a", encoding="utf-8") as fo:
                    fo.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # Tracker CSVs
            for tf in prob_dir.glob("*_tracker.csv"):
                df = pd.read_csv(tf)
                if "iter_depth" not in df.columns:
                    continue
                # Parse sample index from filename sample_{idx}_tracker.csv
                parts = tf.stem.split("_")
                sidx = int(parts[1]) if len(parts) >= 3 and parts[0] == "sample" else -1
                out_len = sample_output_len.get((prob_dir.name, sidx))
                if isinstance(out_len, int) and out_len >= 0:
                    df = df[(df["iter_depth"] == 0).cumsum() <= out_len]
                    df.to_csv(tf, index=False)
                depth_counts = df["iter_depth"].value_counts().to_dict()
                for ic in range(1, 6):
                    delta = depth_counts.get(ic - 1, 0) - depth_counts.get(ic, 0)
                    if delta > 0:
                        iter_dist[ic] += delta
                all_tracker_files.append(tf)

    # Fix avg_output_length per problem from actual results
    by_problem: Dict[str, list] = {}
    for r in all_results:
        by_problem.setdefault(r["problem_id"], []).append(r)
    for pid, rows in by_problem.items():
        if pid in problem_stats:
            problem_stats[pid]["avg_output_length"] = sum(r["output_tokens"] for r in rows) / len(rows)

    total = len(all_results)
    total_correct = sum(1 for r in all_results if r["is_correct"])
    avg_len = sum(r["output_tokens"] for r in all_results) / total if total else 0.0

    # Write combined evaluation_stats.csv
    with open(output_dir / "evaluation_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["problem_id", "accuracy", "correct_count", "total_samples", "avg_output_length"])
        for pid, s in sorted(problem_stats.items()):
            writer.writerow([pid, s["accuracy"], s["correct_count"], s["total_samples"], f"{s['avg_output_length']:.2f}"])
        writer.writerow([])
        writer.writerow(["Total Accuracy", f"{total_correct/total:.3f}" if total else "0.000",
                         total_correct, total, f"{avg_len:.2f}"])

    # Write combined detailed_results.csv
    fieldnames = ["problem_id", "sample_idx", "correct_answer", "predicted_answer",
                  "has_answer", "is_correct", "input_tokens", "output_tokens", "processing_time"]
    with open(output_dir / "detailed_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (x["problem_id"], x["sample_idx"])):
            writer.writerow(r)

    # iter_count_distribution.csv
    total_tokens = sum(iter_dist.values())
    if total_tokens > 0:
        with open(output_dir / "iter_count_distribution.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iter_count", "token_count", "percentage"])
            for ic in sorted(iter_dist):
                cnt = iter_dist[ic]
                if cnt > 0:
                    writer.writerow([ic, cnt, cnt / total_tokens * 100])
            writer.writerow([])
            writer.writerow(["Total Tokens", total_tokens, 100.0])

    # all_trackers.csv
    if all_tracker_files and save_all_trackers:
        with open(output_dir / "all_trackers.csv", "w", newline="", encoding="utf-8") as fo:
            header_written = False
            csv_writer = None
            for tf in all_tracker_files:
                with open(tf, "r", encoding="utf-8") as fi:
                    rows = list(csv.reader(fi))
                if not rows:
                    continue
                if not header_written:
                    csv_writer = csv.writer(fo)
                    csv_writer.writerow(rows[0] + ["data_id"])
                    header_written = True
                for row in rows[1:]:
                    csv_writer.writerow(row + [tf.parent.name])

    print(f"Combined {job_nums} jobs: accuracy={total_correct/total:.4f}" if total else f"Combined {job_nums} jobs: no results")

    # results.json
    results_file = output_dir / "detailed_results.csv"
    if results_file.exists():
        metrics = compute_metrics(str(results_file))
        if metrics:
            with open(output_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            n = metrics["repeat_size"]
            print(f"pass@{n}={metrics[f'pass@{n}']}, cons@{n}={metrics[f'cons@{n}']}, avg@{n}={metrics[f'avg@{n}']}")

    if del_job_dir:
        for job_id in range(job_nums):
            shutil.rmtree(output_dir / f"job_{job_id}", ignore_errors=True)
