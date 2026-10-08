"""On-the-fly SFT data preparation.

Two input formats are supported, detected from the dataset columns:

1. Raw conversations (e.g. open-thoughts/OpenThoughts3-1.2M with its
   ``conversations`` column), tokenized here with the model's chat template.
2. Pre-tokenized data produced by ``script2/data/regenerate.py build``
   (``real_token`` + ``mask`` columns).

Labels are generated online: only assistant/response tokens are supervised;
there is no offline labeling step (no iter_count_strategy, no teacher top-k
sidecar).

Caching: the processed (tokenized) dataset is saved to disk once and
memory-mapped on every later run — the cache key is built EXPLICITLY from the
raw data files (relative name/size), tokenizer vocab size + chat template, and
all preprocessing params. Absolute paths are deliberately excluded so the
same shared dataset can reuse one cache through different mount points. This
does not depend on HF datasets' function fingerprinting (which hashes the
tokenizer object and breaks across transformers versions). A ``length`` column
is materialized at the same time so the dynamic batcher never has to walk
``input_ids`` again.
"""

import hashlib
import json
import os
import re
import shutil
from functools import partial
from typing import Dict, List

from accelerate import Accelerator
from datasets import load_dataset, load_from_disk

_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}

# Bump to invalidate all processed-dataset caches when the preprocessing
# logic below changes in a way that affects outputs.
# v2: strip the template-injected redundant empty think block (see
# _strip_redundant_think) — bump invalidates older tokenization caches.
_PREPROC_VERSION = 2


def _infer_num_proc(dataset_size: int, upper_bound: int = 32) -> int | None:
    if dataset_size < 500:
        return None
    cpu_count = os.cpu_count() or 1
    return min(cpu_count, upper_bound, max(2, dataset_size // 250))


def _load_raw_dataset(path: str):
    """Load a raw dataset from a parquet file/dir or a saved-to-disk dataset."""
    if path.endswith(".parquet"):
        return load_dataset("parquet", data_files=path, split="train")
    if os.path.isdir(path):
        parquet_files = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".parquet")
        )
        if parquet_files:
            return load_dataset("parquet", data_files=parquet_files, split="train")
        return load_from_disk(path)
    raise ValueError(
        f"Unsupported dataset path: {path} (expect .parquet file/dir or saved dataset dir)"
    )


def _list_data_files(path: str) -> List[str]:
    """The concrete files behind a dataset path (for cache-key stat'ing)."""
    if path.endswith(".parquet"):
        return [path]
    if os.path.isdir(path):
        parquet_files = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".parquet")
        )
        if parquet_files:
            return parquet_files
        # saved-to-disk dataset: stat ONLY canonical files — HF map drops
        # cache-*.arrow litter here, which would change the key every run.
        return sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if (
                (name.startswith("data-") and name.endswith(".arrow"))
                or name in ("dataset_info.json", "state.json")
            )
        )
    return [path]


def _tokenizer_signature(tokenizer) -> Dict:
    # Deliberately NOT keyed on name_or_path: same-family models (e.g. Qwen3
    # 1.7B/4B/8B, local dir vs HF repo id) share one tokenizer, so vocab size
    # + chat template are enough and let them reuse one cache.
    template = getattr(tokenizer, "chat_template", None) or ""
    return {
        "vocab_size": int(len(tokenizer)),
        "chat_template_sha1": hashlib.sha1(template.encode("utf-8")).hexdigest(),
    }


def _cache_key_payload(meta: Dict) -> Dict:
    """Return the path-independent part of cache metadata used for identity.

    ``data_path`` is diagnostic metadata only. File paths are reduced to their
    names because ``_list_data_files`` only returns files directly inside the
    dataset directory. This also normalizes legacy metadata, which stored
    absolute file paths, so old compatible caches remain reusable.
    """
    payload = {
        key: value
        for key, value in meta.items()
        if key not in {"data_path", "num_samples"}
    }
    payload["data_files"] = [
        {
            "path": os.path.basename(os.path.normpath(file_meta["path"])),
            "size": file_meta["size"],
        }
        for file_meta in meta.get("data_files", [])
    ]
    return payload


def _find_compatible_cache(cache_root: str, stem: str, meta: Dict) -> str | None:
    """Find a complete cache written with an older, path-sensitive key."""
    if not os.path.isdir(cache_root):
        return None
    expected = _cache_key_payload(meta)
    prefix = f"{stem}_"
    for name in sorted(os.listdir(cache_root)):
        if not name.startswith(prefix):
            continue
        cache_dir = os.path.join(cache_root, name)
        meta_path = os.path.join(cache_dir, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as f:
                cached_meta = json.load(f)
        except (OSError, ValueError, TypeError):
            continue
        if _cache_key_payload(cached_meta) == expected:
            return cache_dir
    return None


def _processed_cache_dir(
    data_path: str,
    data_config: Dict,
    tokenizer,
    select_num: int | None,
    select_ratio: float,
) -> tuple[str | None, Dict]:
    """Deterministic cache dir for one processed split + the meta describing it.

    Returns (cache_dir, meta). cache_dir is None when caching is disabled
    (``data_config["cache_dir"] = false``) or the data path cannot be stat'ed.
    """
    cache_root = data_config.get("cache_dir", None)
    if cache_root is False:
        return None, {}
    try:
        # Identity = relative name + size ONLY. Absolute paths are deliberately
        # excluded because shared files may have different mount prefixes.
        # mtime is also excluded: NFS client attr caching can make it jitter
        # by ±1s between runs.
        files = [
            {"path": os.path.basename(f), "size": os.path.getsize(f)}
            for f in _list_data_files(data_path)
        ]
    except OSError:
        return None, {}

    meta = {
        "preproc_version": _PREPROC_VERSION,
        "data_path": os.path.abspath(data_path),
        "data_files": files,
        "tokenizer": _tokenizer_signature(tokenizer),
        "max_length": data_config.get("max_length", None),
        "max_length_action": (
            data_config.get("max_length_action", "cutoff") or "cutoff"
        ).lower(),
        "conversations_key": data_config.get("conversations_key", "conversations"),
        "fixed_iter_count": data_config.get("fixed_iter_count", None),
        "select_num": select_num,
        "select_ratio": select_ratio,
    }
    if data_config.get("fixed_iter_count") is not None:
        meta["fixed_iter_count_format_version"] = 2
    if cache_root is None:
        base = data_path if os.path.isdir(data_path) else os.path.dirname(data_path)
        cache_root = os.path.join(base, ".tah_cache")
    stem = os.path.splitext(os.path.basename(os.path.normpath(data_path)))[0]
    key = hashlib.sha1(
        json.dumps(_cache_key_payload(meta), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = os.path.join(cache_root, f"{stem}_{key}")
    if os.path.isfile(os.path.join(cache_dir, "meta.json")):
        return cache_dir, meta

    # Reuse caches produced before absolute paths were removed from the key.
    # Besides avoiding needless work, this prevents the migration itself from
    # creating a third copy next to two mount-alias duplicates.
    compatible_cache = _find_compatible_cache(cache_root, stem, meta)
    return compatible_cache or cache_dir, meta


def _cache_complete(cache_dir: str) -> bool:
    # meta.json is written LAST by the producer, so its presence implies a
    # complete save_to_disk.
    return os.path.isfile(os.path.join(cache_dir, "meta.json"))


def _conversation_to_messages(conversation) -> List[Dict[str, str]]:
    """Normalize ShareGPT-style ({from, value}) or OpenAI-style ({role, content}) turns."""
    messages = []
    for turn in conversation:
        role = turn.get("role", turn.get("from"))
        content = turn.get("content", turn.get("value"))
        if role is None or content is None:
            continue
        role = _ROLE_MAP.get(str(role).lower())
        if role is None:
            continue
        messages.append({"role": role, "content": content})
    return messages


# Qwen3's template injects an empty think block before assistant content that
# has no *closed* </think>. OpenThoughts3 carries many assistant turns whose
# own <think> never closes, so the render becomes
# "<think>\n\n</think>\n\n<think>..." — drop the injected empty pair and keep
# the content's original tags verbatim.
_REDUNDANT_THINK_RE = re.compile(r"<think>\n\n</think>\n\n(?=\s*<think>)")


def _strip_redundant_think(text: str) -> str:
    return _REDUNDANT_THINK_RE.sub("", text)


def _tokenize_messages(
    messages: List[Dict[str, str]], tokenizer
) -> tuple[List[int], List[int]]:
    """Tokenize a conversation with the chat template, supervising assistant turns only."""

    def _render_ids(msgs, add_generation_prompt):
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return tokenizer(_strip_redundant_think(text), add_special_tokens=False)[
            "input_ids"
        ]

    input_ids: List[int] = []
    labels: List[int] = []
    rendered_len = 0
    for idx, message in enumerate(messages):
        ids = _render_ids(messages[: idx + 1], add_generation_prompt=False)
        if message["role"] == "assistant" and idx > 0:
            prompt_ids = _render_ids(messages[:idx], add_generation_prompt=True)
            prompt_len = max(len(prompt_ids), rendered_len)
            seg_labels = [-100] * (prompt_len - rendered_len) + ids[prompt_len:]
        else:
            seg_labels = [-100] * (len(ids) - rendered_len)
        input_ids.extend(ids[rendered_len:])
        labels.extend(seg_labels)
        rendered_len = len(ids)
    return input_ids, labels


def _survival_profile(cfg: Dict, max_iter: int) -> list | None:
    """Target routing profile [P(count>=2), ..., P(count>=max_iter)], or None.

    Three ways to specify it (avg iter count = 1 + sum of the profile):
      ``survival: [0.5, 0.45, ...]``  explicit, non-increasing, len max_iter-1
      ``continue_prob: 0.8``          geometric, P(count>=k) = p^(k-1)
      ``deep_frac: 0.6``              all-or-nothing (the profile real training
                                      collapses to: a token either stops at 1 or
                                      runs to max_iter), avg = 1 + f*(max_iter-1)
    """
    explicit = cfg.get("survival", None)
    if explicit is not None:
        profile = [float(x) for x in explicit]
        if len(profile) != max_iter - 1:
            raise ValueError(
                f"fixed_iter_count.survival needs {max_iter - 1} entries "
                f"(P(count>=2..{max_iter})), got {len(profile)}"
            )
        return profile
    if cfg.get("continue_prob", None) is not None:
        p = float(cfg["continue_prob"])
        return [p**k for k in range(1, max_iter)]
    if cfg.get("deep_frac", None) is not None:
        return [float(cfg["deep_frac"])] * (max_iter - 1)
    return None


def _fixed_random_iter_counts(labels: list, sample_index: int, cfg: Dict) -> list:
    """Deterministic random per-token iter counts for routing-controlled tests.

    Seeded by (cfg.seed, sample_index) so every run / rank / tp size generates
    the identical sequence. Default: supervised tokens get uniform [1, max_iter],
    prompt tokens get 1 (route once). With a survival profile
    (:func:`_survival_profile`) counts are inverse-CDF sampled from it, so the
    batch hits the requested per-depth rates exactly in expectation; the profile
    applies to *every* token (prompt included) so avg_iter_count — the quantity
    that drives KV/activation memory — is the knob.
    """
    import numpy as np

    max_iter = int(cfg.get("max_iter", 2))
    seed = int(cfg.get("seed", 1234))
    rng = np.random.default_rng([seed, int(sample_index)])
    profile = _survival_profile(cfg, max_iter)
    if profile is not None:
        u = rng.random(len(labels))
        counts = 1 + (u[:, None] < np.asarray(profile)[None, :]).sum(axis=1)
        return [int(c) for c in counts]
    counts = rng.integers(1, max_iter + 1, size=len(labels))
    return [int(c) if l != -100 else 1 for c, l in zip(counts, labels)]


def preprocess_for_sft_batch(
    examples: Dict,
    indices,
    tokenizer,
    max_length: int | None,
    max_length_action: str = "cutoff",
    conversations_key: str = "conversations",
    fixed_iter_count: Dict | None = None,
) -> Dict:
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_iter_counts = [] if fixed_iter_count else None
    action = (max_length_action or "cutoff").lower()

    for sample_index, conversation in zip(indices, examples[conversations_key]):
        messages = _conversation_to_messages(conversation)
        if not any(m["role"] == "assistant" for m in messages):
            continue

        input_ids, labels = _tokenize_messages(messages, tokenizer)

        if max_length is not None and len(input_ids) > max_length:
            if action == "filter":
                continue
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        if all(label == -100 for label in labels):
            continue

        batch_input_ids.append(input_ids)
        batch_attention_mask.append([1] * len(input_ids))
        batch_labels.append(labels)
        if fixed_iter_count:
            batch_iter_counts.append(
                _fixed_random_iter_counts(labels, sample_index, fixed_iter_count)
            )

    result = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
        # Materialized here so the dynamic batcher reads one int column
        # instead of walking the full input_ids column (minutes at 1M+ rows).
        "length": [len(ids) for ids in batch_input_ids],
    }
    if fixed_iter_count:
        result["iter_count"] = batch_iter_counts
    return result


def preprocess_pretokenized_batch(
    examples: Dict,
    indices,
    max_length: int | None,
    max_length_action: str = "cutoff",
    fixed_iter_count: Dict | None = None,
) -> Dict:
    """Preprocess data tokenized by ``script2/data/regenerate.py build``.

    Expects ``real_token`` (token ids) and ``mask`` (1 = supervised response
    token, 0 = prompt token) columns.
    """
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_iter_counts = [] if fixed_iter_count else None
    action = (max_length_action or "cutoff").lower()

    for sample_index, input_ids, mask in zip(
        indices, examples["real_token"], examples["mask"]
    ):
        if max_length is not None and len(input_ids) > max_length:
            if action == "filter":
                continue
            input_ids = input_ids[:max_length]
            mask = mask[:max_length]

        if not any(mask):
            continue

        labels = [token if m else -100 for token, m in zip(input_ids, mask)]
        batch_input_ids.append(list(input_ids))
        batch_attention_mask.append([1] * len(input_ids))
        batch_labels.append(labels)
        if fixed_iter_count:
            batch_iter_counts.append(
                _fixed_random_iter_counts(labels, sample_index, fixed_iter_count)
            )

    result = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
        "length": [len(ids) for ids in batch_input_ids],
    }
    if fixed_iter_count:
        result["iter_count"] = batch_iter_counts
    return result


def _build_preprocess_fn(
    dataset,
    tokenizer,
    max_length: int | None,
    max_length_action: str,
    conversations_key: str,
    fixed_iter_count: Dict | None = None,
):
    if conversations_key in dataset.column_names:
        return partial(
            preprocess_for_sft_batch,
            tokenizer=tokenizer,
            max_length=max_length,
            max_length_action=max_length_action,
            conversations_key=conversations_key,
            fixed_iter_count=fixed_iter_count,
        )
    if "real_token" in dataset.column_names:
        return partial(
            preprocess_pretokenized_batch,
            max_length=max_length,
            max_length_action=max_length_action,
            fixed_iter_count=fixed_iter_count,
        )
    raise ValueError(
        f"Dataset columns {dataset.column_names} contain neither "
        f"'{conversations_key}' (raw conversations) nor 'real_token' (pre-tokenized)."
    )


def _apply_even_strategy(
    dataset,
    even_strategy: str,
    dp: int,
    accelerator: Accelerator,
    dataset_name: str = "dataset",
):
    """Make dataset size divisible by dp via "drop" (truncate) or "pad" (duplicate)."""
    if even_strategy == "none" or even_strategy is None or dp <= 1:
        return dataset

    original_size = len(dataset)
    remainder = original_size % dp

    if remainder == 0:
        accelerator.print(
            f"{dataset_name}: size {original_size} is already divisible by dp={dp}"
        )
        return dataset

    if even_strategy == "drop":
        new_size = original_size - remainder
        dataset = dataset.select(range(new_size))
        accelerator.print(
            f"{dataset_name}: dropped {remainder} samples ({original_size} -> {new_size}) to be divisible by dp={dp}"
        )
    elif even_strategy == "pad":
        pad_count = dp - remainder
        pad_indices = list(range(original_size)) + list(range(pad_count))
        dataset = dataset.select(pad_indices)
        accelerator.print(
            f"{dataset_name}: padded {pad_count} samples ({original_size} -> {len(dataset)}) to be divisible by dp={dp}"
        )
    else:
        accelerator.print(
            f"Warning: Unknown even_strategy '{even_strategy}', no change applied"
        )

    return dataset


def _load_or_process_split(
    data_path: str,
    data_config: Dict,
    tokenizer,
    accelerator: Accelerator,
    select_num: int | None,
    select_ratio: float,
    dataset_name: str,
):
    """Return the processed (tokenized) dataset for one split, cached on disk.

    Cache hit: memory-map the saved dataset directly (no raw load, no map).
    Cache miss: rank 0 loads + tokenizes + ``save_to_disk``; all other ranks
    wait at the ``main_process_first`` barrier and then load the cache, so the
    expensive map runs exactly once per shared filesystem.
    """
    cache_dir, meta = _processed_cache_dir(
        data_path, data_config, tokenizer, select_num, select_ratio
    )

    max_length = data_config.get("max_length", None)
    max_length_action = (
        data_config.get("max_length_action", "cutoff") or "cutoff"
    ).lower()
    if max_length_action not in {"cutoff", "filter"}:
        max_length_action = "cutoff"

    with accelerator.main_process_first():
        if cache_dir and _cache_complete(cache_dir):
            accelerator.print(f"[data] {dataset_name}: cache hit -> {cache_dir}")
            return load_from_disk(cache_dir)

        dataset = _load_raw_dataset(data_path)
        if select_num is not None:
            accelerator.print(f"Using {select_num} of {dataset_name}")
            dataset = dataset.select(range(select_num))
        elif select_ratio != 1.0:
            accelerator.print(f"Using {select_ratio} of {dataset_name}")
            dataset = dataset.select(range(int(len(dataset) * select_ratio)))

        preprocess_fn = _build_preprocess_fn(
            dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            max_length_action=max_length_action,
            conversations_key=data_config.get("conversations_key", "conversations"),
            fixed_iter_count=data_config.get("fixed_iter_count", None),
        )
        num_proc = data_config.get("num_proc", None) or _infer_num_proc(len(dataset))
        batch_size = 2000 if len(dataset) >= 2000 else max(128, len(dataset))
        accelerator.print(
            f"[data] {dataset_name}: processing {len(dataset)} samples "
            f"(num_proc={num_proc})..."
        )
        map_kwargs = {}
        map_tmp_dir = None
        if cache_dir:
            # Route HF map's temp arrows OUT of the source dataset dir (they
            # would feed back into the cache key). Sibling of cache_dir, not
            # inside it: save_to_disk refuses to overwrite its own backing files.
            map_tmp_dir = cache_dir + ".maptmp"
            shutil.rmtree(map_tmp_dir, ignore_errors=True)  # killed-run leftovers
            os.makedirs(map_tmp_dir, exist_ok=True)
            map_kwargs["cache_file_name"] = os.path.join(map_tmp_dir, "map_tmp.arrow")
        dataset = dataset.map(
            preprocess_fn,
            batched=True,
            with_indices=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc=f"Processing {dataset_name}",
            # HF's fingerprint cache can serve stale results across code edits;
            # our save_to_disk cache is the only cache layer.
            load_from_cache_file=False,
            **map_kwargs,
        )
        if "length" not in dataset.column_names:
            raise RuntimeError(
                f"preprocessing produced no 'length' column ({dataset.column_names})"
            )
        if cache_dir and accelerator.is_main_process:
            n_samples = len(dataset)
            dataset.save_to_disk(cache_dir)
            if map_tmp_dir:
                shutil.rmtree(map_tmp_dir, ignore_errors=True)
            # meta.json written LAST marks the cache complete.
            with open(os.path.join(cache_dir, "meta.json"), "w") as f:
                json.dump({**meta, "num_samples": n_samples}, f, indent=2)
            accelerator.print(f"[data] {dataset_name}: cache saved -> {cache_dir}")

    if cache_dir and _cache_complete(cache_dir):
        return load_from_disk(cache_dir)
    return dataset


def preprocess_dataset(
    data_config: Dict, tokenizer, accelerator: Accelerator, eval_only: bool = False
):
    """Load raw conversation data and tokenize it for SFT (disk-cached).

    Args:
        eval_only: If True, only load and process eval dataset, skip train dataset entirely.
    """
    eval_data_path = data_config.get("eval_data_path", None)
    eval_data_ratio = data_config.get("eval_data_ratio", 0.05)
    even_strategy = (data_config.get("even_strategy", "none") or "none").lower()
    dp = data_config.get("dp", 8)

    processed_eval_dataset = None
    use_separate_eval = False
    if eval_data_path and eval_data_path.strip():
        try:
            processed_eval_dataset = _load_or_process_split(
                eval_data_path,
                data_config,
                tokenizer,
                accelerator,
                select_num=None,
                select_ratio=1.0,
                dataset_name="eval dataset",
            )
            use_separate_eval = True
            accelerator.print(
                f"Using separate evaluation dataset from: {eval_data_path}"
            )
        except Exception as e:
            if eval_only:
                raise ValueError(
                    f"eval_only=True but could not load eval dataset from {eval_data_path}: {e}"
                )
            accelerator.print(
                f"Warning: Could not load eval dataset from {eval_data_path}: {e}"
            )
            accelerator.print("Will split train dataset instead")
    if eval_only:
        if not use_separate_eval:
            raise ValueError("eval_only=True but no separate eval_data_path provided")
        accelerator.print(f"Eval dataset size: {len(processed_eval_dataset)}")
        return None, processed_eval_dataset

    processed_train_dataset = _load_or_process_split(
        data_config["train_data_path"],
        data_config,
        tokenizer,
        accelerator,
        select_num=data_config.get("train_data_num", None),
        select_ratio=data_config.get("train_data_ratio", 1.0),
        dataset_name="train dataset",
    )

    if not use_separate_eval:
        accelerator.print(
            "No separate eval dataset provided, will split train dataset using ratio"
        )
        if eval_data_ratio > 0:
            # Keep split index arrays in memory: per-rank indices cache files
            # + file locks deadlock multi-rank startup on NFS.
            import datasets as hf_datasets

            caching_was_enabled = hf_datasets.is_caching_enabled()
            hf_datasets.disable_caching()
            try:
                split_dataset = processed_train_dataset.train_test_split(
                    test_size=eval_data_ratio, seed=42
                )
            finally:
                if caching_was_enabled:
                    hf_datasets.enable_caching()
            processed_train_dataset = split_dataset["train"]
            processed_eval_dataset = split_dataset["test"]

    if even_strategy == "drop":
        # Train: never truncate before the shuffle — a different dataset length
        # changes the whole permutation, so per-step data would depend on dp.
        # BalancedGlobalBatchSampler drops the tail after the per-epoch shuffle.
        accelerator.print(
            "Train dataset: even_strategy=drop deferred to the batch sampler "
            "(post-shuffle tail drop; per-step data is dp-independent)"
        )
    else:
        processed_train_dataset = _apply_even_strategy(
            processed_train_dataset, even_strategy, dp, accelerator, "Train dataset"
        )
    accelerator.print(f"Train dataset size: {len(processed_train_dataset)}")
    if processed_eval_dataset is not None:
        processed_eval_dataset = _apply_even_strategy(
            processed_eval_dataset, even_strategy, dp, accelerator, "Eval dataset"
        )
        accelerator.print(f"Eval dataset size: {len(processed_eval_dataset)}")

    return processed_train_dataset, processed_eval_dataset
