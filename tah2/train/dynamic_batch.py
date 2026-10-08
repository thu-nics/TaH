"""Dynamic / length-grouped batch samplers for SFT.

The defining constraint here is a *per-microbatch padded token budget*: each
batch must satisfy ``len(batch) * max_len_in_batch <= token_budget``. With the
current sft_base recipe (max sample length around 8200, budget = 9000), this means

  - 8K samples -> mb=1
  - 4K samples -> mb=2
  - 2K samples -> mb=4
  - 1K samples -> mb=8

This is the standard "token-batching" shape optimization. Loss is still
token-normalized in the trainer, so the gradient direction is unchanged when
global-batch composition is held fixed.

For DDP / DeepSpeed every rank must yield the same number of microbatches per
optimizer step (otherwise NCCL deadlocks). We enforce this by:

  1. Building fixed-size global batches with ``global_batch_size`` samples.
  2. Packing each global batch by length under the padded token budget.
  3. Force-splitting microbatches until the count is a multiple of
     ``num_replicas``.

Accelerate then strides those microbatches across ranks, and the trainer uses
the sampler's per-rank microbatch counts as variable gradient accumulation.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


def _auto_cache_path(
    dataset,
    data_config: Optional[dict] = None,
    cache_root: Optional[str] = None,
) -> Optional[str]:
    """Derive a default cache path that encodes the parameters affecting lengths.

    Filename layout (under caller-provided cache root, ``$TAH_LENGTHS_CACHE_DIR``,
    or ``~/.cache/tah_lengths``):

      ``{data_basename}_ml{max_length}_{max_length_action}_n{N}_{fp8}.npy``

    Each visible token corresponds to a setting that, if changed, invalidates
    the cache. The trailing ``fp8`` is the first 8 chars of HF's
    ``dataset._fingerprint`` and acts as a safety net for any preprocessing
    change not captured by the explicit fields above (e.g. tokenizer change,
    new column transforms). Files for different runs sit side-by-side in the
    cache dir, so it's easy to ``ls`` and see what's been computed.
    """
    fp = getattr(dataset, "_fingerprint", None)
    if not fp:
        return None
    cache_root = (
        cache_root
        or os.environ.get("TAH_LENGTHS_CACHE_DIR")
        or os.path.expanduser("~/.cache/tah_lengths")
    )
    parts: List[str] = []
    if data_config:
        dpath = str(data_config.get("train_data_path") or "")
        if dpath:
            parts.append(os.path.basename(dpath.rstrip("/")) or "ds")
        ml = data_config.get("max_length")
        if ml is not None:
            parts.append(f"ml{int(ml)}")
        action = str(data_config.get("max_length_action") or "cutoff").lower()
        parts.append(action)
    parts.append(f"n{len(dataset)}")
    parts.append(str(fp)[:8])
    return os.path.join(cache_root, "_".join(parts) + ".npy")


def _materialize_lengths(
    dataset,
    cache_path: Optional[str] = None,
    data_config: Optional[dict] = None,
    cache_root: Optional[str] = None,
) -> np.ndarray:
    """Return a 1-D int64 array of input_ids length per sample.

    Fast path: datasets processed by ``tah2.utils.data_prepare`` carry a
    ``length`` column — read it directly. Otherwise walk ``input_ids`` (slow)
    with a .npy cache (explicit ``cache_path``, or auto-derived from the data
    params + HF fingerprint; validated against ``len(dataset)``).
    """
    try:
        if "length" in dataset.column_names:
            if getattr(dataset, "_indices", None) is None:
                # No row remapping: read the arrow column directly (~instant).
                arr = (
                    dataset.data.column("length")
                    .to_numpy(zero_copy_only=False)
                    .astype(np.int64)
                )
            else:
                arr = np.asarray(
                    dataset.with_format("numpy", columns=["length"])["length"],
                    dtype=np.int64,
                )
            if len(arr) == len(dataset):
                print(f"[lengths] using precomputed 'length' column (n={len(arr)})")
                return arr
    except Exception:
        pass

    auto = False
    if cache_path is None:
        cache_path = _auto_cache_path(
            dataset,
            data_config=data_config,
            cache_root=cache_root,
        )
        auto = True

    if cache_path and os.path.isfile(cache_path):
        try:
            arr = np.load(cache_path)
            if len(arr) == len(dataset):
                print(
                    f"[lengths] cache hit{' (auto)' if auto else ''}: {cache_path} "
                    f"(n={len(arr)})"
                )
                return arr.astype(np.int64, copy=False)
        except Exception:
            pass

    print(
        f"[lengths] cache miss; computing {len(dataset)} sample lengths "
        f"(will save to {cache_path})"
        if cache_path
        else f"[lengths] computing {len(dataset)} sample lengths (no cache path)"
    )
    try:
        col = dataset["input_ids"]
    except Exception:
        col = [s["input_ids"] for s in dataset]
    arr = np.fromiter((len(x) for x in col), dtype=np.int64, count=len(dataset))

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, arr)
        except Exception:
            pass
    return arr


class BalancedGlobalBatchSampler(Sampler[List[int]]):
    """Same-composition-as-baseline dynamic batcher.

    Each *global batch* contains exactly ``global_batch_size`` samples drawn
    from a deterministic shuffle. Within that 64-sample (or whatever)
    group, samples are sorted by length and greedy-packed into microbatches
    under a *padded* token budget (``len(batch) * max_len <= token_budget``),
    capped by ``max_batch_size``.

    To survive accelerate's stride-based ``BatchSamplerShard`` and keep
    every rank's mb-count per global-batch identical, the natural mb count
    ``M`` is rounded up to the next multiple of ``num_replicas`` by
    force-splitting the largest mb. Per-rank mbs per global batch is then
    ``target_M // num_replicas``; the trainer reads the sequence via
    :meth:`per_rank_mb_counts` and uses it as variable gradient-accumulation.

    Loss equivalence to the fixed-batch baseline at the same shuffle seed
    is exact: the same 64 samples are summed per opt-step, ``num_items_in_batch``
    is the same total token count, and the per-token CE summed across the
    microbatches is identical — only the microbatch grouping changes.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        global_batch_size: int,
        token_budget: int,
        max_batch_size: Optional[int] = None,
        num_replicas: int = 1,
        seed: int = 0,
        drop_last: bool = False,
    ):
        # dp alignment is done HERE, after the per-epoch shuffle (see _build):
        # ``drop_last=True`` drops the partial last global batch, otherwise only
        # ``(n % gbs) % num_replicas`` samples are dropped so the partial batch
        # stays rank-aligned for force_split. Either way the full global batches
        # are identical for any ``num_replicas`` under the same seed; only the
        # tail differs. Do not truncate ``lengths`` upstream: a different
        # ``len(lengths)`` changes the whole permutation.
        if global_batch_size <= 0:
            raise ValueError(f"global_batch_size must be > 0, got {global_batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if global_batch_size % num_replicas != 0:
            raise ValueError(
                f"global_batch_size ({global_batch_size}) must be a multiple of "
                f"num_replicas ({num_replicas}) so force-split can always reach a "
                f"clean rank-aligned mb count."
            )
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.gbs = int(global_batch_size)
        self.budget = int(token_budget)
        self.cap = int(max_batch_size) if max_batch_size else None
        self.dp = int(num_replicas)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self._all_mbs: List[List[int]] = []
        self._per_rank_counts: List[int] = []
        self._build()

    # ------------------------------------------------------------------ #

    def set_epoch(self, epoch: int) -> None:
        new_epoch = int(epoch)
        if new_epoch == self.epoch and self._all_mbs:
            return
        self.epoch = new_epoch
        self._build()

    def per_rank_mb_counts(self) -> List[int]:
        return list(self._per_rank_counts)

    def __iter__(self) -> Iterator[List[int]]:
        for mb in self._all_mbs:
            yield mb

    def __len__(self) -> int:
        return len(self._all_mbs)

    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed + self.epoch)
        all_indices = rng.permutation(len(self.lengths))

        all_mbs: List[List[int]] = []
        per_rank_counts: List[int] = []

        # Shuffle-then-drop: truncate AFTER the permutation so every full
        # global batch is the same for any num_replicas / drop_last; only the
        # tail is affected.
        n = len(all_indices)
        if self.drop_last:
            n_keep = (n // self.gbs) * self.gbs
        else:
            n_keep = n - (n % self.gbs) % self.dp
        view = all_indices[:n_keep]

        for start in range(0, len(view), self.gbs):
            chunk = view[start : start + self.gbs]
            if self.drop_last and len(chunk) < self.gbs:
                break

            # Partial last global batch (drop_last=False) is rank-aligned by the
            # post-shuffle truncation above, so force_split can always reach a
            # clean target_M.
            if len(chunk) % self.dp != 0:
                raise RuntimeError(
                    f"Partial global batch of size {len(chunk)} is not divisible by "
                    f"num_replicas={self.dp} (post-shuffle truncation bug)."
                )

            # Sort by length descending within the global batch.
            order = np.argsort(-self.lengths[chunk], kind="stable")
            sorted_chunk = chunk[order]

            mbs = self._pack(sorted_chunk)
            target_M = ((len(mbs) + self.dp - 1) // self.dp) * self.dp
            if len(mbs) < target_M:
                mbs = self._force_split(mbs, target_M)
            if len(mbs) != target_M:
                # When ``len(chunk) % dp == 0`` (full or aligned-partial gb) the
                # invariant "at least one mb has >=2 samples whenever natural
                # M < len(chunk)" guarantees we can always reach target_M.
                raise RuntimeError(
                    f"force_split could not reach target_M={target_M} (got {len(mbs)}); "
                    f"chunk_start={start} chunk_size={len(chunk)} "
                    f"lengths={self.lengths[chunk].tolist()}"
                )

            all_mbs.extend(mbs)
            per_rank_counts.append(target_M // self.dp)

        self._all_mbs = all_mbs
        self._per_rank_counts = per_rank_counts

    def _pack(self, indices: np.ndarray) -> List[List[int]]:
        batches: List[List[int]] = []
        cur: List[int] = []
        cur_max = 0
        budget = self.budget
        cap = self.cap
        L = self.lengths
        for idx in indices:
            i = int(idx)
            ell = int(L[i])
            new_max = ell if ell > cur_max else cur_max
            new_bs = len(cur) + 1
            new_padded = new_bs * new_max
            would_overflow_pad = new_padded > budget and len(cur) > 0
            would_overflow_cap = (cap is not None) and new_bs > cap
            if would_overflow_pad or would_overflow_cap:
                batches.append(cur)
                cur = [i]
                cur_max = ell
                continue
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    @staticmethod
    def _force_split(mbs: List[List[int]], target: int) -> List[List[int]]:
        """Repeatedly halve the mb with the most samples until count == target.

        Pre: at least one mb has >=2 samples whenever len(mbs) < total_samples.
        Post: len(out) == target (raises if a singleton-only state is reached
        with len(out) < target — caller validates).
        """
        out = list(mbs)
        while len(out) < target:
            best_idx = 0
            best_size = len(out[0])
            for i in range(1, len(out)):
                if len(out[i]) > best_size:
                    best_size = len(out[i])
                    best_idx = i
            if best_size < 2:
                break
            mb = out[best_idx]
            h = len(mb) // 2
            out = out[:best_idx] + [mb[:h], mb[h:]] + out[best_idx + 1 :]
        return out


class LengthGroupedDistributedSampler(Sampler[int]):
    """Drop-in replacement for DistributedSampler that returns indices in
    length-sorted megabatch order. Use when you want fixed batch_size + length
    grouping (no token budget)."""

    def __init__(
        self,
        lengths: Sequence[int],
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        megabatch_factor: int = 50,
        batch_size: int = 1,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.megabatch_factor = int(megabatch_factor)
        self.batch_size = int(batch_size)
        self.epoch = 0
        self.num_samples = len(self.lengths) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        n = len(self.lengths)
        perm = rng.permutation(n)
        mega = max(self.batch_size * self.megabatch_factor, 100)
        for start in range(0, n, mega):
            chunk = perm[start : start + mega]
            order = np.argsort(self.lengths[chunk], kind="stable")
            perm[start : start + len(order)] = chunk[order]
        # Shard
        shard = perm[self.rank :: self.num_replicas][: self.num_samples]
        for i in shard:
            yield int(i)

    def __len__(self) -> int:
        return self.num_samples
