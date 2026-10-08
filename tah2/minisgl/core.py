from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from tah2.minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from tah2.minisgl.kvcache import BaseCacheHandle, BaseKVCachePool


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        initial_len = len(self.input_ids)
        self.device_len = initial_len
        self.max_device_len = initial_len + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        # Preallocate input_ids to max_device_len so append_host is O(1)
        # (slot slice) instead of O(N) per step (torch.cat).
        if initial_len < self.max_device_len:
            padded = torch.empty(self.max_device_len, dtype=self.input_ids.dtype)
            padded[:initial_len] = self.input_ids
            self.input_ids = padded
        # TaH iteration state.  Tensor state (hidden, accum, rem) lives in
        # engine-owned pools keyed by table_idx; only scalars stay on Req.
        self.tah_iter_depth: int = 0
        self.tah_iter0_position: int = -1
        # TaH duo-attention state. ``tah_iter_counts[d-1]`` = number of
        # persisted iter-``d`` KV slots for this request (DUO mode, d>=1).
        # Slot indices live in engine's
        # ``_tah_iter_page_tables[d-1][req.table_idx, :tah_iter_counts[d-1]]``.
        # The list has up to ``max_iter-1`` entries and grows lazily as a
        # request reaches deeper iterations; counts persist across sampled
        # tokens (one slot per token that reached iter d) and are released only
        # when the request frees. In causal mode the counts reset after each sample.
        self.tah_iter_counts: List[int] = []

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    @property
    def is_iterating(self) -> bool:
        return self.tah_iter_depth > 0

    @property
    def tah_total_iter_slots(self) -> int:
        """Total persisted extra-iteration KV slots across all streams (DUO)."""
        return sum(self.tah_iter_counts)

    def tah_visible_iter_slots(self, depth: int) -> int:
        """Number of extra KV slots visible to a query at ``cur_iter==depth``.

        Duo predicate: a depth-``d`` query attends streams 1..d, so it sees
        ``sum(tah_iter_counts[:d])`` extra slots (streams d+1.. belong to
        deeper iterations and are excluded).
        """
        return sum(self.tah_iter_counts[:depth])

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        # complete_one() has already bumped device_len; new token lands at
        # device_len - 1 in the preallocated buffer.
        self.input_ids[self.device_len - 1 : self.device_len] = next_token

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def begin_iteration(self) -> None:
        """Route the token to recurrent decode without advancing its sequence."""
        self.tah_iter0_position = self.device_len - 1
        self.tah_iter_depth = 1

    def finish_iteration(self) -> None:
        """Clear recurrent state before committing the sampled token."""
        self.tah_iter_depth = 0
        self.tah_iter0_position = -1

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode", "mixed_decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        # Mixed decode should follow decode kernels/metadata paths.
        return self.phase in ("decode", "mixed_decode")

    @property
    def is_mixed_decode(self) -> bool:
        return self.phase == "mixed_decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)
    # TaH DUO: one page table per extra iteration (max_iter-1 entries).
    # ``tah_iter_page_tables[d-1][req.table_idx, :]`` holds iter-``d`` KV slot
    # indices; ``None`` unless ``iter_attention_mode == "duo"``.
    # See ``Req.tah_iter_counts`` for the active length per stream per row.
    tah_iter_page_tables: List[torch.Tensor] | None = field(default=None, init=False)
    # "causal" | "duo" — mirrored from the TaH model for attention backends.
    tah_iter_attention_mode: str = field(default="causal", init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
