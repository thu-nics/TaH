"""KV cache for ``TaHForCausalLM``.

The cache stores key/value tensors plus per-row position and valid metadata
keyed by ``(layer_idx, iter_depth)``. The TaH attention mask uses two views
of the cache:

* ``up_to(iter_depth)`` — concatenated K/V from iterations ``0…iter_depth``
  (the visible KV slots for query iter ``iter_depth``).
* ``iter_index_up_to(iter_depth)`` — a per-slot iteration index used to
  enforce the per-iter visibility rule in the additive attention mask.

The wrapper writes one slot per iteration; HF causal-LM layers call
:meth:`update` to append the current iteration's K/V into the cache, then
read the up-to view from inside SDPA.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache


# A slot is keyed by (layer_idx, iter_depth). Storing one flat dict per
# tensor type (key/value/pos/valid) is more uniform than nested dicts and
# makes ``to()`` / ``__len__`` / device introspection trivial loops.
_Slot = Tuple[int, int]


class TaHCache(DynamicCache):
    """Per-(layer, iteration) KV cache with up-to-N-iter views.

    Public contract used by ``TaHForCausalLM``:
      * Set ``current_iter_depth`` / ``position_ids_to_cache`` /
        ``valid_mask_to_cache`` before triggering any per-layer
        :meth:`update`.
      * Read ``get_cache_upto_iter(layer_idx, iter_depth)`` for KV,
        ``get_position_id_upto_iter`` and ``get_valid_mask_upto_iter`` for
        attention metadata, ``get_cache_iter_index_upto_iter`` for the
        per-slot iter index that the duo-mode mask compares against.
    """

    def __init__(self):
        super().__init__()
        self._k: Dict[_Slot, torch.Tensor] = {}
        self._v: Dict[_Slot, torch.Tensor] = {}
        self._pos: Dict[_Slot, torch.Tensor] = {}
        self._valid: Dict[_Slot, torch.Tensor] = {}

        # Set by the wrapper before each per-layer .update() pass.
        self.current_iter_depth: int = 0
        self.position_ids_to_cache: Optional[torch.Tensor] = None
        self.valid_mask_to_cache: Optional[torch.Tensor] = None

        self.batch_size: Optional[int] = None

    def has_layer(self, layer_idx: int = 0) -> bool:
        """``True`` iff at least one iter slot has been written for ``layer_idx``."""
        return any(l == layer_idx for (l, _) in self._k)

    # ── write path ────────────────────────────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append (key, value) for this layer at the current iter depth.

        Returns the concatenated K/V from iterations ``0…current_iter_depth``
        so the calling SDPA attention sees the full visible window.
        """
        del cache_kwargs  # protocol arg, not used here
        if self.batch_size is None:
            self.batch_size = key_states.shape[0]
        else:
            assert key_states.shape[0] == self.batch_size, (
                f"batch size mismatch: cached {self.batch_size}, got {key_states.shape[0]}"
            )

        slot: _Slot = (layer_idx, self.current_iter_depth)
        if slot in self._k:
            # Same (layer, iter) called more than once (e.g. autoregressive
            # decode): concatenate along the sequence dimension.
            self._k[slot] = torch.cat([self._k[slot], key_states], dim=-2)
            self._v[slot] = torch.cat([self._v[slot], value_states], dim=-2)
            self._pos[slot] = torch.cat([self._pos[slot], self.position_ids_to_cache], dim=-1)
            self._valid[slot] = torch.cat([self._valid[slot], self.valid_mask_to_cache], dim=-1)
        else:
            self._k[slot] = key_states
            self._v[slot] = value_states
            self._pos[slot] = self.position_ids_to_cache
            self._valid[slot] = self.valid_mask_to_cache

        return self.get_cache_upto_iter(layer_idx, self.current_iter_depth)

    # ── read views ────────────────────────────────────────────────────────

    def get_cache_upto_iter(
        self, layer_idx: int, upto_iter_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Concatenated K/V from iter 0..upto_iter_idx (inclusive). ``(None, None)`` if empty."""
        keys = [self._k[(layer_idx, i)] for i in range(upto_iter_idx + 1) if (layer_idx, i) in self._k]
        if not keys:
            return None, None
        vals = [self._v[(layer_idx, i)] for i in range(upto_iter_idx + 1) if (layer_idx, i) in self._v]
        return torch.cat(keys, dim=-2), torch.cat(vals, dim=-2)

    def _meta_upto_iter(
        self,
        store: Dict[_Slot, torch.Tensor],
        layer_idx: int,
        upto_iter_idx: int,
        init_batch_size: int,
        empty_dim: int,
    ) -> torch.Tensor:
        """Concatenate per-iter metadata tensors (positions or valid masks).

        ``empty_dim`` is the singleton dim for an absent iter (1 for B-prefixed
        tensors of shape (B, T)) so the cat dimension stays well-defined.
        """
        chunks = []
        batch = init_batch_size
        for i in range(upto_iter_idx + 1):
            t = store.get((layer_idx, i))
            if t is None:
                shape = (batch, 0) if empty_dim == 2 else (0,)
                chunks.append(torch.empty(shape, device=self.device, dtype=torch.long))
            else:
                batch = t.shape[0] if empty_dim == 2 else batch
                chunks.append(t)
        return torch.cat(chunks, dim=-1)

    def get_position_id_upto_iter(self, layer_idx: int, upto_iter_idx: int, init_batch_size: int = 1) -> torch.Tensor:
        return self._meta_upto_iter(self._pos, layer_idx, upto_iter_idx, init_batch_size, empty_dim=2)

    def get_valid_mask_upto_iter(self, layer_idx: int, upto_iter_idx: int, init_batch_size: int = 1) -> torch.Tensor:
        return self._meta_upto_iter(self._valid, layer_idx, upto_iter_idx, init_batch_size, empty_dim=2)

    def get_cache_iter_index_upto_iter(self, layer_idx: int, upto_iter_idx: int) -> torch.Tensor:
        """Per-slot iteration index for KV slots in iter 0..upto_iter_idx.

        Shape ``(total_kv_len,)``; element ``j`` is the iter the slot belongs
        to. Used by the duo-mode mask: a query at iter ``i`` can attend to a
        KV slot iff its iter index is ``<= i``.
        """
        per_iter_lens = [
            self._k[(layer_idx, i)].shape[-2] if (layer_idx, i) in self._k else 0
            for i in range(upto_iter_idx + 1)
        ]
        if sum(per_iter_lens) == 0:
            return torch.empty((0,), device=self.device, dtype=torch.long)
        return torch.cat([
            torch.full((n,), i, device=self.device, dtype=torch.long)
            for i, n in enumerate(per_iter_lens) if n > 0
        ])

    # ── lengths & misc DynamicCache contract ─────────────────────────────

    def get_cache_length(self, layer_idx: Optional[int] = 0, iter_idx: Optional[int] = None) -> int:
        """Sequence length of stored K/V; sum across iters when ``iter_idx is None``."""
        if iter_idx is not None:
            t = self._k.get((layer_idx, iter_idx))
            return t.shape[-2] if t is not None else 0
        return sum(t.shape[-2] for (l, _), t in self._k.items() if l == layer_idx)

    def get_cache_length_upto_iter(self, layer_idx: Optional[int] = 0, iter_depth: int = 0) -> int:
        return sum(
            self._k[(layer_idx, i)].shape[-2]
            for i in range(iter_depth + 1) if (layer_idx, i) in self._k
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0, iter_idx: Optional[int] = 0) -> int:
        """HF protocol: max position + 1 for the layer's iter-0 slot."""
        t = self._k.get((layer_idx, iter_idx))
        return t.shape[-2] if t is not None else 0

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        del new_seq_length
        return self.get_cache_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        return None  # dynamic — no cap

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        return cache_position.shape[0] + self.get_cache_length(layer_idx), 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the batch axis of all stored tensors for beam search."""
        for store in (self._k, self._v):
            for slot, t in store.items():
                store[slot] = t.index_select(0, beam_idx.to(t.device))

    # ── device / dtype ────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        for t in self._k.values():
            return t.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for t in self._k.values():
            return t.dtype
        return torch.bfloat16

    def to(self, *args, **kwargs) -> "TaHCache":
        """Move all stored tensors. Mirrors ``torch.Tensor.to`` calling conventions."""
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        # K/V get both device and dtype; position/valid stay long but follow device.
        for store in (self._k, self._v):
            for slot, t in store.items():
                store[slot] = t.to(*(([device] if device is not None else []) + ([dtype] if dtype is not None else [])))
        if device is not None:
            for store in (self._pos, self._valid):
                for slot, t in store.items():
                    store[slot] = t.to(device)
        return self
