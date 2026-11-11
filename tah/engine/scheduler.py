"""Host request ledger and paged swapping into a fixed pinned host arena."""
import math
from dataclasses import dataclass, field
import torch

from .host_arena import HostArena
from .kv_cache import KVCapacityError
from .sampling import SamplingParams


@dataclass(frozen=True)
class Request:
    """Optional per-request controls; plain strings/token lists also work."""
    prompt: object
    max_new_tokens: int = 1024
    seed: int = 0


@dataclass
class Result:
    request_id: int
    token_ids: list
    text: str
    prompt_pass_counts: list
    generated_pass_counts: list
    finish_reason: str
    timing: dict


@dataclass
class State:
    index: int
    prompt: list
    params: SamplingParams
    max_new_tokens: int
    seed: int
    submitted: float
    cache: object = None
    row: int = -1
    position: int = 0
    depth: int = 1
    generated: list = field(default_factory=list)
    prompt_counts: list = field(default_factory=list)
    generated_counts: list = field(default_factory=list)
    prefill_seconds: float = 0.
    admitted: float = 0.
    swaps: int = 0
    restarts: int = 0
    wasted_tokens: int = 0
    snapshot: object = None
    inflight: object = None  # overlap: last launched step id containing this row
    spec_a: object = None  # overlap: (step, stream, position) unresolved deeper-pass slot


@dataclass
class SwapSnapshot:
    positions: tuple
    pages: list  # arena page indices holding this snapshot's byte stream, in order
    payload: tuple  # (dtype, shape) of each per-row state tensor, for inspection
    nbytes: int  # arena bytes held: len(pages) * page_bytes
    event: object
    swap: object = field(default=None, repr=False)

    def _payload(self, index):
        # Inspection and tests only: synchronizes and copies; never on the swap path.
        if self.event is not None:
            self.event.synchronize()
        return self.swap.host_payload(self, index)

    logits = property(lambda self: self._payload(0))
    embeddings = property(lambda self: self._payload(1))
    seeds = property(lambda self: self._payload(2))
    counters = property(lambda self: self._payload(3))


class HostSwapOverflow(KVCapacityError):
    """Budget refusal before allocating or changing any request state."""


class HostSwap:
    """Paged swapping into a fixed host arena with stream-ordered copies.

    The arena (``HostArena``: exactly ``budget_bytes``, pinned once at startup)
    is split into pages of ``chunk_entries`` KV entries; a tail shorter than a
    page is never used. A snapshot is one byte stream laid over the pages it
    holds: KV chunk k (contiguous [layers, 2, n, 1, heads, dim]) fills page k,
    then the per-row payload (logits, embeddings, seed, counter) follows the
    last chunk as raw bytes and may continue on one more page. It fits exactly
    when ceil(stream bytes / page bytes) pages are free. Pages are
    interchangeable, so freed pages never fragment the arena.

    Reuse needs no host synchronization. During generation the host never
    reads or writes arena bytes: pages are written only by swap-out
    device-to-host copies and read only by restore host-to-device copies, all
    enqueued on the engine's single CUDA stream. A page freed by restore or
    discard may still have such a copy pending, but the next operation that
    touches it is a later swap-out's copy on the same stream, which CUDA runs
    only after every earlier operation of that stream. Slot indices are read
    from the pool's device stream tables (written stream-ordered by
    allocate/reserve), so there is no host staging buffer to protect either.
    """
    def __init__(self, pool, budget_bytes, chunk_entries, arena=None):
        if budget_bytes < 0 or chunk_entries < 1:
            raise ValueError('host swap budget must be nonnegative; chunk_entries positive')
        self.arena = HostArena(budget_bytes, pool.device) if arena is None else arena
        if self.arena.nbytes != budget_bytes:
            raise ValueError('host arena size must equal the host swap budget')
        self.pool, self.budget_bytes = pool, budget_bytes
        self.chunk_entries = min(chunk_entries, pool.capacity)
        self.entry_elements = pool.values[:, :, :1].numel()
        self.entry_bytes = self.entry_elements * pool.values.element_size()
        self.page_bytes = self.chunk_entries * self.entry_bytes
        self.pages = budget_bytes // self.page_bytes
        self._free = list(range(self.pages - 1, -1, -1))
        self.used_bytes = self.peak_bytes = 0
        self.scratch = torch.empty(self.chunk_entries * self.entry_elements,
                                   dtype=pool.dtype, device=pool.device)

    @property
    def free_pages(self):
        return len(self._free)

    def _chunk(self, count):
        # Contiguous view, so each chunk copy is one direct DMA (no temporaries).
        shape = (self.pool.values.shape[0], 2, count, *self.pool.values.shape[3:])
        return self.scratch[:count * self.entry_elements].view(shape)

    @staticmethod
    def _row_bytes(device_state, row):
        return [t[row:row + 1].view(-1).view(torch.uint8) for t in device_state]

    def _spans(self, pages, start, data):
        """Pair stream bytes [start, start + len(data)) of a snapshot with data, split at pages."""
        done = 0
        while done < data.numel():
            page, offset = divmod(start + done, self.page_bytes)
            count = min(data.numel() - done, self.page_bytes - offset)
            yield self.arena.view(pages[page] * self.page_bytes + offset, count), data[done:done + count]
            done += count

    def _pages_for(self, entries, payload_bytes):
        return -(-(entries * self.entry_bytes + payload_bytes) // self.page_bytes)

    def required_bytes(self, state, device_state):
        payload = sum(t[state.row].numel() * t.element_size() for t in device_state)
        return self._pages_for(state.cache.size, payload) * self.page_bytes

    def out(self, state, device_state):
        pool, cache, row = self.pool, state.cache, state.row
        count = cache.size
        payload = self._row_bytes(device_state, row)
        need = self._pages_for(count, sum(p.numel() for p in payload))
        if need > len(self._free):
            raise HostSwapOverflow(f'host swap arena full: need {need} pages of {self.page_bytes} bytes, '
                                   f'{len(self._free)} of {self.pages} free (host_swap_bytes={self.budget_bytes})')
        positions = tuple(tuple(t.host_positions) for t in cache.streams)
        pages = [self._free.pop() for _ in range(need)]
        ids = pool.slot_ids(cache)
        for start in range(0, count, self.chunk_entries):
            chunk = self._chunk(min(self.chunk_entries, count - start))
            torch.index_select(pool.values, 2, ids[start:start + chunk.shape[2]], out=chunk)
            for host, source in self._spans(pages, start * self.entry_bytes, chunk.view(-1).view(torch.uint8)):
                host.copy_(source, non_blocking=True)
        offset = count * self.entry_bytes
        for data in payload:
            for host, source in self._spans(pages, offset, data):
                host.copy_(source, non_blocking=True)
            offset += data.numel()
        event = None
        if pool.device.type == 'cuda':
            event = torch.cuda.Event()
            event.record()
        snapshot = SwapSnapshot(positions, pages, tuple((t.dtype, tuple(t.shape[1:])) for t in device_state),
                                need * self.page_bytes, event, self)
        self.used_bytes += snapshot.nbytes
        self.peak_bytes = max(self.used_bytes, self.peak_bytes)
        pool.release(cache)
        state.cache, state.snapshot = None, snapshot
        state.swaps += 1

    def restore(self, state, device_state, context_limit):
        pool, snapshot = self.pool, state.snapshot
        count = sum(map(len, snapshot.positions))
        if count + 1 > pool.free_slots:
            raise KVCapacityError('not enough free slots to restore request and reserve its next pass')
        if snapshot.event is not None:
            torch.cuda.current_stream(pool.device).wait_event(snapshot.event)
        cache = pool.create_request(state.index, context_limit=context_limit)
        try:
            for stream, positions in enumerate(snapshot.positions):
                pool.allocate(cache, stream, positions)
            ids = pool.slot_ids(cache)
            for start in range(0, count, self.chunk_entries):
                chunk = self._chunk(min(self.chunk_entries, count - start))
                for host, target in self._spans(snapshot.pages, start * self.entry_bytes,
                                                chunk.view(-1).view(torch.uint8)):
                    target.copy_(host, non_blocking=True)
                pool.values.index_copy_(2, ids[start:start + chunk.shape[2]], chunk)
            offset = count * self.entry_bytes
            for data in self._row_bytes(device_state, state.row):
                for host, target in self._spans(snapshot.pages, offset, data):
                    target.copy_(host, non_blocking=True)
                offset += data.numel()
        except Exception:
            pool.release(cache)
            raise
        state.cache, state.snapshot = cache, None
        self._release(snapshot)

    def discard(self, state):
        if state.snapshot is not None:
            self._release(state.snapshot)
            state.snapshot = None

    def _release(self, snapshot):
        # Immediate reuse is safe: see the class docstring (same-stream order).
        self._free.extend(reversed(snapshot.pages))
        self.used_bytes -= snapshot.nbytes
        snapshot.pages = []

    def host_payload(self, snapshot, index):
        """Copy of one per-row state tensor from a snapshot (inspection only)."""
        count = sum(map(len, snapshot.positions))
        offset = count * self.entry_bytes
        for dtype, shape in snapshot.payload[:index]:
            offset += math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        dtype, shape = snapshot.payload[index]
        data = torch.empty(math.prod(shape) * torch.empty((), dtype=dtype).element_size(), dtype=torch.uint8)
        for host, target in self._spans(snapshot.pages, offset, data):
            target.copy_(host)
        return data.view(dtype).reshape(shape)
