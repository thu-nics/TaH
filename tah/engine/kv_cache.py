"""Page-size-one shared KV storage and auditable request/stream ownership.

Host allocation is admission/planning work; tensor writes and swap transfers are
on the storage device. No model/reference imports or implicit CUDA allocation.
"""
from bisect import bisect_left
from dataclasses import dataclass, field
import math

import torch
from .tensors import upload


class KVCapacityError(RuntimeError):
    pass


def triton_blocks(count, block=128):
    return (count + block - 1) // block


@dataclass
class StreamTable:
    slots: torch.Tensor
    positions: torch.Tensor
    host_positions: list = field(default_factory=list)
    host_slots: list = field(default_factory=list)

    @property
    def size(self):
        return len(self.host_slots)


@dataclass
class RequestCache:
    request_id: object
    streams: list
    context_limit: int
    revision: int = 0

    @property
    def size(self):
        return sum(s.size for s in self.streams)


@dataclass
class KVSnapshot:
    """Stream-major, position-major payload, independent of physical slot IDs."""
    positions: tuple
    values: torch.Tensor  # [layers, 2, entries, 1, kv_heads, head_dim]
    context_limit: int


class SlotPool:
    def __init__(self, capacity, num_layers, num_kv_heads, head_dim, max_iter, *,
                 device, dtype=torch.bfloat16, trash_slot=False):
        dimensions = (capacity, num_layers, num_kv_heads, head_dim, max_iter)
        if any(not isinstance(x, int) or x < 1 for x in dimensions):
            raise ValueError("KV dimensions and max_iter must be positive integers")
        if capacity >= 2**31:
            raise ValueError("page-size-one slot IDs must fit int32")
        self.capacity, self.max_iter = capacity, max_iter
        self.device, self.dtype = torch.device(device), dtype
        # Optional write-only slot beyond capacity: never in a free list or a
        # stream table, so no attention row can read it (overlap ghost rows).
        self.trash_slot = capacity if trash_slot else None
        self.values = torch.empty(num_layers, 2, capacity + bool(trash_slot), 1, num_kv_heads, head_dim,
                                  device=self.device, dtype=dtype)
        # Canonicalize e.g. device='cuda' to the actual indexed CUDA device.
        self.device = self.values.device
        self._free = list(range(capacity - 1, -1, -1))
        self._owners = {}
        self.requests = {}
        self.allocations = self.frees = 0

    @staticmethod
    def capacity_from_bytes(budget_bytes, num_layers, num_kv_heads, head_dim, dtype=torch.bfloat16):
        """WP6 supplies measured free-memory budget after reserving workspaces."""
        per_slot = 2 * num_layers * num_kv_heads * head_dim * torch.empty((), dtype=dtype).element_size()
        return math.floor(budget_bytes / per_slot)

    @property
    def free_slots(self):
        return len(self._free)

    def _request(self, request):
        if self.requests.get(request.request_id) is not request:
            raise ValueError("request cache is released or belongs to another pool")

    def create_request(self, request_id, *, context_limit):
        if request_id in self.requests:
            raise ValueError("request ID already owns a cache")
        if not isinstance(context_limit, int) or context_limit < 1:
            raise ValueError("context_limit must be a positive model context bound")
        streams = [StreamTable(torch.empty(0, dtype=torch.int32, device=self.device),
                               torch.empty(0, dtype=torch.long, device=self.device))
                   for _ in range(self.max_iter)]
        request = RequestCache(request_id, streams, context_limit)
        self.requests[request_id] = request
        return request

    @torch.inference_mode()
    def allocate(self, request, stream, positions):
        """Append strictly increasing occupied positions to a zero-based stream.

        A stopped token has no entry in later streams. Capacity failure is atomic.
        Work is proportional to NEW entries, never to an existing context scan.
        """
        self._request(request)
        if not isinstance(stream, int) or not 0 <= stream < self.max_iter:
            raise ValueError("stream outside max_iter")
        positions = list(positions)
        table = request.streams[stream]
        previous = table.host_positions[-1] if table.size else -1
        for position in positions:
            if not isinstance(position, int) or not previous < position < request.context_limit:
                raise ValueError("positions must increase strictly and fit model context_limit")
            previous = position
        if request.size + len(positions) > self.capacity:
            raise KVCapacityError("request cannot fit the KV pool alone; increase KV budget or reject it")
        if len(positions) > self.free_slots:
            raise KVCapacityError("KV pool exhausted; swap a request before allocating")
        if not positions:
            return torch.empty(0, dtype=torch.long, device=self.device)
        needed = table.size + len(positions)
        if needed > table.slots.numel():
            # Geometric request-local growth, unrelated to other request lengths.
            size = min(request.context_limit, max(needed, max(8, 2 * table.slots.numel())))
            slots = torch.empty(size, dtype=torch.int32, device=self.device)
            pos = torch.empty(size, dtype=torch.long, device=self.device)
            slots[:table.size].copy_(table.slots[:table.size])
            pos[:table.size].copy_(table.positions[:table.size])
            table.slots, table.positions = slots, pos
        ids = list(reversed(self._free[-len(positions):]))
        slot_ids = upload(ids, self.device, torch.int32)
        table.slots[table.size:needed].copy_(slot_ids)
        table.positions[table.size:needed].copy_(upload(positions, self.device))
        del self._free[-len(positions):]
        for slot, position in zip(ids, positions):
            self._owners[slot] = (request.request_id, stream, position)
        table.host_slots.extend(ids)
        table.host_positions.extend(positions)
        request.revision += 1
        self.allocations += len(ids)
        return slot_ids.long()

    @torch.inference_mode()
    def reserve(self, entries):
        """Batch allocate [(request, stream, position)] with ONE device table write.

        Same per-stream rules as allocate(); entries for one stream must be in
        increasing order. Atomic on validation/capacity failure. Returns slot IDs.
        """
        entries = list(entries)
        if len(entries) > self.free_slots:
            raise KVCapacityError("KV pool exhausted; swap a request before allocating")
        last, added = {}, {}
        for request, stream, position in entries:
            self._request(request)
            if not isinstance(stream, int) or not 0 <= stream < self.max_iter:
                raise ValueError("stream outside max_iter")
            table = request.streams[stream]
            key = (id(request), stream)
            previous = last.get(key, table.host_positions[-1] if table.size else -1)
            if not isinstance(position, int) or not previous < position < request.context_limit:
                raise ValueError("positions must increase strictly and fit model context_limit")
            last[key] = position
            added[id(request)] = added.get(id(request), 0) + 1
            if request.size + added[id(request)] > self.capacity:
                raise KVCapacityError("request cannot fit the KV pool alone; increase KV budget or reject it")
        if not entries:
            return []
        counts = {}
        for request, stream, _ in entries:
            counts[(id(request), stream)] = counts.get((id(request), stream), 0) + 1
        grown = set()
        for request, stream, _ in entries:
            key = (id(request), stream)
            if key in grown:
                continue
            grown.add(key)
            table = request.streams[stream]
            needed = table.size + counts[key]
            if needed > table.slots.numel():
                size = min(request.context_limit, max(needed, max(8, 2 * table.slots.numel())))
                slots = torch.empty(size, dtype=torch.int32, device=self.device)
                pos = torch.empty(size, dtype=torch.long, device=self.device)
                slots[:table.size].copy_(table.slots[:table.size])
                pos[:table.size].copy_(table.positions[:table.size])
                table.slots, table.positions = slots, pos
        ids = self._free[-len(entries):][::-1]
        del self._free[-len(entries):]
        writes = []
        for (request, stream, position), slot in zip(entries, ids):
            table = request.streams[stream]
            writes.append((table.slots.data_ptr(), table.positions.data_ptr(), table.size, slot, position))
            if self.device.type != 'cuda':
                table.slots[table.size] = slot
                table.positions[table.size] = position
            self._owners[slot] = (request.request_id, stream, position)
            table.host_slots.append(slot)
            table.host_positions.append(position)
            request.revision += 1
        self.allocations += len(ids)
        if self.device.type == 'cuda':
            from .attention_kernels import write_table_entries
            # Fresh pinned tensor: its allocator tracks the asynchronous DMA.
            host = torch.tensor(writes, dtype=torch.long, pin_memory=True)
            device = host.to(self.device, non_blocking=True)
            write_table_entries[(triton_blocks(len(writes)),)](device, len(writes), BLOCK=128)
        return ids

    def pop_last(self, request, stream, position):
        """Release the newest entry of one stream (an unused speculative slot).

        The device table entry beyond the host size is ignored by every reader.
        """
        self._request(request)
        table = request.streams[stream]
        if not table.size or table.host_positions[-1] != position:
            raise ValueError("pop_last requires the stream's newest position")
        slot = table.host_slots.pop()
        table.host_positions.pop()
        del self._owners[slot]
        self._free.append(slot)
        self.frees += 1
        request.revision += 1
        return slot

    def release(self, request):
        self._request(request)
        for table in request.streams:
            for slot in table.host_slots:
                del self._owners[slot]
                self._free.append(slot)
                self.frees += 1
            table.host_slots.clear()
            table.host_positions.clear()
            table.slots = torch.empty(0, dtype=torch.int32, device=self.device)
            table.positions = torch.empty(0, dtype=torch.long, device=self.device)
        del self.requests[request.request_id]
        request.revision += 1

    def audit(self):
        """Explicit host-only leak/duplicate/ownership check (never a decode step)."""
        free = set(self._free)
        expected = {slot: (r.request_id, d, p) for r in self.requests.values()
                    for d, t in enumerate(r.streams) for slot, p in zip(t.host_slots, t.host_positions)}
        assert len(free) == len(self._free), "duplicate free slot"
        assert len(expected) == sum(r.size for r in self.requests.values()), "duplicate owned slot"
        assert expected == self._owners, "ownership/table mismatch"
        assert not free.intersection(expected), "allocated slot in free list"
        assert free.union(expected) == set(range(self.capacity)), "lost slot"
        assert self.allocations - self.frees == len(expected), "allocation accounting mismatch"
        return dict(capacity=self.capacity, free=len(free), owned=len(expected),
                    requests=len(self.requests), allocations=self.allocations, frees=self.frees)

    @torch.inference_mode()
    def write(self, layer, slots, k, v):
        """k/v [rows, kv_heads, head_dim]; unique live slots required."""
        self.values[layer, 0, :, 0].index_copy_(0, slots, k)
        self.values[layer, 1, :, 0].index_copy_(0, slots, v)

    def slot_ids(self, request):
        self._request(request)
        return torch.cat([s.slots[:s.size] for s in request.streams]).long()

    @torch.inference_mode()
    def gather(self, request, *, out=None):
        self._request(request)
        shape = (self.values.shape[0], 2, request.size, *self.values.shape[3:])
        if out is None:
            out = torch.empty(shape, dtype=self.dtype, device=self.device)
        if out.shape != shape or out.dtype != self.dtype or out.device != self.device or not out.is_contiguous():
            raise ValueError("gather out must be a contiguous, same-device KV buffer of the exact shape/dtype")
        torch.index_select(self.values, 2, self.slot_ids(request), out=out)
        return KVSnapshot(tuple(tuple(s.host_positions) for s in request.streams), out, request.context_limit)

    @torch.inference_mode()
    def scatter(self, request, snapshot):
        self._request(request)
        if tuple(tuple(s.host_positions) for s in request.streams) != snapshot.positions:
            raise ValueError("snapshot coordinates do not match destination request")
        shape = (self.values.shape[0], 2, request.size, *self.values.shape[3:])
        if snapshot.values.shape != shape or snapshot.values.dtype != self.dtype or snapshot.values.device != self.device:
            raise ValueError("scatter requires the exact same-device KV shape/dtype")
        self.values.index_copy_(2, self.slot_ids(request), snapshot.values)

    def restore(self, request_id, snapshot):
        """Reserve all streams atomically; WP6 copies host payload to device first."""
        count = sum(map(len, snapshot.positions))
        expected_shape = (self.values.shape[0], 2, count, *self.values.shape[3:])
        if len(snapshot.positions) != self.max_iter or snapshot.values.shape != expected_shape or snapshot.values.dtype != self.dtype or snapshot.values.device != self.device:
            raise ValueError("snapshot does not match this pool")
        if count > self.free_slots:
            raise KVCapacityError("not enough free KV slots to restore request")
        request = self.create_request(request_id, context_limit=snapshot.context_limit)
        try:
            for depth, positions in enumerate(snapshot.positions):
                self.allocate(request, depth, positions)
            self.scatter(request, snapshot)
        except Exception:
            self.release(request)
            raise
        return request

    def visible_count(self, request, position, depth):
        self._request(request)
        return sum(bisect_left(s.host_positions, position + 1) for s in request.streams[:depth])
