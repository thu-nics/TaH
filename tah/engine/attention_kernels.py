"""Device-only ragged duo-visibility count and stable compaction.

One program per query row loops over its OWN real KV entries, tile by tile.
No batch-by-longest buffers; no dynamic allocation or host reads in kernels.
"""
import triton
import triton.language as tl


@triton.jit
def count_visible(pos, depths, starts, lengths, query_pos, query_depth, counts,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(starts + row)
    length = tl.load(lengths + row)
    p = tl.load(query_pos + row)
    d = tl.load(query_depth + row)
    lane = tl.arange(0, BLOCK)
    total = tl.full((), 0, tl.int32)
    for base in range(0, length, BLOCK):
        offset = base + lane
        kp = tl.load(pos + start + offset, offset < length, other=2147483647)
        kd = tl.load(depths + start + offset, offset < length, other=2147483647)
        valid = (offset < length) & (kp <= p) & (kd <= d)
        total += tl.sum(valid.to(tl.int32), 0)
    tl.store(counts + row, total)


@triton.jit
def pack_visible(slots, pos, depths, starts, lengths, query_pos, query_depth,
                 indptr, indices, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(starts + row)
    length = tl.load(lengths + row)
    p = tl.load(query_pos + row)
    d = tl.load(query_depth + row)
    dest = tl.load(indptr + row)
    lane = tl.arange(0, BLOCK)
    for base in range(0, length, BLOCK):
        offset = base + lane
        kp = tl.load(pos + start + offset, offset < length, other=2147483647)
        kd = tl.load(depths + start + offset, offset < length, other=2147483647)
        valid = (offset < length) & (kp <= p) & (kd <= d)
        rank = tl.cumsum(valid.to(tl.int32), 0) - 1
        slot = tl.load(slots + start + offset, offset < length, other=0)
        tl.store(indices + dest + rank, slot, valid)
        dest += tl.sum(valid.to(tl.int32), 0)


@triton.jit
def stage_sources(descriptors, slots, positions, depths, starts, lengths,
                  query_pos, query_depth, STREAMS: tl.constexpr, BLOCK: tl.constexpr):
    """One launch copies ragged device tables described by O(B*N) pointers.

    Pointer ownership/lifetime is retained by the bucket's live request handles.
    Reprepare descriptors after table growth/relocation, before using a graph.
    """
    row = tl.program_id(0)
    descriptor = descriptors + row * (3 * STREAMS + 4)
    dest = tl.load(descriptor + 3 * STREAMS)
    tl.store(starts + row, dest)
    tl.store(lengths + row, tl.load(descriptor + 3 * STREAMS + 1))
    tl.store(query_pos + row, tl.load(descriptor + 3 * STREAMS + 2))
    tl.store(query_depth + row, tl.load(descriptor + 3 * STREAMS + 3))
    lane = tl.arange(0, BLOCK)
    for stream in range(STREAMS):
        source_slots = tl.cast(tl.load(descriptor + stream), tl.pointer_type(tl.int32))
        source_pos = tl.cast(tl.load(descriptor + STREAMS + stream), tl.pointer_type(tl.int64))
        size = tl.load(descriptor + 2 * STREAMS + stream)
        for base in range(0, size, BLOCK):
            offset = base + lane
            mask = offset < size
            tl.store(slots + dest + offset, tl.load(source_slots + offset, mask, other=0), mask)
            tl.store(positions + dest + offset, tl.load(source_pos + offset, mask, other=0), mask)
            tl.store(depths + dest + offset, stream + 1, mask)
        dest += size


@triton.jit
def write_table_entries(writes, count, BLOCK: tl.constexpr):
    """Scatter (slots_ptr, positions_ptr, index, slot, position) rows into tables."""
    entry = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = entry < count
    base = writes + entry * 5
    slots_ptr = tl.load(base, mask, other=0)
    pos_ptr = tl.load(base + 1, mask, other=0)
    index = tl.load(base + 2, mask, other=0)
    slot = tl.load(base + 3, mask, other=0)
    position = tl.load(base + 4, mask, other=0)
    slot_target = tl.cast(slots_ptr, tl.pointer_type(tl.int32)) + index
    pos_target = tl.cast(pos_ptr, tl.pointer_type(tl.int64)) + index
    tl.store(slot_target, slot.to(tl.int32), mask)
    tl.store(pos_target, position, mask)
