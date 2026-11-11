"""FlashInfer page-size-one attention, with CPU SDPA development oracle.

DecodeBucket owns fixed-address ragged scratch buffers. prepare() stages stream
lists and plans outside capture; build() computes exact duo visibility entirely
on the device; run() only executes the already-planned FlashInfer attention.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class AttentionRow:
    request: object
    position: int
    depth: int  # 1-based pass number


def static_plan_integer_bytes(batch_size):
    # FlashInfer 0.2.5 unsplit DecodePlan: int32 request/tile arrays (B),
    # output indptr (B+1), chunk scalar, each with <=16-byte alignment.
    return max(4096, ((12 * batch_size + 68 + 255) // 256) * 256)


def validate_rows(pool, rows):
    if not rows:
        raise ValueError("attention requires at least one row")
    for row in rows:
        pool._request(row.request)
        if not isinstance(row.depth, int) or not 1 <= row.depth <= pool.max_iter:
            raise ValueError("attention depth outside max_iter")
        if not isinstance(row.position, int) or not 0 <= row.position < row.request.context_limit:
            raise ValueError("attention position outside request context_limit")
        if pool.visible_count(row.request, row.position, row.depth) == 0:
            raise ValueError("attention row has no visible KV; allocate current pass before planning")


def _sdpa(q, cache, ids, mask=None):
    # Development-only CPU path: no GPU fallback or repeated dense runtime KV.
    k, v = cache[0, ids, 0], cache[1, ids, 0]
    groups = q.shape[1] // k.shape[1]
    k = k.transpose(0, 1).repeat_interleave(groups, 0)[None]
    v = v.transpose(0, 1).repeat_interleave(groups, 0)[None]
    return F.scaled_dot_product_attention(q.transpose(0, 1)[None], k, v,
        attn_mask=None if mask is None else mask[None, None],
        scale=q.shape[-1] ** -.5)[0].transpose(0, 1)


class DecodeBucket:
    def __init__(self, pool, batch_size, max_entries, num_q_heads, *,
                 workspace_bytes=128 * 1024 * 1024, workspace=None, static_plan=False, buffers=None):
        if batch_size < 1 or max_entries < batch_size or max_entries >= 2**31 or num_q_heads < 1 or num_q_heads % pool.values.shape[-2]:
            raise ValueError("invalid decode bucket dimensions/GQA ratio")
        self.pool, self.batch_size, self.max_entries = pool, batch_size, max_entries
        self.num_q_heads = num_q_heads
        device = pool.device
        def empty(name, n, dtype=torch.int32):
            if buffers is not None:
                return buffers[name][:n]
            return torch.empty(n, dtype=dtype, device=device)
        self.source_slots = empty('source_slots', max_entries)
        self.source_pos = empty('source_pos', max_entries, torch.long)
        self.source_depth = empty('source_depth', max_entries)
        self.starts, self.lengths = empty('starts', batch_size), empty('lengths', batch_size)
        self.query_pos, self.query_depth = empty('query_pos', batch_size, torch.long), empty('query_depth', batch_size)
        self.counts = empty('counts', batch_size)
        self.indptr = empty('indptr', batch_size + 1)
        self.indices = empty('indices', max_entries)
        self.last = torch.ones(batch_size, dtype=torch.int32, device=device)
        self.output = (buffers['output'][:batch_size] if buffers is not None else
                       torch.empty(batch_size, num_q_heads, pool.values.shape[-1], device=device, dtype=pool.dtype))
        self.descriptors = (buffers['descriptors'][:batch_size] if buffers is not None else
                            torch.empty(batch_size, 3 * pool.max_iter + 4, dtype=torch.long, device=device))
        self.wrapper = None
        self.rows = None
        self.static_plan = static_plan
        if device.type == "cuda":
            import flashinfer
            self.workspace = workspace if workspace is not None else torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace, "NHD", use_cuda_graph=not static_plan,
                paged_kv_indptr_buffer=self.indptr,
                paged_kv_indices_buffer=self.indices,
                paged_kv_last_page_len_buffer=self.last)
            if static_plan:
                self.wrapper.reset_workspace_buffer(self.workspace,
                    torch.empty(static_plan_integer_bytes(batch_size), dtype=torch.uint8, device=device))
                # A one-page non-graph plan has no split-KV tiles: one CTA per
                # request/head reads the LIVE indptr. Its launch shape depends
                # only on B, never context lengths. plan() itself stays outside
                # capture; run() remains capturable. FlashInfer 0.2.5 contract.
                self.wrapper.plan(torch.arange(batch_size + 1, dtype=torch.int32),
                    self.indices, torch.ones(batch_size, dtype=torch.int32),
                    num_q_heads, pool.values.shape[-2], pool.values.shape[-1], 1,
                    q_data_type=pool.dtype, kv_data_type=pool.dtype)
                assert not self.wrapper._plan_info[-1], "static decode requires an unsplit FlashInfer plan"
                self.indptr = self.wrapper._paged_kv_indptr_buf
                self.last = self.wrapper._paged_kv_last_page_len_buf

    @torch.inference_mode()
    def prepare(self, rows, query=None):
        """Host O(B*N) descriptors + one ragged staging launch, outside capture.

        Host planning counts use bisect on the allocation ledger (O(log T) per
        stream), with no device-to-host sync. Device kernels independently count
        and compact entries; the host counts are ONLY FlashInfer planning input.
        Reprepare after table growth, slot changes, or row position/depth changes.
        query=(positions, depths) device tensors replace the rows' host query
        coordinates before visibility is built (the overlap scheduler selects
        each row's next pass on the device). Static plans then skip host counts.
        """
        validate_rows(self.pool, rows)
        if len(rows) != self.batch_size:
            raise ValueError("row count does not match fixed decode bucket")
        required = sum(r.request.size for r in rows)
        if required > self.max_entries:
            raise ValueError("ragged decode metadata exceeds bucket capacity; reserve a larger bucket outside capture")
        starts, lengths, counts = [], [], []
        descriptors = []
        offset = 0
        for row in rows:
            starts.append(offset)
            lengths.append(row.request.size)
            if query is None or not self.static_plan:
                counts.append(self.pool.visible_count(row.request, row.position, row.depth))
            tables = row.request.streams
            descriptors.append([*[t.slots.data_ptr() for t in tables],
                                *[t.positions.data_ptr() for t in tables],
                                *[t.size for t in tables], offset, row.request.size,
                                row.position, row.depth])
            for depth, table in enumerate(row.request.streams, 1):
                end = offset + table.size
                if self.pool.device.type != 'cuda':
                    self.source_slots[offset:end].copy_(table.slots[:table.size])
                    self.source_pos[offset:end].copy_(table.positions[:table.size])
                    self.source_depth[offset:end].fill_(depth)
                offset = end
        if self.pool.device.type == 'cuda':
            from .attention_kernels import stage_sources
            # Fresh pinned tensor: its allocator tracks the asynchronous DMA;
            # it cannot be overwritten while a prior transfer still reads it.
            host_descriptors = torch.tensor(descriptors, dtype=torch.long, pin_memory=True)
            self.descriptors.copy_(host_descriptors, non_blocking=True)
            stage_sources[(self.batch_size,)](self.descriptors, self.source_slots,
                self.source_pos, self.source_depth, self.starts, self.lengths,
                self.query_pos, self.query_depth, STREAMS=self.pool.max_iter, BLOCK=256)
        else:
            for dest, values in ((self.starts, starts), (self.lengths, lengths),
                                 (self.query_pos, [r.position for r in rows]),
                                 (self.query_depth, [r.depth for r in rows])):
                dest.copy_(torch.tensor(values, dtype=dest.dtype))
        if query is not None:
            self.query_pos.copy_(query[0])
            self.query_depth.copy_(query[1])
        self.rows = tuple(rows)
        self.revisions = tuple(r.request.revision for r in rows)
        self.build()
        if self.wrapper is not None and not self.static_plan:
            if query is not None:
                raise ValueError("device query coordinates require a static decode plan")
            host_ptr = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32)
            # CPU indptr/last avoid FlashInfer's hidden device->host transfers.
            self.wrapper.plan(host_ptr, self.indices, torch.ones(self.batch_size, dtype=torch.int32),
                self.num_q_heads, self.pool.values.shape[-2], self.pool.values.shape[-1], 1,
                q_data_type=self.pool.dtype, kv_data_type=self.pool.dtype)

    @torch.inference_mode()
    def build(self):
        """Static-shape device metadata construction; safe to put inside a graph."""
        if self.rows is None:
            raise RuntimeError("prepare decode bucket before building metadata")
        if self.pool.device.type == "cuda":
            from .attention_kernels import count_visible, pack_visible
            count_visible[(self.batch_size,)](self.source_pos, self.source_depth,
                self.starts, self.lengths, self.query_pos, self.query_depth, self.counts, BLOCK=256)
            self.indptr[0].zero_()
            torch.cumsum(self.counts, 0, dtype=torch.int32, out=self.indptr[1:])
            pack_visible[(self.batch_size,)](self.source_slots, self.source_pos, self.source_depth,
                self.starts, self.lengths, self.query_pos, self.query_depth, self.indptr, self.indices, BLOCK=256)
        else:
            offset = 0
            self.indptr[0] = 0
            for row in range(self.batch_size):
                start, length = int(self.starts[row]), int(self.lengths[row])
                sl = slice(start, start + length)
                valid = (self.source_pos[sl] <= self.query_pos[row]) & (self.source_depth[sl] <= self.query_depth[row])
                ids = self.source_slots[sl][valid]
                self.counts[row] = len(ids)
                self.indices[offset:offset + len(ids)].copy_(ids)
                offset += len(ids)
                self.indptr[row + 1] = offset

    @torch.inference_mode()
    def run(self, q, layer, *, out=None):
        if self.rows is None:
            raise RuntimeError("prepare decode bucket before running attention")
        if any(r.request.revision != rev for r, rev in zip(self.rows, self.revisions)):
            raise RuntimeError("cache changed since planning; prepare decode bucket again outside capture")
        if q.shape != self.output.shape or q.device != self.pool.device or q.dtype != self.pool.dtype:
            raise ValueError("query does not match decode bucket shape/device/dtype")
        out = self.output if out is None else out
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError("decode output must match query")
        if self.wrapper is not None:
            return self.wrapper.run(q, (self.pool.values[layer, 0], self.pool.values[layer, 1]), out=out)
        for row in range(self.batch_size):
            ids = self.indices[int(self.indptr[row]):int(self.indptr[row + 1])].long()
            out[row:row + 1].copy_(_sdpa(q[row:row + 1], self.pool.values[layer], ids))
        return out


class PrefillAttention:
    """Ragged rows grouped by (request, depth), using an exact GPU custom mask.

    Query order is retained: consecutive groups may refer to the same request.
    Workspace/masks sum actual group sizes; no B*longest-request dimension.
    """
    def __init__(self, pool, rows, num_q_heads, *, workspace_bytes=128 * 1024 * 1024, workspace=None):
        validate_rows(pool, rows)
        if num_q_heads < 1 or num_q_heads % pool.values.shape[-2]:
            raise ValueError("invalid prefill GQA ratio")
        self.pool, self.rows = pool, tuple(rows)
        self.revisions = tuple(r.request.revision for r in rows)
        self.num_q_heads = num_q_heads
        self.groups = []
        begin = 0
        for end in range(1, len(rows) + 1):
            if end == len(rows) or (rows[end].request is not rows[begin].request or rows[end].depth != rows[begin].depth):
                request, depth = rows[begin].request, rows[begin].depth
                tables = request.streams[:depth]
                ids = torch.cat([t.slots[:t.size] for t in tables]).long()
                positions = torch.cat([t.positions[:t.size] for t in tables])
                query_pos = torch.tensor([r.position for r in rows[begin:end]], device=pool.device)
                mask = positions[None] <= query_pos[:, None]
                self.groups.append((begin, end, ids, mask))
                begin = end
        self.indices = torch.cat([g[2] for g in self.groups]).int()
        self.mask = torch.cat([g[3].flatten() for g in self.groups])
        self.wrapper = None
        if pool.device.type == "cuda":
            from .prefill_kernel import make_prefill_wrapper
            qo = torch.tensor([0, *[g[1] for g in self.groups]], dtype=torch.int32)
            kv = torch.tensor([0, *torch.tensor([len(g[2]) for g in self.groups]).cumsum(0).tolist()], dtype=torch.int32)
            self.workspace = workspace if workspace is not None else torch.empty(workspace_bytes, dtype=torch.uint8, device=pool.device)
            self.wrapper = make_prefill_wrapper(self.workspace, pool.dtype, pool.values.shape[-1])
            # FlashInfer 0.2.5 segment_packbits requires DEVICE mask pointers.
            # Prefill planning may synchronize; it is outside decode/capture.
            # Source slot IDs, visibility comparisons and packing stay on GPU.
            self.wrapper.plan(qo.to(pool.device), kv.to(pool.device), self.indices,
                torch.ones(len(self.groups), dtype=torch.int32, device=pool.device),
                num_q_heads, pool.values.shape[-2], pool.values.shape[-1], 1,
                custom_mask=self.mask, q_data_type=pool.dtype, kv_data_type=pool.dtype)

    @torch.inference_mode()
    def run(self, q, layer):
        if any(r.request.revision != rev for r, rev in zip(self.rows, self.revisions)):
            raise RuntimeError("cache changed since prefill planning")
        expected = (len(self.rows), self.num_q_heads, self.pool.values.shape[-1])
        if q.shape != expected or q.dtype != self.pool.dtype or q.device != self.pool.device:
            raise ValueError("prefill query shape/device/dtype mismatch")
        if self.wrapper is not None:
            from .prefill_kernel import run_prefill
            return run_prefill(self.wrapper, q, (self.pool.values[layer, 0], self.pool.values[layer, 1]))
        out = torch.empty_like(q)
        for begin, end, ids, mask in self.groups:
            out[begin:end] = _sdpa(q[begin:end], self.pool.values[layer], ids, mask)
        return out
