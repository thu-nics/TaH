"""Deferred FlashInfer/Triton checks; never run while another GPU job owns it."""
import os
import json
from pathlib import Path

import pytest
import torch

from tah.engine.attention import AttentionRow, DecodeBucket, PrefillAttention
from tah.engine.tests.test_attention import SHAPES, dense_oracle, make_case

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES') == '', reason='CUDA explicitly hidden')]


@pytest.mark.parametrize('name,heads,kv_heads,dim', SHAPES[:2])
@pytest.mark.parametrize('seed', range(10))
def test_cuda_randomized_attention(name, heads, kv_heads, dim, seed):
    pool, rows, q = make_case(seed, heads, kv_heads, dim, device='cuda')
    bucket = DecodeBucket(pool, len(rows), sum(r.request.size for r in rows), heads)
    bucket.prepare(rows)
    expected = dense_oracle(pool, rows, q)
    decoded = bucket.run(q, 0).clone()
    torch.testing.assert_close(decoded.float(), expected.float(), atol=.015, rtol=.02)
    ptr, indices = bucket.indptr.cpu(), bucket.indices.cpu()
    for i, row in enumerate(rows):
        wanted = [slot for d, t in enumerate(row.request.streams) for slot, p in zip(t.host_slots, t.host_positions)
                  if p <= row.position and d < row.depth]
        assert indices[int(ptr[i]):int(ptr[i + 1])].tolist() == wanted
    chunk_rows = [AttentionRow(r.request, p, r.depth) for r in rows for p in [0, r.position, r.request.streams[0].host_positions[-1]]]
    chunk_q = q.repeat_interleave(3, 0)
    prefill = PrefillAttention(pool, chunk_rows, heads)
    prefilled = prefill.run(chunk_q, 0)
    prefill_expected = dense_oracle(pool, chunk_rows, chunk_q)
    torch.testing.assert_close(prefilled.float(), prefill_expected.float(), atol=.015, rtol=.02)
    destination = Path('data/engine/wp4/gpu-attention')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f'{name}-{seed}.json').write_text(json.dumps(dict(
        production_gates_eligible=False, device=torch.cuda.get_device_name(),
        decode_max_abs=float((decoded.float() - expected.float()).abs().max()),
        prefill_max_abs=float((prefilled.float() - prefill_expected.float()).abs().max()),
        decode_queries=len(rows), prefill_queries=len(chunk_rows), status='PASS'), indent=2))


def test_cuda_metadata_graph_mutation_and_replan():
    pool, rows, q = make_case(24, 16, 8, 128, device='cuda')
    bucket = DecodeBucket(pool, len(rows), 512, 16)
    bucket.prepare(rows)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            bucket.build()
            bucket.run(q, 0)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bucket.build()
        bucket.run(q, 0)
    graph.replay()
    eager = bucket.output.clone()
    torch.testing.assert_close(eager.float(), dense_oracle(pool, rows, q).float(), atol=.015, rtol=.02)
    # Change depths, positions, source slot IDs, and counts under the same addresses.
    rows = [AttentionRow(r.request, r.request.streams[0].host_positions[-1], 3) for r in rows]
    bucket.prepare(rows)  # planning deliberately outside capture
    q.normal_()
    bucket.build()
    eager = bucket.run(q, 0).clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(bucket.output, eager, atol=0, rtol=0)
    torch.testing.assert_close(bucket.output.float(), dense_oracle(pool, rows, q).float(), atol=.015, rtol=.02)
    snap = pool.gather(rows[0].request)
    old_ids = pool.slot_ids(rows[0].request).clone()
    pool.release(rows[0].request)
    block = pool.create_request('block', context_limit=128)
    pool.allocate(block, 0, list(range(len(old_ids))))
    restored = pool.restore(0, snap)
    rows[0] = AttentionRow(restored, rows[0].position, rows[0].depth)
    assert not torch.equal(pool.slot_ids(restored), old_ids)
    bucket.prepare(rows)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(bucket.output, eager, atol=0, rtol=0)
    for r in [row.request for row in rows] + [block]:
        pool.release(r)
    assert pool.audit()['free'] == pool.capacity


def test_cuda_pinned_snapshot_round_trip():
    from tah.engine.kv_cache import KVSnapshot
    pool, rows, q = make_case(5, 16, 8, 128, device='cuda', num_layers=3)
    request = rows[-1].request
    snapshot = pool.gather(request)
    original = snapshot.values.clone()
    host = torch.empty(snapshot.values.shape, dtype=pool.dtype, pin_memory=True)
    event = torch.cuda.Event()
    host.copy_(snapshot.values, non_blocking=True)
    event.record()
    event.synchronize()  # release/reuse only AFTER WP6 observes transfer completion
    pool.release(request)
    gpu = torch.empty_like(original)
    gpu.copy_(host, non_blocking=True)
    relocated = pool.restore(request.request_id, KVSnapshot(snapshot.positions, gpu, snapshot.context_limit))
    torch.testing.assert_close(pool.gather(relocated).values, original, atol=0, rtol=0)
    for r in [row.request for row in rows[:-1]] + [relocated]:
        pool.release(r)
    assert pool.audit()['free'] == pool.capacity


def test_cuda_cached_model_pass_capture(tmp_path):
    from tah.engine.model import EagerTaHModel
    from tah.engine.tests.test_model import reference
    from tah.engine.tests.test_cached_model import new_pool
    # FlashInfer's CUDA kernels use tensor-core tiles; match production head
    # width while keeping the reference constructor's layers/vocab tiny.
    reference(tmp_path, head_dim=128)
    model = EagerTaHModel.from_pretrained(tmp_path, device='cuda')
    pool = new_pool(model, 64)
    a, b = [pool.create_request(i, context_limit=32) for i in range(2)]
    model.forward_chunk([3, 7], pool, a, iter_count_labels=[3, 1])
    model.forward_chunk([5, 9], pool, b, iter_count_labels=[1, 2])
    model.forward_chunk([12], pool, b, start_position=2, iter_count_labels=[1])
    slots = torch.cat([pool.allocate(a, 0, [2]), pool.allocate(b, 1, [2])])
    pos = torch.tensor([2, 2], device='cuda')
    depths = torch.tensor([1, 2], device='cuda')
    embeddings = model.model.embed_tokens(torch.tensor([11, 13], device='cuda'))
    rows = [AttentionRow(a, 2, 1), AttentionRow(b, 2, 2)]
    bucket = DecodeBucket(pool, 2, 64, 4)
    bucket.prepare(rows)
    for _ in range(3):
        model.forward_cached(embeddings, pos, depths, slots, pool, bucket)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            model.forward_cached(embeddings, pos, depths, slots, pool, bucket)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bucket.build()
        captured = model.forward_cached(embeddings, pos, depths, slots, pool, bucket)
    graph.replay()
    expected = model.forward_cached(embeddings, pos, depths, slots, pool, bucket)
    torch.testing.assert_close(captured.logits, expected.logits, atol=0, rtol=0)
    # New slot allocations, same fixed row input/metadata addresses.
    slots.copy_(torch.cat([pool.allocate(a, 1, [2]), pool.allocate(b, 0, [3])]))
    depths.copy_(torch.tensor([2, 1], device='cuda'))
    pos.copy_(torch.tensor([2, 3], device='cuda'))
    embeddings.copy_(model.model.embed_tokens(torch.tensor([17, 19], device='cuda')))
    bucket.prepare([AttentionRow(a, 2, 2), AttentionRow(b, 3, 1)])
    expected = model.forward_cached(embeddings, pos, depths, slots, pool, bucket)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured.logits, expected.logits, atol=0, rtol=0)
    for r in [a, b]:
        pool.release(r)
    assert pool.audit()['free'] == pool.capacity


def test_cuda_long_stream_staging_growth():
    """Tile boundaries, empty streams and pointer changes after table growth."""
    from tah.engine.kv_cache import SlotPool
    lengths = [255, 256, 257, 2049, 4096]
    pool = SlotPool(sum(lengths) * 3 + 32, 1, 8, 128, 3, device='cuda')
    rows = []
    for index, length in enumerate(lengths):
        request = pool.create_request(index, context_limit=length + 8)
        pool.allocate(request, 0, list(range(length)))
        pool.allocate(request, 1, list(range(0, length, 3)) if index else [])
        pool.allocate(request, 2, list(range(0, length, 7)) if index > 1 else [])
        rows.append(AttentionRow(request, length - 3, 1 + index % 3))
    pool.values.normal_()
    q = torch.randn(len(rows), 16, 128, dtype=torch.bfloat16, device='cuda')
    bucket = DecodeBucket(pool, len(rows), pool.capacity, 16)
    def check():
        bucket.prepare(rows)
        ptr, ids = bucket.indptr.cpu(), bucket.indices.cpu()
        for index, row in enumerate(rows):
            wanted = [slot for d, t in enumerate(row.request.streams)
                      for slot, p in zip(t.host_slots, t.host_positions) if d < row.depth and p <= row.position]
            assert ids[int(ptr[index]):int(ptr[index + 1])].tolist() == wanted
        torch.testing.assert_close(bucket.run(q, 0).float(), dense_oracle(pool, rows, q).float(), atol=.015, rtol=.02)
    check()
    old_pointers = [r.request.streams[0].slots.data_ptr() for r in rows]
    for index, row in enumerate(rows):
        slots = pool.allocate(row.request, 0, [lengths[index]])
        pool.write(0, slots, torch.randn(1, 8, 128, device='cuda', dtype=pool.dtype),
                   torch.randn(1, 8, 128, device='cuda', dtype=pool.dtype))
        rows[index] = AttentionRow(row.request, lengths[index], 3)
    assert all(old != row.request.streams[0].slots.data_ptr() for old, row in zip(old_pointers, rows))
    check()
    for row in rows:
        pool.release(row.request)
    assert pool.audit()['free'] == pool.capacity
