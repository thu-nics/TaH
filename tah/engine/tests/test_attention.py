"""Independent dense masked SDPA over all owned slots, including invisible KV."""
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tah.engine.attention import AttentionRow, DecodeBucket, PrefillAttention
from tah.engine.kv_cache import SlotPool


SHAPES = [('Qwen3-0.6B', 16, 8, 128), ('Qwen3-1.7B', 16, 8, 128), ('tiny', 4, 2, 8)]


def make_case(seed, heads, kv_heads, dim, device='cpu', dtype=torch.bfloat16, num_layers=1):
    # Host generator keeps CPU/GPU layouts identical and never initializes CUDA.
    gen = torch.Generator(device='cpu').manual_seed(seed)
    pool = SlotPool(512, num_layers, kv_heads, dim, 3, device=device, dtype=dtype)
    rows = []
    for request_id, length in enumerate([1, 7, 13, 31, 47]):
        request = pool.create_request(request_id, context_limit=128)
        for d in range(3):
            positions = [i for i in range(length) if d == 0 or float(torch.rand((), generator=gen)) < .6]
            pool.allocate(request, d, positions)
        position = int(torch.randint(length, (), generator=gen))
        depth = 1 + request_id % 3
        rows.append(AttentionRow(request, position, depth))
    pool.values.copy_(torch.randn(pool.values.shape, generator=gen).to(device=device, dtype=dtype))
    q = torch.randn(len(rows), heads, dim, generator=gen).to(device=device, dtype=dtype)
    return pool, rows, q


def dense_oracle(pool, rows, q):
    output = []
    for i, row in enumerate(rows):
        entries = [(slot, position, d + 1) for d, table in enumerate(row.request.streams)
                   for slot, position in zip(table.host_slots, table.host_positions)]
        ids = torch.tensor([e[0] for e in entries], device=q.device)
        mask = torch.tensor([p <= row.position and d <= row.depth for _, p, d in entries], device=q.device)
        k = pool.values[0, 0, ids, 0].transpose(0, 1).repeat_interleave(q.shape[1] // pool.values.shape[-2], 0)
        v = pool.values[0, 1, ids, 0].transpose(0, 1).repeat_interleave(q.shape[1] // pool.values.shape[-2], 0)
        out = F.scaled_dot_product_attention(q[i][None, :, None], k[None], v[None], attn_mask=mask[None, None, None])
        output.append(out[0, :, 0])
    return torch.stack(output)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU test initialized CUDA'))
    torch.set_num_threads(2)


@pytest.mark.parametrize('name,heads,kv_heads,dim', SHAPES)
@pytest.mark.parametrize('seed', range(10))
def test_random_visibility_decode_prefill(name, heads, kv_heads, dim, seed):
    pool, rows, q = make_case(seed, heads, kv_heads, dim)
    expected = dense_oracle(pool, rows, q)
    bucket = DecodeBucket(pool, len(rows), sum(r.request.size for r in rows), heads)
    bucket.prepare(rows)
    for i, row in enumerate(rows):
        actual_ids = bucket.indices[int(bucket.indptr[i]):int(bucket.indptr[i + 1])].tolist()
        exact = [slot for d, t in enumerate(row.request.streams) for slot, p in zip(t.host_slots, t.host_positions)
                 if p <= row.position and d < row.depth]
        assert actual_ids == exact
    decode = bucket.run(q, 0).clone()
    # Different query positions of the same request/depth exercise actual chunk masks.
    chunk_rows = [AttentionRow(r.request, p, r.depth) for r in rows for p in [0, r.position, r.request.streams[0].host_positions[-1]]]
    chunk_q = q.repeat_interleave(3, 0)
    prefill = PrefillAttention(pool, chunk_rows, heads)
    chunk_expected = dense_oracle(pool, chunk_rows, chunk_q)
    actual = prefill.run(chunk_q, 0)
    # A priori BF16 dev tolerance: about 2 ulps near unit-sized outputs.
    torch.testing.assert_close(decode.float(), expected.float(), atol=.015625, rtol=.02)
    torch.testing.assert_close(actual.float(), chunk_expected.float(), atol=.015625, rtol=.02)
    assert prefill.mask.numel() == sum((end - begin) * len(ids) for begin, end, ids, _ in prefill.groups)
    report = Path('data/engine/wp4/cpu-attention')
    report.mkdir(parents=True, exist_ok=True)
    (report / f'{name}-{seed}.json').write_text(json.dumps(dict(production_gates_eligible=False,
        decode_max_abs=float((decode.float() - expected.float()).abs().max()),
        prefill_max_abs=float((actual.float() - chunk_expected.float()).abs().max()),
        rows=len(rows), prefill_rows=len(chunk_rows), entries=sum(r.request.size for r in rows))))


def test_reprepare_static_addresses_and_invalid_metadata():
    pool, rows, q = make_case(0, 4, 2, 8)
    bucket = DecodeBucket(pool, len(rows), 512, 4)
    buffers = [bucket.indices, bucket.indptr, bucket.source_slots, bucket.output]
    ptrs = [t.data_ptr() for t in buffers]
    bucket.prepare(rows)
    bucket.run(q, 0)
    rows = [AttentionRow(r.request, r.request.streams[0].host_positions[-1], 3) for r in rows]
    bucket.prepare(rows)
    torch.testing.assert_close(bucket.run(q, 0), dense_oracle(pool, rows, q), atol=.015625, rtol=.02)
    assert ptrs == [t.data_ptr() for t in buffers]
    pool.allocate(rows[0].request, 0, [1])
    with pytest.raises(RuntimeError, match='changed'):
        bucket.run(q, 0)
    with pytest.raises(ValueError, match='capacity'):
        small = DecodeBucket(pool, len(rows), len(rows), 4)
        small.prepare(rows)
    with pytest.raises(ValueError, match='depth'):
        PrefillAttention(pool, [AttentionRow(rows[0].request, 0, 0)], 4)
