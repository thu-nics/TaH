import json
from pathlib import Path

import pytest
import torch

from tah.engine.attention import AttentionRow, DecodeBucket, PrefillAttention
from tah.engine.fixtures import FixtureSuite
from tah.engine.kv_cache import SlotPool
from tah.engine.model import EagerTaHModel
from tah.engine.tests.test_model import reference


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU test initialized CUDA'))


def new_pool(model, capacity):
    cfg = model.config
    return SlotPool(capacity, cfg['num_hidden_layers'], cfg['num_key_value_heads'], cfg['head_dim'],
                    model.max_iter, device=model.device, dtype=model.dtype)


def replay_chunks(model, ids, labels, chunk_size, decode=False):
    pool = new_pool(model, len(ids) * model.max_iter)
    request = pool.create_request('sequence', context_limit=len(ids))
    chunks = []
    for start in range(0, len(ids), chunk_size):
        chunks.append(model.forward_chunk(ids[start:start + chunk_size], pool, request,
            start_position=start, iter_count_labels=None if labels is None else labels[start:start + chunk_size],
            hidden_positions=list(range(min(chunk_size, len(ids) - start))), decode=decode))
    pool.audit()
    pool.release(request)
    assert pool.audit()['free'] == pool.capacity
    return chunks


def compare_chunks(full, chunks):
    logits = torch.cat([c.logits for c in chunks])
    assert torch.equal(torch.cat([c.iter_count for c in chunks]), full.iter_count)
    for name in ['visited', 'decision']:
        assert torch.equal(torch.cat([getattr(c, name) for c in chunks], 1), getattr(full, name))
    torch.testing.assert_close(logits.float(), full.logits.float(), atol=.03125, rtol=.02)
    assert torch.equal(logits.argmax(-1), full.logits.argmax(-1))
    torch.testing.assert_close(torch.cat([c.continue_probability for c in chunks], 1), full.continue_probability,
                               atol=.0078125, rtol=.02, equal_nan=True)
    if full.decider_hidden.numel():
        torch.testing.assert_close(torch.cat([c.decider_hidden for c in chunks], 1), full.decider_hidden,
                                   atol=.03125, rtol=.02, equal_nan=True)
    return dict(max_abs=float((logits.float() - full.logits.float()).abs().max()),
                mae=float((logits.float() - full.logits.float()).abs().mean()),
                top1_agreement=float((logits.argmax(-1) == full.logits.argmax(-1)).float().mean()))


@pytest.mark.parametrize('name', ['mlp1', 'mlp3', 'label3'])
@pytest.mark.parametrize('chunk_size,decode', [(1, True), (1, False), (7, False), (32, False)])
def test_tiny_fixture_chunked_incremental(name, chunk_size, decode):
    root = Path('data/engine/fixtures/cpu-tiny')
    if not root.exists():
        pytest.skip('tiny development fixtures absent')
    suite = FixtureSuite(root)
    assert suite.manifest['production_gates_eligible'] is False
    model = EagerTaHModel.from_pretrained(suite.manifest['checkpoints'][name]['path'])
    results = []
    for record in suite.records(kind='teacher_forced', checkpoint=name):
        if record['variant'] != 'single':
            continue
        t = suite.load(record)
        ids, labels = t['input_ids'], t.get('iter_count_labels')
        full = model(ids, iter_count_labels=labels, hidden_positions=list(range(len(ids))))
        chunks = replay_chunks(model, ids, labels, chunk_size, decode)
        results.append(compare_chunks(full, chunks))
    assert len(results) == 9
    destination = Path('data/engine/wp4/cpu-cached')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f'{name}-{chunk_size}-{decode}.json').write_text(json.dumps(dict(
        production_gates_eligible=False, records=results), indent=2))


@pytest.mark.parametrize('options', [dict(max_iter=1, tied=False), dict(max_iter=3),
    dict(max_iter=5, label=False, normalize=True, layers=3),
    dict(max_iter=3, patterns=True, bias=True, targets='all-linear', tied=False)])
def test_constructor_chunks(tmp_path, options):
    oracle = reference(tmp_path, **options)
    model = EagerTaHModel.from_pretrained(tmp_path)
    ids = torch.tensor([3, 7, 11, 13, 17, 19, 23, 29, 31, 5, 8, 12, 9])
    labels = torch.tensor([1, 2, 3, 4, 5, 2, 1, -100, 3, 2, 1, 4, 5])
    expected = oracle(input_ids=ids[None], iter_count_labels=labels[None]).logits[0]
    full = model(ids, iter_count_labels=labels, hidden_positions=list(range(len(ids))))
    torch.testing.assert_close(expected, full.logits, atol=0, rtol=0)
    compare_chunks(full, replay_chunks(model, ids, labels, 4))


def test_mixed_depth_cached_pass_and_swap(tmp_path):
    reference(tmp_path, patterns=True, bias=True, targets='all-linear', tied=False)
    model = EagerTaHModel.from_pretrained(tmp_path)
    pool = new_pool(model, 200)
    a, b = [pool.create_request(i, context_limit=32) for i in range(2)]
    model.forward_chunk([3, 7, 11], pool, a, iter_count_labels=[3, 1, 2])
    model.forward_chunk([5, 8], pool, b, iter_count_labels=[2, 3])
    embeddings = model.model.embed_tokens(torch.tensor([13, 9]))
    pos = torch.tensor([3, 2])
    depths = torch.tensor([1, 1])
    slots = torch.cat([pool.allocate(a, 0, [3]), pool.allocate(b, 0, [2])])
    rows = [AttentionRow(a, 3, 1), AttentionRow(b, 2, 1)]
    bucket = DecodeBucket(pool, 2, a.size + b.size + 2, 4)
    bucket.prepare(rows)
    first = model.forward_cached(embeddings, pos, depths, slots, pool, bucket)
    # Same-token next pass for a; next new token for b. Per-row LoRA differs.
    values, indices = first.logits[:1].topk(97)
    soft = (values.softmax(-1)[..., None] * model.model.embed_tokens.weight[indices]).sum(-2)
    embeddings = torch.cat([soft, model.model.embed_tokens(torch.tensor([12]))])
    slots = torch.cat([pool.allocate(a, 1, [3]), pool.allocate(b, 0, [3])])
    rows = [AttentionRow(a, 3, 2), AttentionRow(b, 3, 1)]
    pos, depths = torch.tensor([3, 3]), torch.tensor([2, 1])
    bucket.prepare(rows)
    actual = model.forward_cached(embeddings, pos, depths, slots, pool, bucket)
    prefill = PrefillAttention(pool, rows, 4)
    alternate = model.forward_cached(embeddings, pos, depths, slots, pool, prefill)
    torch.testing.assert_close(actual.logits, alternate.logits, atol=.03125, rtol=.02)
    # Relocate a to different physical slots and reproduce the same pass exactly.
    snapshot = pool.gather(a)
    old = pool.slot_ids(a).clone()
    pool.release(a)
    blocker = pool.create_request('block', context_limit=32)
    pool.allocate(blocker, 0, list(range(10)))
    a = pool.restore(0, snapshot)
    assert not torch.equal(old, pool.slot_ids(a))
    slots[0] = a.streams[1].slots[a.streams[1].size - 1]
    rows = [AttentionRow(a, 3, 2), AttentionRow(b, 3, 1)]
    bucket.prepare(rows)
    restored = model.forward_cached(embeddings, pos, depths, slots, pool, bucket)
    torch.testing.assert_close(actual.logits, restored.logits, atol=0, rtol=0)
    for r in [a, b, blocker]:
        pool.release(r)
    assert pool.audit()['free'] == pool.capacity


def test_cached_pass_has_no_host_tensor_reads(tmp_path, monkeypatch):
    reference(tmp_path)
    model = EagerTaHModel.from_pretrained(tmp_path)
    pool = new_pool(model, 16)
    request = pool.create_request('static', context_limit=16)
    slots = pool.allocate(request, 0, [0, 1])
    embeddings = model.model.embed_tokens(torch.tensor([3, 7]))
    positions, depths = torch.tensor([0, 1]), torch.tensor([1, 2])
    from tah.engine.attention import _sdpa
    class FixedAttention:
        def __init__(self):
            self.pool = pool
        def run(self, q, layer):
            return _sdpa(q, pool.values[layer], slots, mask)
    mask = torch.tensor([[True, False], [True, True]])
    attention = FixedAttention()
    def forbidden(*args, **kwargs):
        raise AssertionError('cached pass read a tensor value on the host')
    for name in ['item', 'tolist', 'cpu', 'numpy', '__bool__', '__int__', 'nonzero']:
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    output = model.forward_cached(embeddings, positions, depths, slots, pool, attention)
    assert output.logits.shape == (2, 97)
