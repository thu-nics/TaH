import pytest
import torch

from tah.engine.kv_cache import KVCapacityError, SlotPool


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU test initialized CUDA'))
    torch.set_num_threads(2)


def pool(capacity=32):
    return SlotPool(capacity, 2, 2, 8, 3, device='cpu')


def test_ownership_capacity_atomic_and_lifecycle():
    p = pool(8)
    a, b = [p.create_request(i, context_limit=100) for i in range(2)]
    p.allocate(a, 0, [0, 1, 2])
    p.allocate(a, 2, [1, 2])
    p.allocate(b, 0, [0, 1])
    before = p.audit()
    with pytest.raises(KVCapacityError, match='swap'):
        p.allocate(b, 1, [0, 1])
    assert p.audit() == before
    with pytest.raises(KVCapacityError, match='alone'):
        p.allocate(a, 1, [0, 1, 2, 3])
    assert p.audit() == before
    for stream, positions in [(3, [0]), (0, [2]), (0, [4, 3]), (1, [-1]), (1, [100])]:
        with pytest.raises(ValueError):
            p.allocate(a, stream, positions)
    assert p.audit() == before
    p.release(a)
    with pytest.raises(ValueError, match='released'):
        p.release(a)
    p.release(b)
    assert p.audit()['free'] == 8
    assert p.allocations == p.frees == 7
    assert SlotPool.capacity_from_bytes(1024, 2, 2, 8) == 8


def test_snapshot_all_layers_streams_contiguous_relocation():
    p = pool()
    a = p.create_request('a', context_limit=100)
    for d, positions in enumerate(([0, 1, 3, 5], [1, 5], [5])):
        ids = p.allocate(a, d, positions)
        for layer in range(2):
            k = torch.arange(len(ids) * 16).reshape(len(ids), 2, 8).to(torch.bfloat16) + 100 * d + layer
            p.write(layer, ids, k, -k)
    snap = p.gather(a)
    assert snap.values.is_contiguous()
    original_ids = p.slot_ids(a).clone()
    original = snap.values.clone()
    out = torch.empty_like(snap.values)
    assert p.gather(a, out=out).values.data_ptr() == out.data_ptr()
    p.release(a)
    blocker = p.create_request('block', context_limit=100)
    p.allocate(blocker, 0, list(range(7)))
    restored = p.restore('a', snap)
    assert not torch.equal(original_ids, p.slot_ids(restored))
    assert torch.equal(original, p.gather(restored).values)
    p.values.zero_()
    p.scatter(restored, snap)
    assert torch.equal(original, p.gather(restored).values)
    p.release(restored)
    p.release(blocker)
    assert p.audit()['free'] == p.capacity


def test_long_request_tables_not_batch_times_longest():
    p = pool(20000)
    long = p.create_request(0, context_limit=17000)
    p.allocate(long, 0, list(range(16384)))
    short = [p.create_request(i + 1, context_limit=17000) for i in range(64)]
    for r in short:
        p.allocate(r, 0, [0, 1, 2])
    assert p.audit()['owned'] == 16384 + 64 * 3
    assert sum(s.slots.numel() for r in p.requests.values() for s in r.streams) == 16384 + 64 * 8
    for r in [long, *short]:
        p.release(r)
    assert p.audit()['free'] == 20000


def test_batch_reserve_pop_last_and_trash_slot():
    p = SlotPool(16, 2, 2, 8, 3, device='cpu', trash_slot=True)
    assert p.trash_slot == 16 and p.values.shape[2] == 17 and p.free_slots == 16
    a, b = [p.create_request(i, context_limit=10) for i in range(2)]
    p.allocate(a, 0, [0, 1])
    # Mixed requests/streams in one call; same-stream entries must increase.
    ids = p.reserve([(a, 1, 1), (a, 0, 2), (b, 0, 0), (b, 0, 1)])
    assert len(set(ids)) == 4 and p.free_slots == 10
    assert a.streams[0].host_positions == [0, 1, 2] and b.streams[0].host_positions == [0, 1]
    assert a.streams[0].slots[:3].tolist() == a.streams[0].host_slots
    assert a.streams[0].positions[:3].tolist() == [0, 1, 2]
    assert b.streams[0].slots[:2].tolist() == b.streams[0].host_slots
    p.audit()
    before = p.audit()
    for bad in ([(a, 0, 2)], [(b, 0, 3), (b, 0, 2)], [(a, 3, 5)], [(a, 1, 10)]):
        with pytest.raises(ValueError):
            p.reserve(bad)
        assert p.audit() == before
    with pytest.raises(KVCapacityError, match='exhausted'):
        p.reserve([(b, 2, i) for i in range(11)])
    assert p.audit() == before
    # Popping the newest entry returns its slot; non-newest positions refuse.
    with pytest.raises(ValueError, match='newest'):
        p.pop_last(a, 0, 1)
    slot = a.streams[0].host_slots[-1]
    assert p.pop_last(a, 0, 2) == slot and p.free_slots == 11
    assert a.streams[0].host_positions == [0, 1]
    p.reserve([(a, 0, 2)])  # the freed coordinate can be reserved again
    assert p.trash_slot not in p._owners and p.audit()['owned'] == 6
    # Many entries in one stream grow its table geometrically.
    p.reserve([(b, 1, i) for i in range(1, 9)])
    assert b.streams[1].positions[:8].tolist() == list(range(1, 9))
    for r in (a, b):
        p.release(r)
    assert p.audit()['free'] == 16 and p.allocations == p.frees
