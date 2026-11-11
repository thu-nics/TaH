"""Paged host swap arena: page accounting edges, reuse and bit-exact round trips (CPU)."""
import pytest
import torch

from tah.engine import TaHEngine, Request
from tah.engine.host_arena import HostArena
from tah.engine.kv_cache import SlotPool
from tah.engine.sampling import SamplingParams
from tah.engine.scheduler import HostSwap, HostSwapOverflow, State
from tah.engine.tests.test_model import reference

# Pool entry: 2 layers x K/V x 2 heads x 8 dims x bf16 = 128 bytes; chunk 3 -> 384-byte pages.
ENTRY, PAGE = 128, 384


def setup(budget, *, vocab=40, hidden=16, rows=4, capacity=64):
    torch.manual_seed(0)
    pool = SlotPool(capacity, 2, 2, 8, 3, device='cpu')
    pool.values.view(torch.int16).copy_(torch.randint(-2**15, 2**15, pool.values.shape, dtype=torch.int16))
    state = (torch.randn(rows, vocab).bfloat16(), torch.randn(rows, hidden).bfloat16(),
             torch.randint(0, 2**62, (rows,)), torch.randint(0, 2**62, (rows,)))
    return pool, state, HostSwap(pool, budget, 3)


def request(pool, index, row, counts):
    """counts[s] entries in stream s (positions 0..n-1)."""
    state = State(index, [1], SamplingParams(), 4, 0, 0.)
    state.cache, state.row = pool.create_request(index, context_limit=4096), row
    for stream, n in enumerate(counts):
        pool.allocate(state.cache, stream, list(range(n)))
    return state


def contents(pool, state, device_state):
    return pool.gather(state.cache).values.clone(), [t[state.row].clone() for t in device_state]


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(-1).view(torch.uint8), b.view(-1).view(torch.uint8))


def test_arena_is_exactly_the_budget_and_tail_is_unused():
    _, _, swap = setup(5 * PAGE + 383)
    assert swap.arena.buffer.numel() == 5 * PAGE + 383 and swap.page_bytes == PAGE
    assert swap.pages == swap.free_pages == 5
    assert HostArena(0, 'cpu').buffer is None and HostSwap(SlotPool(8, 2, 2, 8, 3, device='cpu'), 0, 3).pages == 0
    with pytest.raises(ValueError, match='nonnegative integer'):
        HostArena(1.5, 'cpu')


@pytest.mark.parametrize('budget,entries,fits', [
    (2 * PAGE, 5, True),            # 5 * 128 + 128 payload = 768: exact fit, arena full
    (2 * PAGE, 6, False),           # one entry over
    (2 * PAGE + PAGE - 1, 6, False),  # a tail one byte short of a page adds no room
    (3 * PAGE, 6, True),
])
def test_exact_fit_and_one_entry_over(budget, entries, fits):
    pool, device_state, swap = setup(budget)  # payload = 80 + 32 + 8 + 8 = 128 bytes
    state = request(pool, 0, 1, [entries])
    if not fits:
        cache = state.cache
        with pytest.raises(HostSwapOverflow, match='arena full'):
            swap.out(state, device_state)
        assert state.cache is cache and state.snapshot is None and state.swaps == 0
        assert swap.used_bytes == 0 and swap.free_pages == swap.pages and pool.free_slots == 64 - entries
        return
    swap.out(state, device_state)
    assert swap.used_bytes == swap.peak_bytes == state.snapshot.nbytes == -(-(entries + 1) * ENTRY // PAGE) * PAGE
    assert swap.free_pages == swap.pages - len(state.snapshot.pages)
    if budget == 2 * PAGE:
        assert swap.free_pages == 0


@pytest.mark.parametrize('entries,pages', [(1, 2), (2, 2), (3, 2), (4, 3), (6, 3)])
def test_partial_last_page_and_payload_placement(entries, pages):
    # Real-sized payload ratio: 97-token logits (194 B) + 64 B + 16 B = 274 B.
    pool, device_state, swap = setup(10 * PAGE, vocab=97, hidden=32)
    state = request(pool, 0, 0, [entries])
    assert swap.required_bytes(state, device_state) == pages * PAGE
    swap.out(state, device_state)
    assert len(state.snapshot.pages) == pages  # entries=3 fills a page: payload takes one more


def test_round_trip_bit_exact_across_streams_rows_and_split_payload():
    pool, device_state, swap = setup(20 * PAGE, vocab=97, hidden=32)
    filler = request(pool, 9, 3, [5])
    state = request(pool, 0, 1, [7, 4, 2])  # 13 entries: payload straddles a page boundary
    kv, rows = contents(pool, state, device_state)
    positions = tuple(tuple(t.host_positions) for t in state.cache.streams)
    swap.out(state, device_state)
    for got, expected in zip((state.snapshot.logits, state.snapshot.embeddings,
                              state.snapshot.seeds, state.snapshot.counters), rows):
        assert same(got, expected)
    for t in device_state:
        t[1].zero_()
    pool.values.zero_()
    pool.release(filler.cache)
    request(pool, 8, 0, [11])  # destination slots differ from the source slots
    state.row = 2              # and so does the row
    swap.restore(state, device_state, 4096)
    assert tuple(tuple(t.host_positions) for t in state.cache.streams) == positions
    got_kv, got_rows = contents(pool, state, device_state)
    assert same(got_kv, kv) and all(same(a, b) for a, b in zip(got_rows, rows))
    assert swap.used_bytes == 0 and swap.free_pages == swap.pages and state.snapshot is None


def test_pages_do_not_fragment_after_mixed_out_restore_discard():
    pool, device_state, swap = setup(4 * PAGE)  # 128-byte payload
    a, b, c = (request(pool, i, i, [n]) for i, n in enumerate([2, 5, 2]))  # 1, 2, 1 pages
    saved = {}
    for s in (a, b, c):
        saved[s.index] = contents(pool, s, device_state)
        swap.out(s, device_state)
    assert swap.free_pages == 0 and swap.used_bytes == swap.peak_bytes == 4 * PAGE
    assert sorted(a.snapshot.pages + c.snapshot.pages) == [0, 3]  # the two free-able pages are not adjacent
    swap.restore(a, device_state, 4096)
    swap.discard(c)
    assert swap.free_pages == 2 and swap.used_bytes == 2 * PAGE
    d = request(pool, 3, 3, [5])  # needs 2 pages: any two free pages serve
    saved[3] = contents(pool, d, device_state)
    swap.out(d, device_state)
    assert sorted(d.snapshot.pages) == [0, 3] and swap.free_pages == 0
    for s in (b, d):
        pool.values.zero_()
        for t in device_state:
            t[s.row].zero_()
        swap.restore(s, device_state, 4096)
        kv, rows = contents(pool, s, device_state)
        assert same(kv, saved[s.index][0]) and all(same(x, y) for x, y in zip(rows, saved[s.index][1]))
    assert swap.used_bytes == 0 and swap.free_pages == 4 and swap.peak_bytes == 4 * PAGE


def test_budget_zero_refuses_without_state_change():
    pool, device_state, swap = setup(0)
    state = request(pool, 0, 0, [1])
    with pytest.raises(HostSwapOverflow):
        swap.out(state, device_state)
    assert state.cache is not None and state.snapshot is None and swap.peak_bytes == 0


@pytest.mark.parametrize('overlap', [False, True])
def test_engine_swaps_without_host_allocation_after_init(tmp_path, monkeypatch, overlap):
    reference(tmp_path, max_iter=3, label=False)
    eng = TaHEngine(tmp_path, device='cpu', kv_capacity=50, max_batch_size=3,
                    prefill_chunk_size=2, swap_chunk_entries=3, eos_token_id=None,
                    host_swap_bytes=20000, overlap=overlap)
    assert eng.memory_budget['host_swap'] == dict(arena_bytes=20000, pinned=False,
                                                  page_bytes=eng.swap.page_bytes, pages=eng.swap.pages)
    real_empty, inside, host_allocations = torch.empty, [False], []
    def empty(*args, **kwargs):
        # The old per-snapshot path passed pin_memory explicitly (False on CPU).
        if 'pin_memory' in kwargs:
            raise AssertionError('pinned (or would-be pinned) allocation after engine construction')
        if inside[0] and 'device' not in kwargs:  # pool tables pass device=
            host_allocations.append((args, kwargs))
        return real_empty(*args, **kwargs)
    monkeypatch.setattr(torch, 'empty', empty)
    for name in ('out', 'restore'):
        method = getattr(eng.swap, name)
        def scoped(*args, _method=method):
            inside[0] = True
            try:
                return _method(*args)
            finally:
                inside[0] = False
        setattr(eng.swap, name, scoped)
    results = eng.generate([Request([3, 7], 12, i) for i in range(4)], dict(temperature=0))
    assert eng.last_run['swap_outs'] > 0 and eng.last_run['swap_ins'] > 0
    assert host_allocations == []
    assert eng.last_run['host_swap_arena_bytes'] == 20000
    assert 0 < eng.last_run['host_swap_peak_bytes'] <= eng.swap.pages * eng.swap.page_bytes
    assert all(len(r.token_ids) == 12 for r in results) and eng.swap.used_bytes == 0


@pytest.mark.parametrize('release', ['del', 'close', 'with'])
def test_engines_do_not_accumulate_arenas(tmp_path, monkeypatch, release):
    import gc
    import weakref
    import tah.engine.host_arena as host_arena
    reference(tmp_path, max_iter=3, label=False)
    live = weakref.WeakSet()
    original_init = host_arena.HostArena.__init__
    def tracked(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        live.add(self)
    monkeypatch.setattr(host_arena.HostArena, '__init__', tracked)
    options = dict(device='cpu', kv_capacity=50, max_batch_size=3, prefill_chunk_size=2,
                   swap_chunk_entries=3, eos_token_id=None, host_swap_bytes=20000)
    peak = 0
    gc.collect()
    gc.disable()  # reference counting alone must release them
    try:
        for i in range(4):
            overlap = bool(i % 2)
            if release == 'with':
                with TaHEngine(tmp_path, overlap=overlap, **options) as eng:
                    peak = max(peak, len(live))
                    eng.generate([Request([3, 7], 12, j) for j in range(4)], dict(temperature=0))
                assert eng.host_arena.closed and eng.host_arena.buffer is None
                del eng
            else:
                eng = TaHEngine(tmp_path, overlap=overlap, **options)
                peak = max(peak, len(live))
                eng.generate([Request([3, 7], 12, j) for j in range(4)], dict(temperature=0))
                assert eng.last_run['swap_outs'] > 0
                if release == 'close':
                    eng.close()
                    eng.close()
                    assert eng.host_arena.buffer is None
                    with pytest.raises(RuntimeError, match='closed'):
                        eng.generate([Request([3], 1)], dict(temperature=0))
                del eng
            assert len(live) == 0, 'engine arena outlived its engine'
    finally:
        gc.enable()
    assert peak == 1
