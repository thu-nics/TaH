"""CUDA engine state, unsplit static plans and the single-status-sync bound."""
import json
import os

import pytest
import torch

from tah.engine import TaHEngine, Request, SamplingParams
from tah.engine.attention import AttentionRow, DecodeBucket
from tah.engine.kv_cache import SlotPool
from tah.engine.tests.test_model import reference

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES') == '' or not torch.cuda.is_available(), reason='CUDA required')]
# Tests that do not size the host arena on purpose: at most 2 snapshots of a
# 400-entry pool of 2 KiB entries (1.6 MB), so it never overflows. The arena is
# pinned at construction; the most simultaneous engines in one test is two.
TINY_SWAP_BYTES = 4 * 1024**2


def test_cuda_overflow_restart_repeated_completion_and_cleanup(tmp_path):
    reference(tmp_path, max_iter=3, label=False, head_dim=128)
    torch.set_num_threads(2)
    eng = TaHEngine(tmp_path, max_batch_size=3, kv_capacity=50,
                    prefill_chunk_size=2, swap_chunk_entries=3, eos_token_id=None,
                    host_swap_bytes=26000, overlap=False)
    requests = [Request([3, 7], 12, 100 + i) for i in range(6)]
    sources = []
    restart = eng._restart

    def observed(state, running, swapped, free_rows, retries):
        sources.append(state.snapshot is not None)
        assert state.index == max(s.index for s in (*running, *swapped))
        restart(state, running, swapped, free_rows, retries)
        assert state.row == -1 and state.cache is None and state.snapshot is None
        assert eng.swap.used_bytes <= eng.swap.budget_bytes

    def bounded(engine, running, swapped, free_rows):
        assert engine.last_run['steps'] < 2000, 'CUDA restart livelock'
    eng._restart, eng._step_hook = observed, bounded
    callbacks = []
    results = eng.generate(requests, dict(temperature=.6, top_k=20, top_p=.95), callbacks.append)
    assert sorted(r.request_id for r in callbacks) == list(range(6))
    assert eng.last_run['restarts'] >= 2 and eng.last_run['wasted_tokens'] > 0
    assert any(sources) and not all(sources)  # both resident and pinned snapshot victims
    assert eng.last_run['swap_outs'] > 0 and eng.last_run['graph_steps'] == eng.last_run['steps']
    assert sum(r.timing['restarts'] for r in results) == eng.last_run['restarts']
    assert sum(r.timing['wasted_tokens'] for r in results) == eng.last_run['wasted_tokens']
    for r in results:
        assert r.finish_reason == 'length' and len(r.token_ids) == 12
        assert r.prompt_pass_counts == [3, 3] and r.generated_pass_counts == [3] * 12
    audit = eng.pool.audit()
    assert audit['free'] == 50 and audit['owned'] == audit['requests'] == 0
    assert audit['allocations'] == audit['frees'] and eng.swap.used_bytes == 0
    print('Forced restart CUDA report:', json.dumps(dict(run=eng.last_run,
        kv_audit=audit, host_swap_used_bytes=eng.swap.used_bytes,
        callbacks=len(callbacks), swapped_victims=sum(sources),
        resident_victims=len(sources) - sum(sources))))
    # All rows/buffers are reusable after many dropped attempts.
    eng.generate([Request([3, 7], 4, 100)], dict(temperature=.6, top_k=20, top_p=.95))
    assert eng.last_run['restarts'] == 0 and eng.swap.used_bytes == 0


def test_static_plan_grows_ragged_context_without_replan():
    pool = SlotPool(4096, 1, 2, 128, 3, device='cuda')
    requests = [pool.create_request(i, context_limit=1024) for i in range(3)]
    bucket = DecodeBucket(pool, 3, pool.capacity, 4, static_plan=True)
    assert bucket.wrapper._int_workspace_buffer.numel() == 4096
    assert bucket.wrapper._pin_memory_int_workspace_buffer.numel() == 4096
    def fail(*args, **kwargs):
        raise AssertionError('decode replanned a static bucket')
    bucket.wrapper.plan = fail
    torch.manual_seed(5)
    pool.values.normal_()
    q = torch.randn(3, 4, 128, device='cuda', dtype=torch.bfloat16)
    for length in [1, 3, 127, 256, 1024]:
        rows = []
        for i, req in enumerate(requests):
            target = max(1, length - 17 * i)
            for d in range(3):
                positions = range(req.streams[d].size, target) if d == 0 else []
                pool.allocate(req, d, positions)
            rows.append(AttentionRow(req, target - 1, 1))
        bucket.prepare(rows)
        actual = bucket.run(q, 0).clone()
        # Component oracle uses independent paged decode plans, already
        # established against dense SDPA in WP4; retain its numeric tolerance.
        oracle = DecodeBucket(pool, 3, pool.capacity, 4)
        oracle.prepare(rows)
        torch.testing.assert_close(actual, oracle.run(q, 0), atol=.015, rtol=.02)
        del oracle
    for req in requests:
        pool.release(req)
    assert pool.audit()['free'] == pool.capacity


def test_cuda_engine_forced_swap_and_single_sync(tmp_path, monkeypatch):
    reference(tmp_path, max_iter=3, label=False, head_dim=128)
    # Long enough tiny context for short generation, without changing runtime.
    cfg = json.loads((tmp_path / 'config.json').read_text())
    cfg['max_position_embeddings'] = 256
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    torch.set_num_threads(2)
    eng = TaHEngine(tmp_path, host_swap_bytes=TINY_SWAP_BYTES, max_batch_size=3, kv_capacity=128,
                    prefill_chunk_size=2, swap_chunk_entries=2, eos_token_id=None, overlap=False)
    requests = [Request([3, 7], 8, 81), Request([5, 8], 8, 12)]
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    eng.cuda_graphs = False
    expected = eng.generate(requests, params)
    eng.cuda_graphs = True
    exercised = []
    def hook(engine, running, swapped, free_rows):
        if engine.last_run['steps'] == 2:
            state = running[-1]
            engine._swap_out(state, running, swapped, free_rows)
            assert engine.swap.used_bytes > 0
            engine._restore(running, swapped, free_rows)
            assert not swapped
            exercised.append(True)
    eng._step_hook = hook
    from tah.engine.engine import StepBucket
    original_run = StepBucket.run
    def run(bucket, engine, rotary):
        def fail(*args, **kwargs):
            raise AssertionError('host scalar/read or dynamic shape inside CUDA step')
        with monkeypatch.context() as m:
            for name in ['item', 'tolist', 'cpu', 'numpy', '__bool__', '__int__', 'nonzero']:
                m.setattr(torch.Tensor, name, fail)
            return original_run(bucket, engine, rotary)
    monkeypatch.setattr(StepBucket, 'run', run)
    actual = eng.generate(requests, params)
    assert exercised == [True]
    assert [(r.token_ids, r.generated_pass_counts) for r in actual] == [(r.token_ids, r.generated_pass_counts) for r in expected]
    assert eng.swap.used_bytes == 0 and eng.pool.audit()['owned'] == 0
    # Audit the ENTIRE decode including planning for hidden local_scalar syncs.
    eng._step_hook = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        eng.generate(requests, params)
    scalar = [e for e in prof.events() if e.name == 'aten::_local_scalar_dense']
    # Prefill's validation/ragged bookkeeping is outside the decode bound.
    # Explicit per-decode inspection below includes host->device allocation,
    # static planning, dynamic RoPE, all layers, sampling and status transfer.
    from torch.profiler import record_function
    original_decode = eng._decode
    def decode(running):
        with record_function('engine_decode_step'):
            return original_decode(running)
    eng._decode = decode
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        eng.generate(requests, params)
    def in_decode(event):
        parent = event.cpu_parent
        while parent:
            if parent.name == 'engine_decode_step':
                return True
            parent = parent.cpu_parent
        return False
    scalars = [e for e in prof.events() if e.name == 'aten::_local_scalar_dense' and in_decode(e)]
    assert not scalars
    # The sole blocking status copy produces one cudaStreamSynchronize each.
    steps = [e for e in prof.events() if e.name == 'engine_decode_step']
    for step in steps:
        syncs = [e for e in prof.events() if e.name in ('cudaStreamSynchronize', 'cudaDeviceSynchronize') and in_decode(e) and step.time_range.start <= e.time_range.start < step.time_range.end]
        assert len(syncs) <= 1, [(e.name, e.time_range) for e in syncs]
    assert eng.last_run['graph_steps'] == eng.last_run['steps'] > 0


def test_compile_fullgraph_eager_and_cuda_graph_identity(tmp_path):
    reference(tmp_path, max_iter=3, label=False, head_dim=128)
    torch.set_num_threads(2)
    eng = TaHEngine(tmp_path, host_swap_bytes=TINY_SWAP_BYTES, max_batch_size=3, kv_capacity=256,
                    prefill_chunk_size=2, eos_token_id=None,
                    graph_batch_sizes=[1, 2, 3], compile=True,
                    compile_cache_dir=tmp_path / 'inductor-cache')
    assert eng.compiled_step.compile_graphs >= 1
    assert eng.memory_budget['graphs']['compile_seconds'] >= 0
    assert (tmp_path / 'inductor-cache').is_dir()
    requests = [Request([3, 7], 8, 81), Request([5, 8], 8, 12)]
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    eng.cuda_graphs = False
    eager = eng.generate(requests, params)
    eager_trace = [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in eager]
    assert eng.last_run['compile']['enabled'] is True
    assert eng.last_run['eager_steps'] > 0
    eng.cuda_graphs = True
    graphed = eng.generate(requests, params)
    graph_trace = [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in graphed]
    assert graph_trace == eager_trace
    assert eng.last_run['graph_steps'] == eng.last_run['steps'] > 0
    assert eng.pool.audit()['owned'] == 0 and eng.swap.used_bytes == 0
    general = SamplingParams(temperature=.6, top_k=0, top_p=.95)
    eng.cuda_graphs = False
    general_eager = eng.generate(requests, general)
    eng.cuda_graphs = True
    general_fallback = eng.generate(requests, general)
    assert [(r.token_ids, r.generated_pass_counts) for r in general_fallback] == [
        (r.token_ids, r.generated_pass_counts) for r in general_eager]
    assert eng.last_run['eager_steps'] > 0  # general sampling has no WP7 CUDA graph capture
    assert eng.last_run['compile']['graphs'] > 0


@pytest.mark.parametrize('rope', [None, {'rope_type': 'dynamic', 'factor': 2.},
                                 {'rope_type': 'longrope', 'factor': 2.,
                                  'short_factor': [1.] * 64, 'long_factor': [2.] * 64}])
def test_graph_shared_pool_arbitrary_order_rotary_params_and_fallback(tmp_path, rope):
    reference(tmp_path, max_iter=3, label=False, head_dim=128, rope=rope)
    cfg = json.loads((tmp_path / 'config.json').read_text())
    cfg['max_position_embeddings'] = 128
    if rope and rope['rope_type'] == 'longrope':
        cfg['original_max_position_embeddings'] = 4
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    torch.set_num_threads(2)
    eng = TaHEngine(tmp_path, host_swap_bytes=TINY_SWAP_BYTES, max_batch_size=3, kv_capacity=256, prefill_chunk_size=2,
                    eos_token_id=None, graph_batch_sizes=[1, 3], overlap=False)
    assert set(eng.graphs.info['batch_sizes']) == {1, 3}
    assert eng.memory_budget['reservations']['cuda_graphs'] >= eng.graphs.info['private_reserved_bytes']
    trace = []
    decode = eng._decode
    def observed(running):
        result = decode(running)
        trace.extend((s.index, s.position, s.depth, bool(row[0])) for s, row in zip(running, result))
        return result
    eng._decode = observed
    for size, params in [(1, dict(temperature=0)), (3, dict(temperature=.6, top_k=20)),
                         (2, dict(temperature=.8, top_k=7, top_p=.7)),
                         (3, dict(temperature=.6, top_k=0)), (1, dict(temperature=0)),
                         (3, dict(temperature=.3, top_k=12, min_p=.1)),
                         (3, [dict(temperature=0), dict(temperature=.4, top_k=7),
                              dict(temperature=.8, top_k=20)])]:
        requests = [Request([3 + i, 7], 7 - i, 91 + i) for i in range(size)]
        eng.cuda_graphs = False
        trace.clear()
        expected = eng.generate(requests, params)
        expected_trace = trace.copy()
        trace.clear()
        eng.cuda_graphs = True
        actual = eng.generate(requests, params)
        assert trace == expected_trace
        assert [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in actual] == [
            (r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in expected]
        if size == 2 or isinstance(params, dict) and params.get('top_k') == 0:
            assert eng.last_run['eager_steps'] > 0
        else:
            assert eng.last_run['graph_steps'] > 0
        assert eng.pool.audit()['owned'] == 0 and eng.swap.used_bytes == 0


def test_graph_natural_swap_fifo_admission_identity(tmp_path):
    reference(tmp_path, max_iter=3, label=False, head_dim=128)
    torch.set_num_threads(2)
    eng = TaHEngine(tmp_path, host_swap_bytes=TINY_SWAP_BYTES, max_batch_size=3, kv_capacity=50, prefill_chunk_size=2,
                    swap_chunk_entries=3, eos_token_id=None, overlap=False)
    requests = [Request([3, 7], 12, i) for i in range(4)]
    params = SamplingParams(temperature=0)
    eng.cuda_graphs = False
    expected = eng.generate(requests, params)
    restored, admissions = [], []
    restore, prefill = eng.swap.restore, eng._prefill
    def observed_restore(state, *args):
        restored.append(state.index)
        return restore(state, *args)
    def observed_prefill(state):
        admissions.append((state.index, eng.swap.used_bytes))
        return prefill(state)
    eng.swap.restore, eng._prefill = observed_restore, observed_prefill
    eng.cuda_graphs = True
    actual = eng.generate(requests, params)
    assert [(r.token_ids, r.generated_pass_counts) for r in actual] == [(r.token_ids, r.generated_pass_counts) for r in expected]
    assert restored[0] == 2 and all(used == 0 for _, used in admissions)
    assert eng.last_run['blocked_admission_steps'] > 0 and eng.last_run['graph_steps'] > 0
    assert eng.pool.audit()['free'] == 50 and eng.swap.used_bytes == 0


def overlap_checkpoint(tmp_path, threshold=.5):
    reference(tmp_path, max_iter=3, label=False, head_dim=128)
    cfg = json.loads((tmp_path / 'config.json').read_text())
    cfg['max_position_embeddings'] = 256
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    tah = json.loads((tmp_path / 'tah_config.json').read_text())
    tah['iter_decider_kwargs']['threshold'] = threshold
    (tmp_path / 'tah_config.json').write_text(json.dumps(tah))
    torch.set_num_threads(2)


def test_cuda_overlap_identity_swap_restart_graph_and_eager(tmp_path):
    from tah.engine.tests.overlap_replay import assert_same_execution, overlap_vs_replay
    overlap_checkpoint(tmp_path)
    eng = TaHEngine(tmp_path, host_swap_bytes=TINY_SWAP_BYTES, max_batch_size=3, kv_capacity=400, prefill_chunk_size=2,
                    swap_chunk_entries=3, eos_token_id=None, overlap=True)
    requests = [Request([3 + i, 7, 9][:1 + i % 3], 10 + 3 * (i % 4), 40 + i) for i in range(7)]
    params = SamplingParams(temperature=1.0, top_k=8, top_p=.95)
    from collections import Counter
    tokens = Counter(t for r in eng.generate(requests, params) for t in r.token_ids[2:])
    eng.eos_token_id = tokens.most_common(1)[0][0]
    events = []
    def hook(engine, running, swapped, free_rows):
        step = engine.last_run['steps']
        if step == 5:
            victim = max(running, key=lambda s: s.index)
            engine._swap_out(victim, running, swapped, free_rows)
            events.append(('swap', victim.index))
        elif step == 11:
            victim = min(running, key=lambda s: len(s.generated))
            engine._restart(victim, running, swapped, free_rows, engine._restarted)
            events.append(('restart', victim.index))
    traces = {}
    for graphs in (True, False):
        eng.cuda_graphs = graphs
        events.clear()
        overlap, replay = overlap_vs_replay(eng, requests, params, hook)
        assert len(events) == 4 and events[:2] == events[2:]
        assert_same_execution(overlap, replay)
        assert overlap['run']['ghost_rows'] > 0 and overlap['run']['swap_outs'] == 1
        assert {r.finish_reason for r in overlap['results']} == {'eos', 'length'}
        if graphs:
            assert overlap['run']['graph_steps'] == overlap['run']['steps']
        traces[graphs] = overlap
    assert traces[True]['decisions'] == traces[False]['decisions']
    assert_same_execution(traces[True], traces[False])


def test_cuda_overlap_overflow_restart_and_single_sync(tmp_path):
    overlap_checkpoint(tmp_path)
    eng = TaHEngine(tmp_path, max_batch_size=3, kv_capacity=50, prefill_chunk_size=2,
                    swap_chunk_entries=3, eos_token_id=None, host_swap_bytes=26000, overlap=True)
    requests = [Request([3, 7], 12, 100 + i) for i in range(6)]
    def bounded(engine, running, swapped, free_rows):
        assert engine.last_run['steps'] < 2000, 'CUDA overlap restart livelock'
        assert engine.pool.trash_slot not in engine.pool._owners
    eng._step_hook = bounded
    callbacks = []
    results = eng.generate(requests, dict(temperature=.6, top_k=20, top_p=.95), callbacks.append)
    assert sorted(r.request_id for r in callbacks) == list(range(6))
    assert eng.last_run['restarts'] >= 1 and eng.last_run['swap_outs'] > 0
    assert all(r.finish_reason == 'length' and len(r.token_ids) == 12 for r in results)
    audit = eng.pool.audit()
    assert audit['free'] == 50 and audit['owned'] == audit['requests'] == 0
    assert audit['allocations'] == audit['frees'] and eng.swap.used_bytes == 0
    print('Overlap forced restart CUDA report:', json.dumps(dict(run=eng.last_run, kv_audit=audit)))
    # Sync audit: iterations without admission contain at most ONE blocking
    # wait (the lagged status event) and no hidden scalar reads.
    eng._step_hook = None
    big = TaHEngine(tmp_path, host_swap_bytes=TINY_SWAP_BYTES, max_batch_size=3, kv_capacity=400, prefill_chunk_size=2,
                    eos_token_id=None, overlap=True)
    from torch.profiler import record_function
    iteration, prefill = big._overlap_iteration, big._prefill
    admitted = []
    def scoped(ctx):
        with record_function('engine_overlap_iteration'):
            before, launched = len(admitted), big.last_run['steps']
            result = iteration(ctx)
            # Steady-state turn: one launch, no admission (prefill syncs).
            scopes.append(len(admitted) == before and big.last_run['steps'] == launched + 1)
            return result
    def counted(state):
        admitted.append(state.index)
        return prefill(state)
    big._overlap_iteration, big._prefill = scoped, counted
    scopes = []
    requests = [Request([3, 7], 16, 81), Request([5, 8], 16, 12), Request([4], 16, 7)]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        big.generate(requests, SamplingParams(temperature=.6, top_k=20, top_p=.95))
    events = prof.events()
    steps = [e for e in events if e.name == 'engine_overlap_iteration'
             and e.device_type == torch.autograd.DeviceType.CPU]
    assert len(steps) == len(scopes)
    blocking = ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize', 'aten::_local_scalar_dense')
    checked = 0
    for step, clean in zip(steps, scopes):
        if not clean:
            continue
        inside = [e for e in events if e.name in blocking
                  and step.time_range.start <= e.time_range.start < step.time_range.end]
        assert sum(e.name == 'cudaEventSynchronize' for e in inside) <= 1
        assert not [e for e in inside if e.name != 'cudaEventSynchronize'], [e.name for e in inside]
        checked += 1
    assert checked >= 10 and big.last_run['graph_steps'] == big.last_run['steps']


def test_cuda_arena_close_and_del_unpin(tmp_path):
    import gc
    import weakref
    reference(tmp_path, max_iter=3, label=False, head_dim=128)
    with TaHEngine(tmp_path, max_batch_size=3, kv_capacity=50, eos_token_id=None,
                   host_swap_bytes=TINY_SWAP_BYTES, overlap=False) as eng:
        buffer = eng.host_arena.buffer
        assert buffer.is_pinned() and buffer.numel() == TINY_SWAP_BYTES
        eng.generate([Request([3, 7], 2, 0)], dict(temperature=0))
    assert not buffer.is_pinned() and eng.host_arena.buffer is None  # unregistered by close()
    with pytest.raises(RuntimeError, match='closed'):
        eng.generate([Request([3, 7], 2, 0)], dict(temperature=0))
    eng.close()  # idempotent
    gc.collect()
    gc.disable()
    try:
        eng = TaHEngine(tmp_path, max_batch_size=3, kv_capacity=50, eos_token_id=None,
                        host_swap_bytes=TINY_SWAP_BYTES, overlap=True)
        eng.generate([Request([3, 7], 2, 0)], dict(temperature=0))
        buffer, arena = eng.host_arena.buffer, weakref.ref(eng.host_arena)
        del eng  # no reference cycle: the finalizer unpins without a gc pass
        assert arena() is None and not buffer.is_pinned()
    finally:
        gc.enable()
