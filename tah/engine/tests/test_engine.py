"""Scheduler lifecycle and generation semantics, with a tiny independent oracle."""
from types import SimpleNamespace

import pytest
import torch

from tah.engine import TaHEngine, Request, SamplingParams
from tah.engine.kv_cache import KVCapacityError
from tah.engine.tests.overlap_replay import assert_same_execution, overlap_vs_replay
from tah.engine.tests.test_model import reference
from tah.model.utils import TaHForCasualLM_generate


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU engine initialized CUDA'))


def engine(tmp_path, **kwargs):
    return TaHEngine(tmp_path, device='cpu', kv_capacity=128, max_batch_size=4,
                     prefill_chunk_size=3, eos_token_id=None, **kwargs)


def test_compile_requires_cuda_before_model_load():
    with pytest.raises(ValueError, match='compile=True requires a CUDA device'):
        TaHEngine('unused-checkpoint', device='cpu', compile=True)


@pytest.mark.parametrize('max_iter,label', [(1, True), (3, True), (3, False), (5, False)])
def test_generation_reference_and_counts(tmp_path, max_iter, label):
    oracle = reference(tmp_path, max_iter=max_iter, label=label)
    eng = engine(tmp_path)
    prompts = [[3, 7, 11], [5, 8]]
    results = eng.generate([Request(p, 4, i) for i, p in enumerate(prompts)], SamplingParams(temperature=0))
    for result, prompt in zip(results, prompts):
        ids, _, counts = TaHForCasualLM_generate(oracle,
            SimpleNamespace(eos_token_id=None, decode=lambda ids: ''),
            dict(input_ids=torch.tensor([prompt])), max_new_tokens=4,
            do_sample=False, verbose=False, return_pass_counts=True)
        assert result.token_ids == ids[0]
        assert result.generated_pass_counts == counts[0]
        assert result.prompt_pass_counts == [max_iter if not label else 1] * len(prompt)
        assert result.finish_reason == 'length'
    assert eng.pool.audit()['owned'] == 0
    assert eng.swap.used_bytes == 0


def test_eos_zero_length_and_callback_cleanup(tmp_path):
    reference(tmp_path, max_iter=1)
    eng = engine(tmp_path)
    token = eng.generate([Request([3], 1)], dict(temperature=0))[0].token_ids[0]
    eng.eos_token_id = token
    seen = []
    results = eng.generate([Request([3], 3), Request([4], 0)], dict(temperature=0), seen.append)
    assert [r.request_id for r in seen] == [0, 1]
    assert results[0].token_ids == [] and results[0].generated_pass_counts == []
    assert results[0].finish_reason == 'eos'
    assert results[1].finish_reason == 'length' and results[1].prompt_pass_counts == [1]
    def fail(result):
        raise RuntimeError('callback failure')
    with pytest.raises(RuntimeError, match='callback failure'):
        eng.generate([Request([3], 3), Request([4], 3)], dict(temperature=0), fail)
    assert eng.pool.audit()['free'] == 128
    assert eng.swap.used_bytes == 0 and not eng._busy


def test_forced_swap_identity_and_all_state(tmp_path):
    reference(tmp_path, max_iter=3, label=False)
    eng = engine(tmp_path, swap_chunk_entries=2, overlap=False)
    requests = [Request([3, 7], 7, 81), Request([5, 8], 7, 12)]
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    expected = eng.generate(requests, params)
    exercised = []
    def hook(engine, running, swapped, free_rows):
        if engine.last_run['steps'] != 2:
            return
        state = running[-1]
        assert state.depth == 3
        old_kv = engine.pool.gather(state.cache).values.clone()
        old_state = [t[state.row].clone() for t in engine.device_state]
        scheduler = (state.position, state.depth, state.generated.copy(), state.generated_counts.copy())
        engine._swap_out(state, running, swapped, free_rows)
        assert engine.swap.used_bytes > 0 and state.row == -1
        engine._restore(running, swapped, free_rows)
        assert not swapped
        assert torch.equal(engine.pool.gather(state.cache).values, old_kv)
        for t, expected_tensor in zip(engine.device_state, old_state):
            assert torch.equal(t[state.row], expected_tensor)
        assert scheduler == (state.position, state.depth, state.generated, state.generated_counts)
        exercised.append(True)
    eng._step_hook = hook
    actual = eng.generate(requests, params)
    assert exercised == [True]
    for a, b in zip(actual, expected):
        assert a.token_ids == b.token_ids
        assert a.prompt_pass_counts == b.prompt_pass_counts
        assert a.generated_pass_counts == b.generated_pass_counts
    assert eng.swap.used_bytes == 0 and eng.pool.audit()['free'] == eng.pool.capacity


def test_natural_swap_fifo_blocks_admission_and_leaks(tmp_path):
    reference(tmp_path, max_iter=3, label=False)
    eng = TaHEngine(tmp_path, device='cpu', kv_capacity=50, max_batch_size=3,
                    prefill_chunk_size=2, swap_chunk_entries=3, eos_token_id=None,
                    admission_headroom=.1, overlap=False)
    requests = [Request([3, 7], 12, i) for i in range(4)]
    restored, admissions = [], []
    original_restore, original_prefill = eng.swap.restore, eng._prefill
    def restore(state, *args):
        restored.append(state.index)
        return original_restore(state, *args)
    def prefill(state):
        admissions.append((state.index, eng.swap.used_bytes))
        return original_prefill(state)
    eng.swap.restore, eng._prefill = restore, prefill
    eng.generate(requests, dict(temperature=0))
    assert eng.last_run['swap_outs'] > 0 and eng.last_run['swap_ins'] > 0
    assert eng.last_run['blocked_admission_steps'] > 0
    assert all(used == 0 for _, used in admissions)
    assert restored[0] == 2  # newest resident
    assert eng.pool.audit()['owned'] == 0 and eng.swap.used_bytes == 0


@pytest.mark.parametrize('failure', ['context', 'prompt', 'single_request'])
def test_capacity_errors_are_clear_and_cleanup(tmp_path, failure):
    reference(tmp_path, max_iter=3, label=False)
    eng = TaHEngine(tmp_path, device='cpu', kv_capacity=28, max_batch_size=2,
                    eos_token_id=None, host_swap_bytes=0)
    if failure == 'context':
        requests, match = [Request([3], 20)], 'context limit'
    elif failure == 'prompt':
        requests, match = [Request([3] * 10, 1)], 'prefill'
    else:
        requests, match = [Request([3], 12)], 'fit KV pool alone'
    with pytest.raises(KVCapacityError, match=match):
        eng.generate(requests, dict(temperature=0))
    assert eng.pool.audit()['owned'] == 0 and eng.swap.used_bytes == 0


@pytest.mark.parametrize('host_budget', [0, 4800])
def test_overflow_restart_completes_without_livelock(tmp_path, host_budget):
    reference(tmp_path, max_iter=3, label=False)
    eng = TaHEngine(tmp_path, device='cpu', kv_capacity=50, max_batch_size=3,
                    prefill_chunk_size=2, swap_chunk_entries=3, eos_token_id=None,
                    admission_headroom=.1, host_swap_bytes=host_budget, overlap=False)
    requests = [Request([3, 7], 12, 100 + i) for i in range(6)]
    events, callbacks, admissions, retry_queues = [], [], [], []
    restart, prefill = eng._restart, eng._prefill

    def observed_restart(state, running, swapped, free_rows, retries):
        assert state.index == max(s.index for s in (*running, *swapped))
        events.append((state.index, state.snapshot is not None, len(state.generated)))
        waited = []
        if state.snapshot is not None:
            class PendingCopy:
                def synchronize(self):
                    waited.append(True)
            state.snapshot.event = PendingCopy()
        restart(state, running, swapped, free_rows, retries)
        if events[-1][1]:
            assert waited == [True], 'snapshot dropped before outgoing copy completion'
        retry_queues[:] = [retries]
        assert state.seed == requests[state.index].seed
        assert not state.generated and not state.prompt_counts and not state.generated_counts
        assert state.cache is None and state.snapshot is None and state.row == -1
        assert state.depth == 1 and state.position == 0
        assert eng.swap.used_bytes <= host_budget

    def observed_prefill(state):
        if not state.restarts:
            assert not retry_queues or not retry_queues[0], 'fresh input bypassed a waiting retry'
        elif retry_queues and retry_queues[0]:
            assert state.index < retry_queues[0][0].index, 'retry priority changed'
        # Counter/seed reset is checked immediately after fresh prompt sampling.
        admissions.append((state.index, state.restarts, eng.swap.used_bytes))
        result = prefill(state)
        assert int(eng.seeds[state.row]) == requests[state.index].seed
        assert int(eng.counters[state.row]) == 1
        return result

    def bounded(engine, running, swapped, free_rows):
        assert engine.last_run['steps'] < 2000, 'restart livelock'
    eng._restart, eng._prefill, eng._step_hook = observed_restart, observed_prefill, bounded
    actual = eng.generate(requests, dict(temperature=.6, top_k=20, top_p=.95), callbacks.append)
    assert len(actual) == len(callbacks) == 6
    assert sorted(r.request_id for r in callbacks) == list(range(6))
    assert eng.last_run['restarts'] == len(events) >= 2
    assert eng.last_run['wasted_tokens'] == sum(n for _, _, n in events)
    assert sum(r.timing['restarts'] for r in actual) == len(events)
    assert sum(r.timing['wasted_tokens'] for r in actual) == eng.last_run['wasted_tokens']
    for result in actual:
        assert result.finish_reason == 'length' and len(result.token_ids) == 12
        assert len(result.prompt_pass_counts) == 2
        assert result.generated_pass_counts == [3] * 12
        assert all(0 <= token < eng.model.config['vocab_size'] for token in result.token_ids)
    assert all(used == 0 for _, _, used in admissions)
    if host_budget:
        assert eng.last_run['swap_outs'] > 0
        assert any(was_swapped for _, was_swapped, _ in events)
    else:
        assert not eng.last_run['swap_outs'] and eng.swap.peak_bytes == 0
    assert eng.pool.audit()['owned'] == 0 and eng.pool.free_slots == eng.pool.capacity
    assert eng.pool.audit()['allocations'] == eng.pool.audit()['frees']
    assert eng.swap.used_bytes == 0 and not eng._busy


def test_restart_callback_failure_releases_all_attempts(tmp_path):
    reference(tmp_path, max_iter=3, label=False)
    eng = TaHEngine(tmp_path, device='cpu', kv_capacity=50, max_batch_size=3,
                    eos_token_id=None, host_swap_bytes=4800)
    def fail(result):
        assert eng.last_run['restarts'] > 0
        raise RuntimeError('callback after restart')
    with pytest.raises(RuntimeError, match='callback after restart'):
        eng.generate([Request([3, 7], 12, i) for i in range(4)],
                     dict(temperature=0), fail)
    assert eng.swap.used_bytes == 0 and eng.pool.audit()['owned'] == 0 and not eng._busy
    assert eng.pool.free_slots == eng.pool.capacity
    assert len(eng.generate([Request([3], 2)], dict(temperature=0))) == 1


def test_static_shape_and_no_host_reads_in_step(tmp_path, monkeypatch):
    reference(tmp_path, label=False)
    eng = engine(tmp_path)
    # CPU attention is the development oracle with ragged host indexing;
    # this check isolates the engine update boundary. CUDA tests include the
    # actual fixed-plan attention and enforce the bound on the whole step.
    monkeypatch.setattr('tah.engine.attention.DecodeBucket.run', lambda self, q, layer: torch.zeros_like(q))
    from tah.engine.engine import StepBucket
    original = StepBucket.run
    def checked(bucket, engine, rotary):
        def fail(*args, **kwargs):
            raise AssertionError('host scalar/read or dynamic shape in device step')
        with monkeypatch.context() as m:
            for name in ['item', 'tolist', 'cpu', 'numpy', '__bool__', '__int__', 'nonzero']:
                m.setattr(torch.Tensor, name, fail)
            return original(bucket, engine, rotary)
    monkeypatch.setattr(StepBucket, 'run', checked)
    eng.generate([Request([3], 4), Request([4, 5], 4)], dict(temperature=0))
    assert eng.pool.free_slots == eng.pool.capacity


def test_reference_capture_explicit_prompt_decode_boundaries(tmp_path):
    from tah.engine.reference_capture import capture
    oracle = reference(tmp_path, max_iter=3, label=False)
    sequence = [3, 7, 11, 13, 17, 19]
    a, la = capture(oracle, [sequence], chunk_size=2)
    b, lb = capture(oracle, [sequence], chunk_boundaries=[0, 2, 4, 6])
    assert torch.equal(la[0], lb[0])
    assert torch.equal(a[0]['decision'], b[0]['decision'])
    trace, logits = capture(oracle, [sequence], chunk_boundaries=[0, 3, 4, 5, 6])
    assert torch.equal(trace[0]['pass_count'], torch.full((6,), 3))
    for boundaries in ([1, 6], [0, 2, 2, 6], [0, 5]):
        with pytest.raises(ValueError, match='chunk_boundaries'):
            capture(oracle, [sequence], chunk_boundaries=boundaries)


def test_public_engine_runtime_import_isolation():
    import os
    from pathlib import Path
    import subprocess
    import sys
    model = Path('data/engine/fixture-models/tiny/mlp1')
    if not model.exists():
        pytest.skip('tiny fixture checkpoint absent')
    script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('transformers', 'peft', 'tah.model')):
        raise AssertionError('forbidden runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tah.engine import TaHEngine, Request, SamplingParams
engine = TaHEngine('data/engine/fixture-models/tiny/mlp1', device='cpu', kv_capacity=128,
                  max_batch_size=2, prefill_chunk_size=2, eos_token_id=None)
result = engine.generate([Request([3,7],2,17)], SamplingParams(temperature=0))[0]
assert len(result.token_ids)==2 and result.generated_pass_counts==[1,1]
'''
    subprocess.run([sys.executable, '-c', script], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)


def test_reference_generate_layout_pack_and_capture(tmp_path):
    from tah.engine.reference_capture import capture, pack
    oracle = reference(tmp_path, max_iter=1)
    sequences = [[3, 7, 11, 13, 17], [5, 8, 9, 12]]
    inputs, columns = pack(sequences, 0, 'cpu', prompt_lengths=[3, 2])
    assert inputs['input_ids'].tolist() == [[3, 7, 11, 13, 17], [0, 5, 8, 9, 12]]
    assert [c.tolist() for c in columns] == [[0, 1, 2, 3, 4], [1, 2, 3, 4]]
    traces, logits = capture(oracle, sequences, prompt_lengths=[3, 2], chunk_boundaries=[0, 3, 4, 5])
    for i, sequence in enumerate(sequences):
        single, values = capture(oracle, [sequence], chunk_boundaries=[0, len(sequence) - 2, len(sequence) - 1, len(sequence)])
        assert torch.equal(logits[i], values[0])
        assert torch.equal(traces[i]['pass_count'], single[0]['pass_count'])


def test_profile_counts_physical_work_without_annotation_or_cpu_duplicates():
    from tah.engine.tests.engine_check import trace_profile_totals
    trace = [
        dict(cat='user_annotation', name='engine_decode_step', ts=0, dur=10),
        dict(cat='gpu_user_annotation', name='engine_decode_step', ts=2, dur=7),
        dict(cat='cpu_op', name='aten::mm', ts=3, dur=5),
        dict(cat='cpu_op', name='Unrecognized', ts=3, dur=5),
        dict(cat='kernel', name='matmul', ts=5, dur=2),
        dict(cat='gpu_memcpy', name='Memcpy HtoD', ts=7, dur=1),
        dict(cat='gpu_memset', name='Memset', ts=8, dur=1),
        dict(cat='kernel', name='outside decode', ts=13, dur=4),
    ]
    assert trace_profile_totals(trace) == dict(steps=1, wall_us=10, kernel_us=2, copy_us=2, activities=3)


def test_graph_bucket_selection_validation():
    from tah.engine.graphs import batch_sizes
    assert batch_sizes(4, None) == (1, 2, 3, 4)
    assert batch_sizes(4, [4, 1, 4]) == (1, 4)
    assert batch_sizes(4, []) == ()
    for bad in ([0], [5], [True], [1.5]):
        with pytest.raises(ValueError, match='graph_batch_sizes'):
            batch_sizes(4, bad)


def test_cuda_graph_copy_kernel_is_physical_kernel_not_dma():
    from tah.engine.tests.engine_check import cuda_transfer_activity
    assert not cuda_transfer_activity('memcpy32_post')
    assert not cuda_transfer_activity('void copy_kernel<float>(...)')
    assert cuda_transfer_activity('Memcpy HtoD (Pinned -> Device)')
    assert cuda_transfer_activity('Memcpy DtoD (Device -> Device)')
    assert cuda_transfer_activity('Memset (Device)')


def mixed_depth_reference(tmp_path, max_iter=3, threshold=.5):
    """Tiny MLP-decider checkpoint whose decisions mix continue/stop."""
    import json
    reference(tmp_path, max_iter=max_iter, label=False)
    path = tmp_path / 'tah_config.json'
    config = json.loads(path.read_text())
    config['iter_decider_kwargs']['threshold'] = threshold
    path.write_text(json.dumps(config))


@pytest.mark.parametrize('max_iter', [1, 3, 5])
def test_overlap_identical_to_replayed_schedule(tmp_path, max_iter):
    mixed_depth_reference(tmp_path, max_iter=max_iter)
    eng = TaHEngine(tmp_path, device='cpu', kv_capacity=400, max_batch_size=3,
                    prefill_chunk_size=3, eos_token_id=None, overlap=True)
    # More requests than rows: admissions follow finishes with the lag.
    requests = [Request([3 + i, 7, 2 * i + 1][:1 + i % 3], 6 + i % 5, 50 + i) for i in range(7)]
    params = SamplingParams(temperature=1.0, top_k=8, top_p=.95)
    # A frequent mid-sequence token as EOS gives unpredictable finishes.
    from collections import Counter
    tokens = Counter(t for r in eng.generate(requests, params) for t in r.token_ids[2:])
    eng.eos_token_id = tokens.most_common(1)[0][0]
    overlap, replay = overlap_vs_replay(eng, requests, params)
    assert_same_execution(overlap, replay)
    assert overlap['run']['ghost_rows'] == replay['run']['ghost_rows'] > 0
    reasons = {r.finish_reason for r in overlap['results']}
    assert reasons == {'eos', 'length'}
    if max_iter > 1:
        depths = {d[3] for d in overlap['decisions']}
        assert len(depths) > 1 and any(d[4] for d in overlap['decisions'])


def test_overlap_forced_swap_and_restart_identical_to_replay(tmp_path):
    mixed_depth_reference(tmp_path, max_iter=3)
    eng = TaHEngine(tmp_path, device='cpu', kv_capacity=400, max_batch_size=3,
                    prefill_chunk_size=2, swap_chunk_entries=3, eos_token_id=None, overlap=True)
    requests = [Request([3, 7], 12, 81), Request([5, 8], 11, 12), Request([9], 13, 5), Request([4, 4], 9, 6)]
    params = SamplingParams(temperature=.8, top_k=20, top_p=.95)
    events = []
    def hook(engine, running, swapped, free_rows):
        step = engine.last_run['steps']
        if step == 4:
            victim = max(running, key=lambda s: s.index)
            engine._swap_out(victim, running, swapped, free_rows)
            events.append(('swap', victim.index))
        elif step == 9:
            victim = min(running, key=lambda s: len(s.generated))
            engine._restart(victim, running, swapped, free_rows, engine._restarted)
            events.append(('restart', victim.index))
    overlap, replay = overlap_vs_replay(eng, requests, params, hook)
    assert events[:2] == events[2:] and len(events) == 4
    assert_same_execution(overlap, replay)
    for trace in (overlap, replay):
        assert trace['run']['swap_outs'] == trace['run']['swap_ins'] == 1
        assert trace['run']['restarts'] == 1
    # Ordinary non-overlap (no ghost rows) remains a valid, different schedule.
    eng.overlap = False
    plain = eng.generate(requests, params)
    assert all(len(r.token_ids) == q.max_new_tokens for r, q in zip(plain, requests))


@pytest.mark.parametrize('host_budget', [0, 4800])
def test_overlap_overflow_restart_and_natural_swaps_complete(tmp_path, host_budget):
    mixed_depth_reference(tmp_path, max_iter=3)
    eng = TaHEngine(tmp_path, device='cpu', kv_capacity=50, max_batch_size=3,
                    prefill_chunk_size=2, swap_chunk_entries=3, eos_token_id=None,
                    admission_headroom=.1, host_swap_bytes=host_budget, overlap=True)
    requests = [Request([3, 7], 12, 100 + i) for i in range(6)]
    audits = []
    def bounded(engine, running, swapped, free_rows):
        assert engine.last_run['steps'] < 2000, 'restart livelock'
        # Every owned entry belongs to a live request; no trash slot is owned.
        audit = engine.pool.audit()
        assert engine.pool.trash_slot not in engine.pool._owners
        audits.append(audit['owned'])
    eng._step_hook = bounded
    callbacks = []
    results = eng.generate(requests, dict(temperature=.6, top_k=20, top_p=.95), callbacks.append)
    assert sorted(r.request_id for r in callbacks) == list(range(6))
    for result in results:
        assert result.finish_reason == 'length' and len(result.token_ids) == 12
        assert len(result.generated_pass_counts) == 12
    assert eng.last_run['restarts'] >= 1 or eng.last_run['swap_outs'] >= 1
    if not host_budget:
        assert eng.last_run['swap_outs'] == 0 and eng.last_run['restarts'] >= 1
    assert eng.pool.audit()['owned'] == 0 and eng.pool.free_slots == eng.pool.capacity
    assert eng.pool.audit()['allocations'] == eng.pool.audit()['frees']
    assert eng.swap.used_bytes == 0 and not eng._busy


def test_overlap_drains_lag_when_one_row_needs_both_slots(tmp_path):
    mixed_depth_reference(tmp_path, max_iter=3)
    request = Request([3, 5], 12, 4)
    eng = TaHEngine(tmp_path, device='cpu', kv_capacity=200, max_batch_size=1,
                    eos_token_id=None, admission_headroom=0, overlap=True)
    eng.overlap = False
    expected = eng.generate([request], dict(temperature=0))[0]
    eng.overlap = True
    stolen, needs = [], []
    def squeeze(engine, running, swapped, free_rows):
        # Leave a single free slot: enough for a resolved row, not for both
        # speculative slots; the pipeline must drain instead of failing.
        free = engine.pool._free
        if not needs and engine.last_run['steps'] >= 6 and engine._overlap_need(running) == 2:
            needs.append(engine.last_run['steps'])
            stolen.extend(free[:-1])
            del free[:-1]
        elif stolen and engine.last_run['steps'] > needs[0]:
            free[:0] = stolen
            stolen.clear()
    eng._step_hook = squeeze
    result = eng.generate([request], dict(temperature=0))[0]
    assert needs and eng.last_run['overlap_drains'] > 0
    assert result.token_ids == expected.token_ids
    assert result.generated_pass_counts == expected.generated_pass_counts
    assert eng.pool.free_slots == eng.pool.capacity


def test_overlap_option_validation_and_auto_default(tmp_path):
    with pytest.raises(ValueError, match='overlap must be None'):
        TaHEngine('unused-checkpoint', device='cpu', overlap=1)
    reference(tmp_path / 'static', max_iter=3, label=False)
    assert engine(tmp_path / 'static').overlap is True
    assert engine(tmp_path / 'static', overlap=False).overlap is False
    # Frequencies depending on the context length need the exact host length.
    reference(tmp_path / 'dynamic', max_iter=3, label=False, rope={'rope_type': 'dynamic', 'factor': 2.})
    assert engine(tmp_path / 'dynamic').overlap is False
    with pytest.raises(ValueError, match='static RoPE'):
        engine(tmp_path / 'dynamic', overlap=True)
