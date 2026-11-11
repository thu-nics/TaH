"""Bounded engine gates and workload/profile reports.

Run one checkpoint/gate per process. Production parity uses WP2's immutable
self-noise, with runtime/reference/checkpoint hashes recorded in every report.
"""
import argparse
from bisect import bisect_right
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import yaml

from tah.engine import TaHEngine, Request, SamplingParams
from tah.engine.fixtures import FixtureSuite
from tah.engine.parity import assert_gtf, compare_output, derive_decision_limits, gtf_summary, production_noise, require_production
from tah.engine.model import TeacherForcedOutput
from tah.engine.reference_capture import capture, divergence
from tah.engine.generate_fixtures import load_reference


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_graph_batch_sizes(value):
    if not value.strip():
        return []
    try:
        return [int(part.strip()) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('graph batch sizes must be comma-separated integers') from exc


def cuda_transfer_activity(name):
    # CUDA graphs can launch a real kernel named memcpy32_post. Substring
    # matching misclassifies it as a DMA copy; Kineto DMA names have a word
    # prefix ("Memcpy HtoD ...", "Memset (Device) ...").
    return name.split(' ', 1)[0].lower() in ('memcpy', 'memset')


def trace_profile_totals(trace):
    """Exclusive physical CUDA work and CPU scope duration from a saved trace.

    GPU user-annotation envelopes and nested CPU operation totals are excluded.
    This also permits auditing a report without executing the model again.
    """
    steps = [e for e in trace if e.get('cat') == 'user_annotation'
             and e.get('name') == 'engine_decode_step']
    spans = [(e['ts'], e['ts'] + e['dur']) for e in steps]
    work = [e for e in trace if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
            and any(a <= e['ts'] and e['ts'] + e['dur'] <= b for a, b in spans)]
    return dict(steps=len(steps), wall_us=sum(e['dur'] for e in steps),
                kernel_us=sum(e['dur'] for e in work if e['cat'] == 'kernel'),
                copy_us=sum(e['dur'] for e in work if e['cat'] != 'kernel'), activities=len(work))


class Observer:
    """Bounded device->pinned-host tracing; decode adds NO synchronization."""
    def __init__(self, engine):
        self.engine, self.records = engine, {}

    def host(self, tensor):
        out = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=tensor.is_cuda)
        out.copy_(tensor, non_blocking=True)
        return out

    def prefill(self, state, start, output):
        rows = self.records.setdefault(state.index, [])
        for pos in range(output.logits.shape[0]):
            rows.append((start + pos, self.host(output.logits[pos]),
                         self.host(output.visited[:, pos]), self.host(output.decision[:, pos]),
                         self.host(output.decision_logits[:, pos]), self.host(output.continue_probability[:, pos])))

    def decode(self, states, bucket, status):
        for row, state in enumerate(states):
            self.records[state.index].append((state.position, self.host(bucket.logits[row]),
                                              state.depth, self.host(status[row])))

    def output(self, index, length):
        n, v = self.engine.model.max_iter, self.engine.model.config['vocab_size']
        logits = torch.empty(length, v, dtype=self.engine.model.dtype)
        visited = torch.zeros(n, length, dtype=torch.bool)
        decisions = torch.zeros_like(visited)
        raw = torch.full((n, length), float('nan'), dtype=self.engine.model.dtype)
        prob = torch.full_like(raw, float('nan'))
        for record in self.records[index]:
            pos, values = record[:2]
            logits[pos] = values
            if len(record) == 6:
                visited[:, pos], decisions[:, pos], raw[:, pos], prob[:, pos] = record[2:]
            else:
                depth, status = record[2:]
                visited[depth - 1, pos] = True
                decisions[depth - 1, pos] = bool(status[0])
                raw[depth - 1, pos], prob[depth - 1, pos] = status[2], status[3]
        return TeacherForcedOutput(logits, visited.sum(0), visited, decisions, raw, prob,
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(n, 0, 0, 0))


def record_decisions(engine, decisions):
    """Collect (index, position, depth, keep) of executed live rows in either scheduler."""
    if engine.overlap:
        def resolved(record, rows):
            decisions.extend((index, pos, depth, keep) for _, index, pos, depth, keep in rows)
        engine._resolved_hook = resolved
        return lambda: setattr(engine, '_resolved_hook', None)
    decode = engine._decode
    def observed(running):
        status = decode(running)
        decisions.extend((s.index, s.position, s.depth, bool(row[0])) for s, row in zip(running, status))
        return status
    engine._decode = observed
    return lambda: setattr(engine, '_decode', decode)


def fixture_setup(name):
    suite = FixtureSuite('data/engine/fixtures/reference')
    require_production(suite)
    meta = suite.manifest['checkpoints'][name]
    root = Path(meta['path'])
    for file, info in meta['files'].items():
        assert digest(root / file) == info['sha256'], f'checkpoint changed: {file}'
    for file, value in suite.manifest['reference_source_sha256'].items():
        assert digest(file) == value, f'reference changed: {file}'
    eval_config = Path(suite.manifest['eval_config']['path'])
    assert digest(eval_config) == suite.manifest['eval_config']['sha256']
    override = yaml.safe_load(eval_config.read_text()) if name == 'real' else None
    noise = production_noise(suite, name, json.loads(Path('data/engine/wp2/tolerances.json').read_text()))
    records = [r for r in suite.records(kind='greedy', checkpoint=name) if r['variant'] == 'single']
    assert len(records) == 6
    prompts = [suite.load(r)['prompt_ids'].tolist() for r in records]
    return suite, root, override, noise, records, prompts, eval_config


def parity(args):
    suite, root, override, noise, records, prompts, eval_config = fixture_setup(args.checkpoint)
    engine = TaHEngine(root, override, max_batch_size=6, prefill_chunk_size=128, kv_capacity=20000,
                      cuda_graphs=args.cuda_graphs, compile=args.compile,
                      compile_cache_dir=args.compile_cache_dir,
                      graph_batch_sizes=args.graph_batch_sizes, overlap=args.overlap)
    observer = Observer(engine) if args.gate == 'tf' else None
    engine._observer = observer
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95) if args.gate == 'tf' else SamplingParams(temperature=0)
    results = engine.generate([Request(p, 256, i) for i, p in enumerate(prompts)], params)
    report = dict(gate='G-TF' if observer else 'G-GEN', checkpoint=args.checkpoint,
        production_gates_eligible=True, engine_run=engine.last_run,
        results=[asdict(r) for r in results], manifest_sha256=digest(suite.root / 'manifest.json'),
        wp2_tolerances_sha256=digest('data/engine/wp2/tolerances.json'),
        decision_limits=derive_decision_limits(noise))
    failure = None
    if observer:
        outputs = [observer.output(i, len(prompt) + max(0, len(result.token_ids) - (result.finish_reason == 'length')))
                   for i, (prompt, result) in enumerate(zip(prompts, results))]
        engine._observer = None
        del observer
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        oracle = load_reference(SimpleNamespace(device='cuda', eval_config=eval_config), args.checkpoint, root)
        rows = []
        for i, (prompt, result, output) in enumerate(zip(prompts, results, outputs)):
            sequence = prompt + result.token_ids[:output.logits.shape[0] - len(prompt)]
            # Replay actual engine-generated sequences with WP2 tooling. No
            # engine-vs-engine/cross-layout proxy can establish this gate.
            boundaries = list(range(0, len(prompt), 128)) + list(range(len(prompt), len(sequence) + 1))
            trace, logits = capture(oracle, [sequence], pad_id=oracle.config.eos_token_id,
                                    chunk_boundaries=boundaries)
            row = compare_output(trace[0], output, logits[0], suite.manifest['checkpoints'][args.checkpoint]['threshold'])
            row.update(sequence_id=records[i]['sequence_id'], generated_sequence=True)
            rows.append(row)
            print(f'G-TF {args.checkpoint}: replay {i + 1}/6', flush=True)
        report.update(summary=gtf_summary(rows), comparisons=rows, noise_limits=noise,
                      reference_layout='singleton prompt chunks128, then cached one-token chunks')
        try:
            assert_gtf(rows, noise)
        except AssertionError as exc:
            failure = str(exc)
    else:
        rows = []
        for record, result in zip(records, results):
            baseline = suite.load(record)['generated_ids'].tolist()
            d = divergence(baseline, result.token_ids)
            rows.append(dict(sequence_id=record['sequence_id'], first_divergence=d,
                             generation_length=len(baseline), effective_divergence=len(baseline) if d is None else d))
        baseline_lengths = {r['sequence_id']: len(suite.load(r)['generated_ids']) for r in records}
        reference = suite.manifest['self_noise'][args.checkpoint]['greedy']['comparisons']
        ref_positions = [baseline_lengths[r['sequence_id']] if r['first_divergence'] is None else r['first_divergence'] for r in reference]
        candidate_median, reference_median = statistics.median(r['effective_divergence'] for r in rows), statistics.median(ref_positions)
        report.update(comparisons=rows, median_first_divergence=candidate_median,
            reference_median_first_divergence=reference_median, median_limit=.5 * reference_median)
        if candidate_median < .5 * reference_median or (any(r['effective_divergence'] == 0 for r in rows) and 0 not in ref_positions):
            failure = 'G-GEN first-divergence bound failed'
    report['status'] = 'FAIL' if failure else 'PASS'
    if failure:
        report['failure'] = failure
    return report


# Host swap arena for gates whose swaps are bounded: smoke and overlap-identity run at
# most 16 resident W-smoke requests (prompt <= 347 tokens, cap <= 1024, max_iter 2:
# <= 2742 entries = 0.29 GiB each, <= 44k in all, far below the full pool), so they
# never swap naturally; swap, phase-profile and overlap-identity force one swap of one
# such request. Gates that can
# swap naturally (workload, parity, profile, overlap) keep the production 8 GiB.
# The arena is pinned at engine construction; each gate builds one engine.
BOUNDED_SWAP_BYTES = 1024**3


def workload_prompts(model_path='models/TaH-plus-1.7B', repeat=2):
    baseline = Path('data/engine/wperf_bs16/math500_tah/20261011_155619/TASK_0_64/samples.jsonl')
    if baseline.exists():
        rows = [json.loads(line) for line in baseline.read_text().splitlines()]
        questions = {row['id']: row['problem'] for row in rows}
    else:
        from tah.evaluate.datasets import load_combined_dataset
        rows, _ = load_combined_dataset('math500')
        questions = {row['id']: row['question'] for row in rows[:64]}
    assert len(questions) == 64
    # Exactly the reference evaluation chat template, rendered with its saved
    # tokenizer only in offline workload preparation (not engine runtime).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return [tokenizer.apply_chat_template([dict(role='user', content=questions[f'math500_{i}'])],
                                        tokenize=True, add_generation_prompt=True)
            for i in range(64) for _ in range(repeat)]


def workload(args):
    prompts = workload_prompts(args.model_path)[args.start:args.end]
    if args.gate == 'long-prompt':
        prompts = [(prompts[0] * (16384 // len(prompts[0]) + 1))[:16384]] + prompts[1:33]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine(args.model_path, override, max_batch_size=args.batch_size,
                      prefill_chunk_size=128, profile_steps=0, memory_fraction=args.memory_fraction,
                      cuda_graphs=args.cuda_graphs, compile=args.compile,
                      compile_cache_dir=args.compile_cache_dir,
                      graph_batch_sizes=args.graph_batch_sizes, overlap=args.overlap)
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    # Load/warmup excluded; exercise actual maximum-B linear and bounded sampler.
    engine.generate([Request(prompts[i % len(prompts)], 4, i) for i in range(args.batch_size)], params)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    requests = [Request(p, args.max_new_tokens, i + args.start) for i, p in enumerate(prompts)]
    kv_high_water = engine.pool.capacity - engine.pool.free_slots
    allocate = engine.pool.allocate
    def measured_allocate(*args, **kwargs):
        nonlocal kv_high_water
        result = allocate(*args, **kwargs)
        kv_high_water = max(kv_high_water, engine.pool.capacity - engine.pool.free_slots)
        return result
    engine.pool.allocate = measured_allocate
    completed = []
    def done(result):
        completed.append(result)
        print(f'{args.gate} completed {len(completed)}/{len(requests)}, tokens={sum(len(r.token_ids) for r in completed)}', flush=True)
    results = engine.generate(requests, params, done)
    report = dict(status='PASS', workload='W-perf' if args.gate == 'perf' else 'G-ROBUST',
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, request_range=[args.start, args.end],
        requests=len(requests), total_prompt_tokens=sum(map(len, prompts)),
        memory_budget=engine.memory_budget, run=engine.last_run,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        kv_audit=engine.pool.audit(), host_swap_used_bytes=engine.swap.used_bytes,
        kv_pool_high_water_entries=kv_high_water,
        wp0_baseline_tokens_per_second=78.26, speedup=engine.last_run['tokens_per_second'] / 78.26,
        results=[asdict(r) for r in results])
    assert report['kv_audit']['free'] == engine.pool.capacity and report['host_swap_used_bytes'] == 0
    return report


def swap_check(args):
    """Production-weight forced swap with the same batch composition."""
    if args.overlap:
        raise ValueError('overlap forced swap/restart identity is --gate overlap-identity')
    prompts = workload_prompts(args.model_path)[:4]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine(args.model_path, override, max_batch_size=4, kv_capacity=10000,
                      host_swap_bytes=BOUNDED_SWAP_BYTES, swap_chunk_entries=7, prefill_chunk_size=128, cuda_graphs=args.cuda_graphs,
                      compile=args.compile, compile_cache_dir=args.compile_cache_dir,
                      graph_batch_sizes=args.graph_batch_sizes, overlap=args.overlap)
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    requests = [Request(p, 128, i) for i, p in enumerate(prompts)]
    decisions = []
    original_decode = engine._decode
    def observed_decode(running):
        status = original_decode(running)
        decisions.extend((s.index, s.position, s.depth, bool(row[0])) for s, row in zip(running, status))
        return status
    engine._decode = observed_decode
    engine.cuda_graphs = False
    expected = engine.generate(requests, params)
    expected_decisions = decisions.copy()
    decisions.clear()
    engine.cuda_graphs = args.cuda_graphs
    checks = []
    def hook(engine, running, swapped, free_rows):
        if engine.last_run['steps'] != 32:
            return
        state = running[-1]
        original_state = (state.position, state.depth, state.generated.copy(), state.generated_counts.copy())
        engine._swap_out(state, running, swapped, free_rows)
        host_bytes = engine.swap.used_bytes
        assert host_bytes > 0 and state.cache is None
        engine._restore(running, swapped, free_rows)
        assert not swapped and engine.swap.used_bytes == 0
        assert original_state == (state.position, state.depth, state.generated, state.generated_counts)
        checks.append(dict(position=state.position, depth=state.depth, host_bytes=host_bytes))
    engine._step_hook = hook
    actual = engine.generate(requests, params)
    assert len(checks) == 1
    assert decisions == expected_decisions
    assert [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in actual] == [
        (r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in expected]
    return dict(status='PASS', gate='G-ROBUST forced swap identity', requests=len(requests),
                output_tokens=sum(len(r.token_ids) for r in actual), checks=checks,
                run=engine.last_run, kv_audit=engine.pool.audit(), host_swap_used_bytes=engine.swap.used_bytes,
                decision_rows=len(decisions), decisions_identical=True, memory_budget=engine.memory_budget,
                results=[asdict(r) for r in actual])


def smoke(args):
    """W-smoke graph/eager identity, sharing capacity and request composition."""
    prompts = workload_prompts(args.model_path, repeat=1)[:16]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine(args.model_path, override, max_batch_size=args.batch_size,
                      memory_fraction=args.memory_fraction, cuda_graphs=args.cuda_graphs,
                      compile=args.compile, compile_cache_dir=args.compile_cache_dir,
                      graph_batch_sizes=args.graph_batch_sizes, overlap=args.overlap,
                      host_swap_bytes=BOUNDED_SWAP_BYTES)
    requests = [Request(p, 1024, i) for i, p in enumerate(prompts)]
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    decisions = []
    record_decisions(engine, decisions)
    engine.cuda_graphs = False
    expected = engine.generate(requests, params)
    eager_run, expected_decisions = engine.last_run, decisions.copy()
    decisions.clear()
    engine.cuda_graphs = args.cuda_graphs
    actual = engine.generate(requests, params)
    assert decisions == expected_decisions, 'graph/eager decode decisions differ'
    assert [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts, r.finish_reason) for r in actual] == [
        (r.token_ids, r.prompt_pass_counts, r.generated_pass_counts, r.finish_reason) for r in expected]
    if args.cuda_graphs:
        assert engine.last_run['graph_steps'] == engine.last_run['steps'] > 0
    assert engine.pool.audit()['free'] == engine.pool.capacity and engine.swap.used_bytes == 0
    return dict(status='PASS', workload='W-smoke', requests=16, max_new_tokens=1024,
        tokens_identical=True, decisions_identical=True, pass_counts_identical=True,
        decision_rows=len(decisions), decision_sha256=hashlib.sha256(json.dumps(decisions).encode()).hexdigest(),
        eager_run=eager_run, run=engine.last_run, memory_budget=engine.memory_budget,
        kv_audit=engine.pool.audit(), results=[asdict(r) for r in actual])


def overlap_identity(args):
    """Production overlap vs non-overlap replay of the same per-step row sets.

    24 W-smoke prompts at resident B16 (admissions follow finishes), varied caps
    (finishes at different steps, EOS enabled), one forced swap and one forced
    restart. A plain non-overlap run is reported as a composition diagnostic.
    """
    from tah.engine.tests.overlap_replay import assert_same_execution, overlap_vs_replay
    prompts = workload_prompts(args.model_path, repeat=1)[:24]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine(args.model_path, override, max_batch_size=16, host_swap_bytes=BOUNDED_SWAP_BYTES,
                      memory_fraction=args.memory_fraction, cuda_graphs=args.cuda_graphs,
                      compile=args.compile, compile_cache_dir=args.compile_cache_dir,
                      graph_batch_sizes=args.graph_batch_sizes, overlap=True)
    requests = [Request(p, 256 + 64 * (i % 8), i) for i, p in enumerate(prompts)]
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    events = []
    def hook(engine, running, swapped, free_rows):
        step = engine.last_run['steps']
        if step == 40:
            victim = max(running, key=lambda s: s.index)
            engine._swap_out(victim, running, swapped, free_rows)
            events.append(('swap', step, victim.index))
        elif step == 160:
            victim = min(running, key=lambda s: len(s.generated))
            engine._restart(victim, running, swapped, free_rows, engine._restarted)
            events.append(('restart', step, victim.index))
    overlap, replay = overlap_vs_replay(engine, requests, params, hook)
    assert len(events) == 4 and events[:2] == events[2:], events
    assert_same_execution(overlap, replay)
    if args.cuda_graphs:
        assert overlap['run']['graph_steps'] == overlap['run']['steps']
    engine.overlap = False
    plain = engine.generate(requests, params)
    plain_run = engine.last_run
    engine.overlap = True
    first = [divergence(a.token_ids, b.token_ids) for a, b in zip(overlap['results'], plain)]
    return dict(status='PASS', gate='overlap identity vs replayed row sets', requests=len(requests),
        forced_events=events[:2], steps=len(overlap['rows']), decision_rows=len(overlap['decisions']),
        output_tokens=sum(len(r.token_ids) for r in overlap['results']),
        ghost_rows=overlap['run']['ghost_rows'], finish_reasons=sorted({r.finish_reason for r in overlap['results']}),
        rows_identical=True, decisions_identical=True, tokens_counts_identical=True,
        decision_sha256=hashlib.sha256(json.dumps(overlap['decisions']).encode()).hexdigest(),
        overlap_run=overlap['run'], replay_run=replay['run'], plain_nonoverlap_run=plain_run,
        plain_nonoverlap_first_divergence=first,
        plain_nonoverlap_identical_requests=sum(d is None for d in first),
        kv_audit=engine.pool.audit(), host_swap_used_bytes=engine.swap.used_bytes,
        memory_budget=engine.memory_budget, results=[asdict(r) for r in overlap['results']])


def phase_profile(args):
    """Short real admission/prefill and forced swap launch-gap diagnostic."""
    from torch.profiler import record_function
    prompts = workload_prompts(args.model_path)[:4]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine(args.model_path, override, max_batch_size=4, kv_capacity=10000,
                      host_swap_bytes=BOUNDED_SWAP_BYTES, swap_chunk_entries=256, cuda_graphs=args.cuda_graphs,
                      compile=args.compile, compile_cache_dir=args.compile_cache_dir,
                      graph_batch_sizes=args.graph_batch_sizes, overlap=args.overlap)
    requests = [Request(p, 48, i) for i, p in enumerate(prompts)]
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    timings = dict(prefill=[], swap_out=[], swap_in=[], decode=[])
    for name, owner, method in [('prefill', engine, '_prefill'), ('swap_out', engine.swap, 'out'),
                                ('swap_in', engine.swap, 'restore'), ('decode', engine, '_decode')]:
        original = getattr(owner, method)
        def wrapped(*a, _name=name, _fn=original, **kw):
            with record_function('engine_phase_' + _name):
                start = time.perf_counter()
                result = _fn(*a, **kw)
                timings[_name].append((time.perf_counter() - start) * 1000)
                return result
        setattr(owner, method, wrapped)
    def hook(engine, running, swapped, free_rows):
        if engine.last_run['steps'] == 16:
            engine._swap_out(running[-1], running, swapped, free_rows)
            engine._restore(running, swapped, free_rows)
    engine._step_hook = hook
    engine.generate(requests, params)
    for samples in timings.values():
        samples.clear()
    engine.generate(requests, params)
    unprofiled = {name: values.copy() for name, values in timings.items()}
    for samples in timings.values():
        samples.clear()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        engine.generate(requests, params)
    trace_path = args.output.with_suffix('.trace.json')
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())['traceEvents']
    physical = [e for e in trace if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    phases = {}
    for name, samples in unprofiled.items():
        scopes = [e for e in trace if e.get('cat') == 'user_annotation' and e.get('name') == 'engine_phase_' + name]
        assert len(scopes) == len(samples) > 0
        spans = [(e['ts'], e['ts'] + e['dur']) for e in scopes]
        # Swap transfers may finish after their CPU issue scope. Retain that
        # distinction instead of incorrectly treating the gap as GPU idle.
        work = [e for e in physical if any(a <= e['ts'] and e['ts'] + e['dur'] <= b for a, b in spans)]
        device_ms = sum(e['dur'] for e in work) / 1000 / len(samples)
        phases[name] = dict(calls=len(samples), unprofiled_mean_host_ms=statistics.mean(samples),
            profiled_mean_host_ms=sum(e['dur'] for e in scopes) / 1000 / len(samples),
            mean_kernel_ms=sum(e['dur'] for e in work if e['cat'] == 'kernel') / 1000 / len(samples),
            mean_copy_ms=sum(e['dur'] for e in work if e['cat'] != 'kernel') / 1000 / len(samples),
            estimated_host_minus_in_scope_device_ms=statistics.mean(samples) - device_ms)
    return dict(status='PASS', phases=phases, run=engine.last_run, trace_file=str(trace_path),
        trace_sha256=digest(trace_path), note='Scope wall timings include host admission/plan/issue work. '
        'Decode/prefill finish with status copies; swap is asynchronous, so device work can spill into the following scope. '
        'Negative residuals reflect profiler duration distortion/overlap and are not negative idle time.')


def overlap_probe(engine, requests, params):
    """Measure branch-independent next-step host drafts during live GPU work.

    Never use an assumed decision to mutate KV or state. Both outcomes are
    prepared on the host only; actual admission/allocation still waits for t.
    """
    original = engine.graphs.run
    samples = []
    def run(bucket, engine, rotary):
        measured = engine.last_run['steps'] < 11
        if not measured:
            return original(bucket, engine, rotary)
        # Compare the SAME prepared step, not two independent prefills whose
        # paged layouts/allocator histories differ. These are the only mutable
        # step inputs; global state and current KV writes are overwritten by
        # the second replay, which becomes the actual scheduler result.
        logits, counters = bucket.logits.clone(), bucket.rng.counters.clone()
        baseline = original(bucket, engine, rotary).clone()
        baseline_done = torch.cuda.Event()
        baseline_done.record()
        while not baseline_done.query():
            time.sleep(.0001)
        bucket.logits.copy_(logits)
        bucket.rng.counters.copy_(counters)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(2)] if measured else None
        if events:
            events[0].record()
        status = original(bucket, engine, rotary)
        if events:
            events[1].record()
            ready_before = events[1].query()
            start = time.perf_counter()
            drafts = []
            for continued in (False, True):
                offset = 0
                for row in bucket.attention.rows:
                    keep = continued and row.depth < engine.model.max_iter
                    depth, position = (row.depth + 1, row.position) if keep else (1, row.position + 1)
                    tables = row.request.streams
                    lengths = [t.size + (d == depth - 1) for d, t in enumerate(tables)]
                    visible = sum(bisect_right(t.host_positions, position) for t in tables[:depth]) + 1
                    drafts.append([*[t.slots.data_ptr() for t in tables],
                                   *[t.positions.data_ptr() for t in tables], *lengths,
                                   offset, sum(lengths), position, depth, visible])
                    offset += sum(lengths)
            # The same pinned descriptor construction used by real planning;
            # no H2D upload or shared step-buffer write happens here.
            host = torch.tensor(drafts, dtype=torch.long, pin_memory=True)
            duration = (time.perf_counter() - start) * 1000
            ready_after = events[1].query()
            parity = ((baseline == status) | (baseline.isnan() & status.isnan())).all(0)
            samples.append((duration, ready_before, ready_after, events,
                            host.numel() * host.element_size(), parity, events[0].query()))
        return status
    engine.graphs.run = run
    previous_profile_steps = engine.profile_steps
    engine.profile_steps = 0
    try:
        engine.generate(requests, params)
    finally:
        engine.graphs.run = original
        engine.profile_steps = previous_profile_steps
    parity = torch.stack([s[5] for s in samples]).cpu().tolist()
    assert all(all(fields) for fields in parity), parity
    return dict(steps=len(samples), mean_host_draft_ms=statistics.mean(s[0] for s in samples),
        mean_gpu_step_ms=statistics.mean(s[3][0].elapsed_time(s[3][1]) for s in samples),
        gpu_pending_before=sum(not s[1] for s in samples), gpu_pending_after=sum(not s[2] for s in samples),
        gpu_started_by_draft_end=sum(s[6] for s in samples),
        descriptor_bytes_per_step=[s[4] for s in samples],
        same_step_status_identity=parity,
        parity_control='identical prepared inputs, counters, positions/depths and KV layout; exact keep/token/raw/probability',
        note='Two candidate host descriptor/count drafts overlap GPU step t. They are discarded: '
        'actual next depths/positions, EOS/removal, admission/preemption and table growth depend on its status. '
        'Using these drafts for a full next step would require additional buffer/table lifetime work; no runtime lookahead is enabled.')


def profile(args):
    prompts = workload_prompts(args.model_path)
    if args.profile_context:
        prompts = [(p * (args.profile_context // len(p) + 1))[:args.profile_context] for p in prompts]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine(args.model_path, override, max_batch_size=args.batch_size,
                      prefill_chunk_size=128, profile_steps=0, admission_headroom=0, cuda_graphs=args.cuda_graphs,
                      compile=args.compile, compile_cache_dir=args.compile_cache_dir,
                      graph_batch_sizes=args.graph_batch_sizes, overlap=args.overlap)
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    requests = [Request(prompts[i % len(prompts)], 12, i) for i in range(args.batch_size)]
    engine.generate(requests, params)
    # Overlap: one scope = one pipeline turn (consume lagged status, schedule,
    # plan+launch). Turns are back to back, so scope wall = step period.
    step_name = '_overlap_iteration' if engine.overlap else '_decode'
    original = getattr(engine, step_name)
    from torch.profiler import record_function
    def decode(*a):
        with record_function('engine_decode_step'):
            return original(*a)
    setattr(engine, step_name, decode)
    # Uninstrumented timing samples provide the step wall envelope before the
    # profiler's CPU instrumentation increases launch overhead.
    engine.profile_steps = 100
    engine.generate(requests, params)
    unprofiled = engine.last_run['profile']
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
    started = False
    closed = False
    count = 0
    window = []
    def profiled_decode(*a):
        nonlocal started, closed, count
        # Starting at the first decode avoids recording the large, irrelevant
        # prefill trace (and keeps the profiler's host memory bounded).
        if engine.overlap and not started and engine.last_run['steps'] < 2:
            return original(*a)  # overlap: start once the pipeline is primed
        if not started:
            prof.__enter__()
            started = True
        if engine.overlap:
            window.append(engine.last_run['steps'])
        result = decode(*a)
        count += 1
        if count == (10 if engine.overlap else 11):  # overlap: steps 2..11, before the 12-token tail
            prof.__exit__(None, None, None)
            closed = True
            setattr(engine, step_name, decode)
        return result
    setattr(engine, step_name, profiled_decode)
    try:
        engine.generate(requests, params)
    finally:
        if started and not closed:
            prof.__exit__(None, None, None)
    # Work directly from raw Kineto activities. Expanding CUDA graph nodes
    # into Python FunctionEvent trees can use gigabytes and add no information
    # to the physical kernel/copy accounting needed here.
    raw_events = prof.profiler.kineto_results.events()
    steps = [e for e in raw_events if e.name() == 'engine_decode_step'
             and e.device_type() == torch.autograd.DeviceType.CPU]
    def covered(start, end, scopes):
        return any(a <= start and end <= b for a, b in scopes)
    scopes = [(e.start_ns(), e.end_ns()) for e in steps]
    # CPU inclusive device totals can duplicate activities (including an
    # "Unrecognized" CPU event) and include GPU user-annotation envelopes.
    # Count actual CUDA events once; annotation spans are NOT kernels.
    device_events = [e for e in raw_events if e.device_type() == torch.autograd.DeviceType.CUDA
                     and not e.is_user_annotation() and covered(e.start_ns(), e.end_ns(), scopes)]
    gpu_us = sum((e.end_ns() - e.start_ns()) / 1000 for e in device_events)
    kernel_us = sum((e.end_ns() - e.start_ns()) / 1000 for e in device_events
                    if not cuda_transfer_activity(e.name()))
    transfer_us = gpu_us - kernel_us
    wall_us = sum((s.end_ns() - s.start_ns()) / 1000 for s in steps)
    assert 0 < kernel_us <= gpu_us <= wall_us
    if engine.overlap:
        # Launch ids of the profiled turns; same deterministic schedule unprofiled.
        def pick(entries):
            by_step = {e['step']: dict(e, total_ms=e['iteration_ms']) for e in entries}
            return [by_step[i] for i in window]
        unprofiled = pick(unprofiled)
        engine.last_run['profile'] = pick(engine.last_run['profile'])
    unprofiled = unprofiled[:len(steps)]
    observed = engine.last_run['profile'][:len(steps)]
    assert [p['batch_size'] for p in unprofiled] == [p['batch_size'] for p in observed]
    unprofiled_mean = statistics.mean(p['total_ms'] for p in unprofiled)
    grouped = {}
    for event in device_events:
        name = event.name()
        grouped[name] = grouped.get(name, 0.) + (event.end_ns() - event.start_ns()) / 1000
    trace_path = args.output.with_suffix('.trace.json')
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())['traceEvents']
    trace_steps = [(e['ts'], e['ts'] + e['dur']) for e in trace
                   if e.get('cat') == 'user_annotation' and e.get('name') == 'engine_decode_step']
    assert len(trace_steps) == len(steps)
    trace_device = [e for e in trace if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
                    and covered(e['ts'], e['ts'] + e['dur'], trace_steps)]
    trace_gpu_us = sum(e['dur'] for e in trace_device)
    args.output.with_suffix('.timings.json').write_text(json.dumps(dict(
        unprofiled_event_profile=unprofiled, event_profile=engine.last_run['profile'],
        memory_budget=engine.memory_budget, run=engine.last_run,
        raw_gpu_us=gpu_us, raw_kernel_us=kernel_us, raw_wall_us=wall_us,
        raw_cuda_activities=len(device_events), trace_cuda_activities=len(trace_device)), indent=2))
    assert len(trace_device) == len(device_events)
    assert abs(trace_gpu_us - gpu_us) < .1, (trace_gpu_us, gpu_us)
    audit = trace_profile_totals(trace)
    assert audit['steps'] == len(steps)
    assert abs(audit['kernel_us'] - kernel_us) < .1
    assert abs(audit['wall_us'] - wall_us) < .1
    report = dict(status='PASS', batch_size=args.batch_size, context=args.profile_context,
        memory_budget=engine.memory_budget, run=engine.last_run,
        context_source='repeated real workload prompts; actual model prefill', admission_headroom=0,
        observed_batch_sizes=[p['batch_size'] for p in engine.last_run['profile'][:len(steps)]], decode_steps=len(steps),
        mean_step_wall_ms=wall_us / len(steps) / 1000,
        mean_kernel_gpu_ms=kernel_us / len(steps) / 1000,
        mean_gpu_copy_ms=transfer_us / len(steps) / 1000,
        mean_python_launch_and_gpu_idle_ms=(wall_us - gpu_us) / len(steps) / 1000,
        profiler_overhead_included=True, event_profile=engine.last_run['profile'],
        unprofiled_mean_step_ms=unprofiled_mean,
        estimated_unprofiled_python_launch_idle_ms=unprofiled_mean - gpu_us / len(steps) / 1000,
        unprofiled_event_profile=unprofiled,
        verification=dict(actual_cuda_device_us=gpu_us, trace_cuda_device_us=trace_gpu_us,
                          actual_cuda_activities=len(device_events), kernel_us=kernel_us,
                          cpu_step_scopes=len(steps), source_trace_sha256=digest(trace_path),
                          raw_kineto_activities=len(raw_events),
                          accounting='raw Kineto physical CUDA activities cross-checked against Chrome trace',
                          trace_file=str(trace_path)),
        top_gpu_kernels=[dict(name=name, total_gpu_ms=duration / 1000)
            for name, duration in sorted(grouped.items(), key=lambda pair: pair[1], reverse=True)[:20]],
        note='GPU kernel sum is device execution time; stream events include CPU launch gaps. The unprofiled residual uses device durations from an equivalent profiled run and is an estimate. CPU and GPU overlap; residuals are not additive CPU utilization. The profiled residual includes profiler overhead.')
    if engine.overlap:
        report['overlap_host_ms'] = {key: statistics.mean(p[key] for p in unprofiled) for key in
            ('iteration_ms', 'consume_wait_host_ms', 'schedule_host_ms', 'prepare_host_ms', 'launch_host_ms', 'gpu_step_ms')}
    elif args.cuda_graphs and report['estimated_unprofiled_python_launch_idle_ms'] > 2:
        report['overlap_probe'] = overlap_probe(engine, requests, params)
    return report


def overlap_check(args):
    if not args.cuda_graphs:
        raise ValueError('overlap diagnostic requires --cuda-graphs')
    prompts = workload_prompts(args.model_path)
    if args.profile_context:
        prompts = [(p * (args.profile_context // len(p) + 1))[:args.profile_context] for p in prompts]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine(args.model_path, override, max_batch_size=args.batch_size,
                      admission_headroom=0, cuda_graphs=True, compile=args.compile,
                      compile_cache_dir=args.compile_cache_dir,
                      graph_batch_sizes=args.graph_batch_sizes, overlap=args.overlap)
    requests = [Request(prompts[i % len(prompts)], 12, i) for i in range(args.batch_size)]
    result = overlap_probe(engine, requests, SamplingParams(temperature=.6, top_k=20, top_p=.95))
    return dict(status='PASS', overlap_probe=result, memory_budget=engine.memory_budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', choices=['gen', 'tf', 'smoke', 'perf', 'robust', 'long-prompt', 'swap', 'profile', 'phase-profile', 'overlap', 'overlap-identity'], required=True)
    parser.add_argument('--checkpoint', choices=['real', 'mlp3', 'mlp1', 'label3'], default='real')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=128)
    parser.add_argument('--memory-fraction', type=float, default=.9)
    parser.add_argument('--profile-context', type=int, default=2048)
    parser.add_argument('--model-path', type=Path, default=Path('models/TaH-plus-1.7B'))
    parser.add_argument('--cuda-graphs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False,
                        help='fuse decode with TorchInductor and reuse its persistent disk cache')
    parser.add_argument('--overlap', action=argparse.BooleanOptionalAction, default=False,
                        help='pipelined scheduler: plan step t+1 while step t runs (one-step status lag)')
    parser.add_argument('--compile-cache-dir', type=Path,
                        help='TorchInductor disk cache (default: TORCHINDUCTOR_CACHE_DIR or ~/.cache/torch/inductor)')
    parser.add_argument('--graph-batch-sizes', type=parse_graph_batch_sizes,
                        help='comma-separated exact batch sizes to capture, e.g. 54,55,58,60,64')
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.perf_counter()
    report = (parity(args) if args.gate in ('gen', 'tf') else profile(args) if args.gate == 'profile'
              else swap_check(args) if args.gate == 'swap' else smoke(args) if args.gate == 'smoke'
              else overlap_check(args) if args.gate == 'overlap'
              else overlap_identity(args) if args.gate == 'overlap-identity'
              else phase_profile(args) if args.gate == 'phase-profile' else workload(args))
    report.update(process_seconds=time.perf_counter() - start, device=torch.cuda.get_device_name(), cuda_graphs=args.cuda_graphs,
        compile=args.compile, overlap=args.overlap,
        compile_cache_dir=str(args.compile_cache_dir) if args.compile_cache_dir else None,
        graph_batch_sizes=args.graph_batch_sizes,
        runtime_sha256={str(p): digest(p) for p in Path('tah/engine').glob('*.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k not in ('runtime_sha256', 'results', 'comparisons', 'noise_limits', 'engine_run')}, flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
