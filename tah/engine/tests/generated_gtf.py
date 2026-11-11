"""Corpus-matched generated G-TF, in ordered generate/noise/compare phases."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from tah.engine import TaHEngine, Request, SamplingParams
from tah.engine.generate_fixtures import load_reference
from tah.engine.model import TeacherForcedOutput
from tah.engine.parity import assert_gtf, compare_output, decision_metrics, derive_decision_limits, gtf_summary
from tah.engine.reference_capture import capture, compare
from tah.engine.tests.engine_check import Observer, digest, fixture_setup


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def model_provenance(suite, name):
    return dict(manifest_sha256=digest(suite.root / 'manifest.json'),
        checkpoint_files=suite.manifest['checkpoints'][name]['files'],
        reference_source_sha256=suite.manifest['reference_source_sha256'],
        runtime_sha256={str(p): digest(p) for p in Path('tah/engine').glob('*.py')})


def generate(args):
    suite, root, override, _, records, prompts, _ = fixture_setup(args.checkpoint)
    destination = args.root / 'gtf-corpus' / args.checkpoint
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        raise ValueError('corpus already exists; preserve it and use another --root')
    engine = TaHEngine(root, override, max_batch_size=6, prefill_chunk_size=128, kv_capacity=20000,
                       compile=args.compile, compile_cache_dir=args.compile_cache_dir,
                       overlap=args.overlap)
    manifest = dict(checkpoint=args.checkpoint, status='partial', generated_positions=0,
        groups=[], sampling=asdict(SamplingParams(temperature=.6, top_k=20, top_p=.95)),
        prefill_chunk_size=128, compile=args.compile, overlap=args.overlap,
        compile_cache_dir=engine.compiled_step.cache_dir if engine.compiled_step else None,
        startup_compile_seconds=engine.compiled_step.compile_seconds if engine.compiled_step else 0.,
        startup_compile_graphs=engine.compiled_step.compile_graphs if engine.compiled_step else 0,
        memory_budget=engine.memory_budget,
        provenance=model_provenance(suite, args.checkpoint))
    historical = args.root / 'tf-real.json'
    previous = json.loads(historical.read_text()) if args.checkpoint == 'real' and historical.exists() else None
    destination.mkdir(parents=True, exist_ok=True)
    while manifest['generated_positions'] < args.min_generated_positions:
        group = len(manifest['groups'])
        observer = Observer(engine)
        engine._observer = observer
        results = engine.generate([Request(p, 256, group * 6 + i) for i, p in enumerate(prompts)], manifest['sampling'])
        if group == 0 and previous:
            assert [r.token_ids for r in results] == [r['token_ids'] for r in previous['results']], 'original tf-real corpus changed'
            manifest['original_tf_real_sha256'] = digest(historical)
        entries = []
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            length = len(prompt) + len(result.token_ids) - (result.finish_reason == 'length')
            output = observer.output(i, length)
            generated = length - len(prompt)
            sequence = prompt + result.token_ids[:generated]
            file = destination / f'g{group:03d}-r{i}.safetensors'
            save_file({**{k: v.contiguous() for k, v in vars(output).items()},
                'input_ids': torch.tensor(sequence), 'prompt_ids': torch.tensor(prompt)}, str(file))
            entries.append(dict(sequence_id=records[i]['sequence_id'], file=file.name, sha256=digest(file),
                                generated_positions=generated, prompt_length=len(prompt), seed=group * 6 + i,
                                result=asdict(result)))
            manifest['generated_positions'] += generated
            del output
            observer.records.pop(i)
        manifest['groups'].append(entries)
        write(manifest_path, manifest)
        engine._observer = None
        del observer
        gc.collect()
        print(f'{args.checkpoint} corpus group {group}: {manifest["generated_positions"]} generated positions', flush=True)
    manifest['status'] = 'complete'
    manifest['kv_audit'] = engine.pool.audit()
    manifest['host_swap_used_bytes'] = engine.swap.used_bytes
    write(manifest_path, manifest)
    return manifest


def metric_noise(rows, decider, model_cap):
    noise = gtf_summary(rows)
    noise['reference_max_flip_margin'] = max(d['max_flip_margin'] for d in noise['per_depth'])
    noise['mlp_count_floor'] = 2
    noise['learned_depths'] = list(range(1, min(decider.max_iter, model_cap))) if hasattr(decider, 'backbone') else []
    # gtf_summary exposes the standard 1.5x errors and disagreement limits;
    # the only count floor is the one prescribed in SPEC for learned decisions.
    return noise


def load_manifest(args):
    destination = args.root / 'gtf-corpus' / args.checkpoint
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['generated_positions'] >= 8192
    return destination, manifest


def oracle_layout_b(oracle, entries, payload):
    sequences = [p['input_ids'].tolist() for p in payload]
    prompt_lengths = [e['prompt_length'] for e in entries]
    continuation = max(len(s) - p for s, p in zip(sequences, prompt_lengths))
    # Reference generate keeps finished rows alive until the batch finishes.
    # Fill their tail with EOS and trim observations to the actual corpus.
    padded = [s + [oracle.config.eos_token_id] * (continuation - len(s) + p) for s, p in zip(sequences, prompt_lengths)]
    width = max(prompt_lengths) + continuation
    boundaries = [0] + list(range(max(prompt_lengths), width + 1))
    traces, logits = capture(oracle, padded, pad_id=oracle.config.eos_token_id,
                             prompt_lengths=prompt_lengths, chunk_boundaries=boundaries)
    for i, seq in enumerate(sequences):
        length = len(seq)
        for key in ['visited', 'decision', 'decision_logits', 'continue_probability']:
            traces[i][key] = traces[i][key][:, :length]
        traces[i]['pass_count'] = traces[i]['pass_count'][:length]
        logits[i] = logits[i][:length]
    return traces, logits


def noise(args):
    suite, root, _, _, _, _, eval_config = fixture_setup(args.checkpoint)
    destination, manifest = load_manifest(args)
    limits_path = args.root / 'gtf-limits' / f'{args.checkpoint}.json'
    if limits_path.exists():
        raise ValueError('saved limits already exist; never overwrite a preregistration')
    oracle = load_reference(SimpleNamespace(device='cuda', eval_config=eval_config), args.checkpoint, root)
    rows, oracle_files = [], []
    for group, entries in enumerate(manifest['groups']):
        # Read token IDs only. Never load engine logits or decision outputs in
        # reference-noise calibration; this phase cannot compare the engine.
        from safetensors import safe_open
        payload = []
        for entry in entries:
            path = destination / entry['file']
            assert digest(path) == entry['sha256']
            with safe_open(str(path), framework='pt', device='cpu') as f:
                payload.append({'input_ids': f.get_tensor('input_ids')})
        b, lb = oracle_layout_b(oracle, entries, payload)
        for i, (entry, values) in enumerate(zip(entries, payload)):
            sequence = values['input_ids'].tolist()
            prompt_length = entry['prompt_length']
            boundaries = list(range(0, prompt_length, 128)) + list(range(prompt_length, len(sequence) + 1))
            a, la = capture(oracle, [sequence], pad_id=oracle.config.eos_token_id, chunk_boundaries=boundaries)
            row = compare(a[0], b[i], la[0], lb[i], suite.manifest['checkpoints'][args.checkpoint]['threshold'])
            for depth, metrics in zip(row['per_depth'], decision_metrics(a[0], b[i], suite.manifest['checkpoints'][args.checkpoint]['threshold'])):
                depth.update(metrics)
            row.update(group=group, row=i)
            rows.append(row)
            path = destination / f'a-g{group:03d}-r{i}.safetensors'
            save_file({**{k: v.contiguous() for k, v in a[0].items()}, 'full_logits': la[0].contiguous()}, str(path))
            oracle_files.append(dict(file=path.name, sha256=digest(path)))
            del a, la
        del b, lb, payload
        gc.collect()
        print(f'{args.checkpoint} reference noise {group + 1}/{len(manifest["groups"])}', flush=True)
    calibrated = metric_noise(rows, oracle._active_iter_decider, oracle.max_iter)
    report = dict(status='PREREGISTERED', checkpoint=args.checkpoint, generated_positions=manifest['generated_positions'],
        corpus_manifest_sha256=digest(destination / 'manifest.json'),
        source='reference layout A vs reference layout B ONLY',
        layouts=dict(A='singleton prompt chunks128 + cached token1', B='batched left-padded full prompt + aligned cached token1; finished rows EOS-filled'),
        noise_limits=calibrated, decision_limits=derive_decision_limits(calibrated), comparisons=rows,
        oracle_files=oracle_files, provenance=model_provenance(suite, args.checkpoint),
        saved_unix_time=time.time())
    write(limits_path, report)
    return report


def comparison(args):
    destination, manifest = load_manifest(args)
    limits_path = args.root / 'gtf-limits' / f'{args.checkpoint}.json'
    limits = json.loads(limits_path.read_text())  # required BEFORE any engine comparison
    limits_sha256 = digest(limits_path)
    assert limits['status'] == 'PREREGISTERED'
    assert limits['corpus_manifest_sha256'] == digest(destination / 'manifest.json')
    assert limits['provenance']['runtime_sha256'] == {str(p): digest(p) for p in Path('tah/engine').glob('*.py')}
    suite, _, _, _, _, _, _ = fixture_setup(args.checkpoint)
    rows = []
    for group, entries in enumerate(manifest['groups']):
        for i, entry in enumerate(entries):
            path = destination / entry['file']
            assert digest(path) == entry['sha256']
            engine = load_file(str(path))
            saved = limits['oracle_files'][group * 6 + i]
            path = destination / saved['file']
            assert digest(path) == saved['sha256']
            a = load_file(str(path))
            output = TeacherForcedOutput(**{k: engine[k] for k in TeacherForcedOutput.__dataclass_fields__})
            row = compare_output(a, output, a['full_logits'], suite.manifest['checkpoints'][args.checkpoint]['threshold'])
            row.update(group=group, row=i)
            rows.append(row)
            del a, engine, output
    report = dict(gate='Generated-sequence G-TF', checkpoint=args.checkpoint,
        generated_positions=manifest['generated_positions'], summary=gtf_summary(rows), comparisons=rows,
        limits_sha256=limits_sha256, limits_saved_unix_time=limits['saved_unix_time'],
        comparison_unix_time=time.time(), decision_limits=limits['decision_limits'], noise_limits=limits['noise_limits'])
    try:
        assert_gtf(rows, limits['noise_limits'])
    except AssertionError as exc:
        report.update(status='FAIL', failure=str(exc))
    else:
        report['status'] = 'PASS'
    assert digest(limits_path) == limits_sha256
    write(args.root / f'tf-{args.checkpoint}-expanded.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['generate', 'noise', 'compare'], required=True)
    parser.add_argument('--checkpoint', choices=['real', 'mlp3', 'mlp1', 'label3'], required=True)
    parser.add_argument('--root', type=Path, default=Path('data/engine/wp6'))
    parser.add_argument('--min-generated-positions', type=int, default=8192)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False,
                        help='use the full-graph TorchInductor decode path')
    parser.add_argument('--overlap', action=argparse.BooleanOptionalAction, default=False,
                        help='generate the corpus with the overlap (pipelined) scheduler')
    parser.add_argument('--compile-cache-dir', type=Path,
                        help='TorchInductor disk cache (default: TORCHINDUCTOR_CACHE_DIR or ~/.cache/torch/inductor)')
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.perf_counter()
    result = {'generate': generate, 'noise': noise, 'compare': comparison}[args.phase](args)
    print(dict(phase=args.phase, checkpoint=args.checkpoint, status=result['status'],
               generated_positions=result['generated_positions'], seconds=time.perf_counter() - start), flush=True)
    if result['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
