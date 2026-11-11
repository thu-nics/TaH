#!/usr/bin/env python3
"""Curate saved confirmatory R1-O aggregates on CPU; never read model tensors.

No per-position, per-sequence or per-problem arrays are copied. Repeating this
command with the same sources produces byte-identical data and manifest entries.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DEFAULT_ROOT = REPO / 'data/engine/oracle/wp23c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(record):
    return (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def collect(root=DEFAULT_ROOT, o1_manifest=None, c1_manifest=None, self_split=None):
    root = Path(root).resolve()
    o1_manifest = Path(o1_manifest or REPO / 'data/engine/oracle/wp23c/inputs/manifest.json')
    c1_manifest = Path(c1_manifest or REPO / 'data/engine/oracle/wp23c/inputs/c1-manifest.json')
    self_split = Path(self_split or REPO / 'data/engine/oracle/wp23c/inputs/self-split.json')
    sources = {}

    def source(path):
        path = Path(path)
        content = path.read_bytes()
        entry = dict(source_path=relative(path), source_sha256=hashlib.sha256(content).hexdigest(), source_bytes=len(content))
        sources[entry['source_path']] = entry
        return entry

    def load(path):
        source(path)
        return json.loads(Path(path).read_text())

    def inventory(path):
        source(path)
        return {name: digest for digest, name in (line.split(maxsplit=1) for line in path.read_text().splitlines())}

    manifest = load(o1_manifest)
    cmanifest = load(c1_manifest)
    by_id = {s['id']: s for s in manifest['sequences']}
    subsets = {}
    for name, ids in manifest['subsets'].items():
        items = [by_id[i] for i in ids]
        subsets[name] = dict(sequences=len(items), positions=sum(len(s['forced']) for s in items),
                             datasets=dict(Counter(s['dataset'] for s in items)),
                             generators=dict(Counter(s['backend'] for s in items)),
                             max_total_length=max(len(s['prompt']) + len(s['forced']) for s in items))
    legacy = load(root / 'o1/stage1/analysis/r1o1.json')
    source(REPO / 'report/PROTOCOL.md')
    looks = [load(path) for path in sorted((root / 'o1').glob('look[12].json'))]
    c1 = load(sorted((root / 'c1').glob('look[12].json'))[-1])
    near = load(root / 'o1/near-tie.json')
    composition = load(root / 'o1/free-composition.json')
    diagnostic = load(self_split)
    frozen = inventory(root / 'source.sha256')
    history = root if (root / 'kvcap-fix.json').exists() else DEFAULT_ROOT
    before_kvcap = source(history / 'source.before-kvcap.sha256')
    fix = load(history / 'kvcap-fix.json')
    for name in ('logs/chain.status', 'logs/eng-forced.kvcap-fail.log'):
        source(history / name)
    failed = REPO / 'data/engine/oracle/wp23b'
    failed_sources = [source(failed / name) for name in ('source.sha256', 'replay-fix.json')]

    # Preserve saved inventories, including frozen checkpoint-file digests. Hash
    # small current sources separately; do not re-read multi-GB checkpoints.
    unavailable_sources = []
    for name in sorted(frozen):
        if Path(name).parts[0] not in {'script', 'tah', 'models'}:
            continue
        path = REPO / name
        if path.suffix in ('.py', '.sh', '.yaml'):
            if not path.exists():
                unavailable_sources.append(name)
                continue
            if sha(path) != frozen[name]:
                raise ValueError('Changed frozen source: ' + name)
            source(path)
    canonical_c1_hash = hashlib.sha256(json.dumps(cmanifest, sort_keys=True).encode()).hexdigest()
    assert all(look['manifest_sha256'] == sha(o1_manifest) for look in looks)
    assert c1['manifest_sha256'] == canonical_c1_hash

    runs = []
    for path in sorted(root.glob('**/run.json')):
        run = load(path)
        is_o1 = 'blocks' in run
        assert run.get('complete') if is_o1 else run['status'] == 'complete'
        cfg = run['config']
        seconds = (sum(b['seconds'] for b in run['blocks']) if is_o1 else
                   sum(b['wall_seconds'] for b in run['telemetry'].values()))
        env = run['environment'] if is_o1 else cfg['environment']
        runs.append(dict(phase=str(path.parent.relative_to(root)), seconds=seconds,
                         timing_scope='sum of saved block seconds; excludes process/model loading and gaps',
                         model_sha256=run.get('model_sha256', cfg.get('model_sha256')),
                         manifest_sha256=run.get('manifest_sha256', cfg.get('manifest_sha256')),
                         source_sha256=cfg.get('source_sha256'), environment=env,
                         settings={k: cfg[k] for k in ('impl', 'implementation', 'dtype', 'mode', 'layout', 'batch', 'chunk',
                                   'block', 'block_size', 'max_batch_size', 'kv_capacity', 'sequence_order', 'fp64_attention',
                                   'max_new_tokens', 'engine_options', 'eval_config_sha256') if k in cfg}))
    forced = {}
    if looks[-1]['forced_sequences'] != 64:
        raise ValueError('This aggregate importer requires the recorded forced stop at 64; '
                         'an expanded forced study needs its pooled raw means collected before import')
    for name in ('M1', 'M2', 'M3', 'bias'):
        stopped = next(look for look in looks if look['criteria'][name]['status'] == 'PASS')
        row = dict(stopped['criteria'][name], stopped_at_look=stopped['look'],
                   sequences=stopped['forced_sequences'], positions=subsets['stage1']['positions'])
        if name == 'bias':
            row.update({k: legacy['bias'][k] for k in ('D', 'engine_bias', 'reference_bias', 'reference_M2')})
        else:
            row.update({k: legacy['primary'][name][k] for k in ('engine', 'reference', 'margin')})
        forced[name] = row
    tail_looks = []
    for look in looks:
        item = dict(look=look['look'], sequences=look['free_sequences'],
                    positions=sum(subsets[f'stage{i}']['positions'] for i in range(1, look['look'] + 1)),
                    criteria=look['tail']['criteria'], original_maximum=look['original_maximum'],
                    next_action=look['free_next_action'])
        tail_looks.append(item)
    c1_summary = {k: c1[k] for k in ('C1', 'C2', 'look', 'censoring', 'next_action', 'validity_met', 'conclusion')}
    c1_summary.update(cap=cmanifest['max_new_tokens'], prompt_limit=cmanifest['prompt_limit'],
                      datasets=dict(Counter(p['dataset'] for p in c1['per_problem'])),
                      censoring_counts={key: sum(p[key] for p in c1['per_problem']) for key in c1['censoring']},
                      early_identical_counts={arm: sum(p[f'{arm}_no_observed_difference'] and not p[f'{arm}_cap_censored']
                                                      for p in c1['per_problem']) for arm in ('engine', 'reference')})
    stable_input_names = {
        'manifest.json': 'data/engine/oracle/wp23c/inputs/manifest.json',
        'c1-manifest.json': 'data/engine/oracle/wp23c/inputs/c1-manifest.json',
        'self-split.json': 'data/engine/oracle/wp23c/inputs/self-split.json',
    }
    stable_inventory = lambda values: {stable_input_names.get(Path(name).name, name): digest
                                        for name, digest in values.items()}
    record = dict(schema_version=1, conclusion='PASS' if looks[-1]['conclusion'] == c1['conclusion'] == 'PASS' else 'not shown',
                  design=dict(delta=legacy['delta'], alpha_per_look=looks[0]['alpha_per_look'], resamples=looks[0]['resamples'],
                              bootstrap_seed=looks[0]['seed'], quantile=near['quantile'], minimum_tail_events=20,
                              validity_factor=0.05, validity_min_sequences=20, validity_min_positions=20000,
                              c1_increment=len(cmanifest['subsets']['stage2']), c1_maximum=len(cmanifest['problems']),
                              withdrawn_c2_margin_pp=1,
                              maximum_teacher_forced_length=manifest['selection']['max_length'],
                              oracle_threshold=legacy['oracle']['threshold'],
                              amended='2026-10-15', maximum_binding=False, C2_withdrawn=True),
                  sizes=subsets, validity=legacy['validity'], forced=forced, tail_looks=tail_looks,
                  secondary_reference_b1=dict(metrics=legacy['reference_b1'], **subsets['b1'],
                      confidence=0.95, binding=False, note='Descriptive four-sequence secondary; original 95% bounds, not confirmatory looks'),
                  tail_stopped_at_look={kind: next(look['look'] for look in looks if look['tail']['criteria'][kind]['status'] == 'PASS')
                                        for kind in ('decision', 'top1')},
                  self_split=diagnostic, composition=dict(engine_vs_engine=composition, engine_vs_reference=near),
                  c1=c1_summary, gpu_time=dict(runs=runs, total_block_seconds=sum(r['seconds'] for r in runs),
                      scope='Saved successful-block wall time on GPU; not CUDA event time or end-to-end chain wall time. '
                            'The chain also incurred loading, CPU analysis, a failed KV-capacity launch and idle gaps.'),
                  provenance=dict(root=relative(root), date='2026-10-15', source_hashes=stable_inventory(frozen),
                      source_inventory=source(root / 'source.sha256'), previous_inventory=before_kvcap,
                      unavailable_historical_sources=unavailable_sources,
                      manifests=dict(o1_file_sha256=sha(o1_manifest), c1_file_sha256=sha(c1_manifest),
                                     c1_canonical_json_sha256=canonical_c1_hash),
                      fp64_attention='odd-kv-masked-pad-v1 (PyTorch #143644): validation-only odd float64 KV width '
                                     'padded with one zero K/V slot masked out; no dependency or reference-model change',
                      kv_capacity=148000, kv_capacity_fix=fix,
                      preserved_failed_root=dict(path=relative(failed), sources=failed_sources,
                          disposition='Native fp64 attention invalid; preserved, excluded, never pooled with this study')))
    return record, [sources[key] for key in sorted(sources)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--o1-manifest', type=Path)
    parser.add_argument('--c1-manifest', type=Path)
    parser.add_argument('--self-split', type=Path)
    args = parser.parse_args()
    record, sources = collect(args.source_root, args.o1_manifest, args.c1_manifest, args.self_split)
    content = encoded(record)
    data = ROOT / 'data'
    (data / 'r1o.json').write_bytes(content)
    manifest_path = data / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['r1o.json'] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
        encoding='curated JSON', purpose='Amended confirmatory R1-O: validity, forced means/bias, both tail looks, '
        'descriptive maxima, C1, diagnostics, block timings and provenance; no raw arrays', sources=sources,
        top_level_keys_or_columns=list(record))
    method_path = data / 'method.json'
    method = json.loads(method_path.read_text())
    method['reproduction']['r1o'] = dict(
        runtime=f'{record["gpu_time"]["total_block_seconds"]/3600:.2f} h measured successful blocks (loading and gaps extra)',
        device='RTX 3090 24 GiB; CPU curation in seconds')
    method_content = encoded(method)
    method_path.write_bytes(method_content)
    entry = manifest['files']['method.json']
    entry.update(sha256=hashlib.sha256(method_content).hexdigest(), bytes=len(method_content),
                 source_sha256=hashlib.sha256(method_content).hexdigest(), source_bytes=len(method_content),
                 purpose='Declared analysis policy and reproduction runtimes; r1o runtime derived from r1o.json saved block times.')
    manifest_path.write_bytes(encoded(manifest))
    print(f'Curated r1o.json: {len(content)} bytes, {len(sources)} hashed sources; {record["conclusion"]}')


if __name__ == '__main__':
    main()
