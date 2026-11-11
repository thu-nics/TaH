#!/usr/bin/env python3
"""Build the compact, confirmatory R1 archive from complete experiment runs.

The chapter stores one long-form sample table and one metadata document. Full
generations and run artifacts remain in the source experiment directories.
"""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = ROOT / 'data'
ENGINE_DIR = 'tah/engine'
RUNS = [
    ('reference-seed1234', 'tah', 1234, 'data/engine/wacc_reference_protocol_seed1234', 'protocol'),
    ('engine-seed42', 'tah_engine', 42, 'data/engine/wacc_engine_protocol_seed42', 'protocol'),
    ('engine-seed1234', 'tah_engine', 1234, 'data/engine/wacc_engine_protocol_seed1234', 'protocol'),
    ('engine-seed2024', 'tah_engine', 2024, 'data/engine/wp10/wacc-seed2024', 'direct'),
]
DATASETS = ('math500', 'gpqa')
SAMPLE_COLUMNS = ('run', 'dataset', 'problem_id', 'sample_idx', 'is_correct', 'pass_stat_tokens',
                  'mean_passes_per_token', 'fraction_tokens_at_least_2_passes')
RESULT_COLUMNS = SAMPLE_COLUMNS[2:]
SETTING_KEYS = ('dtype', 'temperature', 'top_p', 'top_k', 'min_p', 'max_new_tokens', 'batch_size', 'repeat_size',
                'max_iter', 'iter_decider', 'eval_iter_decider', '_random_seed')
ENGINE_KEYS = ('restarts', 'wasted_tokens', 'swap_outs', 'swap_ins', 'steps', 'generated_tokens', 'overlap',
               'host_swap_peak_bytes')
DEVIATIONS = [
    'Engine seed 2024 was run through eval.py before the protocol runner existed (harness batch size 16 instead '
    'of 8, one engine job for both datasets); the engine backend ignores the batch size, other settings match.',
]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def rel(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def source(path):
    raw = path.read_bytes()
    return {'source_path': rel(path), 'source_sha256': sha(raw), 'source_bytes': len(raw)}


def sanitize(value):
    if isinstance(value, str):
        return value.replace(str(REPO.resolve()) + '/', '')
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    return value


def dataset_leaves(base, dataset, backend, layout):
    parent = base / 'canonical' / f'{dataset}_{backend}' if layout == 'protocol' else base / f'{dataset}_{backend}'
    found = sorted(path.parent for path in parent.rglob('run_status.json'))
    if len(found) != 1:
        raise ValueError(f'Expected one canonical {dataset} run status under {parent}; found {len(found)}')
    return found[0]


def config_values(path):
    config = yaml.safe_load(path.read_text())
    out = {key: config[key] for key in (*SETTING_KEYS, 'protocol', 'backend', 'seed', 'groups') if key in config}
    if 'threshold' in config.get('iter_decider_kwargs', {}):
        out['threshold'] = config['iter_decider_kwargs']['threshold']
    return out


def telemetry(paths):
    seen, total, options = set(), {}, None
    for path in paths:
        digest = sha(path.read_bytes())
        if digest in seen:
            continue
        seen.add(digest)
        record = json.loads(path.read_text())
        for key in ENGINE_KEYS:
            if key == 'overlap':
                total[key] = record['engine_run'][key]
            elif key == 'host_swap_peak_bytes':
                total[key] = max(total.get(key, 0), record['engine_run'][key])
            else:
                total[key] = total.get(key, 0) + record['engine_run'][key]
        options = record.get('engine_options')
    return total, options, len(seen)


def protocol_groups(base, backend, seed):
    groups, all_sources = [], []
    for group_dir in sorted((base / 'groups').iterdir()):
        runs = sorted((group_dir / 'runs').iterdir())
        if len(runs) != 1:
            raise ValueError(f'Expected one completed R1 attempt in {group_dir}; found {len(runs)}')
        run_dir = runs[0]
        command_path = run_dir / 'command.json'
        command = json.loads(command_path.read_text())
        if command['status'] != 'complete' or command['exit_code'] != 0:
            raise ValueError(f'Incomplete R1 group command: {command_path}')
        leaves = sorted(run_dir.rglob('eval_output/**/run_status.json'))
        if not leaves:
            # rglob's globstar behavior differs by Python version; use the stable pattern below.
            leaves = sorted((run_dir / 'eval_output').rglob('run_status.json'))
        if len(leaves) != 1:
            raise ValueError(f'Expected one evaluation result for {group_dir}; found {len(leaves)}')
        leaf = leaves[0].parent
        status = json.loads(leaves[0].read_text())
        if status.get('status') != 'complete' or status.get('random_seed') != seed:
            raise ValueError(f'Incomplete or wrong-seed group result: {leaves[0]}')
        config_path = leaf / 'effective_config.yaml'
        metric_paths = sorted(leaf.glob('job_*.metrics.json'))
        record = {
            'group': group_dir.name,
            'dataset': status['dataset'],
            'selection': command['selection']['data_ranges'] or command['selection']['data_ids'],
            'runs': len(runs),
            'status': status['status'],
            'samples': status['completed_samples'],
            'settings': config_values(config_path),
            'source_files': [source(command_path), source(config_path), source(leaves[0])],
        }
        if metric_paths and backend == 'tah_engine':
            total, options, _ = telemetry(metric_paths)
            record.update(engine=total, engine_options=options)
            record['source_files'] += [source(path) for path in metric_paths]
        groups.append(record)
        all_sources.extend(record['source_files'])
    return groups, all_sources


def direct_groups(base, backend, seed):
    groups, all_sources, metrics = [], [], []
    for dataset in DATASETS:
        leaf = dataset_leaves(base, dataset, backend, 'direct')
        status_path = leaf / 'run_status.json'
        status = json.loads(status_path.read_text())
        if status.get('status') != 'complete' or status.get('random_seed') != seed:
            raise ValueError(f'Incomplete or wrong-seed direct result: {status_path}')
        config_path = leaf / 'effective_config.yaml'
        metric_paths = sorted(leaf.glob('job_*.metrics.json'))
        groups.append({
            'group': f'{dataset}_single_job', 'dataset': dataset,
            'selection': 'math500=0:200' if dataset == 'math500' else 'gpqa=all',
            'runs': 1, 'status': status['status'], 'samples': status['completed_samples'],
            'settings': config_values(config_path),
            'source_files': [source(status_path), source(config_path), *[source(path) for path in metric_paths]],
        })
        all_sources.extend(groups[-1]['source_files'])
        metrics.extend(metric_paths)
    totals, options, _ = telemetry(metrics)
    for group in groups:
        group['engine_options'] = options
    return groups, all_sources, totals


def engine_inventory(commit):
    listing = subprocess.check_output(['git', 'ls-tree', '--name-only', commit, ENGINE_DIR + '/'], cwd=REPO,
                                      text=True).split()
    inventory = {path: sha(subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=REPO))
                 for path in listing if path.endswith('.py')}
    return inventory, sha(json.dumps(inventory, sort_keys=True).encode())


def collect(run_specs, inventory, digest, commit, direct_2024=True):
    """Return compact CSV bytes, metadata and provenance lists for the manifest."""
    samples, run_records, all_sources = [], [], []
    input_inventory = {}
    reference_manifest = None
    for name, backend, seed, directory, layout in run_specs:
        base = Path(directory)
        if not base.is_absolute():
            base = REPO / base
        dataset_records, run_sources = {}, []
        for dataset in DATASETS:
            leaf = dataset_leaves(base, dataset, backend, layout)
            csv_path, status_path, config_path = (leaf / 'detailed_results.csv', leaf / 'run_status.json',
                                                   leaf / 'effective_config.yaml')
            with csv_path.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                missing = set(RESULT_COLUMNS) - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(f'Missing required result columns {sorted(missing)} in {csv_path}')
                rows = list(reader)
            status = json.loads(status_path.read_text())
            if status.get('status') != 'complete' or status.get('random_seed') != seed:
                raise ValueError(f'Incomplete or wrong-seed canonical status: {status_path}')
            if len(rows) != status.get('completed_samples') or status.get('completed_samples') != status.get('expected_samples'):
                raise ValueError(f'Sample count differs from run status: {csv_path}')
            config = config_values(config_path)
            config_seed = config.get('_random_seed', config.get('seed'))
            if config_seed != seed:
                raise ValueError(f'Effective config seed differs from run seed: {config_path}')
            data_sources = [source(csv_path), source(status_path), source(config_path)]
            dataset_records[dataset] = {
                'run_status': sanitize(status), 'source_path': rel(csv_path.parent),
                'config_values': config, 'config_source_path': rel(config_path),
                'source_files': data_sources,
            }
            run_sources.extend(data_sources)
            for row in rows:
                samples.append({'run': name, 'dataset': dataset,
                                **{key: row[key] for key in RESULT_COLUMNS}})
        if layout == 'protocol':
            manifest_path = base / 'canonical/protocol_manifest.json'
            manifest_source = source(manifest_path)
            protocol = json.loads(manifest_path.read_text())
            if name == 'reference-seed1234':
                reference_manifest = protocol
            run_sources.append(manifest_source)
            protocol_hash = manifest_source['source_sha256']
        else:
            protocol_hash = None
        if layout == 'protocol':
            groups, group_sources = protocol_groups(base, backend, seed)
            totals = None
        else:
            groups, group_sources, totals = direct_groups(base, backend, seed)
        run_sources.extend(group_sources)
        if backend == 'tah_engine' and totals is None:
            metric_paths = [Path((REPO / directory)) / 'groups']
            all_metrics = [path for parent in metric_paths for path in parent.rglob('job_*.metrics.json')]
            totals = telemetry(all_metrics)[0]
        total_samples = sum(item['run_status']['completed_samples'] for item in dataset_records.values())
        run_records.append({
            'name': name, 'backend': backend, 'seed': seed, 'status': 'complete', 'samples': total_samples,
            'source_path': rel(base), 'protocol_manifest_sha256': protocol_hash,
            'datasets': dataset_records, 'groups': groups, 'engine_totals': totals,
            'source_files': deduplicate(run_sources),
        })
        all_sources.extend(run_sources)
    if reference_manifest is None:
        raise ValueError('R1 reference protocol manifest is missing')
    dataset_order = {dataset: i for i, dataset in enumerate(DATASETS)}
    run_order = {record['name']: i for i, record in enumerate(run_records)}
    samples.sort(key=lambda row: (run_order[row['run']], dataset_order[row['dataset']],
                                  (0, int(row['problem_id'])) if row['problem_id'].isdigit() else (1, row['problem_id']),
                                  int(row['sample_idx'])))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=SAMPLE_COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(samples)
    declarations = [{'name': run_records[0]['name'], 'backend': 'tah', 'seed': run_records[0]['seed']}]
    declarations += [{'name': run['name'], 'backend': 'tah_engine', 'seed': run['seed'],
                      'frozen_engine_sha256': digest} for run in run_records[1:]]
    index = {
        'complete': True,
        'frozen_engine_sha256': digest,
        'frozen_engine_commit': commit,
        'frozen_engine_inventory_scope': f'{ENGINE_DIR}/*.py at the frozen commit; digest is SHA256 of sorted-key inventory JSON',
        'source_inventory': inventory,
        'reference': declarations[0], 'engine': declarations[1:], 'runs': run_records,
        'expected_problem_ids': reference_manifest['expected_problem_ids'],
        'repeats': 2,
        'deviations': DEVIATIONS if direct_2024 else [],
        'scope': 'Confirmatory R1 sample and run metadata.',
    }
    all_sources.append({'source_path': f'git:{commit}:{ENGINE_DIR}/*.py',
                        'source_sha256': digest, 'note': 'frozen engine inventory digest'})
    return out.getvalue().encode(), index, deduplicate(all_sources), samples


def deduplicate(sources):
    unique = {item['source_path']: item for item in sources}
    return [unique[path] for path in sorted(unique)]


def manifest_entry(content, purpose, sources, schema):
    return {'sha256': sha(content), 'bytes': len(content), 'encoding': 'curated CSV' if content.startswith(b'run,') else 'curated JSON',
            'purpose': purpose, 'sources': sources, 'top_level_keys_or_columns': schema}


def store_compact_archive(sample_bytes, index, sources, samples, drop_legacy=False):
    manifest_path = DATA / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'] = {name: entry for name, entry in manifest['files'].items()
                         if not name.startswith(('r1/', 'evaluations/'))}
    method_path = DATA / 'method.json'
    method = json.loads(method_path.read_text())
    method['reproduction'].pop('exploratory', None)
    method_bytes = (json.dumps(method, indent=2, ensure_ascii=False) + '\n').encode()
    method_path.write_bytes(method_bytes)
    manifest['files']['method.json']['sha256'] = sha(method_bytes)
    manifest['files']['method.json']['bytes'] = len(method_bytes)
    samples_path, runs_path = DATA / 'r1/samples.csv', DATA / 'r1/runs.json'
    shutil.rmtree(DATA / 'r1', ignore_errors=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_bytes(sample_bytes)
    runs_bytes = (json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
    runs_path.write_bytes(runs_bytes)
    sample_sources = deduplicate([item for run in index['runs'] for ds in run['datasets'].values()
                                  for item in ds['source_files'] if item['source_path'].endswith('detailed_results.csv')])
    manifest['files']['r1/samples.csv'] = manifest_entry(
        sample_bytes, 'Confirmatory R1 per-sample results; only fields used by analysis and comparison are retained.',
        sample_sources, list(SAMPLE_COLUMNS))
    manifest['files']['r1/runs.json'] = manifest_entry(
        runs_bytes, 'Confirmatory R1 status, groups, settings, telemetry and source provenance.', sources, list(index))
    manifest['seeds'].pop('engine_exploratory', None)
    manifest['seeds'].pop('reference_exploratory', None)
    manifest['seeds'].pop('engine_frozen_r1_input', None)
    manifest['seeds']['r1_reference'] = [1234]
    manifest['seeds']['r1_engine'] = [42, 1234, 2024]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    if drop_legacy:
        shutil.rmtree(DATA / 'evaluations', ignore_errors=True)
    return len(samples), len(manifest['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-commit', default='1cb2562')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.frozen_commit], cwd=REPO, text=True).strip()
    inventory, digest = engine_inventory(commit)
    sample_bytes, index, sources, rows = collect(RUNS, inventory, digest, commit)
    count, file_count = store_compact_archive(sample_bytes, index, sources, rows, drop_legacy=True)
    print(f'Curated {count} samples across {len(index["runs"])} runs in {file_count} data files; frozen engine sha256 {digest}')


if __name__ == '__main__':
    main()
