#!/usr/bin/env python3
"""CPU curation of R2 speed and the recorded host-memory A/B comparison."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = ROOT / 'data'
R2_DATA = REPO / 'report/2-baseline/data'
ENGINE_RUNS = REPO / 'data/engine'
MEMORY_AB = REPO / 'data/engine/memory_ab'
REASONING_BENCHMARKS = {'aime25', 'amc23', 'olympiadbench', 'math500', 'gpqa'}
CODE_BENCHMARKS = {'humaneval', 'mbpp'}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO.resolve())) if path.is_relative_to(REPO.resolve()) else str(path)


def source(path):
    path = Path(path)
    content = path.read_bytes()
    return dict(source_path=rel(path), source_sha256=digest(content), source_bytes=len(content),
                source_mtime_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())


def unique_sources(items):
    by_path = {item['source_path']: item for item in items}
    return [by_path[path] for path in sorted(by_path)]


def write_curated(name, record, purpose, sources):
    path = DATA / name
    content = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
    path.write_bytes(content)
    manifest_path = DATA / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][name] = dict(
        sha256=digest(content), bytes=len(content), encoding='curated JSON', purpose=purpose,
        sources=unique_sources(sources), top_level_keys_or_columns=list(record))
    manifest['archived_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    print(f'Wrote {name} ({len(content)} bytes) and updated MANIFEST.json')


def baseline_analysis():
    spec = importlib.util.spec_from_file_location('baseline_analysis', R2_DATA.parent / 'analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_subruns(group_dir, sources):
    subruns = []
    for run_dir in sorted((group_dir / 'runs').iterdir()):
        command_path = run_dir / 'command.json'
        command = json.loads(command_path.read_text())
        sources.append(source(command_path))
        item = dict(attempt=run_dir.name, status=command['status'])
        paths = sorted(run_dir.rglob('job_*.metrics.json'))
        if paths:
            generated_tokens = generation_seconds = restarts = 0
            for path in paths:
                sources.append(source(path))
                metrics = json.loads(path.read_text())
                generated_tokens += metrics['engine_run']['generated_tokens']
                generation_seconds += metrics['generation_seconds']
                restarts += metrics['engine_run']['restarts']
            item['metrics'] = dict(generated_tokens=generated_tokens,
                                   generation_seconds=generation_seconds, restarts=restarts)
        subruns.append(item)
    return subruns


def r2_speed():
    analysis = baseline_analysis()
    protocol_path = R2_DATA / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())['r2']
    sources = [source(protocol_path)]
    groups = [group for group in protocol['groups'] if group['max_new_tokens'] == 8192]
    if not groups:
        raise ValueError('R2 protocol has no 8192-token groups')
    benchmarks = {}
    kv_capacities, prompt_tokens = [], []
    for group in groups:
        dataset = group['datasets'][0]
        seed_rates = {}
        for seed in (42, 1234, 2024):
            base = ENGINE_RUNS / f'paper_protocol_seed{seed}'
            group_dir = base / 'groups' / group['name']
            subruns = metric_subruns(group_dir, sources)
            totals = analysis.run_metrics(dict(subruns=subruns))
            rate = analysis.generation_rate(totals) if totals['complete'] else None
            seed_rates[str(seed)] = dict(generated_tokens=totals['generated_tokens'],
                                         generation_seconds=totals['generation_seconds'],
                                         tokens_per_second=rate, metrics_complete=totals['complete'],
                                         measured_subruns=sum('metrics' in item for item in subruns),
                                         total_subruns=len(subruns))
            canonical = base / 'canonical' / f'{dataset}_tah_engine'
            leaves = sorted(canonical.glob('*'))
            if len(leaves) != 1:
                raise ValueError(f'Expected one canonical output for {dataset} seed {seed}; found {len(leaves)}')
            csv_path = leaves[0] / 'detailed_results.csv'
            sources.append(source(csv_path))
            with csv_path.open(encoding='utf-8', newline='') as stream:
                import csv
                prompt_tokens.extend(int(row['input_tokens']) for row in csv.DictReader(stream) if row['input_tokens'])
        complete_rates = [v['tokens_per_second'] for v in seed_rates.values() if v['metrics_complete']]
        if dataset in REASONING_BENCHMARKS:
            speed_category = 'reasoning'
        elif dataset in CODE_BENCHMARKS:
            speed_category = 'code'
        else:
            raise ValueError(f'Unexpected benchmark in the 8192-token speed groups: {dataset}')
        benchmark = dict(speed_category=speed_category, max_new_tokens=group['max_new_tokens'], seed_rates=seed_rates,
                         complete_seed_count=len(complete_rates),
                         mean_tokens_per_second=statistics.mean(complete_rates) if len(complete_rates) == len(seed_rates) else None)
        benchmarks[dataset] = benchmark

    capacities = Counter()
    all_metric_paths = sorted(ENGINE_RUNS.glob('paper_protocol_seed*/groups/*/runs/*/eval_output/*_tah_engine/*/job_*.metrics.json'))
    for path in all_metric_paths:
        metrics = json.loads(path.read_text())
        if metrics['engine_options']['max_new_tokens'] == 8192:
            capacities[metrics['memory_budget']['kv_capacity']] += 1
    if not capacities or not prompt_tokens:
        raise ValueError('Missing R2 KV capacity or prompt-length measurements')
    kv_capacity, _ = capacities.most_common(1)[0]
    engine_source = REPO / 'tah/engine/engine.py'
    engine_text = engine_source.read_text()
    match = re.search(r'admission_headroom\s*=\s*([0-9.]+)', engine_text)
    if not match:
        raise ValueError('Could not read the engine admission-headroom default')
    headroom = float(match.group(1))
    speed = json.loads((DATA / 'speed.json').read_text())
    speed_path = DATA / 'speed.json'
    sources.extend([source(speed_path), source(engine_source)])
    graph_run = speed['runs']['graphs']
    mean_prompt_tokens = statistics.mean(prompt_tokens)
    output_cap = max(group['max_new_tokens'] for group in groups)
    long_requests = round(kv_capacity * (1 - headroom) / (output_cap + mean_prompt_tokens))
    record = dict(
        rate_definition='For each seed, generated tokens divided by generation seconds, summed across metric-bearing sub-runs; a benchmark mean is the arithmetic mean of its three complete seed rates, or null when any seed has incomplete metrics.',
        benchmarks=benchmarks,
        capacity_context=dict(kv_capacity_slots=kv_capacity, admission_headroom_fraction=headroom,
                             mean_prompt_tokens=mean_prompt_tokens, long_output_cap=output_cap,
                             estimated_long_requests=long_requests,
                             comparison_resident_requests=graph_run['batch_size'],
                             comparison_workload=graph_run['workload']))
    write_curated('r2_speed.json', record,
                  'R2 generated-token throughput by 8192-token benchmark, plus inputs to the approximate KV-capacity comparison.',
                  sources)


def memory_ab():
    reports = {
        '8g': ('compare_8g.json', 'old-8g.json', 'new-8g.json'),
        '0.5g': ('compare_05g.json', 'old-0.5g.json', 'new-0.5g.json'),
    }
    sources = []
    budgets = {}
    page_bytes = None
    for budget, (comparison_name, old_name, new_name) in reports.items():
        comparison_path = MEMORY_AB / comparison_name
        if not comparison_path.exists():
            raise FileNotFoundError(f'Missing saved comparison {comparison_path}; copy the recorded comparison JSON into data/engine/memory_ab')
        comparison = json.loads(comparison_path.read_text())
        old_path, new_path = MEMORY_AB / old_name, MEMORY_AB / new_name
        old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
        sources.extend([source(comparison_path), source(old_path), source(new_path)])
        if comparison['requests'] != len(old['results']) or comparison['requests'] != len(new['results']):
            raise ValueError(f'Request count mismatch in {comparison_path}')
        new_page_bytes = new['memory_budget_host_swap']['page_bytes']
        page_bytes = new_page_bytes if page_bytes is None else page_bytes
        if new_page_bytes != page_bytes:
            raise ValueError('Swap page size differs between A/B runs')
        budgets[budget] = dict(
            requests=comparison['requests'], identical_outputs=comparison['identical_outputs'],
            swap_outs=dict(old=comparison['counters_old']['swap_outs'], new=comparison['counters_new']['swap_outs']),
            restarts=dict(old=comparison['counters_old']['restarts'], new=comparison['counters_new']['restarts']),
            swap_seconds=comparison['swap_seconds'],
            rss_growth_during_gib=comparison['rss_growth_during_gib'],
            differing_requests=len(comparison['differing_requests']),
            differing_per_request_counters=len(comparison['differing_per_request_counters']))
    large = budgets['8g']
    if not large['identical_outputs'] or large['restarts'] != dict(old=0, new=0):
        raise ValueError('The 8 GiB no-overflow comparison must be token-identical and have no restarts')
    reference_run = json.loads((MEMORY_AB / reports['8g'][2]).read_text())
    args = reference_run['args']
    record = dict(workload=dict(benchmark='MATH500', requests=args['requests'], batch_size=args['batch_size'],
                                max_new_tokens=args['max_new_tokens'], kv_capacity=reference_run['kv_capacity']),
                  page_bytes=page_bytes,
                  budgets=budgets)
    write_curated('memory_ab.json', record,
                  'Old/new host-swap comparisons at 8 GiB and 0.5 GiB, including output identity and RSS growth.',
                  sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=('r2_speed', 'memory_ab'))
    args = parser.parse_args()
    {'r2_speed': r2_speed, 'memory_ab': memory_ab}[args.study]()


if __name__ == '__main__':
    main()
