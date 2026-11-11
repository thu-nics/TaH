#!/usr/bin/env python3
"""CPU-only curation of the R2 runs (report/PROTOCOL.md, R2 and its Amendment) into data/.

Reads the merged canonical outputs and per-group job records of the six registered
runs (R2 with the harness code prompt and the TaH2-prompt code runs, seeds 42, 1234,
2024) and writes one long-format sample table and one run-metadata JSON per study,
plus shared protocol and dataset descriptions. Full generations stay in the source
experiment directories; source paths and hashes are recorded in MANIFEST.json.

    .venv-tah/bin/python report/2-baseline/scripts/curate_r2.py [--runs-root data/engine]

Full generations stay in the run directories named in the manifest. Absolute repository
prefixes are removed from copied text; the original SHA256 is recorded.
"""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = ROOT / 'data'
SEEDS = (42, 1234, 2024)
RECIPES = 'script/tah/recipes/qwen3_1.7'
# (data name prefix, run directory prefix under the runs root, protocol config)
PROTOCOLS = [('r2', 'paper_protocol_seed', f'{RECIPES}/protocol_paper.yaml'),
             ('code-tah2', 'paper_code_tah2_seed', f'{RECIPES}/protocol_paper_code_tah2.yaml')]
CODE = ('humaneval', 'mbpp')
CHECKPOINT = 'models/TaH-plus-1.7B'
SETTING_KEYS = ('dtype', 'temperature', 'top_p', 'top_k', 'min_p', 'max_new_tokens', 'repeat_size', 'max_iter',
                'iter_decider', 'eval_iter_decider', 'code_prompt', '_random_seed')
DEF = re.compile(r'^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\(', re.M)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def mtime(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def source_of(path):
    b = path.read_bytes()
    return dict(source_path=rel(path), source_sha256=sha(b), source_bytes=len(b), source_mtime_utc=mtime(path))


def csv_bytes(header, rows):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue().encode()


class Manifest:
    def __init__(self):
        self.files = {}

    def add(self, name, content, entry):
        target = DATA / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self.files[name] = dict(entry, sha256=sha(content), bytes=len(content))

    def curated(self, name, content, purpose, sources, columns):
        entry = dict(encoding='curated', purpose=purpose, sources=sources,
                     top_level_keys_or_columns=columns)
        self.add(name, content, entry)

    def json(self, name, value, purpose, sources):
        self.curated(name, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode(), purpose, sources, list(value))


def leaf_of(directory):
    (leaf,) = sorted(p.parent for p in directory.rglob('run_status.json'))
    return leaf


def settings(path):
    config = yaml.safe_load(path.read_text())
    out = {k: config[k] for k in SETTING_KEYS if k in config}
    out['threshold'] = config['iter_decider_kwargs']['threshold']
    out.setdefault('code_prompt', 'harness')
    return out


def protocol_record(path):
    config = yaml.safe_load((REPO / path).read_text())
    return dict(name=config['name'], source_path=path, sha256=sha((REPO / path).read_bytes()),
                paper_values=config['paper_values'], paper_average=config.get('paper_average'),
                groups=[dict(name=g['name'], datasets=g['datasets'], repeat_size=g['repeat_size'],
                             max_new_tokens=g['max_new_tokens'], overrides=g.get('overrides', {})) for g in config['groups']])


def group_record(group_dir, seed):
    """Settings, timing and generation metrics of every sub-run of one protocol group."""
    sources, subruns = [], []
    for run in sorted((group_dir / 'runs').iterdir()):
        command = json.loads((run / 'command.json').read_text())
        sources.append(source_of(run / 'command.json'))
        item = dict(attempt=run.name, started_utc=command['started_at'], finished_utc=command.get('finished_at'),
                    status=command['status'], sample_index_remap=command['selection'].get('sample_index_remap'))
        leaves = sorted(p.parent for p in run.rglob('effective_config.yaml'))
        if leaves:
            (leaf,) = leaves
            sources.append(source_of(leaf / 'effective_config.yaml'))
            item['settings'] = settings(leaf / 'effective_config.yaml')
            assert item['settings']['_random_seed'] == seed, run
            metrics = sorted(leaf.glob('job_*.metrics.json'))
            if metrics:
                (path,) = metrics
                sources.append(source_of(path))
                m = json.loads(path.read_text())
                item['metrics'] = dict(generated_tokens=m['engine_run']['generated_tokens'],
                                       generation_seconds=m['generation_seconds'],
                                       restarts=m['engine_run']['restarts'])
            graded = sorted(leaf.glob('code_solutions.eval_results.json'))
            if graded:
                (path,) = graded
                record = json.loads(path.read_text())
                item['evalplus'] = dict(dataset_hash=record.get('hash'), graded_at_local=record.get('date'),
                                        graded_mtime_utc=mtime(path))
        subruns.append(item)
    return dict(group=group_dir.name, subruns=subruns), sources


def defined_functions(text):
    """Function names the raw output defines, in order of first appearance (any indentation)."""
    seen = []
    for name in DEF.findall(text):
        if name not in seen:
            seen.append(name)
    return ' '.join(seen)


def top_level_functions(code):
    code = textwrap.dedent(code)
    try:
        return sorted(n.name for n in ast.parse(code).body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except SyntaxError:
        # EvalPlus solutions can contain valid code followed by malformed
        # example statements. Keep names with a real indented body, but do not
        # count a header followed only by comments as a usable graded function.
        lines = code.splitlines()
        names = []
        for match in DEF.finditer(code):
            line_index = code.count('\n', 0, match.start())
            header = lines[line_index]
            indent = len(header) - len(header.lstrip())
            for candidate in lines[line_index + 1:]:
                if not candidate.strip():
                    continue
                candidate_indent = len(candidate) - len(candidate.lstrip())
                if candidate_indent <= indent:
                    break
                if not candidate.lstrip().startswith('#'):
                    names.append(match.group(1))
                    break
        return sorted(set(names))


SAMPLE_COLUMNS = ('run', 'seed', 'dataset', 'problem_id', 'sample_idx', 'is_correct', 'base_status',
                  'entry_point', 'generated_tokens', 'finish_reason', 'raw_defined_functions',
                  'graded_functions', 'graded_solution_empty')


def code_dataset(leaf, dataset, name, seed, expected):
    """Return the analysis fields for one code study, keeping no generated solution text."""
    graded = json.loads((leaf / 'code_solutions.eval_results.json').read_text())
    samples = {}
    with open(leaf / 'samples.jsonl', encoding='utf-8') as f:
        for line in f:
            s = json.loads(line)
            key = (s['problem_id'], int(s['sample']))
            assert key not in samples, key
            samples[key] = s
    rows = {(r['problem_id'], int(r['sample_idx'])): r
            for r in csv.DictReader(io.StringIO((leaf / 'detailed_results.csv').read_text()))}
    assert len(samples) == len(rows) == expected and set(samples) == set(rows), leaf
    out = []
    for key in sorted(rows, key=lambda k: (int(k[0].rsplit('/', 1)[1]), k[1])):
        task = key[0][len(dataset) + 1:]
        (result,) = graded['eval'][task]
        s, r = samples[key], rows[key]
        solution = result['solution'] or ''
        out.append(dict(run=name, seed=seed, dataset=dataset, problem_id=key[0], sample_idx=key[1],
                        is_correct=(result['base_status'] == result['plus_status'] == 'pass'),
                        base_status=result['base_status'], entry_point=s['entry_point'],
                        generated_tokens=len(s['token_ids']), finish_reason=s['finish_reason'],
                        raw_defined_functions=defined_functions(s['output']),
                        graded_functions=' '.join(top_level_functions(solution)),
                        graded_solution_empty=not solution.strip()))
    assert len(graded['eval']) == expected
    plus = sum(r['is_correct'] for r in out) / len(out)
    assert abs(plus - graded['pass_at_k']['plus']['pass@1']) < 1e-12, (leaf, plus)
    srcs = [source_of(leaf / n) for n in ('code_solutions.eval_results.json', 'samples.jsonl', 'detailed_results.csv')]
    return out, srcs


def curate_run(label, prefix, protocol, seed, runs_root):
    base = runs_root / f'{prefix}{seed}'
    name = f'{label}-seed{seed}'
    pm = json.loads((base / 'canonical/protocol_manifest.json').read_text())
    state = json.loads((base / 'protocol_state.json').read_text())
    assert pm['seed'] == seed and pm['backend'] == 'tah_engine', base
    assert pm['protocol_config_sha256'] == protocol['sha256'], f'protocol config changed since {base} ran'
    datasets, sample_rows, all_sources = {}, [], []
    for dataset, expected in pm['expected_samples'].items():
        leaf = leaf_of(base / 'canonical' / f'{dataset}_tah_engine')
        status = json.loads((leaf / 'run_status.json').read_text())
        assert status['status'] == 'complete' and status['random_seed'] == seed, leaf
        assert status['completed_samples'] == status['expected_samples'] == expected, leaf
        data_sources = [source_of(leaf / 'detailed_results.csv'), source_of(leaf / 'run_status.json'),
                        source_of(leaf / 'eval_tah.yaml')]
        if dataset in CODE:
            coded, more_sources = code_dataset(leaf, dataset, name, seed, expected)
            sample_rows.extend(coded)
            data_sources += more_sources
        else:
            with (leaf / 'detailed_results.csv').open(newline='', encoding='utf-8') as stream:
                results = list(csv.DictReader(stream))
            if len(results) != expected:
                raise ValueError(f'Expected {expected} rows in {leaf / "detailed_results.csv"}; found {len(results)}')
            sample_rows.extend(dict(run=name, seed=seed, dataset=dataset, problem_id=row['problem_id'],
                                    sample_idx=int(row['sample_idx']), is_correct=row['is_correct'].lower() == 'true')
                               for row in results)
        all_sources.extend(data_sources)
        datasets[dataset] = dict(problems=len(pm['expected_problem_ids'][dataset]), samples=expected,
                                 repeat_size=status['repeat_size'], status=status['status'],
                                 expected_samples=status['expected_samples'], config_values=settings(leaf / 'eval_tah.yaml'))
    groups, sources = [], [source_of(base / 'canonical/protocol_manifest.json'), source_of(base / 'protocol_state.json')]
    for spec in protocol['groups']:
        record, more = group_record(base / 'groups' / spec['name'], seed)
        record.update(datasets=spec['datasets'], completed_samples=state['groups'][spec['name']]['completed_samples'])
        groups.append(record)
        sources += more
    merge = json.loads((base / 'canonical/merge_provenance.json').read_text())
    sources.append(source_of(base / 'canonical/merge_provenance.json'))
    dropped = [dict(group=g, problem_id=d['problem_id'], sample_idx=d['sample_idx'], reason=d['reason'],
                    attempt=d['attempt_dir'].rsplit('/', 1)[1])
               for g, v in merge['groups'].items() for d in v['dropped_records']]
    return name, dict(protocol=protocol['name'], seed=seed, status='complete', samples=sum(d['samples'] for d in datasets.values()),
                      source_dir=rel(base), created_utc=pm['created_at'],
                      merged_utc=pm['merged_at'], protocol_config_sha256=pm['protocol_config_sha256'],
                      eval_config_sha256=pm['eval_config_sha256'], datasets=datasets, groups=groups,
                      merge=dict(rule=merge['rule'], selected=merge['selected_count'], dropped=merge['dropped_count'],
                                 dropped_items=dropped)), \
        all_sources + sources, sample_rows


def dataset_versions(protocols, runs):
    configs = json.loads((REPO / 'tah/evaluate/eval_configs/dataset_configs.json').read_text())
    out, sources = {}, [source_of(REPO / 'tah/evaluate/eval_configs/dataset_configs.json')]
    hashes = {}
    for run in runs.values():
        for group in run['groups']:
            for sub in group['subruns']:
                if 'evalplus' in sub:
                    hashes.setdefault(group['datasets'][0], set()).add(sub['evalplus']['dataset_hash'])
    for dataset, cfg in configs.items():
        if dataset not in protocols['r2']['paper_values']:
            continue
        item = dict(source=cfg['path'], split=cfg.get('split_name'), subset=cfg.get('subset'))
        path = REPO / cfg['path']
        if cfg['path'].endswith(('.json', '.jsonl')):
            b = path.read_bytes()
            item.update(kind='local file', sha256=sha(b), rows=sum(1 for line in b.splitlines() if line.strip()))
            sources.append(source_of(path))
        else:
            ref = Path.home() / '.cache/huggingface/hub' / ('datasets--' + cfg['path'].replace('/', '--')) / 'refs/main'
            item.update(kind='Hugging Face dataset', revision=ref.read_text().strip())
        if dataset in CODE:
            module = importlib.import_module('evalplus.data.' + dataset)
            constant = 'HUMANEVAL_PLUS_VERSION' if dataset == 'humaneval' else 'MBPP_PLUS_VERSION'
            (digest,) = hashes[dataset]
            item.update(evalplus_dataset=('HumanEvalPlus ' if dataset == 'humaneval' else 'MbppPlus ') + getattr(module, constant),
                        evalplus_dataset_hash=digest)
        out[dataset] = item
    return out, sources


def checkpoint():
    files = {}
    for path in sorted((REPO / CHECKPOINT).rglob('*')):
        if path.is_file() and '.cache' not in path.parts:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for block in iter(lambda: f.read(1 << 24), b''):
                    h.update(block)
            files[str(path.relative_to(REPO / CHECKPOINT))] = h.hexdigest()
    return dict(path=CHECKPOINT, hub_id='nics-efc/TaH-plus-1.7B', files_sha256=files)


def deduplicate(sources):
    by_path = {item['source_path']: item for item in sources}
    return [by_path[path] for path in sorted(by_path)]


def sample_sort(row, run_order, dataset_order):
    problem = row['problem_id']
    try:
        problem_key = (0, int(problem.rsplit('/', 1)[1]))
    except (IndexError, ValueError):
        problem_key = (1, problem)
    return run_order[row['run']], dataset_order[row['dataset']], problem_key, int(row['sample_idx'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-root', type=Path, default=REPO / 'data/engine',
                        help='directory holding paper_protocol_seed*/ and paper_code_tah2_seed*/ (default: data/engine)')
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    manifest = Manifest()
    protocols = {label: protocol_record(path) for label, _, path in PROTOCOLS}
    manifest.json('protocol.json', protocols, 'Paper values, groups and SHA256 of the two R2 protocol configs.',
                  [dict(source_path=p['source_path'], source_sha256=p['sha256']) for p in protocols.values()])
    all_runs, dataset_sources = {}, []
    for label, prefix, _ in PROTOCOLS:
        study_runs, study_sources, study_samples = {}, [], []
        for seed in SEEDS:
            name, record, sources, samples = curate_run(label, prefix, protocols[label], seed, runs_root)
            study_runs[name] = record
            study_sources += sources
            study_samples += samples
        dataset_order = {d: i for i, g in enumerate(protocols[label]['groups']) for d in g['datasets']}
        run_order = {name: i for i, name in enumerate(study_runs)}
        study_samples.sort(key=lambda row: sample_sort(row, run_order, dataset_order))
        sample_bytes = csv_bytes(SAMPLE_COLUMNS, [[row.get(key, '') for key in SAMPLE_COLUMNS]
                                                  for row in study_samples])
        sources = deduplicate(study_sources)
        sample_sources = deduplicate([item for item in sources if item['source_path'].endswith(
            ('detailed_results.csv', 'samples.jsonl', 'code_solutions.eval_results.json'))])
        manifest.curated(f'{label}/samples.csv', sample_bytes,
                         'Long-format per-sample results; only analysis fields and run/sample identifiers are retained.',
                         sample_sources, list(SAMPLE_COLUMNS))
        manifest.json(f'{label}/runs.json', dict(study=label, seeds=list(SEEDS), runs=study_runs),
                      'Per-run sample coverage, settings, sub-run status and generation metrics.', sources)
        all_runs.update(study_runs)
        dataset_sources.extend(sources)
    datasets, sources = dataset_versions(protocols, all_runs)
    manifest.json('datasets.json', datasets, 'Benchmark sources and versions as loaded by the harness.', sources)
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'flashinfer-python', 'evalplus',
                                                            'datasets')}
    record = dict(schema_version=2, archived_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  archive_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  environment=dict(python=platform.python_version(), **packages), checkpoint=checkpoint(),
                  seeds=dict(r2=list(SEEDS), code_tah2=list(SEEDS)),
                  runs={k: dict(source_dir=v['source_dir'], created_utc=v['created_utc'], merged_utc=v['merged_utc'],
                                samples=v['samples']) for k, v in all_runs.items()},
                  note='Full generations and run artifacts stay in the source experiment directories.',
                  files=manifest.files)
    (DATA / 'MANIFEST.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    size = sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
    print(f'Curated {len(manifest.files)} files ({size / 1e6:.2f} MB) from {runs_root}', file=sys.stderr)


if __name__ == '__main__':
    main()
