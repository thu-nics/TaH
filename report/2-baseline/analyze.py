#!/usr/bin/env python3
"""Deterministic CPU analysis of the R2 reproduction. Reads only data/; standard library.

Writes README.md (from README.template.md), tables/*.tex (booktabs),
tables/r2-analysis.json and figures/r2_vs_paper.pdf. A second run is byte-identical.
Run `python report/2-baseline/analyze.py` from any working directory.
"""
import csv
import hashlib
import io
import json
import math
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
TABLES = ROOT / 'tables'
FIGURES = ROOT / 'figures'
SEEDS = (42, 1234, 2024)
ORDER = ('aime25', 'amc23', 'olympiadbench', 'math500', 'gpqa', 'humaneval', 'mbpp', 'gsm8k', 'mmlu_stem')
CODE = ('humaneval', 'mbpp')
NAMES = dict(aime25='AIME25', amc23='AMC23', olympiadbench='OlympiadBench', math500='MATH500', gpqa='GPQA-Diamond',
             humaneval='HumanEval+', mbpp='MBPP+', gsm8k='GSM8K', mmlu_stem='MMLU-STEM')
PROMPTS = {'r2': 'harness', 'code-tah2': 'TaH2'}
SAMPLE_COLUMNS = ('run', 'seed', 'dataset', 'problem_id', 'sample_idx', 'is_correct', 'base_status',
                  'entry_point', 'generated_tokens', 'finish_reason', 'raw_defined_functions',
                  'graded_functions', 'graded_solution_empty')
# Registered sampling settings (report/PROTOCOL.md, R2); every archived run must match them.
SETTINGS = dict(temperature=0.6, top_p=0.95, top_k=20, threshold=0.9)
CONFIDENCE = 0.95


# ---------------------------------------------------------------- input and integrity

def raw(name):
    return (DATA / name).read_bytes()


def read(name):
    return json.loads(raw(name))


def rows(name):
    return list(csv.DictReader(io.StringIO(raw(name).decode())))


def verify():
    manifest = read('MANIFEST.json')
    actual = {str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'MANIFEST.json'}:
        raise ValueError(f'Unmanifested or missing data files: {actual ^ (set(manifest["files"]) | {"MANIFEST.json"})}')
    for name, item in manifest['files'].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('SHA256 mismatch: ' + name)
    return manifest


def read_runs():
    runs = {}
    for study in PROMPTS:
        record = read(f'{study}/runs.json')
        if record['study'] != study or record['seeds'] != list(SEEDS):
            raise ValueError(f'Unexpected run metadata for {study}')
        runs.update(record['runs'])
    return runs


def check_settings(runs, protocol):
    """Every sub-run used the registered sampling settings, cap, seed and code prompt of its group."""
    for name, run in runs.items():
        label, seed = name.rsplit('-seed', 1)
        spec = {g['name']: g for g in protocol[label]['groups']}
        assert run['seed'] == int(seed) and run['protocol'] == protocol[label]['name'], name
        for group in run['groups']:
            g = spec[group['group']]
            for sub in group['subruns']:
                if 'settings' not in sub:
                    continue
                s = sub['settings']
                found = {k: s[k] for k in SETTINGS}
                prompt = g['overrides'].get('code_prompt', 'harness')
                if (found != SETTINGS or s['max_new_tokens'] != g['max_new_tokens'] or s['_random_seed'] != run['seed']
                        or s['code_prompt'] != prompt
                        or (sub['sample_index_remap'] is None and s['repeat_size'] != g['repeat_size'])):
                    raise ValueError(f"Settings differ from the protocol in {name}/{group['group']}/{sub['attempt']}: {s}")


# ---------------------------------------------------------------- per-sample correctness

def outcomes(run, dataset, info):
    """problem_id -> list of correctness per sample; complete, duplicate-free coverage checked."""
    study = run.rsplit('-seed', 1)[0]
    data = [row for row in rows(f'{study}/samples.csv') if row['run'] == run and row['dataset'] == dataset]
    found = {}
    for row in data:
        group = found.setdefault(row['problem_id'], {})
        sample = int(row['sample_idx'])
        if sample in group:
            raise ValueError(f'Duplicate sample {row["problem_id"]}/{sample} in {run}/{dataset}')
        group[sample] = row['is_correct'].lower() == 'true'
    if len(found) != info['problems'] or any(set(g) != set(range(info['repeat_size'])) for g in found.values()):
        raise ValueError(f'Incomplete sample coverage in {run}/{dataset}')
    return {k: [g[i] for i in sorted(g)] for k, g in found.items()}


def z_value():
    return st.NormalDist().inv_cdf((1 + CONFIDENCE) / 2)


def estimate(label, dataset, datasets):
    """Mean over seeds; standard error over problems with samples pooled over seeds."""
    per_seed = [outcomes(f'{label}-seed{s}', dataset, datasets[f'{label}-seed{s}'][dataset]) for s in SEEDS]
    keys = set(per_seed[0])
    if any(set(p) != keys for p in per_seed):
        raise ValueError(f'Problem sets differ across seeds: {label}/{dataset}')
    scores = [st.mean([x for p in per_seed for x in p[k]]) for k in sorted(keys)]
    mean = st.mean(scores)
    se = st.stdev(scores) / math.sqrt(len(scores))
    return dict(dataset=dataset, prompt=PROMPTS[label] if dataset in CODE else None, problems=len(keys),
                samples_per_problem=len(per_seed[0][next(iter(keys))]), mean=mean, se=se,
                ci=[mean - z_value() * se, mean + z_value() * se],
                seeds={str(s): st.mean([x for v in p.values() for x in v]) for s, p in zip(SEEDS, per_seed)})


def average(items):
    """Unweighted 9-benchmark mean; benchmarks are independent, so standard errors add in quadrature."""
    mean = st.mean(i['mean'] for i in items)
    se = math.sqrt(sum(i['se'] ** 2 for i in items)) / len(items)
    return dict(mean=mean, se=se, ci=[mean - z_value() * se, mean + z_value() * se],
                seeds={str(s): st.mean(i['seeds'][str(s)] for i in items) for s in SEEDS})


def verdict(item, paper):
    low, high = item['ci']
    item['paper'] = paper
    item['reproduced'] = low <= paper <= high
    item['gap'] = item['mean'] - paper
    return 'reproduced' if item['reproduced'] else f'gap {signed(item["gap"])}'


# ---------------------------------------------------------------- code failure breakdown

def classify(row, cap):
    """Failure cause of one graded code sample, or None if it passed base and plus tests.

    naming:  the model's code defines functions, but none with the name the tests call;
    capped:  the graded solution is empty and the output stopped at the token cap;
    no_code: no usable code otherwise (sanitize/extraction left no function the tests can call);
    logic:   the tested function exists, but tests fail (wrong result, exception or timeout).
    """
    if row['is_correct'].lower() == 'true':
        return None
    graded = set(row['graded_functions'].split())
    defined = graded | set(row['raw_defined_functions'].split())
    if row['entry_point'] in graded:
        return 'logic'
    if row['graded_solution_empty'].lower() == 'true' and row['finish_reason'] == 'length':
        if int(row['generated_tokens']) != cap:
            raise ValueError(f'Length-capped output below the cap: {row["problem_id"]}')
        return 'capped'
    if defined and row['entry_point'] not in defined:
        return 'naming'
    return 'no_code'


def breakdown(label, seed, dataset, cap):
    run = f'{label}-seed{seed}'
    data = [row for row in rows(f'{label}/samples.csv') if row['run'] == run and row['dataset'] == dataset]
    causes = [classify(row, cap) for row in data]
    count = {k: causes.count(k) for k in ('naming', 'capped', 'no_code', 'logic')}
    failures = sum(c is not None for c in causes)
    base_failures = sum(r['base_status'] != 'pass' for r in data)
    per_task = {r['problem_id'][len(dataset) + 1:]: (c, r) for r, c in zip(data, causes)}
    return dict(prompt=PROMPTS[label], seed=seed, dataset=dataset, tasks=len(data), failures=failures,
                base_failures=base_failures, **count), per_task


# ---------------------------------------------------------------- formatting

def pct(x, digits=1):
    text = f'{100 * x:.{digits}f}'
    return text[1:] if text.startswith('-') and float(text) == 0 else text


def signed(x, digits=1):
    text = pct(x, digits)
    return text if text.startswith('-') or float(text) == 0 else '+' + text


def share(part, whole):
    return f'{part} ({pct(part / whole, 0)}%)' if whole else '0'


def esc(s):
    return ''.join({'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#', '$': r'\$',
                    '{': r'\{', '}': r'\}', '±': r'$\pm$', '×': r'$\times$', '→': r'$\to$', '−': '-',
                    '≈': r'$\approx$', '`': ''}.get(c, c) for c in str(s))


def table(name, headers, body, align=None):
    body = [[str(c) for c in row] for row in body]
    tex = ['% Generated by analyze.py; requires booktabs.', r'\begin{tabular}{' + (align or 'l' * len(headers)) + '}',
           r'\toprule', ' & '.join(map(esc, headers)) + r' \\', r'\midrule']
    tex += [r'\midrule' if row[0] == '' else ' & '.join(map(esc, row)) + r' \\' for row in body]
    tex += [r'\bottomrule', r'\end{tabular}', '']
    (TABLES / (name + '.tex')).write_text('\n'.join(tex))
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in body if row[0] != ''])


def hours(seconds):
    return f'{seconds / 3600:.1f} h'


def minutes(seconds):
    return f'{seconds / 60:.0f} min'


# ---------------------------------------------------------------- run telemetry

def duration(sub):
    if not sub['finished_utc']:
        return None
    return (datetime.fromisoformat(sub['finished_utc']) - datetime.fromisoformat(sub['started_utc'])).total_seconds()


def runtimes(runs=None):
    """Measured wall time per seed: sum of protocol group durations (process start to finish, incl. grading)."""
    runs = runs or read_runs()
    out = {}
    for label in PROMPTS:
        clean = []
        for seed in SEEDS:
            subs = [s for g in runs[f'{label}-seed{seed}']['groups'] for s in g['subruns']]
            # Seeds without crash-resumed groups; a group whose grading stopped still timed its generation.
            if all(len(g['subruns']) == 1 for g in runs[f'{label}-seed{seed}']['groups']):
                clean.append(sum(duration(s) for s in subs))
        out[label] = dict(min=min(clean), max=max(clean), seeds=len(clean))
    return out


def run_metrics(group):
    """Sum generated tokens, generation seconds and restarts across a group's measured sub-runs."""
    subs = [s for s in group['subruns'] if 'metrics' in s]
    complete = all('metrics' in s for s in group['subruns'] if s['status'] == 'complete' or s['attempt'].endswith('initial'))
    total = {k: sum(s['metrics'][k] for s in subs)
             for k in ('generated_tokens', 'generation_seconds', 'restarts')}
    return dict(total, complete=complete, subruns=len(group['subruns']))


def generation_rate(metrics):
    """Generated tokens per second, matching the run-table definition."""
    if metrics['generation_seconds'] <= 0:
        raise ValueError('Generation time must be positive')
    return metrics['generated_tokens'] / metrics['generation_seconds']


def runs_section(runs, protocol):
    per = {}
    for label in PROMPTS:
        for seed in SEEDS:
            run = runs[f'{label}-seed{seed}']
            caps = {g['name']: g['max_new_tokens'] for g in protocol[label]['groups']}
            for group in run['groups']:
                m = run_metrics(group)
                dataset = group['datasets'][0]
                key = (label if dataset in CODE else 'r2', dataset)
                per.setdefault(key, {})[seed] = dict(m, cap=caps[group['group']])
    body = []
    for dataset in ORDER:
        for label in (('r2', 'code-tah2') if dataset in CODE else ('r2',)):
            seeds = per[(label, dataset)]
            cells = []
            for s in SEEDS:
                m = seeds[s]
                cells.append('—' if not m['complete'] else f"{generation_rate(m):.0f}")
            name = NAMES[dataset] + (f' ({"TaH2" if label == "code-tah2" else "harness"} prompt)' if dataset in CODE else '')
            body.append([name, seeds[SEEDS[0]]['cap'], ' / '.join(cells),
                         ' / '.join(str(seeds[s]['restarts']) + ('' if seeds[s]['complete'] else '*')
                                    for s in SEEDS)])
    summary = table('runs', ['Benchmark', 'Cap', 'tok/s (seeds 42 / 1234 / 2024)', 'Restarts'], body)
    incomplete = [f'{NAMES[d]} seed {s}' for (label, d), seed_values in per.items()
                  for s, values in seed_values.items() if not values['complete']]
    return dict(runs_table=summary, incomplete_runs=', '.join(incomplete) or 'none')


# ---------------------------------------------------------------- figure (deterministic vector PDF)

def pdf_figure(path, items):
    """Dot-and-whisker chart: ours (filled circle, 95% CI) vs paper (open diamond) per benchmark."""
    blue, orange, ink, muted, grid = '0.165 0.471 0.839', '0.922 0.408 0.204', '0.043 0.043 0.043', '0.322 0.318 0.306', '0.86 0.86 0.85'
    width, top, row = 560, 40 + 26 * len(items) + 40, 26
    x0, x1 = 200, 530

    def x(v):
        return x0 + (x1 - x0) * v

    def text(px, py, s, size=9, color=ink):
        s = s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        return f'{color} rg BT /F1 {size} Tf {px:.2f} {py:.2f} Td ({s}) Tj ET\n'

    def circle(cx, cy, r):
        k = 0.5523 * r
        return (f'{cx + r:.2f} {cy:.2f} m {cx + r:.2f} {cy + k:.2f} {cx + k:.2f} {cy + r:.2f} {cx:.2f} {cy + r:.2f} c '
                f'{cx - k:.2f} {cy + r:.2f} {cx - r:.2f} {cy + k:.2f} {cx - r:.2f} {cy:.2f} c '
                f'{cx - r:.2f} {cy - k:.2f} {cx - k:.2f} {cy - r:.2f} {cx:.2f} {cy - r:.2f} c '
                f'{cx + k:.2f} {cy - r:.2f} {cx + r:.2f} {cy - k:.2f} {cx + r:.2f} {cy:.2f} c f\n')

    def diamond(cx, cy, r):
        return f'{cx:.2f} {cy + r:.2f} m {cx + r:.2f} {cy:.2f} l {cx:.2f} {cy - r:.2f} l {cx - r:.2f} {cy:.2f} l h S\n'

    s = text(20, top - 22, 'Accuracy (pass@1, %): ours with 95% CI vs the paper', 12)
    s += f'{blue} rg ' + circle(x0 + 4, top - 42, 4) + text(x0 + 14, top - 45, 'Ours (3 seeds)', 9, muted)
    s += f'{orange} RG 1.5 w ' + diamond(x0 + 124, top - 42, 4.5) + text(x0 + 134, top - 45, 'Paper (TaH+ 1.7B)', 9, muted)
    bottom = top - 60 - row * len(items)
    for tick in range(0, 101, 20):
        s += f'{grid} RG 0.5 w {x(tick / 100):.2f} {bottom:.2f} m {x(tick / 100):.2f} {top - 56:.2f} l S\n'
        s += text(x(tick / 100) - 6, bottom - 14, str(tick), 8, muted)
    for i, (label, item) in enumerate(items):
        y = top - 60 - row * i - row / 2
        s += text(20, y - 3, label, 9)
        s += f'{blue} RG 2 w {x(item["ci"][0]):.2f} {y:.2f} m {x(item["ci"][1]):.2f} {y:.2f} l S\n'
        s += f'{blue} rg ' + circle(x(item['mean']), y, 4)
        s += f'{orange} RG 1.5 w ' + diamond(x(item['paper']), y, 4.5)
    content = s.encode('ascii')
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {top}] /Resources << /Font << /F1 4 0 R >> >> '
               f'/Contents 5 0 R >>'.encode(),
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               f'<< /Length {len(content)} >>\nstream\n'.encode() + content + b'endstream']
    out = bytearray(b'%PDF-1.4\n')
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f'{i} 0 obj\n'.encode() + obj + b'\nendobj\n'
    start = len(out)
    out += f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode()
    for offset in offsets:
        out += f'{offset:010d} 00000 n \n'.encode()
    out += f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n'.encode()
    path.write_bytes(out)


# ---------------------------------------------------------------- main

def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    for stale in ('runs_detail.tex', 'throughput.tex'):
        (TABLES / stale).unlink(missing_ok=True)
    (ROOT / 'APPENDIX.md').unlink(missing_ok=True)
    manifest = verify()
    protocol = read('protocol.json')
    runs = read_runs()
    check_settings(runs, protocol)
    datasets = {name: run['datasets'] for name, run in runs.items()}
    paper = protocol['r2']['paper_values']
    assert all(protocol['code-tah2']['paper_values'][d] == paper[d] for d in CODE)
    values = {}

    # Main table: R2 analysis as registered; code rows for both prompts, the comparison uses TaH2.
    results = {d: estimate('r2', d, datasets) for d in ORDER}
    tah2 = {d: estimate('code-tah2', d, datasets) for d in CODE}
    body, figure, analysis = [], [], {}

    def line(label, item, registered=True):
        status = verdict(item, paper[item['dataset']] if 'dataset' in item else item['paper'])
        shown = status if registered else f'({status})'
        return [label, item.get('size', ''), ' / '.join(pct(item['seeds'][str(s)]) for s in SEEDS),
                f"{pct(item['mean'])} ± {pct(item['se'])}", f"[{pct(item['ci'][0])}, {pct(item['ci'][1])}]",
                pct(item['paper']), shown]

    for d in ORDER:
        variants = [('harness', results[d], d not in CODE)] + ([('TaH2', tah2[d], True)] if d in CODE else [])
        for prompt, item, registered in variants:
            item['size'] = f"{item['problems']} × {item['samples_per_problem']}"
            label = NAMES[d] + (f' ({prompt} prompt)' if d in CODE else '')
            body.append(line(label, item, registered))
            analysis[d + ('' if d not in CODE else '_' + prompt.lower())] = dict(item, registered_comparison=registered)
            if registered or d in CODE:
                figure.append((label.replace(' prompt', ''), item))
    averages = {}
    for prompt, chosen in (('harness', results), ('TaH2', {**results, **tah2})):
        item = average([chosen[d] for d in ORDER])
        item['paper'] = protocol['r2']['paper_average']
        averages[prompt] = item
        body.append([f'Average, {prompt} code prompt', '9 benchmarks', *line('', item)[2:]])
        analysis['average_' + prompt.lower()] = item
    body.insert(len(body) - 2, ['', '', '', '', '', '', ''])
    headers = ['Benchmark', 'Problems × samples', 'Seeds 42 / 1234 / 2024 (%)', 'Ours (%) ± SE', '95% CI (%)', 'Paper (%)',
               'Result']
    values['main_table'] = table('main', headers, body, 'lllllll')
    reproduced = [NAMES[d] for d in ORDER if (tah2 if d in CODE else results)[d]['reproduced']]
    gaps = [(NAMES[d], (tah2 if d in CODE else results)[d]['gap']) for d in ORDER
            if not (tah2 if d in CODE else results)[d]['reproduced']]
    values.update(
        n_reproduced=str(len(reproduced)), reproduced_list=', '.join(reproduced) or 'none',
        gap_list='; '.join(f'{n} {signed(g)} pp' for n, g in gaps) or 'none',
        avg_harness=pct(averages['harness']['mean']), avg_harness_ci=f"[{pct(averages['harness']['ci'][0])}, {pct(averages['harness']['ci'][1])}]",
        avg_tah2=pct(averages['TaH2']['mean']), avg_tah2_ci=f"[{pct(averages['TaH2']['ci'][0])}, {pct(averages['TaH2']['ci'][1])}]",
        avg_paper=pct(protocol['r2']['paper_average']),
        avg_tah2_status=verdict(averages['TaH2'], protocol['r2']['paper_average']),
        avg_harness_status=verdict(averages['harness'], protocol['r2']['paper_average']),
        z=f'{z_value():.2f}', confidence=pct(CONFIDENCE, 0),
        mbpp_harness=pct(results['mbpp']['mean']), mbpp_tah2=pct(tah2['mbpp']['mean']),
        he_harness=pct(results['humaneval']['mean']), he_tah2=pct(tah2['humaneval']['mean']),
        mbpp_paper=pct(paper['mbpp']), he_paper=pct(paper['humaneval']),
        mbpp_tah2_status=verdict(tah2['mbpp'], paper['mbpp']), he_tah2_status=verdict(tah2['humaneval'], paper['humaneval']),
        mbpp_harness_status=verdict(results['mbpp'], paper['mbpp']),
        answers_multi=str(results['aime25']['samples_per_problem'] * len(SEEDS)), answers_single=str(len(SEEDS)),
        gpqa_paper_se=pct(math.sqrt(paper['gpqa'] * (1 - paper['gpqa']) / results['gpqa']['problems'])),
        paper_se_range='–'.join(pct(f(math.sqrt(paper[d] * (1 - paper[d]) / results[d]['problems'])
                                       for d in ORDER if results[d]['samples_per_problem'] == 1)) for f in (min, max)))
    pdf_figure(FIGURES / 'r2_vs_paper.pdf', figure)

    # Code-prompt story: failure breakdown per prompt, benchmark and seed.
    caps = {g['datasets'][0]: g['max_new_tokens'] for g in protocol['code-tah2']['groups']}
    body, tasks = [], {}
    for label in PROMPTS:
        for d in ('mbpp', 'humaneval'):
            for s in SEEDS:
                b, per_task = breakdown(label, s, d, caps[d])
                tasks[(label, d, s)] = per_task
                body.append([b['prompt'], NAMES[d], s, f"{b['failures']} / {b['tasks']}", share(b['naming'], b['failures']),
                             share(b['no_code'] + b['capped'], b['failures']), b['capped'], share(b['logic'], b['failures'])])
                analysis.setdefault('code_failures', []).append(b)
    values['failure_table'] = table('code_failures', ['Prompt', 'Benchmark', 'Seed', 'Failures / tasks',
                                                      'Entry point not defined', 'No usable code',
                                                      '…of which empty at cap', 'Wrong logic'], body)
    fails = {(b['prompt'], b['dataset'], b['seed']): b for b in analysis['code_failures']}
    h42 = fails[('harness', 'mbpp', 42)]
    naming_h = [fails[('harness', 'mbpp', s)]['naming'] for s in SEEDS]
    naming_t = [fails[('TaH2', 'mbpp', s)]['naming'] for s in SEEDS]
    capped_t = {d: [fails[('TaH2', d, s)]['capped'] for s in SEEDS] for d in CODE}
    capped_h = sum(fails[('harness', d, s)]['capped'] for d in CODE for s in SEEDS)
    rescued = [t for s in SEEDS for t, (c, _) in tasks[('r2', 'mbpp', s)].items()
               if c == 'naming' and tasks[('code-tah2', 'mbpp', s)][t][0] is None]
    values.update(
        h42_naming=str(h42['naming']), h42_base=str(h42['base_failures']), h42_failures=str(h42['failures']),
        h42_naming_base_share=pct(h42['naming'] / h42['base_failures'], 0),
        naming_h_range=f'{min(naming_h)}–{max(naming_h)}', naming_t_range=f'{min(naming_t)}–{max(naming_t)}',
        naming_h_total=str(sum(naming_h)),
        failures_h_range='–'.join(str(f(fails[('harness', 'mbpp', s)]['failures'] for s in SEEDS)) for f in (min, max)), rescued=str(len(rescued)),
        capped_he=' / '.join(map(str, capped_t['humaneval'])), capped_mbpp=' / '.join(map(str, capped_t['mbpp'])),
        capped_harness=str(capped_h), cap=str(caps['mbpp']),
        tah2_logic_mbpp=' / '.join(str(fails[('TaH2', 'mbpp', s)]['logic']) for s in SEEDS),
        harness_logic_mbpp=' / '.join(str(fails[('harness', 'mbpp', s)]['logic']) for s in SEEDS))
    # Examples: the first harness-prompt MBPP+ naming failures of seed 42, and the same tasks elsewhere.
    examples = sorted((t for t, (c, _) in tasks[('r2', 'mbpp', 42)].items() if c == 'naming'),
                      key=lambda t: int(t.split('/')[1]))[:4]
    body = []
    for t in examples:
        _, row = tasks[('r2', 'mbpp', 42)][t]
        body.append([t, f"`{row['entry_point']}`", ', '.join(f'`{n}`' for n in row['raw_defined_functions'].split()[:2]),
                     sum(tasks[('r2', 'mbpp', s)][t][0] == 'naming' for s in SEEDS),
                     sum(tasks[('code-tah2', 'mbpp', s)][t][0] is None for s in SEEDS)])
    values['examples_table'] = table('code_examples', ['Task', 'Tests call', 'Harness prompt, seed 42: model defined',
                                                       'Naming failures, harness (of 3 seeds)', 'Passed, TaH2 prompt (of 3 seeds)'],
                                     body)

    # Per-run generation rate and restart counts; engine implementation history belongs in chapter 1.
    values.update(runs_section(runs, protocol))
    dropped = [d for run in runs.values() for d in run['merge']['dropped_items']]
    values['merge_note'] = '; '.join(f"{d['problem_id']}, sample {d['sample_idx']}, an ungraded copy from the crashed "
                                     f"sub-run" for d in dropped) or 'none'
    values['amc_subruns'] = str(next(len(g['subruns']) for g in runs['r2-seed42']['groups'] if g['group'] == 'amc23'))
    t = runtimes(runs)
    values.update(r2_runtime=f"{hours(t['r2']['min'])}–{hours(t['r2']['max'])}".replace(' h–', '–'),
                  code_runtime=f"{minutes(t['code-tah2']['min'])}–{minutes(t['code-tah2']['max'])}".replace(' min–', '–'),
                  r2_runtime_seeds=str(t['r2']['seeds']), code_runtime_seeds=str(t['code-tah2']['seeds']))

    # Datasets as used.
    info = read('datasets.json')
    body = []
    for d in ORDER:
        i = info[d]
        source = i['source'] + (f", {i['subset']}" if i.get('subset') else '') + (f", split {i['split']}" if i.get('split') else '')
        version = (f"revision {i['revision'][:7]}" if i['kind'] == 'Hugging Face dataset' else f"SHA256 {i['sha256'][:12]}…")
        if 'evalplus_dataset' in i:
            version += f"; tests: EvalPlus {i['evalplus_dataset']}"
        cap = next(g['max_new_tokens'] for g in protocol['r2']['groups'] if d in g['datasets'])
        body.append([NAMES[d], results[d]['problems'], results[d]['samples_per_problem'], cap, f'`{source}`', version])
    values['datasets_table'] = table('datasets', ['Benchmark', 'Problems', 'Samples per problem', 'Max new tokens',
                                                  'Source', 'Version'], body)
    values['total_samples'] = f"{sum(r['datasets'][d]['samples'] for r in [runs['r2-seed42']] for d in ORDER):,}"
    values['data_file_count'] = str(len(manifest['files']) + 1)

    analysis['settings'] = SETTINGS
    analysis['confidence'] = CONFIDENCE
    (TABLES / 'r2-analysis.json').write_text(json.dumps(analysis, indent=2, sort_keys=True) + '\n')
    template = (ROOT / 'README.template.md').read_text()
    missing = set(re.findall(r'\{\{(\w+)\}\}', template)) - values.keys()
    if missing:
        raise ValueError('Unfilled template fields: ' + ', '.join(sorted(missing)))
    (ROOT / 'README.md').write_text(re.sub(r'\{\{(\w+)\}\}', lambda m: values[m[1]], template))
    appendix = (ROOT / 'APPENDIX.template.md').read_text()
    missing = set(re.findall(r'\{\{(\w+)\}\}', appendix)) - values.keys()
    if missing:
        raise ValueError('Unfilled appendix fields: ' + ', '.join(sorted(missing)))
    (ROOT / 'APPENDIX.md').write_text(re.sub(r'\{\{(\w+)\}\}', lambda m: values[m[1]], appendix))
    print(f'Verified {len(manifest["files"])} curated data files; wrote README.md, APPENDIX.md, tables and figure.')


if __name__ == '__main__':
    main()
