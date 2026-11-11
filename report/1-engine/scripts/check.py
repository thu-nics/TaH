#!/usr/bin/env python3
"""CPU regression for curated-data integrity, deterministic outputs and the R1 hook.

R1 uses a temporary copy of the compact confirmatory inputs to exercise the
comparison adapter and its validation failures.
"""
import ast
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1]
SCRATCH=ROOT.parents[1]/'data/tmp/report-1-engine/checks'


def crosscheck_r1o_sources(chapter):
    """Independent direct JSON-to-table check, optional on a raw-data checkout."""
    repo=ROOT.parents[1]
    data=chapter.read('r1o.json')
    raw=repo/data['provenance']['root']
    load=lambda path:json.loads(path.read_text())
    legacy=load(raw/'o1/stage1/analysis/r1o1.json')
    looks=[load(raw/f'o1/look{i}.json') for i in (1,2)]
    c=load(raw/'c1/look1.json');diag=load(repo/'data/engine/oracle/wp23c/inputs/self-split.json')
    demo=load(raw/'o1/free-composition.json');near=load(raw/'o1/near-tie.json')
    assert data['validity']==legacy['validity']
    assert data['self_split']==diag
    assert data['composition']==dict(engine_vs_engine=demo,engine_vs_reference=near)
    for saved,look in zip(data['tail_looks'],looks):
        assert saved['criteria']==look['tail']['criteria']
        assert saved['original_maximum']==look['original_maximum']
    for key in ('C1','C2','censoring','look','next_action','validity_met','conclusion'):
        assert data['c1'][key]==c[key]
    for name,row in data['forced'].items():
        assert row['upper975']==looks[0]['criteria'][name]['upper975']
        assert row['stopped_at_look']==looks[1]['criteria'][name]['stopped_for_success_at_look']==1
        original=legacy['bias'] if name=='bias' else legacy['primary'][name]
        for key in (('D','engine_bias','reference_bias','reference_M2') if name=='bias' else ('engine','reference')):
            assert row[key]==original[key]
    manifest=load(repo/'data/engine/oracle/wp23c/inputs/manifest.json');seq={s['id']:s for s in manifest['sequences']}
    for name,ids in manifest['subsets'].items():
        assert data['sizes'][name]['sequences']==len(ids)
        assert data['sizes'][name]['positions']==sum(len(seq[i]['forced']) for i in ids)
    for entry in chapter.read('MANIFEST.json')['files']['r1o.json']['sources']:
        if Path(entry['source_path']).parts[0] not in {'data', 'script', 'tah', 'models'}:
            continue
        assert hashlib.sha256((repo/entry['source_path']).read_bytes()).hexdigest()==entry['source_sha256']
    expected={}
    f=lambda x:format(x,'.6g') if isinstance(x,float) else str(x)
    expected['r1o_validity']=[[m,*[f(r[k]) for k in ('fp32_vs_fp64','reference_vs_fp64','limit')],r['status']]
                             for m,r in legacy['validity']['metrics'].items()]
    rows=[]
    for m in ('M1','M2','M3','bias'):
        r=legacy['bias'] if m=='bias' else legacy['primary'][m]
        b=looks[0]['criteria'][m]
        rows.append([m,*[f(r[k]) for k in (('engine_bias','reference_bias','D') if m=='bias' else ('engine','reference','ratio'))],
                     f(b['upper975']), '0' if m=='bias' else '1.1','1','PASS'])
    expected['r1o_forced']=rows
    tails=[];maxima=[]
    for look in looks:
        for kind in ('decision','top1'):
            r=look['tail']['criteria'][kind];m=look['original_maximum']['criteria'][kind]
            tails.append([kind,str(look['look']),str(look['free_sequences']),f(r['engine_mean_q90']),f(r['reference_mean_q90']),f(r['upper975']),
                f"{r['engine_events']}/{r['reference_events']}",r.get('observed_status',r['status'])+
                (' (descriptive)' if kind=='top1' and look['look']==2 else '')])
            maxima.append([kind,str(look['look']),str(look['free_sequences']),*[f(m[k]) for k in ('engine','reference','limit')],m['status']])
    expected['r1o_tails']=tails;expected['r1o_maximum']=maxima
    expected['r1o_c1']=[[*[f(c['C1'][k]) for k in ('problems','engine_mean','reference_mean','difference','difference_lower','limit')],f(c['look']),c['C1']['status']]]
    expected['r1o_censoring']=[[arm,str(sum(p[arm+'_cap_censored'] for p in c['per_problem'])),
        str(sum(p[arm+'_no_observed_difference'] and not p[arm+'_cap_censored'] for p in c['per_problem'])),
        str(sum(p[arm+'_no_observed_difference'] for p in c['per_problem'])),str(c['C1']['problems'])] for arm in ('engine','reference')]
    expected['r1o_composition']=[[name,kind,*[f(item[k]) for k in ('engine_mean_q90','reference_mean_q90','upper_bound')],
        f(study['composition_correlations'][kind]['disagreement_position_phi']),item['status'],item['original_max']['status']]
        for name,study in [('engine vs engine',demo),('engine vs reference',near)] for kind,item in study['criteria'].items()]
    expected['r1o_self_split']=[[kind,*[f(100*rates[k]) for k in ('max_not_shown','equality_false_alarm','boundary_false_noninferiority','noninferiority_not_shown')]]
        for kind,rates in diag['rates'].items()]
    expected['r1o_runtime']=[]
    for path in sorted(raw.glob('**/run.json')):
        r=load(path)
        seconds=sum(b['seconds'] for b in r['blocks']) if 'blocks' in r else sum(b['wall_seconds'] for b in r['telemetry'].values())
        expected['r1o_runtime'].append([str(path.parent.relative_to(raw)),f(seconds/60)])
    generated=load(ROOT/'tables/r1o-analysis.json')
    counts={}
    for name,rows in expected.items():
        assert generated['tables'][name]['rows']==rows,name
        tex=(ROOT/'tables'/f'{name}.tex').read_text()
        for row in rows:assert ' & '.join(chapter.esc(x) for x in row)+r' \\' in tex,name
        counts[name]=dict(rows=len(rows),cells=sum(len(row) for row in rows))
    result=dict(status='PASS',tables=counts,source_root=str(raw.relative_to(repo)),
                note='Every rendered table cell cross-checked directly against original saved look/validity/diagnostic/run JSON, at displayed precision.')
    SCRATCH.mkdir(parents=True,exist_ok=True)
    (SCRATCH/'r1o-crosscheck.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('R1-O direct source-to-table cross-check:',sum(x['cells'] for x in counts.values()),'cells in',len(counts),'tables PASS')


def snapshot():
    files=[ROOT/'README.md',ROOT/'APPENDIX.md',*sorted((ROOT/'tables').glob('*')),*sorted((ROOT/'figures').glob('*'))]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r1o-sources',action='store_true',help='Also compare every new table cell against original external R1-O sources')
    args=parser.parse_args()
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
    subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True)
    first=snapshot()
    subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True)
    assert first==snapshot(),'Nondeterministic generated outputs'
    assert '{{' not in (ROOT/'README.md').read_text()
    print('README, appendix, tables and PDF: byte-identical on consecutive runs')
    # Every named reproduction plan must be inspectable without GPU imports.
    for step in json.loads((ROOT/'data/method.json').read_text())['reproduction']:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),step,'--dry-run'],check=True,capture_output=True,text=True)
        if step in ('r2_speed','memory_ab'):
            assert 'curate_r2_speed_memory.py' in result.stdout
        elif step=='r1o':
            assert 'script.tah.validation.paired_look' in result.stdout and 'curate_r1o.py' in result.stdout
        else:
            assert 'Then extract' in result.stdout
    print('All named reproduction dry-runs: PASS')
    spec=importlib.util.spec_from_file_location('chapter_analysis',ROOT/'analyze.py')
    chapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(chapter)
    import copy
    chapter.r1o_section()
    assert chapter.R1O_FACTS['r1o_verdict']=='PASS'
    if args.r1o_sources:crosscheck_r1o_sources(chapter)
    SCRATCH.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='r1o-',dir=SCRATCH) as temporary:
        original_data,original_tables=chapter.DATA,chapter.TABLES
        try:
            chapter.DATA=Path(temporary);chapter.TABLES=Path(temporary)/'tables';chapter.TABLES.mkdir()
            shutil.copy(original_data/'method.json',chapter.DATA/'method.json')
            data=json.loads((original_data/'r1o.json').read_text())
            # A descriptive maximum failure must not turn the amended verdict.
            badmax=copy.deepcopy(data)
            row=badmax['tail_looks'][1]['original_maximum']['criteria']['top1']
            row.update(engine=100,status='not shown')
            (chapter.DATA/'r1o.json').write_text(json.dumps(badmax));chapter.r1o_section()
            assert chapter.R1O_FACTS['r1o_verdict']=='PASS'
            # First-look success is permanent, even if the later descriptive
            # observation would not have passed on its own.
            later=copy.deepcopy(data)
            later['tail_looks'][1]['criteria']['top1'].update(upper975=1,observed_status='not shown')
            (chapter.DATA/'r1o.json').write_text(json.dumps(later));chapter.r1o_section()
            assert chapter.R1O_FACTS['r1o_verdict']=='PASS'
            failed=copy.deepcopy(data)
            failed['forced']['M1'].update(upper975=1.2,status='not shown')
            failed['conclusion']='not shown'
            (chapter.DATA/'r1o.json').write_text(json.dumps(failed));chapter.r1o_section()
            assert chapter.R1O_FACTS['r1o_verdict']=='not shown'
            failed['conclusion']='PASS';(chapter.DATA/'r1o.json').write_text(json.dumps(failed))
            try:chapter.r1o_section()
            except AssertionError:pass
            else:raise AssertionError('Incorrect amended PASS accepted')
        finally:chapter.DATA,chapter.TABLES=original_data,original_tables
    print('R1-O: non-binding maximum, permanent tail stopping and binding-error rejection PASS')
    record=next(r for r in chapter.read('equivalence.json')['reports'] if r['checkpoint']=='real' and r['layout']=='full')
    zero=copy.deepcopy(record)
    zero['summary']['common_decision_mismatches']=0
    zero['summary']['pass_count_mismatches']=0
    for depth in zero['summary']['per_depth']:depth['common_decision_mismatches']=0
    assert chapter.ratio_cells([zero])[:2]==['0.00','0.00']
    zero['noise_limits']['pass_count_mismatches']=0
    for depth in zero['decision_limits']['per_depth']:depth['reference_flips']=0
    assert chapter.ratio_cells([zero])[:2]==['exact','exact']
    print('Ratio labels distinguish zero engine errors from exact agreement: PASS')
    chapter.exact_section()
    SCRATCH.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='check-',dir=SCRATCH) as temporary:
        original=chapter.DATA;chapter.DATA=Path(temporary)
        try:
            bad=json.loads((original/'exact.json').read_text());bad['tiny']['mlp3']['top1_mismatches']=1
            (chapter.DATA/'exact.json').write_text(json.dumps(bad))
            try:chapter.exact_section()
            except ValueError:pass
            else:raise AssertionError('Non-exact level-0 data accepted')
        finally:chapter.DATA=original
    print('Level 0: exact data accepted, a single mismatch rejected: PASS')
    policy=chapter.read('method.json')['confirmatory']
    with tempfile.TemporaryDirectory(prefix='check-',dir=SCRATCH) as temporary:
        tmp=Path(temporary);chapter.DATA=tmp/'data';chapter.TABLES=tmp/'tables'
        chapter.DATA.mkdir();chapter.TABLES.mkdir()
        shutil.copy(ROOT/'data/method.json',chapter.DATA/'method.json')
        assert 'Pending' in chapter.r1_section(policy)
        (chapter.DATA/'r1').mkdir()
        index=dict(complete=False)
        (chapter.DATA/'r1/runs.json').write_text(json.dumps(index))
        assert 'Pending' in chapter.r1_section(policy)
        shutil.copy(ROOT/'data/r1/samples.csv',chapter.DATA/'r1/samples.csv')
        shutil.copy(ROOT/'data/r1/runs.json',chapter.DATA/'r1/runs.json')
        path=chapter.DATA/'r1/runs.json';index=json.loads(path.read_text())
        result=chapter.r1_section(policy)
        assert 'ref. 1234' in result and 'Engine' in result and 'Ref. 42' not in result
        assert chapter.R1_FACTS['r1_verdict']=='equivalence not shown'
        output=json.loads((chapter.TABLES/'r1-analysis.json').read_text())
        assert output['comparisons'][0]['pooled']['n_problems']==sum(policy['problems'].values())
        index_bad=json.loads(json.dumps(index));index_bad['engine'][0]['seed']=99;path.write_text(json.dumps(index_bad))
        try:chapter.r1_section(policy)
        except AssertionError:pass
        else:raise AssertionError('Wrong R1 seed accepted')
        path.write_text(json.dumps(index))
        bad=json.loads(json.dumps(index));bad['runs'][1]['groups'][0]['settings']['temperature']=0.7
        path.write_text(json.dumps(bad))
        try:chapter.r1_section(policy)
        except ValueError:pass
        else:raise AssertionError('Settings differing from the protocol accepted')
        path.write_text(json.dumps(index))
        csv=chapter.DATA/'r1/samples.csv';lines=csv.read_text().splitlines(True)
        csv.write_text(''.join(lines+[lines[1]]))
        try:chapter.r1_section(policy)
        except ValueError:pass
        else:raise AssertionError('Duplicate R1 sample accepted')
    print('R1: missing/incomplete skip, compact API analysis, wrong-seed, wrong-settings and duplicate rejection PASS')
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert size<3_000_000,size
    print(f'Chapter size: {size/1e6:.2f} MB')

if __name__=='__main__':main()
