#!/usr/bin/env python3
"""Named GPU experiments followed by CPU extraction into the chapter's data/.
No arguments list steps. --dry-run prints exact commands without executing them.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
sys.dont_write_bytecode=True
from curate import Collector, dump, CHECKPOINTS

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
PY=REPO/'.venv-tah/bin/python'
SCRIPTS=ROOT/'scripts'
DATA=ROOT/'data'


def plan(step,out):
    commands=[];sources={}
    def add(label,args,gpu=True,expected_failure=False):
        commands.append(dict(label=label,args=[str(PY),*map(str,args)],gpu=gpu,expected_failure=expected_failure))
    def check(label,gate,*flags):
        dest=out/(label+'.json')
        add(label,[SCRIPTS/'validation_cli.py','engine_check','--gate',gate,'--output',dest,*flags])
        return str(dest)
    def generated(c,compiled=False):
        dest=out/('compiled' if compiled else 'generated')
        for phase in ('generate','noise','compare'):
            add(f'{dest.name}-{c}-{phase}',[SCRIPTS/'validation_cli.py','generated_gtf','--checkpoint',c,'--root',dest,'--phase',phase,*(['--compile'] if compiled else [])],gpu=phase!='compare',expected_failure=compiled and phase=='compare')
        return str(dest/f'tf-{c}-expanded.json'),str(dest/'gtf-limits'/f'{c}.json'),str(dest/'gtf-corpus'/c/'manifest.json')
    if step=='fixtures':
        args=['-m','tah.engine.generate_fixtures','--profile','full','--seed','20261012','--output',out/'fixtures','--model-output',out/'fixture-models']
        add('prepare',[*args,'--prepare-only'],False)
        for c in CHECKPOINTS:add('fixture-'+c,[*args,'--device','cuda','--checkpoint',c])
        add('fixture-limits',[SCRIPTS/'export_fixture_limits.py'],False)
    elif step=='reference':
        add('reference',[SCRIPTS/'reference_perf.py',out/'reference-perf'])
    elif step=='speed':
        for mode,flags in [('eager',['--no-cuda-graphs']),('graphs',['--cuda-graphs']),('overlap',['--cuda-graphs','--overlap'])]:
            sources['speed/'+mode]=check('speed-'+mode,'perf','--batch-size','64','--max-new-tokens','4096','--no-compile',*flags)
        sources['speed/profile']=check('profile','profile','--batch-size','64','--cuda-graphs','--no-compile')
    elif step=='equivalence':
        for c in CHECKPOINTS:
            add('full-'+c,['-m','pytest','-q','-o','addopts=','-m','gpu and slow','tah/engine/tests/test_gtf.py','-k',c])
            sources['parity/full/'+c]=str(out/'full'/f'{c}.json')
            for chunk in (1,256):
                add(f'cached-{chunk}-{c}',[SCRIPTS/'cached_validation.py','--checkpoint',c,'--chunk',str(chunk)],expected_failure=c=='mlp1' and chunk==1)
                sources[f'parity/cached-{chunk}/{c}']=str(out/f'cached-{chunk}'/f'{c}.json')
            report,limits,corpus=generated(c)
            for suffix,path in [('',report),('-limits',limits),('-corpus',corpus)]:sources[f'parity/generated/{c}{suffix}']=path
            sources['greedy/'+c]=check('greedy-'+c,'gen','--checkpoint',c,'--no-compile')
    elif step=='diagnostics':
        sources['diagnostics']=str(out/'longctx.json')
        add('longctx',[SCRIPTS/'longctx.py','--output',sources['diagnostics']])
    elif step=='robustness':
        for key,gate,cap in [('swap','swap',4096),('stress','robust',16384),('prompt','long-prompt',8)]:
            sources['robustness/'+key]=check('robustness-'+key,gate,'--batch-size','64','--max-new-tokens',str(cap),'--cuda-graphs','--no-compile')
        sources['robustness/overlap']=check('overlap-identity','overlap-identity')
        add('restart',['-m','pytest','-q','-s','-o','addopts=','tah/engine/tests/test_engine_gpu.py::test_cuda_overflow_restart_repeated_completion_and_cleanup','--basetemp',out/'restart-test'])
        sources['robustness/restart']=str(out/'restart.json')
    elif step=='negative':
        sources['negative/graphs']=check('negative-graphs','perf','--batch-size','64','--no-compile')
        sources['negative/perf']=check('negative-compiled','perf','--batch-size','64','--compile','--graph-batch-sizes','54,55,58,60,64')
        report,_,_=generated('mlp3',True);sources['negative/parity']=report
        sources['negative/greedy']=check('negative-greedy','gen','--checkpoint','mlp3','--compile')
        commands[-1]['expected_failure']=True
    elif step=='exact':
        # CPU only. The tests write their reports to the engine validation data directory.
        for label,test in [('exact-model','test_model.py'),('exact-probes','test_cpu_probes.py')]:
            add(label,['-m','pytest','-q','-o','addopts=','tah/engine/tests/'+test],gpu=False)
        sources.update({k:v for k,v in json.loads((SCRIPTS/'sources.json').read_text()).items() if k.startswith('exact/')})
    elif step=='accuracy':
        for backend,seeds in [('tah',[1234]),('tah_engine',[42,1234,2024])]:
            for seed in seeds:
                dest=out/f'r1-{backend}-{seed}'
                add(dest.name,['script/tah/evaluation/run_protocol.py','run','--config','script/tah/recipes/qwen3_1.7/protocol_wacc.yaml','--model_path','models/TaH-plus-1.7B','--backend',backend,'--seed',seed,'--output_root',dest])
                add('merge-'+dest.name,['script/tah/evaluation/run_protocol.py','merge','--output_root',dest],False)
    else:raise ValueError(step)
    return commands,sources


def source_inventory():
    return {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPO/'tah/engine').glob('*.py'))}


def accuracy_inputs(out,inventory,commit):
    sys.path.insert(0,str(SCRIPTS))
    try:
        import curate_r1
    finally:
        sys.path.pop(0)
    digest=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    specs=[]
    for name,backend,seed in [('reference-seed1234','tah',1234),('engine-seed42','tah_engine',42),
                              ('engine-seed1234','tah_engine',1234),('engine-seed2024','tah_engine',2024)]:
        specs.append((name,backend,seed,out/f'r1-{backend}-{seed}','protocol'))
    sample_bytes,index,sources,rows=curate_r1.collect(specs,inventory,digest,commit,direct_2024=False)
    count,file_count=curate_r1.store_compact_archive(sample_bytes,index,sources,rows,drop_legacy=True)
    dump(out/'r1-index.json',index)
    print(f'Imported {count} confirmatory samples in {file_count} chapter data files.')


def reproduce_r1o(out, dry_run, gpu):
    """Committed validation commands for the measured, frozen amended study.

    The full frozen manifests and historical diagnostic are external inputs.
    Run each capture once; strict replay failures are never retried. As in the
    original chain, validity precedes every primary run and continuation follows
    saved decisions. No new selection, pilots or C2 measurement is performed.
    """
    evidence=json.loads((DATA/'r1o.json').read_text())
    out=out/'r1o'
    if out==DATA or out.is_relative_to(DATA):
        raise ValueError('Keep complete R1-O runs outside data/')
    if not dry_run and out.exists():
        raise ValueError('Use a fresh --output-root for R1-O; preserve existing evidence')
    print('R1-O uses committed script/tah/validation modules.',flush=True)
    print('Prerequisites: data/engine/oracle/wp23c/inputs/{manifest,c1-manifest,self-split}.json, checkpoint and preserved launch history. '
          'The CPU pilot self-split is imported, not repeated. OOM/replay errors stop; use the recorded fitting sizes.',flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',
             TMPDIR=str(REPO/'data/tmp/report-1-engine'),TOKENIZERS_PARALLELISM='false')
    if not dry_run:
        (REPO/'data/tmp/report-1-engine').mkdir(parents=True,exist_ok=True)
        # Freeze/check experiment sources, model files and the immutable manifests.
        # Historical inventories may also name removed diagnostics and source archives.
        frozen=evidence['provenance']['source_hashes']
        for name,digest in frozen.items():
            if Path(name).parts[0] not in {'script', 'tah', 'models'} or name in evidence['provenance']['unavailable_historical_sources']:
                continue
            path=REPO/name
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=digest:raise ValueError('Changed frozen R1-O input: '+name)
        inputs=json.loads((DATA/'MANIFEST.json').read_text())['files']['r1o.json']['sources']
        for entry in inputs:
            if entry['source_path'].startswith('data/engine/oracle/wp23c/inputs/'):
                actual=hashlib.sha256((REPO/entry['source_path']).read_bytes()).hexdigest()
                if actual!=entry['source_sha256']:
                    raise ValueError('Changed frozen R1-O manifest input: '+entry['source_path'])
        out.mkdir(parents=True)
        (out/'source.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(frozen.items())))
    def run(label,module,args,cpu=False):
        command=[str(PY),'-m','script.tah.validation.'+module,*map(str,args)]
        print(shlex.join(command),flush=True)
        if not dry_run:
            with (out/(label+'.log')).open('w') as stream:
                subprocess.run(command,cwd=REPO,env=dict(env,CUDA_VISIBLE_DEVICES='' if cpu else gpu),
                               stdout=stream,stderr=subprocess.STDOUT,check=True)
    model='models/TaH-plus-1.7B';config='script/tah/recipes/qwen3_1.7/eval_tah.yaml'
    manifest='data/engine/oracle/wp23c/inputs/manifest.json';cmanifest='data/engine/oracle/wp23c/inputs/c1-manifest.json'
    def cap(label,impl,subset,dest,*flags):
        run(label,'capture',[impl,'--manifest',manifest,'--model',model,'--eval-config',config,
            '--buffer-dir',REPO/'data/tmp/report-1-engine/buffer','--buffer-gb','24','--subset',subset,
            *(['--kv-capacity','148000'] if impl=='engine' else []),*flags,'--out',dest])
    s=out/'o1/stage1';v=s/'oracle-fp64-c128';v32=s/'validity-fp32-c256';vr=s/'validity-reference'
    cap('validity-oracle','reference','validity',v,'--dtype','fp64','--layout','chunk','--chunk','128','--mode','free')
    cap('validity-fp32','reference','validity',v32,'--dtype','fp32','--layout','chunk','--chunk','256','--mode','forced','--target',v)
    cap('validity-reference','reference','validity',vr,'--mode','forced','--target',v)
    print('Validity gate: require all three fp32-vs-fp64 errors <= reference error /20, >=20 sequences and >=20000 positions.',flush=True)
    if not dry_run:
        sys.path.insert(0,str(REPO))
        from types import SimpleNamespace
        from script.tah.validation.analyze import validity
        m=json.loads((REPO/manifest).read_text())
        vocab=json.loads((REPO/model/'config.json').read_text())['vocab_size']
        val=validity(SimpleNamespace(validity_oracle=v,validity_fp32=v32,validity_reference=vr),vocab,
                     {s['id']:s['dataset'] for s in m['sequences']})
        dump(out/'validity.json',val)
        if not val['status'].startswith('PASS'):raise ValueError('Invalid oracle arithmetic: stop entire R1-O study')
    def oracle(stage):
        directory=out/f'o1/stage{stage}';target=directory/'oracle-fp32-c256'
        cap(f'oracle-{stage}','reference',f'stage{stage}',target,'--dtype','fp32','--layout','chunk','--chunk','256','--mode','free')
        if not dry_run:(directory/'oracle-fp32.path').write_text(str(target)+'\n')
        return directory,target
    s,a=oracle(1)
    for mode in ('forced','free'):
        for impl,name in (('reference','ref'),('engine','engine')):
            flags=['--mode',mode]+(['--target',a] if mode=='forced' else [])
            if impl=='engine':flags+=['--block','32','--max-batch-size','32']
            cap(name+'-'+mode,impl,'stage1',s/(name+'-'+mode),*flags)
    cap('ref-b1','reference','b1',s/'ref-b1','--batch','1','--mode','forced','--target',a)
    report=s/'analysis/r1o1.json'
    run('o1-means','analyze',['--manifest',manifest,'--oracle',a,'--reference',s/'ref-forced','--engine',s/'engine-forced',
        '--reference-free',s/'ref-free','--engine-free',s/'engine-free','--reference-b1',s/'ref-b1',
        '--validity-oracle',v,'--validity-fp32',v32,'--validity-reference',vr,'--output',s/'analysis'],True)
    cap('engine-free-reordered','engine','stage1',s/'engine-free-reordered','--block','64','--max-batch-size','32',
        '--sequence-order','reverse','--mode','free')
    for label,candidate,comparator,flags in [('near-tie',s/'engine-free',s/'ref-free',['--alpha','.025']),
                                          ('free-composition',s/'engine-free-reordered',s/'engine-free',[])]:
        run(label,'near_tie',['--oracle',a,'--engine',candidate,'--reference',comparator,*flags,'--output',out/f'o1/{label}.json'],True)
    run('o1-look1','paired_look',['--root',out,'--manifest',manifest,'--look','1'],True)
    def c1(stage):
        directory=out/f'c1/stage{stage}'
        for impl,name,flags in [('engine','engine',[]),('reference','reference',[]),('oracle','oracle-b4',['--oracle-batch','4'])]:
            run(f'c1-{stage}-{impl}','coupled_run',['run',impl,'--manifest',cmanifest,'--model',model,'--eval-config',config,
                '--max-new-tokens','512','--subset',f'stage{stage}',*flags,'--out',directory/name])
        if not dry_run:(directory/'oracle.path').write_text(str(directory/'oracle-b4')+'\n')
        run(f'c1-look{stage}','short_c1',['analyze','--manifest',cmanifest,'--root',out/'c1','--look',str(stage),'--validity-report',report],True)
    c1(1)
    if dry_run:
        actions=dict(forced=False,free=True)
        print('Dry-run shows observed continuation: forced stopped; exactly +136 free sequences; C1 stopped. '
              'Execution reads the newly saved look decisions.',flush=True)
    else:
        look=json.loads((out/'o1/look1.json').read_text())
        actions={arm:look[arm+'_next_action']=='continue_fixed_136' for arm in ('forced','free')}
    if any(actions.values()):
        s,a=oracle(2)
        for mode in ('forced','free'):
            if not actions[mode]:continue
            for impl,name in (('reference','ref'),('engine','engine')):
                flags=['--mode',mode]+(['--target',a] if mode=='forced' else [])
                if impl=='engine':flags+=['--block','34','--max-batch-size','32']
                cap(f'{name}-{mode}-stage2',impl,'stage2',s/(name+'-'+mode),*flags)
        run('o1-look2','paired_look',['--root',out,'--manifest',manifest,'--look','2'],True)
    if not dry_run and json.loads((out/'c1/look1.json').read_text())['next_action']=='continue_fixed_512':c1(2)
    command=[str(PY),str(SCRIPTS/'curate_r1o.py'),'--source-root',str(out)]
    print(shlex.join(command),flush=True)
    print('Then extract R1-O aggregates, update data/MANIFEST.json, and run analyze.py.',flush=True)
    if not dry_run:
        subprocess.run(command,cwd=REPO,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
        subprocess.run([str(PY),str(ROOT/'analyze.py')],cwd=REPO,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)


def main():
    policy=json.loads((DATA/'method.json').read_text())['reproduction']
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',nargs='?',choices=list(policy))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--output-root',type=Path,default=ROOT/'reproduction')
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    if not args.step:
        for name,item in policy.items():
            output = ('reproduction/fixtures/' if name=='fixtures' else
                      'data/r1/samples.csv, data/r1/runs.json' if name=='accuracy' else
                      'data/speed.json' if name=='reference' else
                      'data/r2_speed.json' if name=='r2_speed' else
                      'data/memory_ab.json' if name=='memory_ab' else f'data/{name}.json')
            print(f"{name}: {item['runtime']}; {item['device']}; {output}")
        print('Use STEP --dry-run for exact commands; run fixtures before equivalence/negative. No work executed.')
        return
    if args.step=='r1o':
        out=REPO/'data/tmp/report-1-engine/reproduction' if args.output_root==ROOT/'reproduction' else args.output_root.resolve()
        reproduce_r1o(out,args.dry_run,args.gpu)
        return
    if args.step in ('r2_speed', 'memory_ab'):
        command=[str(PY),SCRIPTS/'curate_r2_speed_memory.py',args.step]
        print(shlex.join(map(str,command)),flush=True)
        if args.step=='r2_speed':
            print('Reads R2 metrics in data/engine/paper_protocol_seed{42,1234,2024}; then analyze.py refreshes the report.',flush=True)
        else:
            print('Reads the saved old/new runs and comparisons in data/engine/memory_ab/; analyze.py refreshes the report.',flush=True)
        if args.dry_run:return
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        subprocess.run(list(map(str,command)),cwd=REPO,env=env,check=True)
        subprocess.run([str(PY),str(ROOT/'analyze.py')],cwd=REPO,env=env,check=True)
        return
    out=args.output_root.resolve()
    if out==DATA or out.is_relative_to(DATA):parser.error('Keep full experiment outputs outside curated data/')
    commands,sources=plan(args.step,out)
    for item in commands:print(shlex.join(item['args']),flush=True)
    print('Then extract '+args.step+' metrics, update data/MANIFEST.json, and run analyze.py.',flush=True)
    if args.dry_run:return
    out.mkdir(parents=True,exist_ok=True)
    if args.step in ('equivalence','negative') and not (out/'fixture-limits/tolerances.json').exists():
        parser.error('Run fixtures first using the same --output-root')
    if args.step=='exact':
        need=['data/engine/fixtures/cpu-tiny/manifest.json','data/engine/fixture-models/tiny/label3','data/engine/fixture-models/full/label3','models/TaH-plus-1.7B']
        absent=[x for x in need if not (REPO/x).exists()]
        if absent:parser.error('Missing prerequisites (tests would skip silently): '+', '.join(absent))
    started=time.time()
    inventory=source_inventory()
    if args.step=='accuracy':
        freeze=out/'r1-engine-freeze.json'
        if freeze.exists() and json.loads(freeze.read_text())!=inventory:raise ValueError('Engine changed since R1 began; do not mix versions')
        dump(freeze,inventory)
    env=dict(os.environ,PYTHONPATH=str(REPO),PYTHONDONTWRITEBYTECODE='1',ENGINE_REPORT_OUTPUT=str(out),TAH_GTF_FIXTURES=str(out/'fixtures'),TAH_GTF_TOLERANCES=str(out/'fixture-limits/tolerances.json'),TAH_GTF_RESULTS=str(out/'full'))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    dump(out/(args.step+'-commands.json'),dict(commands=commands,engine_sources=inventory,git_commit=commit,started_utc=datetime.now(timezone.utc).isoformat()))
    for item in commands:
        env['CUDA_VISIBLE_DEVICES']=args.gpu if item['gpu'] else ''
        log=out/(item['label']+'.log')
        with log.open('w') as f:result=subprocess.run(item['args'],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode and not (item['expected_failure'] and result.returncode==1):
            raise RuntimeError(f"{item['label']} exited {result.returncode}; inspect {log}")
        if item['label']=='restart':
            line=next(x for x in log.read_text().splitlines() if 'Forced restart CUDA report:' in x)
            value=json.loads(line.split('Forced restart CUDA report:',1)[1]);value['status']='PASS';dump(out/'restart.json',value)
    if inventory!=source_inventory():raise ValueError('Engine changed during the experiment; outputs retained but not imported')
    if args.step=='exact':
        stale=[v for k,v in sources.items() if k not in ('exact/tests','exact/fixtures') and not k.startswith('exact/config/') and (REPO/v).stat().st_mtime<started]
        if stale:raise ValueError('Reports not rewritten by this run (tests skipped?): '+', '.join(stale))
    if args.step=='accuracy':accuracy_inputs(out,inventory,commit)
    elif args.step not in ('fixtures','reference'):
        dump(out/(args.step+'-sources.json'),sources);Collector(sources).collect(args.step)
    subprocess.run([str(PY),str(ROOT/'analyze.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),check=True)

if __name__=='__main__':main()
