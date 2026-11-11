#!/usr/bin/env python3
"""CPU-only extraction of the sufficient metrics used by the chapter.

Original outputs remain in experiment directories. Source paths/hashes are
recorded per curated file; no result or validation threshold is fitted here.
"""
import argparse
import ast
import hashlib
import json
import lzma
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
DATA=ROOT/'data'
CHECKPOINTS=('real','mlp3','mlp1','label3')


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def select(value,keys):
    return {k:value[k] for k in keys.split() if k in value}


def performance(value):
    result=select(value,'status device workload requests batch_size max_new_tokens peak_gpu_allocated_bytes peak_gpu_reserved_bytes cuda_graphs compile overlap runtime_sha256')
    result['run']=select(value['run'],'generated_tokens wall_seconds tokens_per_second')
    return result


def parity(value):
    return select(value,'checkpoint status summary noise_limits decision_limits generated_positions limits_sha256 limits_saved_unix_time comparison_unix_time')


def profile(value):
    return select(value,'decode_steps context unprofiled_mean_step_ms mean_kernel_gpu_ms mean_gpu_copy_ms estimated_unprofiled_python_launch_idle_ms overlap_host_ms runtime_sha256')


def robustness(value):
    result=select(value,'status requests callbacks max_new_tokens total_prompt_tokens kv_pool_high_water_entries output_tokens decisions_identical kv_audit host_swap_used_bytes runtime_sha256')
    result['run']=select(value['run'],'generated_tokens swap_outs swap_ins restarts wasted_tokens host_swap_peak_bytes')
    if 'results' in value:
        result['max_prompt_tokens']=max(len(r.get('prompt_pass_counts',[])) for r in value['results'])
    return result


def exact_metrics(m):
    """Counts of one engine-vs-reference exact comparison (all differences expected zero)."""
    return dict(positions=m['top1_compared'],coordinates=m['logit_elements'],decisions=m['decision_compared'],
                depth_cap=len(m['per_depth']),decision_mismatches=m['decision_mismatches'],
                visited_mismatches=m['visited_mismatches'],pass_count_mismatches=m['pass_count_mismatches'],
                top1_mismatches=m['top1_mismatches'],logit_max_abs=m['logit_max_abs'],
                logit_squared_sum=m['logit_squared_sum'])


def constructor_cases(source):
    """Option dictionaries of the constructor-generality test (read from the test source)."""
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(getattr(t,'id',None)=='CASES' for t in node.targets):
            return [{k.arg:k.value.value for k in call.keywords if isinstance(k.value,ast.Constant)} for call in node.value.elts]
    raise ValueError('CASES not found')


class Collector:
    def __init__(self,sources,root=REPO):
        self.sources=sources
        self.root=root
        self.used=[]

    def read(self,key):
        given=Path(self.sources[key]);path=given if given.is_absolute() else self.root/given
        b=path.read_bytes();shown=given if not given.is_absolute() else path
        self.used.append(dict(source_path=str(shown.relative_to(REPO)) if shown.is_absolute() and shown.is_relative_to(REPO) else str(shown),source_sha256=hashlib.sha256(b).hexdigest(),source_bytes=len(b),source_mtime_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()))
        return path,b

    def load(self,key):
        path,b=self.read(key)
        return json.loads(lzma.decompress(b) if path.suffix=='.xz' else b)

    def load_text(self,key):
        return self.read(key)[1].decode()

    def save(self,name,value,purpose):
        dump(DATA/name,value)
        manifest=json.loads((DATA/'MANIFEST.json').read_text())
        b=(DATA/name).read_bytes()
        manifest['files'][name]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),encoding='curated JSON',purpose=purpose,sources=self.used,top_level_keys_or_columns=list(value))
        dump(DATA/'MANIFEST.json',manifest);self.used=[]

    def collect(self,step):
        if step=='speed':
            old=json.loads((DATA/'speed.json').read_text())
            rows={mode:performance(self.load('speed/'+mode)) for mode in ('eager','graphs','overlap') if 'speed/'+mode in self.sources}
            p=profile(self.load('speed/profile'))
            self.save('speed.json',dict(reference=old['reference'],reference_provenance=old['reference_provenance'],runs=rows,profile=p),'Generation counts, seconds, peak bytes and step profile; reference has a different memory/timing scope.')
        elif step=='equivalence':
            reports=[];greedy=[]
            for c in CHECKPOINTS:
                for layout in ('full','cached-1','cached-256','generated'):
                    r=parity(self.load(f'parity/{layout}/{c}'));r['layout']=layout;r['checkpoint']=c
                    if layout=='generated':
                        reg=self.load(f'parity/generated/{c}-limits')
                        assert self.used[-1]['source_sha256']==r['limits_sha256']
                        assert reg['saved_unix_time']==r['limits_saved_unix_time']<r['comparison_unix_time']
                        corpus=self.load(f'parity/generated/{c}-corpus')
                        assert self.used[-1]['source_sha256']==reg['corpus_manifest_sha256']
                        r['registration']=select(reg,'saved_unix_time corpus_manifest_sha256 source layouts')
                    reports.append(r)
                g=self.load('greedy/'+c)
                greedy.append(select(g,'checkpoint status median_first_divergence reference_median_first_divergence median_limit position_zero_divergences reference_position_zero_divergences'))
            self.save('equivalence.json',dict(reports=reports,greedy=greedy),'Per-layout aggregate G-TF metrics and reference limits, generated registration timestamps/hashes, greedy medians.')
        elif step=='diagnostics':
            d=self.load('diagnostics')
            self.save('diagnostics.json',dict(sequence_count=len(d['sequences']),logits=d['logits'],sampler=d['sampler']),'Long-context logit buckets and identical-logit sampler distributions (TV, support, entropy).')
        elif step=='robustness':
            rows={k:robustness(self.load('robustness/'+k)) for k in ('swap','stress','prompt','restart')}
            identity=self.load('robustness/overlap') if 'robustness/overlap' in self.sources else None
            self.save('robustness.json',dict(tests=rows,overlap=select(identity,'status rows_identical decisions_identical tokens_counts_identical forced_events runtime_sha256') if identity else None),'Swap identity, long-generation/prompt and restart completion/cleanup; overlap composition-controlled identity.')
        elif step=='negative':
            self.save('negative.json',dict(compiled=performance(self.load('negative/perf')),graphs=performance(self.load('negative/graphs')),parity=parity(self.load('negative/parity')),greedy=select(self.load('negative/greedy'),'status median_first_divergence reference_median_first_divergence median_limit')),'Compile-enabled throughput and failing learned-decider numerical/greedy checks.')
        elif step=='exact':
            tolerance=dict(atol=0,rtol=0)
            tiny={};fixture=self.load('exact/fixtures')
            for c in CHECKPOINTS[1:]:
                d=self.load('exact/tiny/'+c)
                assert d['production_gates_eligible'] is False
                tiny[c]=dict(exact_metrics(d['summary']),records=sum(r['checkpoint']==c and r['kind']=='teacher_forced' for r in fixture['records']))
            d=self.load('exact/long');long=dict(exact_metrics(d['metrics']),records=1,length=d['length'])
            probes={}
            for c in CHECKPOINTS:
                d=self.load('exact/probe/'+c);probes[c]=dict(exact_metrics(d['metrics']),records=1,length=d['length'])
            layers=sorted({self.load('exact/config/'+c)['num_hidden_layers'] for c in CHECKPOINTS[1:]})
            cases=constructor_cases(self.load_text('exact/tests'))
            self.save('exact.json',dict(tolerance=tolerance,tiny=tiny,tiny_layers=layers,long=long,probes=probes,
                constructor=dict(cases=len(cases),caps=sorted({c['max_iter'] for c in cases}),layers=sorted({c.get('layers',2) for c in cases}))),
                'Engine-vs-reference counts of the CPU comparisons asserted at atol=rtol=0 (tiny sets, 4096-token tiny probe, 8-token probes of full-size checkpoints, constructor cases).')
        elif step=='port':
            text=self.load_text('port/summary')
            def find(pattern):
                m=re.search(pattern,text,re.M)
                if not m:raise ValueError('Port summary lacks: '+pattern)
                return m
            gpu=find(r'^GPU: (.+?), (\d+) GB VRAM, compute capability ([\d.]+)')
            driver=find(r'^Driver: ([\d.]+); toolkit: CUDA ([\d.]+)')
            env=find(r'^Environment: Python ([\d.]+), torch ([^,]+), Transformers ([\d.]+), PEFT ([\d.]+), FlashInfer ([\d.]*\d)')
            suite=find(r'^- Default suite: (\d+) passed, (\d+) skipped, (\d+) deselected \(([\d.]+) s\)')
            smoke=find(r'^- W-smoke: (PASS), (\d+) requests, (\d+) generated tokens\. Tokens, decisions and pass counts identical between eager and CUDA graph')
            overlap=find(r'^- Overlap identity: (PASS), (\d+) requests, (\d+) output tokens, (\d+) steps\.')
            self.save('port.json',dict(status=find(r'^# New-machine engine validation: (\w+)').group(1),commit=find(r'^Commit: ([0-9a-f]{40})').group(1),
                date=find(r'^Date: (\S+)').group(1),device=gpu.group(1),vram_gb=int(gpu.group(2)),compute_capability=gpu.group(3),
                driver=driver.group(1),cuda=driver.group(2),python=env.group(1),torch=env.group(2),transformers=env.group(3),
                peft=env.group(4),flashinfer=env.group(5),
                default_suite=dict(passed=int(suite.group(1)),skipped=int(suite.group(2)),deselected=int(suite.group(3)),seconds=float(suite.group(4))),
                wsmoke=dict(status=smoke.group(1),requests=int(smoke.group(2)),generated_tokens=int(smoke.group(3)),eager_graph_identical=True),
                overlap_identity=dict(status=overlap.group(1),requests=int(overlap.group(2)),output_tokens=int(overlap.group(3)),steps=int(overlap.group(4))),
                report='Parsed from the validation summary written by the operator; raw JSON reports and logs stay on the validation machine.'),
                'Validation of the engine on a second GPU (L40S): environment, default suite, W-smoke graph/eager identity, overlap identity.')
        else:raise ValueError(step)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=['speed','equivalence','diagnostics','robustness','negative','exact','port'])
    p.add_argument('--sources',type=Path,default=ROOT/'scripts/sources.json')
    p.add_argument('--root',type=Path,default=REPO,help='directory against which relative source paths are resolved')
    a=p.parse_args();Collector(json.loads(a.sources.read_text()),a.root.resolve()).collect(a.step)

if __name__=='__main__':main()
