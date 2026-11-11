"""Long-context validation: long-context numerics by position bucket and sampler distributions.

Engine: token-by-token cached decode replay (eager, production decode path) with
forced W-acc tokens. Reference: one full-sequence forward per sequence; noise
baseline: reference chunk256 vs reference full. All metrics are streamed per
bucket; no full-vocabulary logits are written to disk.
"""
import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from tah.engine import TaHEngine, Request, SamplingParams
from tah.engine.engine import StepBucket
from tah.engine.generate_fixtures import load_reference
from tah.engine.reference_capture import capture
from tah.engine.sampling import GraphSampler
from tah.model.utils import sample_next_token

import lzma
import os
CHAPTER = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('ENGINE_REPORT_OUTPUT', str(CHAPTER / 'reproduction')))
MODEL = Path(os.environ.get('ENGINE_REPORT_MODEL', 'models/TaH-plus-1.7B'))
SOURCES = json.loads((CHAPTER / 'scripts/sources.json').read_text())
RUNS = {d: CHAPTER.parents[1] / SOURCES['longctx/' + d] for d in ('math500', 'gpqa')}

BUCKETS = ['0-1k', '1-2k', '2-3k', '3k+']
T, TOP_K, TOP_P = .6, 20, .95


def bucket_of(pos):
    return BUCKETS[min(pos // 1000, 3)]


def select(tokenizer, per_kind):
    chosen = []
    for dataset, run in RUNS.items():
        counts = {'eos': 0, 'length': 0}
        for line in (lzma.open(run, 'rt') if run.suffix == '.xz' else run.open()):
            r = json.loads(line)
            reason = r['finish_reason']
            if len(r['token_ids']) < 3000 or counts.get(reason, per_kind) >= per_kind:
                continue
            rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': r['problem']}],
                                                     tokenize=False, add_generation_prompt=True)
            prompt = tokenizer.encode(rendered)
            if len(prompt) != len(r['prompt_pass_counts']):
                continue  # cannot reconstruct the exact prompt; skip rather than guess
            counts[reason] += 1
            chosen.append(dict(dataset=dataset, id=r['id'], sample=r['sample'], finish=reason,
                               correct=r['is_correct'], prompt=prompt, tokens=r['token_ids'],
                               pass_counts=r['pass_counts']))
    return chosen


class LogitAcc:
    """Full-vocab logit error, top-1 and fp64 sampling-distribution TV per bucket."""
    def __init__(self):
        self.b = {k: dict(positions=0, elements=0, sq=0., abs=0., max_abs=0., top1_mismatch=0,
                          tv_sum=0., tv_max=0., tv_gt_0p01=0) for k in BUCKETS + ['prompt']}

    def add(self, key, ref, cand):
        ref, cand = ref.float(), cand.float()
        d = ref - cand
        a = self.b[key]
        a['positions'] += len(ref)
        a['elements'] += d.numel()
        a['sq'] += float(d.square().double().sum())
        a['abs'] += float(d.abs().double().sum())
        a['max_abs'] = max(a['max_abs'], float(d.abs().max()))
        a['top1_mismatch'] += int((ref.argmax(-1) != cand.argmax(-1)).sum())
        tv = tv64(ref, cand)
        a['tv_sum'] += float(tv.sum())
        a['tv_max'] = max(a['tv_max'], float(tv.max()))
        a['tv_gt_0p01'] += int((tv > .01).sum())

    def report(self):
        out = {}
        for k, a in self.b.items():
            n = max(a['positions'], 1)
            out[k] = dict(positions=a['positions'], rmse=(a['sq'] / max(a['elements'], 1)) ** .5,
                          mae=a['abs'] / max(a['elements'], 1), max_abs=a['max_abs'],
                          top1_mismatch=a['top1_mismatch'], top1_rate=a['top1_mismatch'] / n,
                          sampling_tv_mean=a['tv_sum'] / n, sampling_tv_max=a['tv_max'],
                          sampling_tv_gt_0p01=a['tv_gt_0p01'])
        return out


def dist64(logits):
    """Canonical high-precision T/top-k/top-p distribution as dense fp64 rows."""
    v, i = logits.float().topk(TOP_K, dim=-1)
    p = torch.softmax(v.double() / T, -1)
    c = p.cumsum(-1)
    remove = torch.cat((torch.zeros_like(c[:, :1], dtype=torch.bool), (c > TOP_P)[:, :-1]), -1)
    p = torch.where(remove, 0., p)
    p = p / p.sum(-1, keepdim=True)
    return torch.zeros(logits.shape, dtype=torch.float64, device=logits.device).scatter_(-1, i, p)


def tv64(a, b):
    return .5 * (dist64(a) - dist64(b)).abs().sum(-1)


def reference_probs(logits):
    """Exact reference sample_next_token distribution (multinomial intercepted)."""
    captured = {}
    original = torch.multinomial

    def fake(p, num_samples, *args, **kwargs):
        captured['p'] = p
        return torch.zeros(p.shape[0], num_samples, dtype=torch.long, device=p.device)
    torch.multinomial = fake
    try:
        sample_next_token(logits, temperature=T, top_p=TOP_P, top_k=TOP_K, min_p=0., do_sample=True)
    finally:
        torch.multinomial = original
    p = captured['p']
    assert p.dtype == torch.bfloat16, p.dtype
    p = p.float()
    return p / p.sum(-1, keepdim=True)  # CUDA multinomial normalizes in fp32


class SamplerAcc:
    """Step 2: engine sampler vs reference bf16 sampler on identical logits."""
    def __init__(self, vocab):
        self.sampler = GraphSampler(vocab, max_top_k=128)
        self.batches = {}
        self.rows = {k: [] for k in ('engine_logits', 'reference_logits')}

    def engine_probs(self, logits):
        k = len(logits)
        if k not in self.batches:
            self.batches[k] = self.sampler.make_batch([SamplingParams(temperature=T, top_k=TOP_K, top_p=TOP_P)] * k,
                                                      device=logits.device)
        return self.sampler.probabilities(logits, self.batches[k])

    def add(self, source, logits, positions):
        pe, pr = self.engine_probs(logits), reference_probs(logits)
        top = logits.float().argmax(-1, keepdim=True)

        def stats(p):
            kept = (p > 0).sum(-1)
            h = -(torch.where(p > 0, p * p.clamp_min(1e-30).log(), 0.)).sum(-1)
            return kept, h, p.gather(-1, top).squeeze(-1)
        ke, he, me = stats(pe)
        kr, hr, mr = stats(pr)
        tv = .5 * (pe.double() - pr.double()).abs().sum(-1)
        for row in zip(positions, tv.tolist(), ke.tolist(), kr.tolist(), he.tolist(), hr.tolist(), me.tolist(), mr.tolist()):
            self.rows[source].append(row)

    def report(self):
        out = {}
        for source, rows in self.rows.items():
            groups = {'all': rows}
            for b in BUCKETS:
                groups[b] = [r for r in rows if bucket_of(r[0]) == b]
            out[source] = {}
            for name, g in groups.items():
                if not g:
                    continue
                n = len(g)
                tv = sorted(r[1] for r in g)
                q = lambda f: tv[min(n - 1, int(f * n))]
                out[source][name] = dict(
                    positions=n, tv_mean=sum(tv) / n, tv_p50=q(.5), tv_p90=q(.9), tv_p99=q(.99), tv_max=tv[-1],
                    tv_zero_fraction=sum(t < 1e-6 for t in tv) / n,
                    kept_engine_mean=sum(r[2] for r in g) / n, kept_reference_mean=sum(r[3] for r in g) / n,
                    kept_engine_gt_ref=sum(r[2] > r[3] for r in g), kept_engine_lt_ref=sum(r[2] < r[3] for r in g),
                    entropy_engine_mean=sum(r[4] for r in g) / n, entropy_reference_mean=sum(r[5] for r in g) / n,
                    entropy_diff_mean=sum(r[4] - r[5] for r in g) / n,
                    entropy_diff_se=_se([r[4] - r[5] for r in g]),
                    entropy_engine_gt_ref=sum(r[4] > r[5] + 1e-6 for r in g), entropy_engine_lt_ref=sum(r[4] < r[5] - 1e-6 for r in g),
                    argmax_mass_engine_mean=sum(r[6] for r in g) / n, argmax_mass_reference_mean=sum(r[7] for r in g) / n,
                    argmax_mass_diff_mean=sum(r[6] - r[7] for r in g) / n,
                    argmax_mass_diff_se=_se([r[6] - r[7] for r in g]),
                    argmax_mass_engine_gt_ref=sum(r[6] > r[7] + 1e-6 for r in g), argmax_mass_engine_lt_ref=sum(r[6] < r[7] - 1e-6 for r in g))
        return out


def _se(xs):
    n = len(xs)
    if n < 2:
        return 0.
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1) / n) ** .5


class DecisionAcc:
    def __init__(self, n):
        self.b = {k: dict(pass_count_compared=0, pass_count_mismatch=0,
                          per_depth=[dict(common=0, flips=0, max_flip_margin=0.) for _ in range(n)])
                  for k in BUCKETS + ['prompt']}

    def add(self, ref, cand, prompt_len, length, threshold):
        both = ref['visited'][:, :length] & cand['visited'][:, :length]
        flips = both & (ref['decision'][:, :length] != cand['decision'][:, :length])
        margin = torch.maximum((ref['prob'][:, :length].float() - threshold).abs(),
                               (cand['prob'][:, :length].float() - threshold).abs())
        pc_mis = ref['visited'][:, :length].sum(0) != cand['visited'][:, :length].sum(0)
        for pos in range(length):
            key = 'prompt' if pos < prompt_len else bucket_of(pos)
            a = self.b[key]
            a['pass_count_compared'] += 1
            a['pass_count_mismatch'] += int(pc_mis[pos])
        # vectorized per depth/bucket for decisions
        for key in self.b:
            mask = torch.zeros(length, dtype=torch.bool)
            if key == 'prompt':
                mask[:prompt_len] = True
            else:
                lo = BUCKETS.index(key) * 1000
                hi = length if key == '3k+' else lo + 1000
                mask[max(lo, prompt_len):min(hi, length)] = True
            for d, a in enumerate(self.b[key]['per_depth']):
                a['common'] += int((both[d] & mask).sum())
                f = flips[d] & mask
                a['flips'] += int(f.sum())
                if f.any():
                    a['max_flip_margin'] = max(a['max_flip_margin'], float(margin[d][f].max()))

    def report(self):
        out = {}
        for k, a in self.b.items():
            out[k] = dict(pass_count_compared=a['pass_count_compared'], pass_count_mismatch=a['pass_count_mismatch'],
                          pass_count_mismatch_rate=a['pass_count_mismatch'] / max(a['pass_count_compared'], 1),
                          per_depth=[dict(depth=d + 1, **x, flip_rate=x['flips'] / max(x['common'], 1))
                                     for d, x in enumerate(a['per_depth'])])
        return out


def trace_of(tr):
    return dict(visited=tr['visited'], decision=tr['decision'], prob=tr['continue_probability'])


class Observer:
    """Engine replay hook: records decisions and streams final-logit comparisons."""
    def __init__(self, group, refs, n, logits_acc, sampler_acc):
        self.group, self.refs, self.logits_acc, self.sampler_acc = group, refs, logits_acc, sampler_acc
        self.traces = {}
        for i, s in enumerate(group):
            L = len(s['prompt']) + len(s['tokens'])
            self.traces[i] = dict(visited=torch.zeros(n, L, dtype=torch.bool), decision=torch.zeros(n, L, dtype=torch.bool),
                                  prob=torch.full((n, L), float('nan')))
        self.engine_counts = {}

    def prefill(self, state, start, output):
        tr = self.traces[state.index]
        k = output.logits.shape[0]
        tr['visited'][:, start:start + k] = output.visited.cpu()
        tr['decision'][:, start:start + k] = output.decision.cpu()
        tr['prob'][:, start:start + k] = output.continue_probability.float().cpu()
        ref = self.refs[state.index][start:start + k].to(output.logits.device, non_blocking=False)
        self.logits_acc.add('prompt', ref, output.logits)

    def decode(self, states, bucket, status):
        host = status.cpu()
        final_rows, refs, positions = [], [], []
        for row, state in enumerate(states):
            keep, _, raw, prob = host[row].tolist()
            tr = self.traces[state.index]
            tr['visited'][state.depth - 1, state.position] = True
            tr['decision'][state.depth - 1, state.position] = bool(keep)
            tr['prob'][state.depth - 1, state.position] = prob
            if not keep:
                final_rows.append(row)
                refs.append(self.refs[state.index][state.position])
                positions.append(state.position)
        if not final_rows:
            return
        device = bucket.logits.device
        cand = bucket.logits[torch.tensor(final_rows, device=device)]
        ref = torch.stack(refs).to(device)
        for i, pos in enumerate(positions):
            self.logits_acc.add(bucket_of(pos), ref[i:i + 1], cand[i:i + 1])
        self.sampler_acc.add('engine_logits', cand, positions)
        self.sampler_acc.add('reference_logits', ref, positions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--per-kind', type=int, default=3, help='sequences per (dataset, finish reason)')
    p.add_argument('--group', type=int, default=4)
    p.add_argument('--output', type=Path, default=ROOT / 'longctx.json')
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    seqs = select(tokenizer, args.per_kind)
    print('selected', [(s['id'], s['sample'], s['finish'], len(s['prompt']), len(s['tokens'])) for s in seqs], flush=True)
    config_path = RUNS['math500'] / 'effective_config.yaml'
    config = yaml.safe_load(config_path.read_text())
    engine = TaHEngine(str(MODEL), config_override=config, max_batch_size=args.group, prefill_chunk_size=128,
                       kv_capacity=40000, cuda_graphs=False, seed=0)
    oracle = load_reference(SimpleNamespace(device='cuda', eval_config=config_path), 'real', MODEL)
    threshold = float(oracle._active_iter_decider.threshold)
    engine_threshold = float(engine.decider.threshold)
    n = engine.model.max_iter
    eos = engine.eos_token_id
    print(dict(threshold=threshold, engine_threshold=engine_threshold, max_iter=n, eos=eos), flush=True)

    # Forced-token sampling: the real sampler runs (RNG semantics unchanged),
    # but the emitted token is the recorded W-acc token.
    forced_buf = torch.zeros(args.group, dtype=torch.long, device=engine.device)
    original_sampling = StepBucket.sampling

    class Forced:
        def __init__(self, inner):
            self.inner = inner

        def sample(self, logits, params, rng, active=None):
            self.inner.sample(logits, params, rng, active)
            tok = forced_buf[:logits.shape[0]]
            return tok.clone() if active is None else torch.where(active, tok, -1)

    def sampling(self, eng, params):
        original_sampling(self, eng, params)
        self.sampler = Forced(self.sampler)
    StepBucket.sampling = sampling

    eng_logits = LogitAcc()
    ref_logits = LogitAcc()
    eng_dec, ref_dec = DecisionAcc(n), DecisionAcc(n)
    sampler_acc = SamplerAcc(engine.model.config['vocab_size'])
    per_sequence = []
    for g0 in range(0, len(seqs), args.group):
        group = seqs[g0:g0 + args.group]
        refs, ref_traces = [], []
        t0 = time.perf_counter()
        for s in group:
            seq = s['prompt'] + s['tokens'][:-1]  # engine processes positions [0, P+n-1)
            length, P = len(seq), len(s['prompt'])
            full_t, full_l = capture(oracle, [seq], pad_id=eos, chunk_boundaries=[0, length])
            chunk_t, chunk_l = capture(oracle, [seq], pad_id=eos, chunk_size=256)
            full_t, full_l, chunk_t, chunk_l = full_t[0], full_l[0], chunk_t[0], chunk_l[0]
            assert full_l.dtype == torch.bfloat16
            # noise baseline: reference chunk256 vs reference full (prompt + generated positions)
            for start in range(0, length, 128):
                end = min(start + 128, length)
                a, b = full_l[start:end].cuda(), chunk_l[start:end].cuda()
                for key in set(['prompt' if q < P else bucket_of(q) for q in range(start, end)]):
                    sel = [q - start for q in range(start, end) if ('prompt' if q < P else bucket_of(q)) == key]
                    idx = torch.tensor(sel, device='cuda')
                    ref_logits.add(key, a[idx], b[idx])
            ref_dec.add(trace_of(full_t), trace_of(chunk_t), P, length, threshold)
            del chunk_l
            refs.append(full_l)
            ref_traces.append(full_t)
            torch.cuda.empty_cache()
        t1 = time.perf_counter()
        observer = Observer(group, refs, n, eng_logits, sampler_acc)
        engine._observer = observer
        original_decode, original_prefill = engine._decode, engine._prefill

        def decode(running):
            forced_buf[:len(running)] = torch.tensor([group[s.index]['tokens'][len(s.generated)] for s in running],
                                                     device=engine.device)
            return original_decode(running)

        def prefill(state):
            forced_buf[0] = group[state.index]['tokens'][0]
            return original_prefill(state)
        engine._decode, engine._prefill = decode, prefill
        try:
            results = engine.generate([Request(s['prompt'], len(s['tokens']), i) for i, s in enumerate(group)],
                                      SamplingParams(temperature=T, top_k=TOP_K, top_p=TOP_P))
        finally:
            del engine._decode, engine._prefill
            engine._observer = None
        t2 = time.perf_counter()
        for i, (s, r) in enumerate(zip(group, results)):
            assert r.token_ids == s['tokens'], 'forced replay diverged'
            length = len(s['prompt']) + len(s['tokens']) - 1
            P = len(s['prompt'])
            eng_dec.add(trace_of(ref_traces[i]), observer.traces[i], P, length, threshold)
            orig = s['pass_counts']
            same = sum(a == b for a, b in zip(orig, r.generated_pass_counts))
            ref_pc = ref_traces[i]['pass_count'].tolist()
            # generated token j was sampled at position P+j-1
            ref_gen = [ref_pc[P + j - 1] for j in range(len(s['tokens']))]
            per_sequence.append(dict(id=s['id'], sample=s['sample'], dataset=s['dataset'], finish=s['finish'],
                                     correct=s['correct'], prompt_tokens=P, generated_tokens=len(s['tokens']),
                                     replay_vs_wacc_generated_count_match=same / len(orig),
                                     replay_vs_ref_full_generated_count_match=sum(a == b for a, b in zip(ref_gen, r.generated_pass_counts)) / len(orig),
                                     replay_mean_passes=sum(r.generated_pass_counts) / len(orig),
                                     wacc_mean_passes=sum(orig) / len(orig),
                                     ref_full_mean_passes=sum(ref_gen) / len(orig)))
        print(dict(group=g0 // args.group, reference_seconds=t1 - t0, engine_seconds=t2 - t1,
                   steps=engine.last_run['steps']), flush=True)
        del refs, ref_traces, observer
        torch.cuda.empty_cache()
    report = dict(
        description='engine token-by-token cached replay vs reference full forward; noise = reference chunk256 vs reference full',
        sequences=per_sequence, threshold=threshold,
        logits=dict(engine_vs_ref_full=eng_logits.report(), ref_chunk256_vs_ref_full=ref_logits.report()),
        decisions=dict(engine_vs_ref_full=eng_dec.report(), ref_chunk256_vs_ref_full=ref_dec.report()),
        sampler=sampler_acc.report(), seconds=time.perf_counter() - started)
    args.output.write_text(json.dumps(report, indent=2))
    print('done', args.output, report['seconds'], flush=True)


if __name__ == '__main__':
    main()
