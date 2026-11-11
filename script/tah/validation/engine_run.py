"""Teacher-forced engine runs through the production eager decode step.

``TaHEngine.generate`` schedules the requests as in an evaluation (continuous
admission, up to ``max_batch_size`` resident rows, mixed pass depths, swap when the
pool is full). Patches live on the engine instance only:
  * the sampler of each step bucket still runs (RNG semantics unchanged) but emits
    the recorded token (forced tokens);
  * the decider object's ``forward`` is wrapped: its own decision is recorded and,
    in forced mode, replaced by the target's decision for (request, position, pass);
  * the engine's observer hook records every pass and final logits.
CUDA graphs are off (graph replay is bitwise identical to this eager step per the
engine identity tests) and overlap is off (its identity gate covers it).

Host RAM: each runner's engine pins ``host_swap_bytes`` (default 8 GiB) at
construction, so a process needs the engine baseline RSS + that budget per live
runner. The engine patches below form reference cycles, so ``del`` alone frees the
arena only at the next garbage collection: call ``close()`` before building the next
runner (``capture.py`` does).
"""
from __future__ import annotations

import types
from collections import Counter

import numpy as np
import torch

from tah.engine import Request, SamplingParams, TaHEngine

from .common import T, TOP_K, TOP_P, EngineDistribution, Recorder, inputs_of, top_gap


class EngineRunner:
    def __init__(self, model_path, config_override=None, *, device="cuda", max_batch_size=32,
                 prefill_chunk_size=128, kv_capacity=None, memory_fraction=.9,
                 host_swap_bytes=8 * 1024**3, sampling_topk=128, eos_token_id="tokenizer"):
        self.engine = TaHEngine(str(model_path), config_override=config_override, device=device,
                                max_batch_size=max_batch_size, prefill_chunk_size=prefill_chunk_size,
                                kv_capacity=kv_capacity, memory_fraction=memory_fraction,
                                host_swap_bytes=host_swap_bytes, sampling_topk=sampling_topk,
                                eos_token_id=eos_token_id, cuda_graphs=False, overlap=False,
                                compile=False, seed=0, max_new_tokens=1)
        engine = self.engine
        self.max_iter = engine.model.max_iter
        self.kv_capacity = engine.memory_budget["kv_capacity"]
        self.distribution = EngineDistribution(engine.model.config["vocab_size"], engine.sampling_topk)
        self.forced_tokens = torch.zeros(max_batch_size, dtype=torch.long, device=engine.device)
        self.forced_keep = torch.zeros(max_batch_size, dtype=torch.bool, device=engine.device)
        self._install()

    # ------------------------------------------------------------ patches
    def _install(self):
        runner, engine = self, self.engine

        class Forced:
            def __init__(self, inner):
                self.inner = inner

            def sample(self, logits, params, rng, active=None):
                self.inner.sample(logits, params, rng, active)
                token = runner.forced_tokens[:logits.shape[0]]
                return token.clone() if active is None else torch.where(active, token, -1)

        make_bucket = engine._bucket

        def bucket(size):
            new = size not in engine.buckets
            result = make_bucket(size)
            if new:
                original = result.sampling

                def sampling(self, eng, params):
                    original(eng, params)
                    self.sampler = Forced(self.sampler)
                result.sampling = types.MethodType(sampling, result)
            return result
        engine._bucket = bucket

        decider = engine.decider
        forward = type(decider).forward.__get__(decider)

        def decide(self, logits, depth, *args, **kwargs):
            keep, raw, prob = forward(logits, depth, *args, **kwargs)
            own = keep
            if isinstance(depth, int):  # prefill pass over one chunk (forward_chunk)
                plan = runner._plan
                if depth == 1:
                    runner._active = torch.arange(logits.shape[0], device=logits.device)
                runner._prefill_own[depth - 1, runner._active.cpu().numpy()] = own.cpu().numpy()
                if plan is not None:
                    keep = plan[depth - 1].to(logits.device)[runner._active]
                runner._active = runner._active[keep & (depth < runner.max_iter)]
            else:  # decode step rows
                runner._decode_own = own
                if runner._forcing:
                    keep = runner.forced_keep[:logits.shape[0]]
            return keep, raw, prob
        decider.forward = types.MethodType(decide, decider)

        chunk = engine.model.forward_chunk

        def forward_chunk(input_ids, pool, request, **kwargs):
            start, length = kwargs.get("start_position", 0), len(input_ids)
            runner._prefill_own = np.zeros((runner.max_iter, length), bool)
            forced = runner._forced
            runner._plan = None if forced is None else torch.from_numpy(
                forced[runner._current][:, start:start + length].copy())
            return chunk(input_ids, pool, request, **kwargs)
        engine.model.forward_chunk = forward_chunk

        prefill, decode = engine._prefill, engine._decode

        def prefill_wrapper(state):
            runner._current = state.index
            runner.forced_tokens[0] = runner._seqs[state.index]["forced"][0]
            return prefill(state)

        def decode_wrapper(running):
            seqs = runner._seqs
            runner.forced_tokens[:len(running)] = torch.tensor(
                [seqs[s.index]["forced"][len(s.generated)] for s in running], device=engine.device)
            if runner._forcing:
                runner.forced_keep[:len(running)] = torch.tensor(
                    [bool(runner._forced[s.index][s.depth - 1, s.position]) for s in running], device=engine.device)
            sizes, depths = runner._stats["rows"], runner._stats["depth_mix"]
            sizes[len(running)] += 1
            depths[len({s.depth for s in running}) > 1] += 1
            return decode(running)
        engine._prefill, engine._decode = prefill_wrapper, decode_wrapper
        engine._observer = self

    # ------------------------------------------------------------ observer hooks
    def prefill(self, state, start, output):
        rec = self._recorders[state.index]
        k = output.logits.shape[0]
        end = start + k
        visited = output.visited.cpu().numpy()
        prob = output.continue_probability.double().cpu().numpy()
        prob[~visited] = np.nan
        prob[self.max_iter - 1] = np.nan
        rec.visited[:, start:end] = visited
        rec.decision[:, start:end] = output.decision.cpu().numpy()
        rec.own_decision[:, start:end] = self._prefill_own & visited
        rec.prob[:, start:end] = prob
        if end == len(self._seqs[state.index]["prompt"]):
            self._finals([(state.index, end - 1)], output.logits[-1:])

    def decode(self, running, bucket, status):
        host = status.cpu().numpy()
        own = self._decode_own.cpu().numpy()
        finals, rows = [], []
        for row, state in enumerate(running):
            keep, _, _, prob = host[row]
            keep = bool(keep > .5)
            self._recorders[state.index].decide(state.depth, state.position, keep, bool(own[row]), float(prob))
            if not keep:
                finals.append((state.index, state.position))
                rows.append(row)
        if rows:
            self._finals(finals, bucket.logits[torch.tensor(rows, device=bucket.logits.device)])

    def _finals(self, owners, logits):
        top1, gap = top_gap(logits)
        dist = self.distribution(logits)
        for k, (index, position) in enumerate(owners):
            self._recorders[index].final([position], top1[k:k + 1], gap[k:k + 1], [dist[k]])
            if self._consumer is not None:
                rec = self._recorders[index]
                self._consumer(self._seqs[index], [position - rec.first], logits[k:k + 1])

    # ------------------------------------------------------------ run
    def run(self, seqs, *, forced=None, consumer=None):
        """Return (recorders, stats). forced: per sequence bool [max_iter, L] or None."""
        self._seqs, self._forced, self._forcing = seqs, forced, forced is not None
        self._consumer, self._plan = consumer, None
        self._recorders = [Recorder(s, self.max_iter) for s in seqs]
        self._stats = dict(rows=Counter(), depth_mix=Counter())
        requests = [Request(list(s["prompt"]), len(s["forced"]), int(s.get("seed", 0))) for s in seqs]
        try:
            results = self.engine.generate(requests, SamplingParams(temperature=T, top_k=TOP_K, top_p=TOP_P))
        finally:
            self._consumer = None
        eos = self.engine.eos_token_id
        for seq, result in zip(seqs, results):
            expected = list(seq["forced"])
            if expected and expected[-1] == eos:
                expected = expected[:-1]
            if result.token_ids != expected:
                raise AssertionError(f"{seq['id']}: forced engine replay diverged from the forced tokens")
        for rec, seq in zip(self._recorders, seqs):
            if rec.length != len(inputs_of(seq)[0]):
                raise AssertionError("recorder length mismatch")
        rows = self._stats["rows"]
        steps = sum(rows.values())
        stats = dict(decode_steps=steps, row_steps=sum(b * n for b, n in rows.items()),
                     rows_histogram={str(b): n for b, n in sorted(rows.items())},
                     steps_mixed_depth=self._stats["depth_mix"][True],
                     swap_outs=self.engine.last_run["swap_outs"], restarts=self.engine.last_run["restarts"],
                     kv_capacity=self.kv_capacity)
        return self._recorders, stats

    def close(self):
        self.engine.close()  # unpin the host swap arena now, not at the next gc
        del self.engine
