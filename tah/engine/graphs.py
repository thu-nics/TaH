"""Startup-only exact-shape decode capture, with measured private-pool sizing.

Graphs share transient storage. Only one step runs at a time and its status is
consumed before another replay; no intermediate survives between steps.
"""
import time

import torch

from .attention import AttentionRow
from .sampling import SamplingParams


def batch_sizes(maximum, selected):
    if selected is None:
        return tuple(range(1, maximum + 1))
    sizes = tuple(selected)
    if any(isinstance(b, bool) or not isinstance(b, int) or not 1 <= b <= maximum for b in sizes):
        raise ValueError('graph_batch_sizes must contain integers in [1, max_batch_size]')
    return tuple(sorted(set(sizes)))


class DecodeGraphs:
    def __init__(self):
        self.captures = {}
        self.info = dict(capture_seconds=0., private_allocated_bytes=0,
                         private_reserved_bytes=0, device_overhead_bytes=0, graphs=0,
                         compile_seconds=0., compile_graphs=0)

    @torch.inference_mode()
    def capture(self, engine):
        started = time.perf_counter()
        compile_seconds_before = engine.compiled_step.compile_seconds if engine.compiled_step else 0.
        compile_graphs_before = engine.compiled_step.compile_graphs if engine.compiled_step else 0
        stream = torch.cuda.Stream(device=engine.device)
        handle = torch.cuda.graph_pool_handle()
        caches = []
        sizes = tuple(reversed(engine.graph_batch_sizes))
        if not sizes:
            return
        try:
            for i in range(max(sizes)):
                cache = engine.pool.create_request(('graph-warmup', i), context_limit=1)
                engine.pool.allocate(cache, 0, [0])
                caches.append(cache)
            paths = [SamplingParams(temperature=.6, top_k=min(20, engine.sampling_topk), top_p=.95)]

            def prepare(size, params):
                bucket = engine._bucket(size)
                bucket.rows.copy_(torch.arange(size, device=engine.device))
                bucket.positions.zero_()
                bucket.depths.copy_(torch.arange(size, device=engine.device) % engine.model.max_iter + 1)
                bucket.slots.copy_(torch.arange(size, device=engine.device))
                bucket.embeddings.copy_(engine.model.model.embed_tokens.weight[0].expand(size, -1))
                bucket.logits.zero_()
                bucket.rng.seeds.zero_()
                bucket.rng.counters.zero_()
                bucket.sampling(engine, [params] * size)
                bucket.attention.prepare([AttentionRow(c, 0, 1) for c in caches[:size]])
                rotary = engine.model.model.rotary_emb(bucket.embeddings[None], bucket.positions, sequence_length=1)
                bucket.set_rotary(rotary)
                return bucket

            # Finish JIT and library initialization before capture. Frequency
            # planning and descriptors always stay outside the device boundary.
            stream.wait_stream(torch.cuda.current_stream(engine.device))
            with torch.cuda.stream(stream):
                for size in sizes:
                    for params in paths:
                        bucket = prepare(size, params)
                        for _ in range(2):
                            bucket.run(engine, bucket.rotary)
            stream.synchronize()
            torch.cuda.empty_cache()
            before_free = torch.cuda.mem_get_info(engine.device)[0]
            before_allocated = torch.cuda.memory_allocated(engine.device)
            before_reserved = torch.cuda.memory_reserved(engine.device)
            with torch.cuda.stream(stream):
                for size in sizes:
                    for params in paths:
                        bucket = prepare(size, params)
                        key = (size, bucket.sampling_key)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=handle, stream=stream):
                            status = bucket.run(engine, bucket.rotary)
                        self.captures[key] = (graph, status)
            stream.synchronize()
            torch.cuda.current_stream(engine.device).wait_stream(stream)
            torch.cuda.empty_cache()
            self.info.update(capture_seconds=time.perf_counter() - started,
                private_allocated_bytes=max(0, torch.cuda.memory_allocated(engine.device) - before_allocated),
                private_reserved_bytes=max(0, torch.cuda.memory_reserved(engine.device) - before_reserved),
                device_overhead_bytes=max(0, before_free - torch.cuda.mem_get_info(engine.device)[0]),
                graphs=len(self.captures), batch_sizes=list(engine.graph_batch_sizes),
                compile_seconds=(engine.compiled_step.compile_seconds - compile_seconds_before
                                 if engine.compiled_step else 0.),
                compile_graphs=(engine.compiled_step.compile_graphs - compile_graphs_before
                                if engine.compiled_step else 0),
                sampling_paths=['bounded_and_greedy'])
        finally:
            stream.synchronize()
            for cache in caches:
                engine.pool.release(cache)
        assert engine.pool.free_slots == engine.pool.capacity

    def run(self, bucket, engine, rotary):
        captured = self.captures.get((bucket.size, bucket.sampling_key)) if engine.cuda_graphs else None
        if captured is None:
            engine.last_run['eager_steps'] += 1
            return bucket.run(engine, rotary)
        bucket.set_rotary(rotary)
        graph, status = captured
        graph.replay()
        engine.last_run['graph_steps'] += 1
        return status

    def close(self):
        for graph, _ in self.captures.values():
            graph.reset()
        self.captures.clear()
