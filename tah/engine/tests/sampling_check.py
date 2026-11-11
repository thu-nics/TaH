"""WP5 reproducible sampler microbench (not an end-to-end W-perf gate).

CPU timings distinguish production full-sampler code from the forced bounded
shortlist math. CUDA measures full/fast/greedy samplers in eager and
captured replay, plus the fp32 reference. No production parity fixtures needed.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from tah.engine.sampling import GraphSampler, RequestRNG, SamplingParams
from tah.model.utils import sample_next_token


def measure(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    if device == "cpu":
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - start) * 1000 / repeats
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(251)
    rows = []
    for b in args.buckets:
        logits = torch.randn(b, args.vocab, generator=generator).to(args.device).bfloat16()
        params = [SamplingParams(temperature=.6, top_k=20, top_p=.95)] * b
        for mode in ("general", "bounded", "greedy", "reference"):
            sampler = GraphSampler(args.vocab, max_top_k=20 if mode == "bounded" else None,
                                   greedy_only=mode == "greedy")
            ps = [SamplingParams(temperature=0)] * b if mode == "greedy" else params
            batch = sampler.make_batch(ps, device=args.device)
            rng = RequestRNG.from_seeds(list(range(b)), device=args.device)
            if mode == "reference":
                fn = lambda: sample_next_token(logits.float(), temperature=.6, top_k=20, top_p=.95)
                measured = "fp32 reference sample_next_token"
            elif mode == "bounded" and args.device == "cpu":
                fn = lambda: sampler._bounded_distribution(logits, batch)
                measured = "bounded shortlist path math (forced CPU; excludes RNG)"
            else:
                fn = lambda: sampler.sample(logits, batch, rng)
                measured = "complete sampler"
            record = {"batch": b, "vocab": args.vocab, "mode": mode, "measured": measured}
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
            record["eager_ms"] = measure(fn, args.device, args.warmup, args.repeats)
            if args.device == "cuda":
                record["eager_workspace_peak_bytes"] = torch.cuda.max_memory_allocated() - baseline
                if mode != "reference":
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(args.warmup):
                            fn()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    torch.cuda.reset_peak_memory_stats()
                    baseline = torch.cuda.memory_allocated()
                    with torch.cuda.graph(graph):
                        output = fn()
                    record["graph_ms"] = measure(graph.replay, args.device, args.warmup, args.repeats)
                    record["capture_workspace_peak_bytes"] = torch.cuda.max_memory_allocated() - baseline
                    del graph, output
            rows.append(record)
            print(json.dumps(record), flush=True)
            del fn, sampler, batch, rng
            gc.collect()
        del logits
    if args.device == "cpu":
        assert not torch.cuda.is_initialized()
    output = {"device": args.device, "torch": torch.__version__, "threads": args.threads,
              "warmup": args.warmup, "repeats": args.repeats, "rows": rows,
              "workload": "sampler microbenchmark; not W-perf",
              "production_gates_eligible": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--buckets", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/engine/wp5/cpu-microbench.json"))
    args = parser.parse_args()
    if args.vocab < 20 or min(args.buckets) < 1 or args.repeats < 1 or args.warmup < 1:
        parser.error("vocab>=20, positive buckets/repeats/warmup required")
    run(args)


if __name__ == "__main__":
    main()
