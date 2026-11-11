"""CPU/CUDA ragged attention and swap-buffer microbenchmarks, JSON output."""
import argparse
import json
from pathlib import Path
import time

import torch

from tah.engine.attention import DecodeBucket, PrefillAttention, AttentionRow
from tah.engine.kv_cache import SlotPool


def measure(fn, device, repeats):
    for _ in range(3):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1e6 / repeats


@torch.inference_mode()
def model_case(model, batch, average_context, repeats):
    """Real model/all-layer decode cost; randomized old KV, no generation."""
    cfg = model.config
    torch.cuda.empty_cache()
    lengths = ([average_context] if batch == 1 else
               [round(average_context * (.75 + .5 * i / (batch - 1))) for i in range(batch)])
    coordinates = [[list(range(0, length - 1, 16**depth)) + [length - 1]
                    for depth in range(model.max_iter)] for length in lengths]
    capacity = sum(len(p) for streams in coordinates for p in streams)
    bytes_per_slot = 2 * cfg['num_hidden_layers'] * cfg['num_key_value_heads'] * cfg['head_dim'] * 2
    free_bytes, _ = torch.cuda.mem_get_info()
    if capacity * bytes_per_slot > .85 * free_bytes:
        raise RuntimeError('model benchmark KV exceeds 85% of measured free GPU memory; reduce batch/context')
    torch.cuda.reset_peak_memory_stats()
    pool = SlotPool(capacity, cfg['num_hidden_layers'], cfg['num_key_value_heads'], cfg['head_dim'],
                    model.max_iter, device=model.device, dtype=model.dtype)
    rows, slots = [], []
    for index, (length, streams) in enumerate(zip(lengths, coordinates)):
        request = pool.create_request(index, context_limit=length)
        for depth, positions in enumerate(streams):
            pool.allocate(request, depth, positions)
        # About 1/16 of rows continue: representative of the real workload's
        # approximately 1.07 passes/token, not uniform maximum-depth traffic.
        depth = min(model.max_iter, 2) if index % 16 == 0 else 1
        rows.append(AttentionRow(request, length - 1, depth))
        slots.append(request.streams[depth - 1].slots[len(streams[depth - 1]) - 1:len(streams[depth - 1])])
    pool.values.normal_()
    slots = torch.cat(slots).long()
    positions = torch.tensor([r.position for r in rows], device=model.device)
    depths = torch.tensor([r.depth for r in rows], device=model.device)
    embeddings = model.model.embed_tokens(torch.arange(batch, device=model.device) % cfg['vocab_size'])
    bucket = DecodeBucket(pool, batch, capacity, cfg['num_attention_heads'])
    bucket.prepare(rows)
    fn = lambda: model.forward_cached(embeddings, positions, depths, slots, pool, bucket)
    record = dict(batch=batch, mean_context=sum(lengths) / batch, real_positions=sum(lengths),
                  kv_entries=capacity, layers=cfg['num_hidden_layers'],
                  kv_bytes=capacity * bytes_per_slot,
                  prepare_cpu_wall_us=measure(lambda: bucket.prepare(rows), 'cpu', repeats),
                  cached_pass_eager_us=measure(fn, 'cuda', repeats))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bucket.build()
        output = fn()
    eager = fn()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.logits, eager.logits, atol=0, rtol=0)
    torch.testing.assert_close(output.hidden_states, eager.hidden_states, atol=0, rtol=0)
    record['eager_graph_exact'] = True
    record['cached_pass_graph_us'] = measure(graph.replay, 'cuda', repeats)
    record['metadata_prepare_fraction_of_pass'] = record['prepare_cpu_wall_us'] / record['cached_pass_graph_us']
    record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    for row in rows:
        pool.release(row.request)
    record['pool_audit'] = pool.audit()
    torch.cuda.synchronize()
    del graph, output, eager, pool, bucket
    torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 32])
    parser.add_argument('--avg-context', type=int, default=None,
                        help='Mean real context length; request lengths vary by +/-25%%')
    parser.add_argument('--model-path', type=Path, default=None,
                        help='Optional local engine checkpoint for all-layer cached-pass timing (CUDA only)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if any(b < 1 for b in args.batches) or (args.avg_context is not None and args.avg_context < 16):
        parser.error('batches must be positive and average context must be >=16')
    if args.model_path is not None and args.device != 'cuda':
        parser.error('--model-path requires --device cuda')
    torch.set_num_threads(2)
    torch.manual_seed(20261012)
    results = []
    model = None
    if args.model_path is not None:
        from tah.engine.model import EagerTaHModel
        model = EagerTaHModel.from_pretrained(args.model_path, device='cuda')
    for batch in args.batches:
        if model is not None:
            results.append(model_case(model, batch, args.avg_context or 2048, args.repeats))
            continue
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        lengths = ([31 + 17 * i for i in range(batch)] if args.avg_context is None else
                   [args.avg_context] if batch == 1 else
                   [round(args.avg_context * (.75 + .5 * i / (batch - 1))) for i in range(batch)])
        pool = SlotPool(sum(lengths) * 3, 1, 8, 128, 3, device=args.device)
        rows = []
        for i, length in enumerate(lengths):
            request = pool.create_request(i, context_limit=length + 1)
            for d in range(3):
                pool.allocate(request, d, list(range(d, length, d + 1)))
            rows.append(AttentionRow(request, length - 1, 1 + i % 3))
        pool.values.normal_()
        q = torch.randn(batch, 16, 128, device=args.device, dtype=torch.bfloat16)
        bucket = DecodeBucket(pool, batch, sum(r.request.size for r in rows), 16)
        bucket.prepare(rows)
        record = dict(batch=batch, real_positions=sum(lengths), kv_entries=sum(r.request.size for r in rows),
            mean_context=sum(lengths) / batch,
            prepare_cpu_wall_us=measure(lambda: bucket.prepare(rows), 'cpu', args.repeats),
            metadata_us=measure(bucket.build, args.device, args.repeats),
            decode_us=measure(lambda: bucket.run(q, 0), args.device, args.repeats))
        chunk_rows = [AttentionRow(r.request, p, r.depth) for r, length in zip(rows, lengths) for p in range(length - 8, length)]
        prefill = PrefillAttention(pool, chunk_rows, 16)
        chunk_q = q.repeat_interleave(8, 0)
        record['prefill_8_us'] = measure(lambda: prefill.run(chunk_q, 0), args.device, args.repeats)
        request = rows[-1].request
        snap = pool.gather(request)
        record['gather_us'] = measure(lambda: pool.gather(request, out=snap.values), args.device, args.repeats)
        record['scatter_us'] = measure(lambda: pool.scatter(request, snap), args.device, args.repeats)
        record['payload_bytes'] = snap.values.numel() * snap.values.element_size()
        if args.device == 'cuda':
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.repeats):
                bucket.prepare(rows)
            torch.cuda.synchronize()
            record['prepare_end_to_end_wall_us'] = (time.perf_counter() - start) * 1e6 / args.repeats
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                bucket.build()
                bucket.run(q, 0)
            record['metadata_decode_graph_us'] = measure(graph.replay, args.device, args.repeats)
            metadata_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(metadata_graph):
                bucket.build()
            record['metadata_graph_us'] = measure(metadata_graph.replay, args.device, args.repeats)
            decode_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(decode_graph):
                bucket.run(q, 0)
            record['decode_graph_us'] = measure(decode_graph.replay, args.device, args.repeats)
            record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        for r in rows:
            pool.release(r.request)
        record['pool_audit'] = pool.audit()
        results.append(record)
        if args.device == 'cuda':
            torch.cuda.synchronize()
            del graph, metadata_graph, decode_graph
        del pool, bucket, prefill, q, chunk_q, snap, request, rows, chunk_rows
    report = dict(production_gates_eligible=False, device=args.device, dtype='bfloat16', threads=2,
                  repeats=args.repeats, units='microseconds', cases=results,
                  batches=args.batches, average_context=args.avg_context,
                  model_path=str(args.model_path) if args.model_path else None,
                  model_case_caveat='randomized past KV; pass-local logits/hidden states; no decider or sampling' if model else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.device == 'cpu':
        assert not torch.cuda.is_initialized()


if __name__ == '__main__':
    main()
