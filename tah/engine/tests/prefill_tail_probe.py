"""One synthetic prefill layout per process, suitable for CUDA memcheck.

This probes the reported shape family, not the unavailable remote prompt/state.
The direct path calls FlashInfer without constructing engine cache metadata.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F


def inputs(case):
    if case == 'aligned':
        streams, queries = [list(range(128))], list(range(128))
    elif case == 'tail81':
        streams, queries = [list(range(209))], list(range(128, 209))
    elif case == 'collector80':
        # EnginePairs.start holds back the last of the 209 prompt tokens.
        streams, queries = [list(range(208))], list(range(128, 208))
    else:
        # Synthetic depth-2 example: every first-chunk token continued,
        # and 80 of the 81 second-chunk tokens continued.
        streams = [list(range(209)), list(range(208))]
        queries = list(range(128, 208))
    positions = torch.tensor([p for stream in streams for p in stream])
    mask = positions[None] <= torch.tensor(queries)[:, None]
    gen = torch.Generator().manual_seed(37)
    q = torch.randn(len(queries), 16, 128, generator=gen).bfloat16()
    k = torch.randn(len(positions), 1, 8, 128, generator=gen).bfloat16()
    v = torch.randn(len(positions), 1, 8, 128, generator=gen).bfloat16()
    return streams, queries, mask, q, k, v


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['aligned', 'tail81', 'tail80', 'collector80'], required=True)
    parser.add_argument('--path', choices=['direct', 'engine'], default='direct')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--pad-packed-mask', action='store_true',
                        help='Diagnostic control: extend packed mask storage after planning; CUDA direct only.')
    args = parser.parse_args()
    if args.device == 'cpu' and args.path != 'engine':
        parser.error('CPU validation uses --path engine')
    if args.pad_packed_mask and (args.device != 'cuda' or args.path != 'direct'):
        parser.error('--pad-packed-mask requires CUDA and --path direct')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No usable CUDA GPU; run outside the sandbox if GPU access is restricted.')

    streams, queries, mask, q_cpu, k_cpu, v_cpu = inputs(args.case)
    assert mask.shape == (len(queries), len(k_cpu))
    assert mask.dtype == torch.bool and bool(mask.any(dim=1).all())
    # Independent float32 CPU oracle; no FlashInfer or GPU SDPA dispatch.
    expected = F.scaled_dot_product_attention(
        q_cpu.float().transpose(0, 1)[None],
        k_cpu[:, 0].float().transpose(0, 1).repeat_interleave(2, 0)[None],
        v_cpu[:, 0].float().transpose(0, 1).repeat_interleave(2, 0)[None],
        attn_mask=mask[None, None])[0].transpose(0, 1)
    q, k, v = [t.to(args.device) for t in (q_cpu, k_cpu, v_cpu)]
    record = dict(case=args.case, path=args.path, device=args.device,
                  q_rows=len(q), kv_rows=len(k), mask_bits=mask.numel(),
                  pad_packed_mask=args.pad_packed_mask, torch=torch.__version__,
                  synthetic_layout=True)

    if args.path == 'direct':
        import flashinfer
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024**2, dtype=torch.uint8, device=args.device), 'NHD')
        wrapper.plan(
            torch.tensor([0, len(q)], dtype=torch.int32, device=args.device),
            torch.tensor([0, len(k)], dtype=torch.int32, device=args.device),
            torch.arange(len(k), dtype=torch.int32, device=args.device),
            torch.ones(1, dtype=torch.int32, device=args.device),
            16, 8, 128, 1, custom_mask=mask.flatten().to(args.device),
            q_data_type=q.dtype, kv_data_type=k.dtype)
        if args.pad_packed_mask:
            # FA2 0.2.5 uses CTA_TILE_Q=128 for these one-group layouts.
            # GQA=2: ceil(Q*2/128)*128/2 query rows may reach LogitsMask.
            padded_rows = ((len(q) * 2 + 127) // 128) * 64
            original = wrapper._custom_mask_buf
            padded = torch.zeros((padded_rows * len(k) + 7) // 8,
                                 dtype=torch.uint8, device=args.device)
            padded[:original.numel()].copy_(original)
            wrapper._custom_mask_buf = padded
        record.update(flashinfer=flashinfer.__version__, backend=wrapper._backend,
                      plan_info=[int(x) for x in wrapper._plan_info],
                      packed_mask_bytes=wrapper._custom_mask_buf.numel())
        def run():
            return wrapper.run(q, (k, v))
    else:
        from tah.engine.attention import AttentionRow, PrefillAttention
        from tah.engine.kv_cache import SlotPool
        pool = SlotPool(len(k), 1, 8, 128, len(streams), device=args.device)
        request = pool.create_request('probe', context_limit=209)
        for depth, positions in enumerate(streams):
            pool.allocate(request, depth, positions)
        pool.values[0, 0].copy_(k)
        pool.values[0, 1].copy_(v)
        attention = PrefillAttention(pool,
            [AttentionRow(request, p, len(streams)) for p in queries], 16)
        torch.testing.assert_close(attention.mask.cpu(), mask.flatten())
        assert attention.indices.cpu().tolist() == list(range(len(k)))
        if args.device == 'cuda':
            import flashinfer
            record.update(flashinfer=flashinfer.__version__,
                          backend=attention.wrapper._backend,
                          plan_info=[int(x) for x in attention.wrapper._plan_info],
                          packed_mask_bytes=attention.wrapper._custom_mask_buf.numel())
        def run():
            return attention.run(q, 0)

    if args.device == 'cuda':
        record['gpu'] = torch.cuda.get_device_name()
        record['cuda'] = torch.version.cuda
        record['compute_capability'] = list(torch.cuda.get_device_capability())
        torch.cuda.synchronize()  # Attribute earlier metadata failures to planning.
    print(json.dumps(dict(record, stage='before_attention')), flush=True)
    actual = run()
    if args.device == 'cuda':
        torch.cuda.synchronize()  # Attribute illegal reads to this operation.
    actual = actual.cpu().float()
    torch.testing.assert_close(actual, expected, atol=.015, rtol=.02)
    print(json.dumps(dict(record, status='NUMERICAL_PASS',
                          max_abs_error=(actual - expected).abs().max().item())), flush=True)


if __name__ == '__main__':
    main()
