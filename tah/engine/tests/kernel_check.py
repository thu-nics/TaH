"""FlashInfer feasibility and microbenchmark checks for WP0.

Run with ``.venv-tah/bin/python -m tah.engine.tests.kernel_check`` on the 3090.
The reference path uses dense PyTorch SDPA over exactly the row-specific KV list
or prefill mask supplied to FlashInfer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import flashinfer


DEVICE = "cuda"
DTYPE = torch.bfloat16
WORKSPACE_BYTES = 128 * 1024 * 1024


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> float:
    error = (actual.float() - expected.float()).abs()
    max_error = error.max().item()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.015, msg=label)
    return max_error


def _dense_decode(q, k_pages, v_pages, indices):
    rows = []
    groups = q.shape[1] // k_pages.shape[2]
    for row, ids in enumerate(indices):
        k = k_pages[ids, 0].transpose(0, 1).repeat_interleave(groups, dim=0)
        v = v_pages[ids, 0].transpose(0, 1).repeat_interleave(groups, dim=0)
        out = F.scaled_dot_product_attention(q[row:row + 1].unsqueeze(2),
                                            k.unsqueeze(0), v.unsqueeze(0))
        rows.append(out[0, :, 0])
    return torch.stack(rows)


def _decode_case(num_qo_heads: int, num_kv_heads: int, head_dim: int, graph: bool):
    torch.manual_seed(23 + num_qo_heads + int(graph))
    batch, page_size, pages = 4, 1, 32
    lengths = [1, 3, 7, 11]
    # Noncontiguous, independently permuted page lists exercise arbitrary row indices.
    indices = [[(j * 7 + row * 3 + 2) % pages for j in range(n)]
               for row, n in enumerate(lengths)]
    flat = [i for row in indices for i in row]
    indptr = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=DEVICE,
                          dtype=torch.int32)
    page_indices = torch.tensor(flat, device=DEVICE, dtype=torch.int32)
    last = torch.ones(batch, device=DEVICE, dtype=torch.int32)
    q = torch.randn(batch, num_qo_heads, head_dim, device=DEVICE, dtype=DTYPE)
    k_pages = torch.randn(pages, page_size, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
    v_pages = torch.randn_like(k_pages)
    cache = torch.stack((k_pages, v_pages), dim=1)
    expected = _dense_decode(q, k_pages, v_pages, indices)
    workspace = torch.empty(WORKSPACE_BYTES, device=DEVICE, dtype=torch.uint8)
    graph_obj = None
    if graph:
        idx_buf = torch.empty(pages, device=DEVICE, dtype=torch.int32)
        indptr_buf = torch.empty(batch + 1, device=DEVICE, dtype=torch.int32)
        last_buf = torch.empty(batch, device=DEVICE, dtype=torch.int32)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_cuda_graph=True,
            paged_kv_indptr_buffer=indptr_buf,
            paged_kv_indices_buffer=idx_buf,
            paged_kv_last_page_len_buffer=last_buf,
        )
        # Max index capacity is required by graph mode; padded tail is unused.
        idx_buf[:len(flat)].copy_(page_indices)
        indptr_buf.copy_(indptr)
        last_buf.copy_(last)
        wrapper.plan(indptr, page_indices, last, num_qo_heads, num_kv_heads,
                     head_dim, page_size, q_data_type=DTYPE, kv_data_type=DTYPE)
        output = torch.empty_like(q)
        graph_obj = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_obj):
            wrapper.run(q, cache, out=output)
        graph_obj.replay()
        torch.cuda.synchronize()
        actual = output
    else:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(indptr, page_indices, last, num_qo_heads, num_kv_heads,
                     head_dim, page_size, q_data_type=DTYPE, kv_data_type=DTYPE)
        actual = wrapper.run(q, cache)
    err = _assert_close(actual, expected, f"decode GQA={num_qo_heads}/{num_kv_heads}, graph={graph}")
    return wrapper, q, cache, actual, err, graph_obj


def _prefill_case():
    torch.manual_seed(51)
    q_len, kv_len, pages = 4, 7, 12
    qo_heads, kv_heads, dim = 16, 8, 128
    slot_ids = [9, 1, 11, 4, 6, 0, 8]
    indptr_q = torch.tensor([0, q_len], device=DEVICE, dtype=torch.int32)
    indptr_kv = torch.tensor([0, kv_len], device=DEVICE, dtype=torch.int32)
    indices = torch.tensor(slot_ids, device=DEVICE, dtype=torch.int32)
    last = torch.tensor([1], device=DEVICE, dtype=torch.int32)
    q = torch.randn(q_len, qo_heads, dim, device=DEVICE, dtype=DTYPE)
    k_pages = torch.randn(pages, 1, kv_heads, dim, device=DEVICE, dtype=DTYPE)
    v_pages = torch.randn_like(k_pages)
    cache = torch.stack((k_pages, v_pages), dim=1)
    # Position-based mask with some allowed earlier and later positions.
    mask = torch.tensor([[1, 0, 1, 0, 0, 1, 0],
                         [1, 1, 0, 0, 1, 0, 0],
                         [0, 1, 1, 1, 0, 0, 1],
                         [1, 0, 0, 1, 1, 0, 1]], device=DEVICE, dtype=torch.bool)
    flat_mask = mask.flatten()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(WORKSPACE_BYTES, device=DEVICE, dtype=torch.uint8), "NHD")
    wrapper.plan(indptr_q, indptr_kv, indices, last, qo_heads, kv_heads,
                 dim, 1, custom_mask=flat_mask,
                 q_data_type=DTYPE, kv_data_type=DTYPE)
    actual = wrapper.run(q, cache)
    groups = qo_heads // kv_heads
    k = k_pages[slot_ids, 0].transpose(0, 1).repeat_interleave(groups, dim=0).unsqueeze(0)
    v = v_pages[slot_ids, 0].transpose(0, 1).repeat_interleave(groups, dim=0).unsqueeze(0)
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k, v,
        attn_mask=mask[None, None, :, :],
    ).squeeze(0).transpose(0, 1)
    err = _assert_close(actual, expected, "prefill arbitrary position mask")
    return wrapper, q, cache, actual, err


def _bench(fn, warmup=20, iterations=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} flashinfer={flashinfer.__version__} "
          f"gpu={torch.cuda.get_device_name(0)}")
    records = []
    # Qwen3-0.6B and Qwen3-1.7B both use 16 Q heads, 8 KV heads, head_dim=128.
    for model in ("Qwen3-0.6B", "Qwen3-1.7B"):
        for graph in (False, True):
            wrapper, q, cache, out, err, graph_obj = _decode_case(16, 8, 128, graph)
            if graph:
                fn = graph_obj.replay
            else:
                fn = lambda: wrapper.run(q, cache)
            us = _bench(fn)
            records.append((model, "decode", "graph" if graph else "eager", us, err))
    wrapper, q, cache, out, err = _prefill_case()
    us = _bench(lambda: wrapper.run(q, cache))
    records.append(("GQA 16/8/128", "prefill q=4 kv=7", "arbitrary-mask", us, err))
    print("case\tmode\tmean_us\tmax_abs_error")
    for model, case, mode, us, err in records:
        print(f"{model} {case}\t{mode}\t{us:.2f}\t{err:.6g}")


if __name__ == "__main__":
    main()
