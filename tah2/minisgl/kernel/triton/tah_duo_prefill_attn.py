"""Tiled, paged attention for the TaH DUO iter>=1 *prefill* pass.

Adapted from the training reference attention implementation (``_tah_attn_fwd_kernel``,
flash-attn-v2 online softmax, GQA in-kernel) with two serving-side changes:

* **paged K/V** — K/V rows are gathered straight from the layer's KV pool
  (``(num_slots, Hkv, D)``, page_size == 1) through a per-request slot-index
  row, so no dense copy and no per-query page table is needed;
* **ragged requests + causal block bounds** — queries are grouped into tiles of
  ``BLOCK_M`` consecutive rows of one request; for every (tile, KV segment) the
  host passes ``[start, end)`` = the prefix of the segment that is visible to the
  tile's *last* query (segments are position-sorted), so blocks past the tile's
  causal horizon are never loaded.

Visibility (DUO, iter ``d``): a query at position ``p`` attends every KV entry of
iter 0..d whose position ``<= p`` (its own iter-d entry included).  The host
encodes that as per-request segments ``[iter0 | stream 1 | ... | stream d]``
with per-entry positions; inside the kernel the only predicate left is
``kv_pos <= q_pos`` (plus the block bounds).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn.utils.rnn import pad_sequence


@triton.jit
def _duo_prefill_attn_kernel(
    Q,
    Out,
    K_cache,
    V_cache,
    q_pos_ptr,  # (Tq,)   int32
    tile_start_ptr,
    tile_end_ptr,  # (n_tiles,) int32  row range of each tile in Q
    tile_req_ptr,  # (n_tiles,) int32  request row in kv_idx / kv_pos
    seg_start_ptr,
    seg_end_ptr,  # (n_tiles, N_SEG) int32  visible [start,end) per segment
    kv_idx_ptr,
    kv_pos_ptr,  # (R, max_kv) int32  slot index / position per KV entry
    sm_scale,
    stride_qm,
    stride_qh,
    stride_om,
    stride_oh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_kv_r,
    Hq,
    N_SEG: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    tile = tl.program_id(0)
    off_hq = tl.program_id(1)
    off_hkv = off_hq // GROUP_SIZE

    t_start = tl.load(tile_start_ptr + tile)
    t_end = tl.load(tile_end_ptr + tile)
    req = tl.load(tile_req_ptr + tile)

    offs_m = t_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < t_end

    q = tl.load(
        Q + offs_m[:, None] * stride_qm + off_hq * stride_qh + offs_d[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    q_pos = tl.load(q_pos_ptr + offs_m, mask=m_mask, other=0).to(tl.int32)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    kv_row = req.to(tl.int64) * stride_kv_r
    for s in tl.static_range(N_SEG):
        n_lo = tl.load(seg_start_ptr + tile * N_SEG + s)
        n_hi = tl.load(seg_end_ptr + tile * N_SEG + s)
        for start_n in range(n_lo, n_hi, BLOCK_N):
            n_idx = start_n + offs_n
            n_mask = n_idx < n_hi
            kv_idx = tl.load(kv_idx_ptr + kv_row + n_idx, mask=n_mask, other=0).to(tl.int64)
            kv_pos = tl.load(kv_pos_ptr + kv_row + n_idx, mask=n_mask, other=0).to(tl.int32)
            visible = (kv_pos[None, :] <= q_pos[:, None]) & n_mask[None, :] & m_mask[:, None]

            k = tl.load(
                K_cache + kv_idx[:, None] * stride_kn + off_hkv * stride_kh + offs_d[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
            v = tl.load(
                V_cache + kv_idx[:, None] * stride_vn + off_hkv * stride_vh + offs_d[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )

            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
            qk = tl.where(visible, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            m_ij_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
            alpha = tl.exp(tl.where(m_i == -float("inf"), 0.0, m_i) - m_ij_safe)
            p = tl.where(visible, tl.exp(qk - m_ij_safe[:, None]), 0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    tl.store(
        Out + offs_m[:, None] * stride_om + off_hq * stride_oh + offs_d[None, :],
        out.to(Out.dtype.element_ty),
        mask=m_mask[:, None],
    )


BLOCK_M = 128  # query rows per tile; shared default of the builder and the kernel launch


def duo_prefill_attention(
    q: torch.Tensor,  # (Tq, Hq, D) bf16, tiles are contiguous row ranges
    k_cache: torch.Tensor,  # (num_slots, Hkv, D)
    v_cache: torch.Tensor,  # (num_slots, Hkv, D)
    q_pos: torch.Tensor,  # (Tq,) int32
    tile_start: torch.Tensor,  # (n_tiles,) int32
    tile_end: torch.Tensor,  # (n_tiles,) int32
    tile_req: torch.Tensor,  # (n_tiles,) int32
    seg_start: torch.Tensor,  # (n_tiles, n_seg) int32
    seg_end: torch.Tensor,  # (n_tiles, n_seg) int32
    kv_idx: torch.Tensor,  # (R, max_kv) int32
    kv_pos: torch.Tensor,  # (R, max_kv) int32
    sm_scale: float,
    block_m: int = BLOCK_M,
    block_n: int = 32,  # autotuned best on A800 (local/acceleration/tests/tune_kernel.py)
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    Tq, Hq, D = q.shape
    Hkv = k_cache.shape[1]
    assert Hq % Hkv == 0 and D in (64, 128, 256)
    assert q.stride(2) == 1 and k_cache.stride(2) == 1 and v_cache.stride(2) == 1
    assert kv_idx.stride(1) == 1 and kv_pos.stride(1) == 1 and kv_idx.stride(0) == kv_pos.stride(0)
    n_tiles, n_seg = seg_start.shape
    out = torch.empty_like(q)
    if n_tiles == 0:
        return out
    _duo_prefill_attn_kernel[(n_tiles, Hq)](
        q,
        out,
        k_cache,
        v_cache,
        q_pos,
        tile_start,
        tile_end,
        tile_req,
        seg_start.contiguous(),
        seg_end.contiguous(),
        kv_idx,
        kv_pos,
        sm_scale,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        kv_idx.stride(0),
        Hq,
        N_SEG=n_seg,
        GROUP_SIZE=Hq // Hkv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def build_duo_prefill_tiles(
    req_q_counts: list,  # per request: number of query rows (rows are request-major)
    q_pos: torch.Tensor,  # (Tq,) int32/int64, ascending within each request
    kv_seg_pos: list,  # per request: list of n_seg 1-D position tensors (each sorted asc)
    kv_seg_idx: list,  # per request: list of n_seg 1-D slot-index tensors (same lengths)
    block_m: int = BLOCK_M,
) -> tuple:
    """Host-side tiling + per-(tile, segment) visible bounds.

    Returns ``(tile_start, tile_end, tile_req, seg_start, seg_end, kv_idx, kv_pos)``:
    ``kv_idx`` / ``kv_pos`` are the per-request concatenated slot-index / position rows
    (R, max_kv) (segments back to back, zero padded); ``seg_start/seg_end`` (n_tiles, n_seg)
    are the visible ``[start, end)`` ranges inside that row.  All on ``q_pos.device``.
    Every length/offset is known on the host, so this does no GPU->CPU sync.
    """
    device = q_pos.device
    n_seg = len(kv_seg_pos[0])

    # Tiles: BLOCK_M consecutive query rows of one request.
    ts, te, tr, n_tiles = [], [], [], []
    row = 0
    for r, n in enumerate(req_q_counts):
        for s in range(row, row + n, block_m):
            ts.append(s)
            te.append(min(s + block_m, row + n))
            tr.append(r)
        n_tiles.append(-(-n // block_m))
        row += n
    tile_start, tile_end, tile_req = torch.tensor(
        [ts, te, tr], dtype=torch.int32, device=device
    ).unbind(0)
    # Last query position of every tile (rows ascending within a request).
    qp_max = q_pos.index_select(0, (tile_end - 1).to(torch.int64)).to(torch.int32)

    # Per-request KV rows (segments back to back), zero padded to max_kv.
    seg_len = [[int(p.numel()) for p in segs] for segs in kv_seg_pos]
    seg_lo = [[sum(lens[:s]) for s in range(n_seg)] for lens in seg_len]
    kv_pos = pad_sequence(
        [torch.cat(segs).to(torch.int32) for segs in kv_seg_pos], batch_first=True
    )
    kv_idx = pad_sequence(
        [torch.cat(segs).to(torch.int32) for segs in kv_seg_idx], batch_first=True
    )
    if kv_pos.shape[1] == 0:  # all rows empty: keep a valid (R, 1) layout for the kernel
        kv_pos = kv_pos.new_zeros((kv_pos.shape[0], 1))
        kv_idx = kv_idx.new_zeros((kv_idx.shape[0], 1))

    # Visible [start, end) per (tile, segment): segment prefix with pos <= tile's last q_pos.
    seg_start = torch.tensor([seg_lo[r] for r in tr], dtype=torch.int32, device=device)
    seg_start = seg_start.view(len(tr), n_seg)
    seg_end = torch.empty_like(seg_start)
    t0 = 0
    for r, segs in enumerate(kv_seg_pos):
        n_t = n_tiles[r]
        qm = qp_max[t0 : t0 + n_t]
        for s, pos in enumerate(segs):
            lo = seg_lo[r][s]
            vis = torch.searchsorted(pos.to(torch.int32), qm, right=True) if seg_len[r][s] else 0
            seg_end[t0 : t0 + n_t, s] = lo + vis
        t0 += n_t
    return tile_start, tile_end, tile_req, seg_start, seg_end, kv_idx, kv_pos
