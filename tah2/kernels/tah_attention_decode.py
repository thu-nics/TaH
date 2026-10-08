"""Fused Triton attention kernel for Lq=1 autoregressive decoding.

At decode time the TaH iter>=1 visibility predicate collapses to plain
causal attention over the KV cache, so this is a single-token
flash-attention-v2 — no sentinel, no sparse predicate.

Layout:
    Q:    [B, Hq,  1,   D]
    K, V: [B, Hkv, Lkv, D]
    Out:  [B, Hq,  1,   D]

GQA is handled in-kernel (``Hq % Hkv == 0``); each program handles one
(batch, q_head) pair.  A split-K variant splits the N-range across
``SPLIT_K`` programs per (B, Hq) and a small merge kernel recombines
the partials via log-sum-exp.  Everything accumulates in FP32.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _pick_split_k(Lkv: int) -> int:
    """Split the N-range once the per-(B,Hq) program under-fills the SMs.

    Thresholds hand-tuned on B200 bf16 at ``bs * Hq = 16``.  The merge
    kernel costs ~25 us, so splitting only pays off past these lengths.
    """
    if Lkv >= 16384:
        return 8
    if Lkv >= 8192:
        return 4
    if Lkv >= 4096:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Single-pass kernel (no split-K) — grid = (B * Hq,)
# ---------------------------------------------------------------------------


@triton.jit
def _decode_kernel(
    Q, K, V, Out, Lse,
    cache_seqlens_ptr,           # [B] int32, or null
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_lseb, stride_lseh,
    B, Hq, Hkv, Lkv,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_CACHE_SEQLEN: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_b = off_bh // Hq
    off_hq = off_bh % Hq
    off_hkv = off_hq // GROUP_SIZE

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    # Load the single Q row (kept in registers across the N-loop).
    q = tl.load(
        Q + off_b * stride_qb + off_hq * stride_qh + offs_d * stride_qd,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    if HAS_CACHE_SEQLEN:
        end_n = tl.load(cache_seqlens_ptr + off_b).to(tl.int32)
    else:
        end_n = Lkv

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end_n

        k = tl.load(
            K + off_b * stride_kb + off_hkv * stride_kh
            + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V + off_b * stride_vb + off_hkv * stride_vh
            + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        # qk[n] = <q, k[n]> * sm_scale, then online-softmax update.
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(n_mask, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(tl.where(m_i == -float("inf"), 0.0, m_i) - m_new_safe)
        p = tl.where(n_mask, tl.exp(qk - m_new_safe), 0.0)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe
    m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
    lse = tl.where(l_i == 0.0, -float("inf"), m_i_safe + tl.log(l_safe))

    tl.store(
        Out + off_b * stride_ob + off_hq * stride_oh + offs_d * stride_od,
        out.to(Out.dtype.element_ty), mask=d_mask,
    )
    tl.store(Lse + off_b * stride_lseb + off_hq * stride_lseh, lse)


# ---------------------------------------------------------------------------
# Split-K kernel + merge (for long context)
# ---------------------------------------------------------------------------


@triton.jit
def _decode_split_k_kernel(
    Q, K, V,
    AccOut,   # [B, Hq, SPLIT_K, D]  fp32 partial acc
    LseOut,   # [B, Hq, SPLIT_K]     fp32 partial lse
    cache_seqlens_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_aob, stride_aoh, stride_aos, stride_aod,
    stride_lob, stride_loh, stride_los,
    B, Hq, Hkv, Lkv,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_CACHE_SEQLEN: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_s = tl.program_id(1)  # split-K chunk id
    off_b = off_bh // Hq
    off_hq = off_bh % Hq
    off_hkv = off_hq // GROUP_SIZE

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    if HAS_CACHE_SEQLEN:
        total_n = tl.load(cache_seqlens_ptr + off_b).to(tl.int32)
    else:
        total_n = Lkv

    # Evenly partition [0, total_n) into SPLIT_K chunks.  Last chunk may be
    # a hair smaller; the N-loop handles the off-end with ``n_mask``.
    chunk = (total_n + SPLIT_K - 1) // SPLIT_K
    start_n0 = off_s * chunk
    end_n = tl.minimum(start_n0 + chunk, total_n)

    q = tl.load(
        Q + off_b * stride_qb + off_hq * stride_qh + offs_d * stride_qd,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for start_n in range(start_n0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end_n

        k = tl.load(
            K + off_b * stride_kb + off_hkv * stride_kh
            + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V + off_b * stride_vb + off_hkv * stride_vh
            + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(n_mask, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(tl.where(m_i == -float("inf"), 0.0, m_i) - m_new_safe)
        p = tl.where(n_mask, tl.exp(qk - m_new_safe), 0.0)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new

    # Store *normalized* per-chunk attention output (acc / l_i) + the
    # chunk's lse so the merge can recombine with simple softmax weights.
    # Empty chunks (start_n0 >= total_n) record lse=-inf; acc is 0.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc_norm = acc / l_safe
    m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
    partial_lse = tl.where(l_i == 0.0, -float("inf"), m_i_safe + tl.log(l_i))

    tl.store(
        AccOut + off_b * stride_aob + off_hq * stride_aoh + off_s * stride_aos
               + offs_d * stride_aod,
        acc_norm, mask=d_mask,
    )
    tl.store(
        LseOut + off_b * stride_lob + off_hq * stride_loh + off_s * stride_los,
        partial_lse,
    )


@triton.jit
def _decode_merge_kernel(
    Acc,    # [B, Hq, SPLIT_K, D]  fp32
    Lse,    # [B, Hq, SPLIT_K]     fp32
    Out,    # [B, Hq, D]           dtype
    OutLse, # [B, Hq]              fp32
    stride_ab, stride_ah, stride_as, stride_ad,
    stride_lb, stride_lh, stride_ls,
    stride_ob, stride_oh, stride_od,
    stride_olb, stride_olh,
    B, Hq,
    SPLIT_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    """Merge SPLIT_K partial attention outputs via log-sum-exp.

    Each partial has (acc_s, lse_s) where acc_s = sum_{n in chunk} p(n) * v(n)
    with p(n) normalized only within the chunk's own max.  Recombine:
        m = max_s lse_s;  w_s = exp(lse_s - m)
        out = sum_s w_s * acc_s / sum_s w_s
        lse = m + log(sum_s w_s)
    """
    off_bh = tl.program_id(0)
    off_b = off_bh // Hq
    off_h = off_bh % Hq

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    # Load all SPLIT_K partial lses to find the combined max.
    offs_s = tl.arange(0, SPLIT_K)
    lse_base = Lse + off_b * stride_lb + off_h * stride_lh
    partial_lse = tl.load(lse_base + offs_s * stride_ls)   # [SPLIT_K]
    m = tl.max(partial_lse)                                 # -inf if every chunk was empty
    m_safe = tl.where(m == -float("inf"), 0.0, m)

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    w_sum = 0.0
    for s in tl.static_range(SPLIT_K):
        partial_acc = tl.load(
            Acc + off_b * stride_ab + off_h * stride_ah + s * stride_as
                + offs_d * stride_ad,
            mask=d_mask, other=0.0,
        )
        lse_s = tl.load(lse_base + s * stride_ls)
        # exp(-inf - m_safe) = 0; exp clamped via tl.where on lse_s == -inf.
        w_s = tl.where(lse_s == -float("inf"), 0.0, tl.exp(lse_s - m_safe))
        acc = acc + w_s * partial_acc
        w_sum = w_sum + w_s

    w_safe = tl.where(w_sum == 0.0, 1.0, w_sum)
    out = acc / w_safe
    lse = tl.where(w_sum == 0.0, -float("inf"), m_safe + tl.log(w_safe))

    tl.store(
        Out + off_b * stride_ob + off_h * stride_oh + offs_d * stride_od,
        out.to(Out.dtype.element_ty), mask=d_mask,
    )
    tl.store(OutLse + off_b * stride_olb + off_h * stride_olh, lse)


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------


def tah_attention_decode(
    q: torch.Tensor,                      # [B, Hq, 1, D] or [B, Hq, D]
    k: torch.Tensor,                      # [B, Hkv, Lkv, D]
    v: torch.Tensor,                      # [B, Hkv, Lkv, D]
    cache_seqlens: Optional[torch.Tensor] = None,  # [B] int, valid cache length per batch
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Plain causal flash-attention-v2 decode kernel.

    Lq=1 only (squeeze in / unsqueeze out).  GQA is handled in-kernel
    via the ``Hq // Hkv`` group-size constexpr.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    squeeze_out = q.dim() == 4 and q.shape[2] == 1
    if q.dim() == 4:
        assert q.shape[2] == 1, f"tah_attention_decode requires Lq=1, got {q.shape[2]}"
        q = q.squeeze(2)
    B, Hq, D = q.shape
    Bk, Hkv, Lkv, Dk = k.shape
    assert Bk == B and Dk == D and v.shape == k.shape
    assert Hq % Hkv == 0
    group_size = Hq // Hkv
    sm_scale = float(sm_scale) if sm_scale is not None else (1.0 / D ** 0.5)

    q_ = q.contiguous()
    k_ = k.contiguous()
    v_ = v.contiguous()

    has_cs = cache_seqlens is not None
    cs_ = cache_seqlens.contiguous().to(torch.int32) if has_cs else q_  # dummy pointer

    BLOCK_D = _next_pow2(D)
    SPLIT_K = _pick_split_k(Lkv)
    # Per-chunk N-loop length determines the best BLOCK_N / num_warps.
    # Short chunks like BN=64 nw=4; long chunks need BN=256 nw=8 for BW.
    chunk_len = Lkv // SPLIT_K
    if chunk_len < 512:
        BLOCK_N, NUM_WARPS = 64, 4
    elif chunk_len < 2048:
        BLOCK_N, NUM_WARPS = 128, 4
    else:
        BLOCK_N, NUM_WARPS = 256, 8

    out = torch.empty_like(q_)
    lse = torch.empty((B, Hq), dtype=torch.float32, device=q.device)

    if SPLIT_K == 1:
        _decode_kernel[(B * Hq,)](
            q_, k_, v_, out, lse,
            cs_, sm_scale,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2), k_.stride(3),
            v_.stride(0), v_.stride(1), v_.stride(2), v_.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            B, Hq, Hkv, Lkv,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            D=D, GROUP_SIZE=group_size,
            HAS_CACHE_SEQLEN=has_cs,
            num_warps=NUM_WARPS, num_stages=1,
        )
    else:
        acc_partials = torch.empty((B, Hq, SPLIT_K, D), dtype=torch.float32, device=q.device)
        lse_partials = torch.empty((B, Hq, SPLIT_K), dtype=torch.float32, device=q.device)
        _decode_split_k_kernel[(B * Hq, SPLIT_K)](
            q_, k_, v_,
            acc_partials, lse_partials,
            cs_, sm_scale,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2), k_.stride(3),
            v_.stride(0), v_.stride(1), v_.stride(2), v_.stride(3),
            acc_partials.stride(0), acc_partials.stride(1), acc_partials.stride(2), acc_partials.stride(3),
            lse_partials.stride(0), lse_partials.stride(1), lse_partials.stride(2),
            B, Hq, Hkv, Lkv,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            D=D, GROUP_SIZE=group_size,
            SPLIT_K=SPLIT_K,
            HAS_CACHE_SEQLEN=has_cs,
            num_warps=NUM_WARPS, num_stages=1,
        )
        _decode_merge_kernel[(B * Hq,)](
            acc_partials, lse_partials, out, lse,
            acc_partials.stride(0), acc_partials.stride(1), acc_partials.stride(2), acc_partials.stride(3),
            lse_partials.stride(0), lse_partials.stride(1), lse_partials.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            B, Hq,
            SPLIT_K=SPLIT_K, BLOCK_D=BLOCK_D, D=D,
            num_warps=2, num_stages=1,
        )

    if squeeze_out:
        out = out.unsqueeze(2)
    return (out, lse) if return_lse else out
