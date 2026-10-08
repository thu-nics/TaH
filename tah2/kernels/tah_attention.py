"""Fused Triton attention for TaH iter >= 1.

At iter_depth >= 1 the TaH wrapper needs attention where only a subset
of KV tokens is visible per query — the "visible" predicate defined in
``tah2.utils.sparse_ops.create_tah_sdpa_attention_mask``.  The stock
SDPA path has to materialize this as a 4-D additive mask, which forces
the slow math fallback (no flash-attn / no mem-efficient).

This module re-implements the same predicate directly in Triton and
consumes the 1-D index tensors (``kv_iter``, ``kv_pos``, ``kv_valid``,
``q_pos``) plus the scalar ``cur_iter`` without building a 4-D mask.
Flash-attn-v2 style tiling with online softmax; GQA handled in-kernel
without ``repeat_kv``.

ITER1_DIAG fast path
--------------------
At ``cur_iter == 1`` with MODE == causal, the KV tensor is laid out as
``[iter-0 full cache | iter-1 active queries]`` with ``kv_iter = 0``
for ``n < L_cache`` and ``1`` for ``n >= L_cache``.  The iter-1
portion is *diagonal* wrt Q (``kv_pos[L_cache + m] == q_pos[m]`` by
construction in ``create_tah_sdpa_attention_mask``), so under causal +
same-pos visibility exactly one iter-1 entry per row contributes.  We
replace the full iter-1 N-scan with a single per-row dot.

We also tighten the iter-0 loop bound to ``min(L_cache, qp_max+1)``:
in iter-0 ``kv_pos[n] == n``, so any n past the block's ``q_pos.max()``
is causal-invisible.  Big win when ``q_pos`` is sorted ascending
(which it is in real training — ``to_active`` uses a stable sort on
the original position index); no-op on random ``q_pos``.

Forward + backward are exposed via :func:`tah_attention` (an
``autograd.Function`` wrapper).  Tuned for ``head_dim == 128`` + bf16
on B200.  Two-pass backward (no atomics): dK/dV parallel over N, dQ
parallel over M.  Both write disjoint rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _tah_attn_fwd_kernel(
    Q, K, V, Out,
    Lse,  # log-sum-exp per row, needed by backward
    kv_iter_ptr, kv_pos_ptr, kv_valid_ptr, q_pos_ptr, kv_continue_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lseb, stride_lseh, stride_lsem,
    stride_kvipos_b, stride_kvipos_n,
    stride_kvval_b, stride_kvval_n,
    stride_qpos_b, stride_qpos_m,
    stride_cont_b, stride_cont_n,
    B, Hq, Hkv, Lq, Lkv, L_cache,
    cur_iter,
    MODE: tl.constexpr,  # 0=causal, 1=duo, 2=duo_posterior, 5=same_iter
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,  # pow-2 >= D
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ITER1_DIAG: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // Hq
    off_hq = off_bh % Hq
    off_hkv = off_hq // GROUP_SIZE

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    m_mask = offs_m < Lq

    q_ptrs = Q + off_b * stride_qb + off_hq * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    q_pos = tl.load(
        q_pos_ptr + off_b * stride_qpos_b + offs_m * stride_qpos_m,
        mask=m_mask, other=0,
    ).to(tl.int32)

    # Online-softmax state.
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if ITER1_DIAG:
        # Fast path: scan iter-0 only, up to min(L_cache, qp_max+1).
        qp_max = tl.max(tl.where(m_mask, q_pos, -1), axis=0)
        iter0_end = tl.minimum(L_cache, qp_max + 1)
        for start_n in range(0, iter0_end, BLOCK_N):
            n_idx = start_n + offs_n
            # Use L_cache as the hard bound; iter0_end may be mid-block and
            # the (kv_pos <= q_pos) predicate handles entries past qp_max.
            n_mask = n_idx < L_cache

            kv_pos_v = tl.load(
                kv_pos_ptr + off_b * stride_kvipos_b + n_idx * stride_kvipos_n,
                mask=n_mask, other=0,
            ).to(tl.int32)
            kv_valid_v = tl.load(
                kv_valid_ptr + off_b * stride_kvval_b + n_idx * stride_kvval_n,
                mask=n_mask, other=0,
            ).to(tl.int32)

            # kv_iter is 0 here, so visibility simplifies for both MODEs to
            # position + validity + in-range + row-valid.
            visible = (
                (kv_pos_v[None, :] <= q_pos[:, None])
                & (kv_valid_v[None, :] == 1)
                & n_mask[None, :]
                & m_mask[:, None]
            )

            k_ptrs = K + off_b * stride_kb + off_hkv * stride_kh + n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kk
            v_ptrs = V + off_b * stride_vb + off_hkv * stride_vh + n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            m_i, l_i, acc = _fwd_softmax_update(q, k, v, visible, sm_scale, m_i, l_i, acc)

        # Diagonal iter-1 contribution: one entry per row at kv index
        # L_cache + m (same position as q by construction).
        iter1_n_idx = L_cache + offs_m
        iter1_valid = m_mask & (iter1_n_idx < Lkv)

        k1_ptrs = K + off_b * stride_kb + off_hkv * stride_kh + iter1_n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v1_ptrs = V + off_b * stride_vb + off_hkv * stride_vh + iter1_n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k1 = tl.load(k1_ptrs, mask=iter1_valid[:, None] & d_mask[None, :], other=0.0)
        v1 = tl.load(v1_ptrs, mask=iter1_valid[:, None] & d_mask[None, :], other=0.0)
        kv_valid_1 = tl.load(
            kv_valid_ptr + off_b * stride_kvval_b + iter1_n_idx * stride_kvval_n,
            mask=iter1_valid, other=0,
        ).to(tl.int32)
        vis1 = iter1_valid & (kv_valid_1 == 1)

        # Per-row dot (D=128, cheap); no tensor cores but negligible.
        qk1 = tl.sum(q.to(tl.float32) * k1.to(tl.float32), axis=1) * sm_scale
        qk1 = tl.where(vis1, qk1, -float("inf"))

        m_new = tl.maximum(m_i, qk1)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(tl.where(m_i == -float("inf"), 0.0, m_i) - m_new_safe)
        p1 = tl.where(vis1, tl.exp(qk1 - m_new_safe), 0.0)

        acc = acc * alpha[:, None] + p1[:, None] * v1.to(tl.float32)
        l_i = l_i * alpha + p1
        m_i = m_new
    else:
        # General path: scan the full KV range with the full predicate.
        for start_n in range(0, Lkv, BLOCK_N):
            n_idx = start_n + offs_n
            n_mask = n_idx < Lkv

            kv_iter_v = tl.load(kv_iter_ptr + n_idx, mask=n_mask, other=0).to(tl.int32)
            kv_pos_v = tl.load(
                kv_pos_ptr + off_b * stride_kvipos_b + n_idx * stride_kvipos_n,
                mask=n_mask, other=0,
            ).to(tl.int32)
            kv_valid_v = tl.load(
                kv_valid_ptr + off_b * stride_kvval_b + n_idx * stride_kvval_n,
                mask=n_mask, other=0,
            ).to(tl.int32)
            # Only the no-grad duo posterior pass reads the continue flag.
            if MODE == 2:
                kv_continue_v = tl.load(
                    kv_continue_ptr + off_b * stride_cont_b + n_idx * stride_cont_n,
                    mask=n_mask, other=0,
                ).to(tl.int32)
            else:
                kv_continue_v = kv_valid_v

            visible = _visibility_mask(
                q_pos, kv_pos_v, kv_iter_v, kv_valid_v, kv_continue_v, n_mask, m_mask, cur_iter, MODE,
            )

            k_ptrs = K + off_b * stride_kb + off_hkv * stride_kh + n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kk
            v_ptrs = V + off_b * stride_vb + off_hkv * stride_vh + n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            m_i, l_i, acc = _fwd_softmax_update(q, k, v, visible, sm_scale, m_i, l_i, acc)

    # Finalize: divide by denominator; write output and LSE.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
    lse = tl.where(l_i == 0.0, -float("inf"), m_i_safe + tl.log(l_safe))

    o_ptrs = Out + off_b * stride_ob + off_hq * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    lse_ptrs = Lse + off_b * stride_lseb + off_hq * stride_lseh + offs_m * stride_lsem
    tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :])
    tl.store(lse_ptrs, lse, mask=m_mask)


# ---------------------------------------------------------------------------
# Shared helpers — inlined by triton JIT
# ---------------------------------------------------------------------------


@triton.jit
def _visibility_mask(q_pos, kv_pos_v, kv_iter_v, kv_valid_v, kv_continue_v, n_mask, m_mask, cur_iter, MODE: tl.constexpr):
    """Full TaH visibility predicate ([M, N] bool).

    MODE: 0=causal, 1=duo, 2=duo_posterior (no-grad posterior side pass),
    5=same_iter (within-iteration causal only).
    ``kv_continue_v`` is only read for MODE == 2; callers pass a dummy otherwise.
    """
    qp = q_pos[:, None]
    kp = kv_pos_v[None, :]
    ki = kv_iter_v[None, :]
    position_mask = kp <= qp
    if MODE == 0:
        # Causal: iter-0 causal + same-position for any kv_iter <= cur_iter.
        root_visible = (ki == 0) & position_mask
        same_pos_visible = (kp == qp) & (ki <= cur_iter)
        visible = root_visible | same_pos_visible
    elif MODE == 1:
        # Duo: everything causal for kv_iter <= cur_iter.
        visible = position_mask & (ki <= cur_iter)
    elif MODE == 2:
        # Duo posterior: prior-iter always causal; current-iter visible iff continued or own.
        kc = kv_continue_v[None, :]
        visible = position_mask & ((ki < cur_iter) | (kc == 1) | (kp == qp))
    else:
        # Same-iter (MODE==5): only the current iteration's KV, causal. Prior-iter
        # cache entries are fully hidden; no continue flag needed.
        visible = position_mask & (ki == cur_iter)
    return visible & (kv_valid_v[None, :] == 1) & n_mask[None, :] & m_mask[:, None]


@triton.jit
def _fwd_softmax_update(q, k, v, visible, sm_scale, m_i, l_i, acc):
    """Flash-attn-v2 online-softmax update for a single (Q-block, KV-block).

    BF16 inputs → FP32 accumulator via ``out_dtype=tl.float32``.
    """
    qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
    qk = tl.where(visible, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    # Sanitize -inf to 0 for arithmetic; rows with no visible keys stay at
    # m_i = -inf, alpha = 1, acc unchanged.
    m_ij_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
    alpha = tl.exp(tl.where(m_i == -float("inf"), 0.0, m_i) - m_ij_safe)
    p = tl.where(visible, tl.exp(qk - m_ij_safe[:, None]), 0.0)

    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    return m_ij, l_i, acc


# ---------------------------------------------------------------------------
# Backward kernels (FA2-style two-pass, no atomics)
#
# dK/dV: parallel over (B, Hkv, N-block).  Each program accumulates dK, dV
#        for its N rows across all Q heads in the KV group and all M rows.
# dQ   : parallel over (B, Hq, M-block).  Each program accumulates dQ for
#        its M rows across all N-blocks.
#
# Both kernels compute `delta[b,h,m] = sum_d O*dO` in PyTorch on the call
# side — a cheap reduction.  Total FLOPs are ~2× the fused single-kernel
# form, but empirically 3–5× faster than `tl.atomic_add` on dQ.
# ---------------------------------------------------------------------------


@triton.jit
def _tah_attn_bwd_dkdv_kernel(
    Q, K, V, DO,
    DK, DV,
    Lse, Delta,
    kv_iter_ptr, kv_pos_ptr, kv_valid_ptr, q_pos_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    stride_lseb, stride_lseh, stride_lsem,
    stride_db, stride_dh, stride_dm,
    stride_kvipos_b, stride_kvipos_n,
    stride_kvval_b, stride_kvval_n,
    stride_qpos_b, stride_qpos_m,
    B, Hq, Hkv, Lq, Lkv, L_cache,
    cur_iter,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ITER1_DIAG: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // Hkv
    off_hkv = off_bh % Hkv

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    n_mask = offs_n < Lkv

    k_ptrs = K + off_b * stride_kb + off_hkv * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_b * stride_vb + off_hkv * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    kv_valid_v = tl.load(
        kv_valid_ptr + off_b * stride_kvval_b + offs_n * stride_kvval_n,
        mask=n_mask, other=0,
    ).to(tl.int32)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # Diagonal fast path: N-block entirely in the iter-1 region.  Each kv
    # entry n is only seen by Q[m = n - L_cache] (the diagonal), so we
    # skip the full M-loop and gather the single Q per n instead.
    if ITER1_DIAG and start_n * BLOCK_N >= L_cache:
        diag_m = offs_n - L_cache
        diag_valid = (diag_m >= 0) & (diag_m < Lq) & n_mask & (kv_valid_v == 1)

        for hq_in_group in tl.static_range(GROUP_SIZE):
            off_hq = off_hkv * GROUP_SIZE + hq_in_group

            q_ptrs = Q + off_b * stride_qb + off_hq * stride_qh + diag_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
            do_ptrs = DO + off_b * stride_dob + off_hq * stride_doh + diag_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
            q_d = tl.load(q_ptrs, mask=diag_valid[:, None] & d_mask[None, :], other=0.0)
            do_d = tl.load(do_ptrs, mask=diag_valid[:, None] & d_mask[None, :], other=0.0)
            lse_d = tl.load(
                Lse + off_b * stride_lseb + off_hq * stride_lseh + diag_m * stride_lsem,
                mask=diag_valid, other=0.0,
            ).to(tl.float32)
            delta_d = tl.load(
                Delta + off_b * stride_db + off_hq * stride_dh + diag_m * stride_dm,
                mask=diag_valid, other=0.0,
            ).to(tl.float32)

            vis = diag_valid & (lse_d > -1e30)

            # Per-n scalars — see docstring; no tensor cores but negligible.
            qk = tl.sum(q_d.to(tl.float32) * k.to(tl.float32), axis=1) * sm_scale
            p = tl.where(vis, tl.exp(qk - lse_d), 0.0)

            dv = dv + p[:, None] * do_d.to(tl.float32)

            dp = tl.sum(do_d.to(tl.float32) * v.to(tl.float32), axis=1)
            ds = tl.where(vis, p * (dp - delta_d), 0.0)
            dk = dk + (ds * sm_scale)[:, None] * q_d.to(tl.float32)

        _store_dk_dv(DK, DV, dk, dv, off_b, off_hkv, offs_n, offs_d,
                     stride_dkb, stride_dkh, stride_dkn, stride_dkk,
                     stride_dvb, stride_dvh, stride_dvn, stride_dvk,
                     n_mask, d_mask)
        return

    # General path: loop over every M-block across the GQA group.  Kept for
    # iter-0 blocks (under ITER1_DIAG) and for blocks that straddle L_cache,
    # as well as the non-ITER1_DIAG case.
    kv_iter_v = tl.load(kv_iter_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    kv_pos_v = tl.load(
        kv_pos_ptr + off_b * stride_kvipos_b + offs_n * stride_kvipos_n,
        mask=n_mask, other=0,
    ).to(tl.int32)

    for hq_in_group in tl.static_range(GROUP_SIZE):
        off_hq = off_hkv * GROUP_SIZE + hq_in_group

        for start_m in range(0, Lq, BLOCK_M):
            m_idx = start_m + offs_m
            m_mask = m_idx < Lq

            q_ptrs = Q + off_b * stride_qb + off_hq * stride_qh + m_idx[:, None] * stride_qm + offs_d[None, :] * stride_qk
            do_ptrs = DO + off_b * stride_dob + off_hq * stride_doh + m_idx[:, None] * stride_dom + offs_d[None, :] * stride_dok
            q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            lse = tl.load(
                Lse + off_b * stride_lseb + off_hq * stride_lseh + m_idx * stride_lsem,
                mask=m_mask, other=0.0,
            ).to(tl.float32)
            delta = tl.load(
                Delta + off_b * stride_db + off_hq * stride_dh + m_idx * stride_dm,
                mask=m_mask, other=0.0,
            ).to(tl.float32)
            q_pos = tl.load(
                q_pos_ptr + off_b * stride_qpos_b + m_idx * stride_qpos_m,
                mask=m_mask, other=0,
            ).to(tl.int32)

            # MODE==2 (posterior side pass) is no-grad and never reaches backward;
            visible = _visibility_mask(
                q_pos, kv_pos_v, kv_iter_v, kv_valid_v, kv_valid_v, n_mask, m_mask, cur_iter, MODE,
            )
            visible = visible & (lse > -1e30)[:, None]

            # Recompute P = exp(QK * scale - LSE).  BF16 inputs → FP32 acc.
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
            p = tl.where(visible, tl.exp(qk - lse[:, None]), 0.0)

            # dV += P^T @ dO
            dv = tl.dot(tl.trans(p.to(do.dtype)), do, acc=dv, out_dtype=tl.float32)

            # dS = P * (dO @ V^T - delta)
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = tl.where(visible, p * (dp - delta[:, None]), 0.0)

            # dK += dS^T @ Q * sm_scale.  Pre-scale ds in FP32 before the bf16
            # cast — the "(tl.dot).to(fp32) * scalar" shape hits a compiler
            # pothole that drops values (was 216× off in dK).
            ds_scaled = (ds * sm_scale).to(q.dtype)
            dk = tl.dot(tl.trans(ds_scaled), q, acc=dk, out_dtype=tl.float32)

    _store_dk_dv(DK, DV, dk, dv, off_b, off_hkv, offs_n, offs_d,
                 stride_dkb, stride_dkh, stride_dkn, stride_dkk,
                 stride_dvb, stride_dvh, stride_dvn, stride_dvk,
                 n_mask, d_mask)


@triton.jit
def _store_dk_dv(DK, DV, dk, dv, off_b, off_hkv, offs_n, offs_d,
                 stride_dkb, stride_dkh, stride_dkn, stride_dkk,
                 stride_dvb, stride_dvh, stride_dvn, stride_dvk,
                 n_mask, d_mask):
    dk_ptrs = DK + off_b * stride_dkb + off_hkv * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    dv_ptrs = DV + off_b * stride_dvb + off_hkv * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
    tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=n_mask[:, None] & d_mask[None, :])
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=n_mask[:, None] & d_mask[None, :])


@triton.jit
def _tah_attn_bwd_dq_kernel(
    Q, K, V, DO,
    DQ,
    Lse, Delta,
    kv_iter_ptr, kv_pos_ptr, kv_valid_ptr, q_pos_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_lseb, stride_lseh, stride_lsem,
    stride_db, stride_dh, stride_dm,
    stride_kvipos_b, stride_kvipos_n,
    stride_kvval_b, stride_kvval_n,
    stride_qpos_b, stride_qpos_m,
    B, Hq, Hkv, Lq, Lkv, L_cache,
    cur_iter,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ITER1_DIAG: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // Hq
    off_hq = off_bh % Hq
    off_hkv = off_hq // GROUP_SIZE

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    m_mask = offs_m < Lq

    q_ptrs = Q + off_b * stride_qb + off_hq * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    do_ptrs = DO + off_b * stride_dob + off_hq * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    lse = tl.load(
        Lse + off_b * stride_lseb + off_hq * stride_lseh + offs_m * stride_lsem,
        mask=m_mask, other=0.0,
    ).to(tl.float32)
    delta = tl.load(
        Delta + off_b * stride_db + off_hq * stride_dh + offs_m * stride_dm,
        mask=m_mask, other=0.0,
    ).to(tl.float32)
    q_pos = tl.load(
        q_pos_ptr + off_b * stride_qpos_b + offs_m * stride_qpos_m,
        mask=m_mask, other=0,
    ).to(tl.int32)
    row_has_keys = lse > -1e30

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if ITER1_DIAG:
        # Fast path: scan iter-0 only up to min(L_cache, qp_max+1); add the
        # one diagonal iter-1 contribution afterwards.
        qp_max = tl.max(tl.where(m_mask, q_pos, -1), axis=0)
        iter0_end = tl.minimum(L_cache, qp_max + 1)
        for start_n in range(0, iter0_end, BLOCK_N):
            n_idx = start_n + offs_n
            n_mask = n_idx < L_cache

            k_ptrs = K + off_b * stride_kb + off_hkv * stride_kh + n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kk
            v_ptrs = V + off_b * stride_vb + off_hkv * stride_vh + n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            kv_pos_v = tl.load(
                kv_pos_ptr + off_b * stride_kvipos_b + n_idx * stride_kvipos_n,
                mask=n_mask, other=0,
            ).to(tl.int32)
            kv_valid_v = tl.load(
                kv_valid_ptr + off_b * stride_kvval_b + n_idx * stride_kvval_n,
                mask=n_mask, other=0,
            ).to(tl.int32)

            visible = (
                (kv_pos_v[None, :] <= q_pos[:, None])
                & (kv_valid_v[None, :] == 1)
                & n_mask[None, :]
                & m_mask[:, None]
                & row_has_keys[:, None]
            )

            dq = _bwd_dq_block_update(q, k, v, do, lse, delta, visible, sm_scale, dq)

        # Diagonal iter-1 step.
        iter1_n_idx = L_cache + offs_m
        iter1_valid = m_mask & (iter1_n_idx < Lkv) & row_has_keys

        k1_ptrs = K + off_b * stride_kb + off_hkv * stride_kh + iter1_n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v1_ptrs = V + off_b * stride_vb + off_hkv * stride_vh + iter1_n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k1 = tl.load(k1_ptrs, mask=iter1_valid[:, None] & d_mask[None, :], other=0.0)
        v1 = tl.load(v1_ptrs, mask=iter1_valid[:, None] & d_mask[None, :], other=0.0)
        kv_valid_1 = tl.load(
            kv_valid_ptr + off_b * stride_kvval_b + iter1_n_idx * stride_kvval_n,
            mask=iter1_valid, other=0,
        ).to(tl.int32)
        vis1 = iter1_valid & (kv_valid_1 == 1)

        qk1 = tl.sum(q.to(tl.float32) * k1.to(tl.float32), axis=1) * sm_scale
        p1 = tl.where(vis1, tl.exp(qk1 - lse), 0.0)
        dp1 = tl.sum(do.to(tl.float32) * v1.to(tl.float32), axis=1)
        ds1 = tl.where(vis1, p1 * (dp1 - delta), 0.0)
        dq = dq + (ds1 * sm_scale)[:, None] * k1.to(tl.float32)
    else:
        # General path.
        for start_n in range(0, Lkv, BLOCK_N):
            n_idx = start_n + offs_n
            n_mask = n_idx < Lkv

            k_ptrs = K + off_b * stride_kb + off_hkv * stride_kh + n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kk
            v_ptrs = V + off_b * stride_vb + off_hkv * stride_vh + n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            kv_iter_v = tl.load(kv_iter_ptr + n_idx, mask=n_mask, other=0).to(tl.int32)
            kv_pos_v = tl.load(
                kv_pos_ptr + off_b * stride_kvipos_b + n_idx * stride_kvipos_n,
                mask=n_mask, other=0,
            ).to(tl.int32)
            kv_valid_v = tl.load(
                kv_valid_ptr + off_b * stride_kvval_b + n_idx * stride_kvval_n,
                mask=n_mask, other=0,
            ).to(tl.int32)

            # MODE==2 (posterior side pass) is no-grad and never reaches backward;
            visible = _visibility_mask(
                q_pos, kv_pos_v, kv_iter_v, kv_valid_v, kv_valid_v, n_mask, m_mask, cur_iter, MODE,
            )
            visible = visible & row_has_keys[:, None]

            dq = _bwd_dq_block_update(q, k, v, do, lse, delta, visible, sm_scale, dq)

    dq_ptrs = DQ + off_b * stride_dqb + off_hq * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def _bwd_dq_block_update(q, k, v, do, lse, delta, visible, sm_scale, dq):
    """dQ accumulation for one (M-block, N-block)."""
    qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
    p = tl.where(visible, tl.exp(qk - lse[:, None]), 0.0)
    dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
    ds = tl.where(visible, p * (dp - delta[:, None]), 0.0)
    # Pre-scale ds in fp32 before bf16 cast (same pothole avoidance as dK).
    ds_scaled = (ds * sm_scale).to(k.dtype)
    return tl.dot(ds_scaled, k, acc=dq, out_dtype=tl.float32)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


# The diagonal step runs without tensor cores (per-row dot); it only pays off
# once enough iter-1 N-blocks are saved to amortize its cost.  Measured
# break-even on B200 bf16 is Lq ~ 400 — 512 is a round threshold above that.
_ITER1_DIAG_MIN_LQ = 512


def _iter1_diag_enabled(mode_int: int, cur_iter: int, L_cache: int, Lq: int) -> bool:
    """Can we take the ITER1_DIAG fast path for this call?

    Requires MODE=causal (duo is not diagonal), cur_iter=1 (iter-0 cache
    is pure iter-0), a valid iter-0 cache, and Lq >= _ITER1_DIAG_MIN_LQ.
    """
    return (
        mode_int == 0
        and cur_iter == 1
        and L_cache >= 0
        and Lq >= _ITER1_DIAG_MIN_LQ
    )


# Hand-tuned block configs (B200, bf16, head_dim=128) via tests/kernel_tune.py.
# Forward: BM=128 under-fills small Lq; we drop to 64 there (see _TaHAttention).
_FWD_NUM_WARPS = 8
_FWD_NUM_STAGES = 2
_FWD_BLOCK_N = 64
_DKDV_BLOCK_M, _DKDV_BLOCK_N, _DKDV_NUM_WARPS, _DKDV_NUM_STAGES = 64, 64, 8, 3
_DQ_BLOCK_M, _DQ_BLOCK_N, _DQ_NUM_WARPS, _DQ_NUM_STAGES = 64, 32, 8, 2


class _TaHAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,         # [B, Hq, Lq, D]
        k: torch.Tensor,         # [B, Hkv, Lkv, D]
        v: torch.Tensor,         # [B, Hkv, Lkv, D]
        kv_iter: torch.Tensor,   # [Lkv]
        kv_pos: torch.Tensor,    # [B, Lkv]
        kv_valid: torch.Tensor,  # [B, Lkv]
        q_pos: torch.Tensor,     # [B, Lq]
        kv_is_continue: torch.Tensor,  # [B, Lkv] int (1=continued); only read for "duo_posterior"
        cur_iter: int,
        sm_scale: float,
        mode: str,  # "causal", "duo", "duo_posterior", or "same_iter"
    ) -> torch.Tensor:
        assert q.is_cuda and k.is_cuda and v.is_cuda
        B, Hq, Lq, D = q.shape
        Bk, Hkv, Lkv, Dk = k.shape
        assert Bk == B and Dk == D and v.shape == k.shape
        assert Hq % Hkv == 0, "num_q_heads must be divisible by num_kv_heads"

        group_size = Hq // Hkv
        mode_int = {"causal": 0, "duo": 1, "duo_posterior": 2, "same_iter": 5}[mode]
        L_cache = Lkv - Lq
        iter1_diag = _iter1_diag_enabled(mode_int, int(cur_iter), L_cache, Lq)

        # Pack inputs to the kernel's expected dtype / layout.
        q_ = q.contiguous()
        k_ = k.contiguous()
        v_ = v.contiguous()
        kv_iter_ = kv_iter.contiguous().to(torch.int32)
        kv_pos_ = kv_pos.contiguous().to(torch.int32)
        kv_valid_ = kv_valid.contiguous().to(torch.int32)
        q_pos_ = q_pos.contiguous().to(torch.int32)
        if mode_int == 2:
            assert kv_is_continue is not None, f"mode={mode} requires kv_is_continue"
            kv_continue_ = kv_is_continue.contiguous().to(torch.int32)
        else:
            # Placeholder: the kernel never reads it for causal/duo/same_iter
            # (branch compiled out). Reuse kv_valid_ to avoid an allocation in
            # the hot path.
            kv_continue_ = kv_valid_

        out = torch.empty_like(q_)
        lse = torch.empty((B, Hq, Lq), dtype=torch.float32, device=q.device)

        BLOCK_M = 128 if Lq >= 1024 else 64
        BLOCK_D = _next_pow2(D)

        _tah_attn_fwd_kernel[(triton.cdiv(Lq, BLOCK_M), B * Hq)](
            q_, k_, v_, out, lse,
            kv_iter_, kv_pos_, kv_valid_, q_pos_, kv_continue_,
            sm_scale,
            q_.stride(0), q_.stride(1), q_.stride(2), q_.stride(3),
            k_.stride(0), k_.stride(1), k_.stride(2), k_.stride(3),
            v_.stride(0), v_.stride(1), v_.stride(2), v_.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            kv_pos_.stride(0), kv_pos_.stride(1),
            kv_valid_.stride(0), kv_valid_.stride(1),
            q_pos_.stride(0), q_pos_.stride(1),
            kv_continue_.stride(0), kv_continue_.stride(1),
            B, Hq, Hkv, Lq, Lkv, L_cache,
            int(cur_iter),
            MODE=mode_int,
            BLOCK_M=BLOCK_M, BLOCK_N=_FWD_BLOCK_N, BLOCK_D=BLOCK_D,
            D=D, GROUP_SIZE=group_size,
            ITER1_DIAG=iter1_diag,
            num_warps=_FWD_NUM_WARPS, num_stages=_FWD_NUM_STAGES,
        )

        ctx.save_for_backward(q_, k_, v_, out, lse, kv_iter_, kv_pos_, kv_valid_, q_pos_)
        ctx.sm_scale = sm_scale
        ctx.mode_int = mode_int
        ctx.cur_iter = int(cur_iter)
        ctx.group_size = group_size
        ctx.shape = (B, Hq, Hkv, Lq, Lkv, D)
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, o, lse, kv_iter, kv_pos, kv_valid, q_pos = ctx.saved_tensors
        B, Hq, Hkv, Lq, Lkv, D = ctx.shape
        sm_scale = ctx.sm_scale
        mode_int = ctx.mode_int
        cur_iter = ctx.cur_iter
        group_size = ctx.group_size
        # Posterior labels use no-grad attention.
        assert mode_int != 2, "duo_posterior is no-grad; backward is not supported"

        do_ = do.contiguous()
        BLOCK_D = _next_pow2(D)
        L_cache = Lkv - Lq
        iter1_diag = _iter1_diag_enabled(mode_int, cur_iter, L_cache, Lq)

        # delta[b, h, m] = sum_d O * dO — cheap reduction, done in PyTorch.
        delta = (o.to(torch.float32) * do_.to(torch.float32)).sum(dim=-1)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # Pass 1: dK / dV, parallel over (B, Hkv, N-block).
        _tah_attn_bwd_dkdv_kernel[(triton.cdiv(Lkv, _DKDV_BLOCK_N), B * Hkv)](
            q, k, v, do_,
            dk, dv,
            lse, delta,
            kv_iter, kv_pos, kv_valid, q_pos,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do_.stride(0), do_.stride(1), do_.stride(2), do_.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            kv_pos.stride(0), kv_pos.stride(1),
            kv_valid.stride(0), kv_valid.stride(1),
            q_pos.stride(0), q_pos.stride(1),
            B, Hq, Hkv, Lq, Lkv, L_cache,
            int(cur_iter),
            MODE=mode_int,
            BLOCK_M=_DKDV_BLOCK_M, BLOCK_N=_DKDV_BLOCK_N, BLOCK_D=BLOCK_D,
            D=D, GROUP_SIZE=group_size,
            ITER1_DIAG=iter1_diag,
            num_warps=_DKDV_NUM_WARPS, num_stages=_DKDV_NUM_STAGES,
        )

        # Pass 2: dQ, parallel over (B, Hq, M-block).
        _tah_attn_bwd_dq_kernel[(triton.cdiv(Lq, _DQ_BLOCK_M), B * Hq)](
            q, k, v, do_,
            dq,
            lse, delta,
            kv_iter, kv_pos, kv_valid, q_pos,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do_.stride(0), do_.stride(1), do_.stride(2), do_.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            kv_pos.stride(0), kv_pos.stride(1),
            kv_valid.stride(0), kv_valid.stride(1),
            q_pos.stride(0), q_pos.stride(1),
            B, Hq, Hkv, Lq, Lkv, L_cache,
            int(cur_iter),
            MODE=mode_int,
            BLOCK_M=_DQ_BLOCK_M, BLOCK_N=_DQ_BLOCK_N, BLOCK_D=BLOCK_D,
            D=D, GROUP_SIZE=group_size,
            ITER1_DIAG=iter1_diag,
            num_warps=_DQ_NUM_WARPS, num_stages=_DQ_NUM_STAGES,
        )

        # Grads for: q, k, v, kv_iter, kv_pos, kv_valid, q_pos, kv_is_continue,
        # cur_iter, sm_scale, mode  (11 inputs → 11 returns).
        return dq, dk, dv, None, None, None, None, None, None, None, None


def tah_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_iter: torch.Tensor,
    kv_pos: torch.Tensor,
    kv_valid: torch.Tensor,
    q_pos: torch.Tensor,
    cur_iter: int,
    sm_scale: float,
    mode: str = "causal",
    kv_is_continue: torch.Tensor = None,
) -> torch.Tensor:
    """TaH iter-`cur_iter` attention.

    Args:
        q:        Query tensor [B, Hq, Lq, D].
        k:        Key tensor   [B, Hkv, Lkv, D]  (GQA: Hq % Hkv == 0).
        v:        Value tensor [B, Hkv, Lkv, D].
        kv_iter:  Iteration index per KV token [Lkv].
        kv_pos:   Position id per (batch, KV token) [B, Lkv].
        kv_valid: Validity flag per (batch, KV token) [B, Lkv] (1 = valid).
        q_pos:    Position id per (batch, Q token) [B, Lq].
        cur_iter: Current iteration depth (>= 1).
        sm_scale: Softmax scale (typically 1 / sqrt(D)).
        mode:     ``"causal"``, ``"duo"``, ``"duo_posterior"`` or
                  ``"same_iter"`` — see the visibility predicate above.
                  ``"same_iter"`` attends only within the current iteration
                  (causal), so prior-iter cache entries are fully masked.
        kv_is_continue: Per-(batch, KV) flag [B, Lkv] (1 = continued).
                  Required only for the no-grad ``duo_posterior`` side pass.

    Returns:
        Output tensor [B, Hq, Lq, D] matching q's dtype.
    """
    return _TaHAttention.apply(
        q, k, v, kv_iter, kv_pos, kv_valid, q_pos, kv_is_continue, cur_iter, sm_scale, mode
    )
