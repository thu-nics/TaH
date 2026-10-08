"""DUO iter>=1 prefill cascade for TaH+ Qwen3.

After iter0 prefill, run one extra pass per iteration ``d`` in ``1..max_iter-1``:
the iter_decider selects which active positions continue, the input_updater
builds iter-``d`` input embeds, and a fused attention call writes iter-``d`` K/V
into ``Engine._tah_iter_page_tables[d-1]``.  This closes the training/serving
gap so later decode iter-``d`` queries see prompt-side iter-``d`` K.

DUO: each iter-``d`` query attends iter0 KV + streams ``1..d`` truncated to
positions ``<= q_pos`` — served by the tiled paged Triton kernel in
``kernel/triton/tah_duo_prefill_attn.py``.

With ``stop_prob_mix`` the last prompt token's logits row is replaced by its
stop-prob mixture across executed iters (matching HF's first-token sample).
Chunked prefill: prior-chunk stream slots are at strictly earlier positions.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Dict, List

import torch
import torch.nn.functional as F

from tah2.minisgl.core import Batch, get_global_ctx
from tah2.minisgl.kernel.triton.tah_duo_prefill_attn import (
    build_duo_prefill_tiles,
    duo_prefill_attention,
)

if TYPE_CHECKING:
    from tah2.minisgl.engine.engine import Engine


def run_duo_iter_prefill(
    engine: "Engine",
    batch: Batch,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    base_embeds: torch.Tensor,
) -> None:
    from tah2.minisgl.models.tah_qwen3 import Qwen3MLPIterDecider, TaHQwen3ForCausalLM

    model: TaHQwen3ForCausalLM = engine.model  # type: ignore[assignment]
    if not isinstance(model, TaHQwen3ForCausalLM):
        return
    iter_pts = engine._tah_iter_page_tables
    if iter_pts is None:  # not a DUO checkpoint
        return
    if not isinstance(model.tah_decider, Qwen3MLPIterDecider):
        # Random / entropy deciders aren't trained to gate prompt positions.
        return
    max_iter = model._tah_max_iter
    if max_iter < 2:
        return

    device = hidden.device
    lm_head = model.lm_head
    page_table = engine.page_table
    cache_mgr = engine._tah_cache_manager
    assert cache_mgr is not None, "Cache manager hook missing on engine"

    reqs = batch.padded_reqs
    extend_lens = [r.extend_len for r in reqs]
    offsets = [0]
    for L in extend_lens:
        offsets.append(offsets[-1] + L)

    # Per-req active local offsets (ascending within the req's extend span).
    # ``None`` means the req contributes no active positions (dummy / empty).
    # Start: every valid prompt position is a candidate for iter1.
    active_local: List[torch.Tensor | None] = []
    for i, req in enumerate(reqs):
        s, e = offsets[i], offsets[i + 1]
        if req is engine.dummy_req or e <= s:
            active_local.append(None)
        else:
            active_local.append(torch.arange(e - s, dtype=torch.long, device=device))

    # Per-position iter counts over each req's extend span (1 = iter0 only);
    # flushed to ``engine.tah_prompt_iter_log`` after the cascade so the
    # scheduler can ship prompt-side counts in the usage payload.
    prompt_counts: List[torch.Tensor | None] = [
        None if la is None else torch.ones(int(la.numel()), dtype=torch.long, device=device)
        for la in active_local
    ]

    # ``cur_hidden`` holds the previous iteration's hidden over the full prompt
    # buffer; scattered in place at active positions after each iter forward.
    # The caller only reads ``logits`` afterwards, so mutating ``hidden`` is safe.
    cur_hidden = hidden

    # Snapshots needed to reconstruct prior-stream visible lengths in later iters:
    # per iter d, the active local offsets, their absolute positions, and each
    # req's stream-(d-1) slot count before this chunk.
    iter_pos_local: Dict[int, List[torch.Tensor | None]] = {}
    chunk_start: Dict[int, List[int]] = {}
    max_L0 = max(r.cached_len + r.extend_len for r in reqs)
    iter0_pos = torch.arange(max_L0, device=device, dtype=torch.int32)

    # ── last-prompt-token mixture (closes the first-token serving gap) ──
    # HF samples the first token from the stop_prob_mix mixture over the last
    # prompt token's executed iters; reconstruct it along the cascade.
    fix_last = engine._tah_weighted_method == "stop_prob_mix"
    last_state: Dict[int, Dict] = {}
    if fix_last:
        for i in range(len(reqs)):
            if active_local[i] is None:
                continue
            last_state[i] = {
                "lo": int(extend_lens[i]) - 1,  # local offset of the last prompt token
                "log_accum": None,  # (V,) float32 mixture accumulator
                "rem": 1.0,  # remaining continue mass
                "logp_cur": torch.log_softmax(logits[i].float(), dim=-1),
                "await": False,  # waiting for this pass's hidden
                "done": False,
            }

    def _finalize_last(st: Dict, i: int) -> None:
        final = math.log(max(st["rem"], 1e-10)) + st["logp_cur"]
        if st["log_accum"] is not None:
            final = torch.logaddexp(st["log_accum"], final)
        logits[i] = final.to(logits.dtype)
        st["done"] = True

    for d in range(1, max_iter):
        stream = d - 1

        # ── decider over the currently-active set (uses iter d-1 hidden) ──
        global_chunks: List[torch.Tensor] = []
        req_slices: List[tuple[int, int]] = []
        cursor = 0
        for i in range(len(reqs)):
            la = active_local[i]
            if la is None or la.numel() == 0:
                req_slices.append((0, 0))
                continue
            n = int(la.numel())
            global_chunks.append(la + offsets[i])
            req_slices.append((cursor, cursor + n))
            cursor += n
        if cursor == 0:
            break
        active_g = torch.cat(global_chunks)
        h_act = cur_hidden.index_select(0, active_g)
        base_act = base_embeds.index_select(0, active_g)
        # The gate's probability features require the global vocabulary under TP.
        logits_act = lm_head.project(h_act)
        needs, cont_probs = model.tah_decider.forward(
            logits_act,
            hidden_states=h_act,
            input_embeds=base_act,
            return_continue_probs=True,
        )

        # ── last-token mixture bookkeeping at this decision point ──
        if fix_last:
            for i, st in last_state.items():
                la = active_local[i]
                if st["done"] or la is None or la.numel() == 0 or int(la[-1]) != st["lo"]:
                    continue
                idx = req_slices[i][1] - 1  # last prompt token is the max active offset
                p = float(cont_probs[idx])
                if bool(needs[idx]):
                    # iter d-1 contributes with stop weight rem*(1-p).
                    term = math.log(max(st["rem"] * (1.0 - p), 1e-10)) + st["logp_cur"]
                    st["log_accum"] = (
                        term
                        if st["log_accum"] is None
                        else torch.logaddexp(st["log_accum"], term)
                    )
                    st["rem"] *= p
                    st["await"] = True
                else:
                    _finalize_last(st, i)  # sampled at iter d-1 with weight rem

        # ── filter each req's active set to the continuing positions ──
        new_active_local: List[torch.Tensor | None] = []
        for i in range(len(reqs)):
            la = active_local[i]
            lo_b, hi_b = req_slices[i]
            if la is None or hi_b == lo_b:
                new_active_local.append(None)
                continue
            sel = needs[lo_b:hi_b].nonzero(as_tuple=True)[0]
            new_active_local.append(la.index_select(0, sel) if sel.numel() else None)
        active_local = new_active_local
        for i in range(len(reqs)):
            la = active_local[i]
            if la is not None and prompt_counts[i] is not None:
                prompt_counts[i][la] += 1
        active_reqs = [
            (i, req, int(active_local[i].numel()))
            for i, req in enumerate(reqs)
            if active_local[i] is not None and active_local[i].numel() > 0
        ]
        total_active = sum(n for _, _, n in active_reqs)
        if total_active == 0:
            break

        # ── snapshot iter-d active set + stream start counts (pre-chunk) ──
        for req in reqs:
            while len(req.tah_iter_counts) < d:
                req.tah_iter_counts.append(0)
        chunk_start[d] = [req.tah_iter_counts[stream] for req in reqs]

        # ── allocate iter-d slots; record them in the stream page tables ──
        new_slots = cache_mgr._allocate(total_active)  # (total_active,) int32 GPU
        active_g_chunks: List[torch.Tensor] = []
        act_start: Dict[int, int] = {}  # req idx -> chunk start within active_g_d
        iter_pos_local[d] = [None] * len(reqs)
        cursor = 0
        for i, req, n in active_reqs:
            la = active_local[i]
            act_start[i] = cursor
            start_d = chunk_start[d][i]
            pos_d = (la + req.cached_len).to(torch.int32)  # absolute token positions
            iter_pts[stream][req.table_idx, start_d : start_d + n] = new_slots[cursor : cursor + n]
            req.tah_iter_counts[stream] += n
            iter_pos_local[d][i] = pos_d
            active_g_chunks.append(la + offsets[i])
            cursor += n
        active_g_d = torch.cat(active_g_chunks)
        iterd_out_loc = new_slots.to(torch.int32)
        iter_positions = batch.positions.index_select(0, active_g_d)

        # ── build per-query visible KV rows for the attention path ──
        # Per-request KV segment rows: [iter0 slots (pos 0..L0-1) | stream 1 | ... |
        # stream d].  Inside a stream, slots from earlier chunks get position 0
        # (always causal-visible) and this chunk's slots their real position, so the
        # kernel's ``kv_pos <= q_pos`` reproduces the legacy rank-based visibility
        # exactly (iter-d active ⊆ iter-j active, j < d).
        q_counts: List[int] = []
        seg_idx_rows: List[List[torch.Tensor]] = []
        seg_pos_rows: List[List[torch.Tensor]] = []
        for i, req, n in active_reqs:
            L0 = req.cached_len + req.extend_len
            idx_segs = [page_table[req.table_idx, :L0]]
            pos_segs = [iter0_pos[:L0]]
            for j in range(1, d + 1):
                n_prior = chunk_start[j][i]
                pos_j = iter_pos_local[j][i]
                assert pos_j is not None
                idx_segs.append(iter_pts[j - 1][req.table_idx, : req.tah_iter_counts[j - 1]])
                pos_segs.append(F.pad(pos_j, (n_prior, 0)) if n_prior else pos_j)
            q_counts.append(n)
            seg_idx_rows.append(idx_segs)
            seg_pos_rows.append(pos_segs)
        tiles = build_duo_prefill_tiles(q_counts, iter_positions, seg_pos_rows, seg_idx_rows)
        attn = _triton_attn(engine, iter_positions, tiles)

        # ── iter-d input embedding + fused forward ──
        iter_input = model.tah_updater.forward(
            base_embeds.index_select(0, active_g_d),
            cur_hidden.index_select(0, active_g_d),
        )
        iter_batch = _IterPrefillBatch(
            input_ids=iter_input,
            positions=iter_positions,
            out_loc=iterd_out_loc,
        )
        ctx = get_global_ctx()
        saved_batch = ctx._batch
        ctx._batch = iter_batch  # type: ignore[assignment]
        try:
            with _swap_attn_for_iter(engine, iterd_out_loc, attn):
                new_hidden = model._layers_forward(iter_input)
        finally:
            ctx._batch = saved_batch

        # iter-d hidden becomes the previous-iteration hidden for the next step.
        cur_hidden.index_copy_(0, active_g_d, new_hidden)

        # ── last-token mixture: fold in this pass's iter-d distribution ──
        if fix_last:
            for i, st in last_state.items():
                if not st["await"]:
                    continue
                la = active_local[i]  # post-filter set: last token continued
                assert la is not None and int(la[-1]) == st["lo"]
                h_last = new_hidden[act_start[i] + int(la.numel()) - 1]
                logit_last = lm_head.project(h_last.unsqueeze(0)).squeeze(0)
                st["logp_cur"] = torch.log_softmax(logit_last.float(), dim=-1)
                st["await"] = False
                if d == max_iter - 1:
                    _finalize_last(st, i)  # iteration cap: sample with weight rem

    # ── flush prompt-side iter counts (extend covers chunked prefill) ──
    log = engine.tah_prompt_iter_log
    if len(log) > engine._tah_iter_log_max:  # abort-leak backstop
        for k in list(log.keys())[: len(log) - engine._tah_iter_log_max]:
            del log[k]
    for i, req in enumerate(reqs):
        if req is engine.dummy_req or prompt_counts[i] is None:
            continue
        log.setdefault(req.uid, []).extend(prompt_counts[i].tolist())


class _IterPrefillBatch:
    """Minimal Batch shim — the patched attention only reads ``positions``
    and ``out_loc``."""

    def __init__(self, input_ids, positions, out_loc):
        self.input_ids = input_ids
        self.positions = positions
        self.out_loc = out_loc
        self.attn_metadata = None
        self.padded_reqs = []
        self.padded_size = positions.shape[0]
        self.size = positions.shape[0]


@contextmanager
def _swap_attn_for_iter(
    engine,
    iter_out_loc: torch.Tensor,
    attn: Callable[[torch.Tensor, int], torch.Tensor],
):
    """Replace ``attn_backend.forward`` for the iter-d pass: store this pass's K/V
    at ``iter_out_loc`` then run ``attn(q, layer_id)`` — one call per layer for
    all requests."""
    backend = engine.attn_backend
    kvcache = engine.kv_cache
    original_forward = backend.forward

    def fused_forward(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch
    ) -> torch.Tensor:
        kvcache.store_kv(k, v, iter_out_loc, layer_id)
        return attn(q, layer_id)

    backend.forward = fused_forward
    try:
        yield
    finally:
        backend.forward = original_forward


def _triton_attn(engine, q_pos: torch.Tensor, tiles: tuple):
    """Tiled paged DUO kernel over per-request KV segment rows (page_size == 1)."""
    kvcache = engine.kv_cache
    q_pos = q_pos.to(torch.int32)

    def attn(q: torch.Tensor, layer_id: int) -> torch.Tensor:
        kc = kvcache.k_cache(layer_id)
        vc = kvcache.v_cache(layer_id)
        kc = kc.view(-1, kc.shape[-2], kc.shape[-1])  # (num_slots, Hkv, D)
        vc = vc.view(-1, vc.shape[-2], vc.shape[-1])
        return duo_prefill_attention(q, kc, vc, q_pos, *tiles, 1.0 / math.sqrt(q.shape[-1]))

    return attn


def run_uniform_prefill(engine, batch, hidden, logits, base_embeds):
    """Causal uniform: only the last prompt token's recurrent states affect sampling.

    Extra iterations see root KV and the same token's earlier iterations. Other
    prompt tokens' extra KV cannot be visible, so temporary slots suffice.
    """
    model = engine.model
    depth = model.tah_max_iter
    if depth == 1:
        return
    reqs = batch.reqs
    indices = batch.attn_metadata.get_last_indices(batch.size).long()
    positions = batch.positions.index_select(0, indices)
    base = base_embeds.index_select(0, indices)
    current_hidden = hidden.index_select(0, indices)
    log_mix = torch.log_softmax(logits[: batch.size].float(), dim=-1)
    cache_mgr = engine._tah_cache_manager
    slots = cache_mgr._allocate(batch.size * (depth - 1)).view(depth - 1, batch.size)
    saved_batch = get_global_ctx()._batch
    try:
        for d in range(1, depth):
            seg_idx = []
            seg_pos = []
            for i, req in enumerate(reqs):
                length = req.cached_len + req.extend_len
                seg_idx.append([engine.page_table[req.table_idx, :length], slots[:d, i]])
                seg_pos.append(
                    [
                        torch.arange(length, dtype=torch.int32, device=hidden.device),
                        positions[i].expand(d).to(torch.int32),
                    ]
                )
            tiles = build_duo_prefill_tiles([1] * batch.size, positions, seg_pos, seg_idx)
            attn = _triton_attn(engine, positions, tiles)
            inputs = model.compute_iter_embeds(base, current_hidden)
            get_global_ctx()._batch = _IterPrefillBatch(inputs, positions, slots[d - 1])
            with _swap_attn_for_iter(engine, slots[d - 1], attn):
                current_hidden = model._layers_forward(inputs)
            # All ranks must participate in vocabulary gathering under TP.
            get_global_ctx()._batch = batch
            last_hidden = hidden.clone()
            last_hidden.index_copy_(0, indices, current_hidden)
            current_logits = model.lm_head.forward(last_hidden)[: batch.size]
            log_mix = torch.logaddexp(log_mix, torch.log_softmax(current_logits.float(), dim=-1))
        logits[: batch.size] = (log_mix - math.log(depth)).to(logits.dtype)
    finally:
        get_global_ctx()._batch = saved_batch
        cache_mgr._free(slots.flatten())
