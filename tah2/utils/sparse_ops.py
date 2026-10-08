from __future__ import annotations

from typing import Optional

import torch

from tah2.model.causal_cache import TaHCache


def gather_active_2d(tensor: Optional[torch.Tensor], gather_idx: torch.LongTensor, pad_mask: torch.BoolTensor, pad_value):
    if tensor is None:
        return None
    out = torch.gather(tensor, 1, gather_idx)
    return out.masked_fill(pad_mask, pad_value)


def gather_active_3d(tensor: Optional[torch.Tensor], gather_idx: torch.LongTensor, pad_mask: torch.BoolTensor, pad_value):
    if tensor is None:
        return None
    hidden = tensor.shape[-1]
    out = torch.gather(tensor, 1, gather_idx.unsqueeze(-1).expand(-1, -1, hidden))
    return out.masked_fill(pad_mask.unsqueeze(-1), pad_value)


def to_active(
    current_iter_mask: torch.BoolTensor,
    input_embeds: torch.Tensor,
    position_ids: torch.LongTensor,
    valid_mask: torch.LongTensor,
    iter_count: Optional[torch.LongTensor],
    labels_shifted: Optional[torch.LongTensor] = None,
    iter_count_labels: Optional[torch.LongTensor] = None,
    labels_all_shifted: Optional[torch.LongTensor] = None,
    remaining_mass: Optional[torch.Tensor] = None,
    first_iter_embeds: Optional[torch.Tensor] = None,
):
    bsz, seq_len, _ = input_embeds.shape
    device = input_embeds.device
    max_len = int(current_iter_mask.sum(dim=1).max())
    if max_len == 0:
        empty_embeds = input_embeds.new_empty(bsz, 0, input_embeds.shape[-1])
        empty_long = position_ids.new_empty(bsz, 0)
        return empty_embeds, empty_long, empty_long, None, None, None, None, None, None

    sentinel = seq_len
    base_idx = torch.arange(seq_len, device=device).expand(bsz, seq_len).masked_fill(~current_iter_mask, sentinel)
    gather_idx = torch.sort(base_idx, dim=1, stable=True).values[:, :max_len]
    pad_mask = gather_idx.eq(sentinel)
    gather_idx = gather_idx.clamp(max=seq_len - 1)

    return (
        gather_active_3d(input_embeds, gather_idx, pad_mask, 0),
        gather_active_2d(position_ids, gather_idx, pad_mask, 0),
        gather_active_2d(valid_mask, gather_idx, pad_mask, 0),
        gather_active_2d(iter_count, gather_idx, pad_mask, 0),
        gather_active_2d(labels_shifted, gather_idx, pad_mask, -100),
        gather_active_2d(iter_count_labels, gather_idx, pad_mask, 0),
        gather_active_2d(labels_all_shifted, gather_idx, pad_mask, -100),
        gather_active_2d(remaining_mass, gather_idx, pad_mask, 0),
        gather_active_3d(first_iter_embeds, gather_idx, pad_mask, 0),
    )


def assign_active(current_iter_mask: torch.BoolTensor, src: torch.Tensor, dest: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    # Vectorized, sync-free.  ``src`` is (B, max_n, ...) left-packed; the first
    # ``active_counts[b]`` slots of row b are valid, matching the row's
    # ``current_iter_mask`` positions 1-to-1 in row-major order.
    #
    # The final ``dest[current_iter_mask] = src[src_valid_mask]`` still triggers
    # a small nonzero-based sync; a prior gather+torch.where replacement OOM'd
    # under autograd because ``where`` retains both branches through backward
    # (unlike masked_scatter which is in-place).
    if not inplace:
        dest = dest.clone()
    if current_iter_mask.shape[0] == 0 or src.shape[1] == 0:
        return dest
    B, max_n = src.shape[0], src.shape[1]
    active_counts = current_iter_mask.sum(dim=1)
    col_idx = torch.arange(max_n, device=src.device).expand(B, max_n)
    src_valid_mask = col_idx < active_counts.unsqueeze(1)
    dest[current_iter_mask] = src[src_valid_mask]
    return dest


def _gather_active_to_dense(current_iter_mask: torch.BoolTensor, src: torch.Tensor) -> torch.Tensor:
    # Gather from left-packed ``src`` into a dense (B, S, ...) layout where each
    # row's active position pulls the matching slot from ``src``.  The index is
    # ``cumsum(mask) - 1`` clamped to 0 (any value is fine for masked-out slots,
    # since the caller scatters only where ``final_mask``).
    src_idx = (current_iter_mask.to(torch.int64).cumsum(dim=1) - 1).clamp_(min=0)
    if src.dim() == 2:
        return torch.gather(src, 1, src_idx)
    trailing = src.shape[2:]
    return torch.gather(
        src, 1, src_idx.view(src_idx.shape + (1,) * len(trailing)).expand(-1, -1, *trailing)
    )


def assign_active_with_mask(
    current_iter_mask: torch.BoolTensor,
    assignment_mask: torch.BoolTensor,
    src: torch.Tensor,
    dest: torch.Tensor,
) -> torch.Tensor:
    final_mask = current_iter_mask & assignment_mask
    if final_mask.numel() == 0 or src.shape[1] == 0:
        return dest
    gathered = _gather_active_to_dense(current_iter_mask, src)
    dest[final_mask] = gathered[final_mask]
    return dest


def add_active_with_mask(
    current_iter_mask: torch.BoolTensor,
    assignment_mask: torch.BoolTensor,
    src: torch.Tensor,
    dest: torch.Tensor,
) -> torch.Tensor:
    final_mask = current_iter_mask & assignment_mask
    if final_mask.numel() == 0 or src.shape[1] == 0:
        return dest
    gathered = _gather_active_to_dense(current_iter_mask, src)
    dest[final_mask] = dest[final_mask] + gathered[final_mask].to(dest.dtype)
    return dest


def create_tah_sdpa_attention_mask(
    iter_attention_mode: str,
    active_position_ids: torch.Tensor,
    active_valid_mask: torch.LongTensor,
    cache: Optional[TaHCache],
    iter_depth: int,
    dtype: torch.dtype = torch.bfloat16,
    iter_attention_impl: str = "sdpa",
) -> Optional[torch.Tensor]:
    batch_size, query_length = active_position_ids.shape
    device = active_position_ids.device

    if cache is not None and (0 in cache._tah_position_id_cache):
        iter_index_cache = cache.get_cache_iter_index_upto_iter(layer_idx=0, upto_iter_idx=iter_depth)
        pos_cache = cache.get_position_id_upto_iter(layer_idx=0, upto_iter_idx=iter_depth, init_batch_size=batch_size)
        valid_cache = cache.get_valid_mask_upto_iter(layer_idx=0, upto_iter_idx=iter_depth, init_batch_size=batch_size)
        kv_cache_len = iter_index_cache.shape[-1]
    else:
        iter_index_cache = torch.empty(size=(0,), device=device, dtype=torch.long)
        pos_cache = torch.empty(size=(batch_size, 0), device=device, dtype=torch.long)
        valid_cache = torch.empty(size=(batch_size, 0), device=device, dtype=torch.long)
        kv_cache_len = 0

    kv_len = kv_cache_len + query_length
    if kv_len == 0:
        return None

    # Fast path: at iter_depth=0 with no KV cache, "visible" reduces to standard
    # causal + padding.  Return the 2D (B, L) padding mask so the base model can
    # pick its fast SDPA/flash-attn kernel; a 4D additive mask would force the
    # quadratic math fallback.
    if iter_depth == 0 and kv_cache_len == 0:
        return active_valid_mask

    # Build the concatenated 1-D metadata once; both the 4-D additive-mask
    # path and the Triton sentinel path consume these directly.
    kv_positions_full = torch.cat((pos_cache, active_position_ids), dim=-1)  # (B, kv_len)
    kv_valid_full = torch.cat((valid_cache, active_valid_mask), dim=-1)       # (B, kv_len)
    kv_iter_full = torch.cat(
        (iter_index_cache, torch.full((query_length,), iter_depth, dtype=torch.long, device=device)),
        dim=-1,
    )  # (kv_len,)

    if iter_attention_impl == "triton":
        # Triton path: sentinel tensor carrying visibility metadata as
        # attributes; hf_adapter picks it up and dispatches to the kernel.
        from tah2.kernels.hf_adapter import _make_tah_triton_mask, register_tah_attention_impl
        register_tah_attention_impl()
        return _make_tah_triton_mask(
            kv_iter=kv_iter_full,
            kv_pos=kv_positions_full,
            kv_valid=kv_valid_full,
            q_pos=active_position_ids,
            cur_iter=iter_depth,
            mode=iter_attention_mode,
            dtype=dtype,
            device=device,
        )

    # SDPA 4-D additive-mask path.  Fold the visibility predicate into one
    # boolean expression and build the additive mask with a single fused
    # ``torch.where`` on scalars — avoids a bool-indexed write
    # (``mask[visible] = 0.0``) which goes through nonzero + index_put and
    # forces a small CPU sync.
    kv_positions = kv_positions_full[:, None, :]
    kv_valid = kv_valid_full[:, None, :]
    kv_iter = kv_iter_full[None, None, :]
    query_positions = active_position_ids[:, :, None]

    causal_valid = (kv_positions <= query_positions) & (kv_valid == 1)
    if iter_attention_mode == "causal":
        # Visible iff: causal + padded-valid + (root layer OR same-pos in a
        # not-yet-future iteration).
        visible = causal_valid & (
            (kv_iter == 0) | ((kv_positions == query_positions) & (kv_iter <= iter_depth))
        )
    elif iter_attention_mode == "duo":
        visible = causal_valid & (kv_iter <= iter_depth)
    elif iter_attention_mode == "same_iter":
        visible = causal_valid & (kv_iter == iter_depth)
    else:
        raise ValueError(f"Invalid iter attention mode: {iter_attention_mode}")

    zero = torch.zeros((), dtype=dtype, device=device)
    neg_inf = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
    attention_mask = torch.where(visible, zero, neg_inf)
    return attention_mask[:, None, :, :]


def create_posterior_side_attention_mask(
    iter_attention_mode: str,
    active_position_ids: torch.Tensor,
    active_valid_mask: torch.LongTensor,
    active_continue_mask: torch.BoolTensor,
    cache: Optional[TaHCache],
    iter_depth: int,
    dtype: torch.dtype = torch.bfloat16,
    iter_attention_impl: str = "sdpa",
) -> Optional[torch.Tensor]:
    """Attention mask for the posterior side-pass (hypothetical next iteration).

    ``"duo"`` → duo_posterior: prior-iter KV causal; current-iter visible iff continued or own.
    Returns a triton sentinel or 4-D SDPA additive mask.
    """
    assert iter_attention_mode == "duo", (
        f"posterior side pass supports duo, got {iter_attention_mode}"
    )
    batch_size = active_position_ids.shape[0]
    device = active_position_ids.device
    active_len = active_position_ids.shape[-1]

    prior_iter_depth = max(int(iter_depth) - 1, 0)
    if cache is not None and (0 in cache._tah_position_id_cache):
        pos_cache = cache.get_position_id_upto_iter(
            layer_idx=0,
            upto_iter_idx=prior_iter_depth,
            init_batch_size=batch_size,
        )
        valid_cache = cache.get_valid_mask_upto_iter(
            layer_idx=0,
            upto_iter_idx=prior_iter_depth,
            init_batch_size=batch_size,
        )
        iter_index_cache = cache.get_cache_iter_index_upto_iter(
            layer_idx=0, upto_iter_idx=prior_iter_depth
        )
    else:
        pos_cache = torch.empty(size=(batch_size, 0), device=device, dtype=torch.long)
        valid_cache = torch.empty(size=(batch_size, 0), device=device, dtype=torch.long)
        iter_index_cache = torch.empty(size=(0,), device=device, dtype=torch.long)

    kv_positions_full = torch.cat((pos_cache, active_position_ids), dim=-1)
    kv_valid_full = torch.cat((valid_cache, active_valid_mask), dim=-1)
    kv_len = kv_positions_full.shape[-1]
    if kv_len == 0:
        return None
    kv_iter_full = torch.cat(
        (
            iter_index_cache,
            torch.full((active_len,), int(iter_depth), dtype=torch.long, device=device),
        ),
        dim=-1,
    )  # (kv_len,)

    active_continue = active_continue_mask.to(device=device, dtype=torch.bool) & (
        active_valid_mask == 1
    )

    kv_is_continue = torch.cat(
        [torch.zeros_like(pos_cache, dtype=torch.int32), active_continue.to(torch.int32)], dim=-1
    )
    if iter_attention_impl == "triton":
        from tah2.kernels.hf_adapter import _make_tah_triton_mask, register_tah_attention_impl
        register_tah_attention_impl()
        return _make_tah_triton_mask(
            kv_iter=kv_iter_full,
            kv_pos=kv_positions_full,
            kv_valid=kv_valid_full,
            q_pos=active_position_ids,
            cur_iter=int(iter_depth),
            mode="duo_posterior",
            dtype=dtype,
            device=device,
            kv_is_continue=kv_is_continue.to(torch.int32),
        )

    # SDPA 4-D additive-mask path.
    kv_positions = kv_positions_full[:, None, :]
    kv_valid = kv_valid_full[:, None, :]
    query_positions = active_position_ids[:, :, None]
    causal_valid = (kv_positions <= query_positions) & (kv_valid == 1)

    kv_iter = kv_iter_full[None, None, :]
    current_kv = kv_iter == int(iter_depth)
    continue_kv = kv_is_continue[:, None, :].bool()
    visible_current = continue_kv | (kv_positions == query_positions)
    visible = causal_valid & ((~current_kv) | visible_current)

    zero = torch.zeros((), dtype=dtype, device=device)
    neg_inf = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
    attention_mask = torch.where(visible, zero, neg_inf)
    return attention_mask[:, None, :, :]
