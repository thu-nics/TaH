"""Persistent-cache model passes and bounded teacher-forced chunk tooling.

forward_cached is the row-mixed layer primitive. forward_chunk is eager tooling,
not a scheduler: it finishes one chunk's passes before admitting its next chunk.
"""
from dataclasses import dataclass

import torch

from .attention import AttentionRow, DecodeBucket, PrefillAttention
from .model import TeacherForcedOutput


@dataclass
class CachedPassOutput:
    logits: torch.Tensor  # pass-local increments, [rows, V], model dtype
    hidden_states: torch.Tensor  # [rows, layers+1, hidden_size], HF indexing


@torch.inference_mode()
def forward_cached(model, embeddings, position_ids, depths, slots, pool, attention, *, rotary=None):
    """Run one pass per row (mixed depths/requests allowed).

    Caller allocates unique slots and prepares attention OUTSIDE graph capture.
    Provide precomputed rotary=(cos,sin) for dynamic/longrope during capture;
    their frequency planning in the full model otherwise reads a host scalar.
    """
    if model.training:
        raise RuntimeError("cached model is inference-only; call eval()")
    rows = embeddings.shape[0]
    if embeddings.shape != (rows, model.config['hidden_size']) or embeddings.dtype != model.dtype or embeddings.device != model.device:
        raise ValueError("cached embeddings must be [rows, hidden_size] in model dtype/device")
    if position_ids.shape != (rows,) or depths.shape != (rows,) or slots.shape != (rows,):
        raise ValueError("cached positions/depths/slots must have one value per row")
    if position_ids.dtype != torch.long or depths.dtype != torch.long or slots.dtype != torch.long or any(t.device != model.device for t in (position_ids, depths, slots)):
        raise ValueError("cached positions/depths/slots must be model-device int64")
    if pool.device != model.device or pool.dtype != model.dtype or pool.max_iter != model.max_iter or pool.values.shape[0] != len(model.model.layers) or pool.values.shape[-2:] != (model.config['num_key_value_heads'], model.config['head_dim']):
        raise ValueError("KV pool does not match model layers/device/dtype/head config")
    if attention.pool is not pool:
        raise ValueError("attention belongs to a different KV pool")
    x = embeddings[None]
    lora_mask = (depths >= 2)[None]
    cos, sin = model.model.rotary_emb(x, position_ids) if rotary is None else rotary
    hidden = []
    for index, layer in enumerate(model.model.layers):
        hidden.append(x)
        q, k, v = layer.self_attn.project(layer.input_layernorm(x), cos, sin, lora_mask)
        pool.write(index, slots, k[0].transpose(0, 1), v[0].transpose(0, 1))
        out = attention.run(q[0].transpose(0, 1).contiguous(), index)
        x = x + layer.self_attn.o_proj(out.reshape(1, rows, -1), lora_mask)
        x = x + layer.mlp(layer.post_attention_layernorm(x), lora_mask)
    x = model.model.norm(x)
    hidden.append(x)
    return CachedPassOutput(model.lm_head(x, lora_mask)[0], torch.stack(hidden, dim=-2)[0])


@torch.inference_mode()
def forward_chunk(model, input_ids, pool, request, *, start_position=0,
                  iter_count_labels=None, hidden_positions=None, decode=False, workspace=None):
    """Bounded single-request teacher forcing with a persistent all-stream KV.

    start_position is an absolute RoPE/cache coordinate, increasing across chunks.
    All returned traces/hidden_positions are indexed locally in this chunk.
    decode=True exercises DecodeBucket on a one-token chunk; graph scheduling
    and model state buffers are deliberately owned by the later engine layer.
    """
    if model.training:
        raise RuntimeError("cached model is inference-only; call eval()")
    pool._request(request)
    ids = torch.as_tensor(input_ids, device=model.device)
    if ids.ndim != 1 or not ids.numel() or ids.dtype != torch.long or bool(((ids < 0) | (ids >= model.config['vocab_size'])).any()):
        raise ValueError("chunk requires nonempty 1D int64 vocabulary token IDs")
    length = ids.numel()
    if not isinstance(start_position, int) or start_position < 0 or start_position + length > request.context_limit:
        raise ValueError("chunk exceeds request model context_limit")
    if pool.max_iter != model.max_iter:
        raise ValueError("pool stream count does not match model max_iter")
    if decode and length != 1:
        raise ValueError("decode chunk must contain exactly one token")
    labels = None if iter_count_labels is None else torch.as_tensor(iter_count_labels, device=model.device)
    if labels is not None and (labels.dtype != torch.long or labels.shape != ids.shape):
        raise ValueError("iter_count_labels must be int64 with chunk input shape")
    selected = torch.empty(0, device=model.device, dtype=torch.long) if hidden_positions is None else torch.as_tensor(hidden_positions, device=model.device)
    if selected.ndim != 1 or selected.dtype != torch.long or bool(((selected < 0) | (selected >= length)).any()):
        raise ValueError("hidden_positions must index the current chunk")
    decider = model.eval_iter_decider if model.eval_iter_decider is not None else model.iter_decider
    layers = decider.selected_layers(len(model.model.layers)) if hasattr(decider, 'selected_layers') else []
    hidden_indices = torch.tensor(layers, device=model.device, dtype=torch.long)
    trace_hidden = torch.full((model.max_iter, len(selected), len(layers), model.config['hidden_size']),
                              float('nan'), device=model.device, dtype=model.dtype)
    visited = torch.zeros(model.max_iter, length, device=model.device, dtype=torch.bool)
    decision = torch.zeros_like(visited)
    raw = torch.full(visited.shape, float('nan'), device=model.device, dtype=model.dtype)
    probability = torch.full_like(raw, float('nan'))
    cumulative = torch.zeros(length, model.config['vocab_size'], device=model.device, dtype=model.dtype)
    counts = torch.zeros_like(ids)
    active = torch.arange(length, device=model.device)
    embeddings = model.model.embed_tokens(ids, False)
    for depth in range(1, model.max_iter + 1):
        # Only NEW chunk rows visit Python, never the existing request context.
        host_positions = [start_position + i for i in active.tolist()]
        slots = pool.allocate(request, depth - 1, host_positions)
        rows = [AttentionRow(request, p, depth) for p in host_positions]
        if decode:
            attention = DecodeBucket(pool, 1, request.size, model.config['num_attention_heads'])
            attention.prepare(rows)
        else:
            attention = PrefillAttention(pool, rows, model.config['num_attention_heads'], workspace=workspace)
        output = forward_cached(model, embeddings,
            torch.tensor(host_positions, device=model.device),
            torch.full((len(rows),), depth, device=model.device, dtype=torch.long),
            slots, pool, attention)
        cumulative[active] = cumulative[active] + output.logits
        keep, depth_raw, depth_prob = decider(cumulative[active], depth, output.hidden_states,
            None if labels is None else labels[active])
        if depth == model.max_iter:
            keep = torch.zeros_like(keep)
            depth_prob = torch.zeros_like(depth_prob)
        visited[depth - 1, active] = True
        decision[depth - 1, active] = keep
        raw[depth - 1, active] = depth_raw
        probability[depth - 1, active] = depth_prob
        counts[active] += 1
        if layers and selected.numel():
            a, b = (active[:, None] == selected[None]).nonzero(as_tuple=True)
            trace_hidden[depth - 1, b] = output.hidden_states[a].index_select(-2, hidden_indices)
        active = active[keep]
        if not active.numel():
            break
        values, indices = cumulative[active].topk(min(model.input_topk, cumulative.shape[-1]), dim=-1)
        embeddings = (values.softmax(-1)[..., None] * model.model.embed_tokens.weight[indices]).sum(-2).to(model.dtype)
    return TeacherForcedOutput(cumulative, counts, visited, decision, raw, probability,
                               selected, hidden_indices, trace_hidden)
