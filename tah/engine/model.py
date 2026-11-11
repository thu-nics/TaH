"""Full-sequence teacher forcing plus explicit persistent-cache entrypoints.

Pass-local K/V histories implement the normative (position, depth) mask.
The full forward keeps only local histories. Cached execution delegates to the
cache layer; scheduling, sampling and generation belong to the engine layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Embedding, Linear, RMSNorm, activate
from .rope import RotaryEmbedding, rotate


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config["head_dim"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        size = config["hidden_size"]
        bias = config["attention_bias"]
        self.q_proj = Linear(size, self.heads * self.head_dim, bias=bias)
        self.k_proj = Linear(size, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = Linear(size, self.kv_heads * self.head_dim, bias=bias)
        self.o_proj = Linear(self.heads * self.head_dim, size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])

    def project(self, x, cos, sin, lora_mask):
        shape = (1, x.shape[1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(x, lora_mask).view(shape))
        k = self.k_norm(self.k_proj(x, lora_mask).view(shape))
        v = self.v_proj(x, lora_mask).view(shape).transpose(1, 2)
        return (rotate(q, cos, sin).transpose(1, 2),
                rotate(k, cos, sin).transpose(1, 2), v)

    def attend(self, q, k, v, mask, lora_mask):
        groups = self.heads // self.kv_heads
        def repeat(x):
            return x[:, :, None].expand(1, self.kv_heads, groups, x.shape[2], self.head_dim).reshape(
                1, self.heads, x.shape[2], self.head_dim)
        out = F.scaled_dot_product_attention(q.contiguous(), repeat(k).contiguous(), repeat(v).contiguous(),
                                             attn_mask=mask, dropout_p=0., is_causal=False,
                                             scale=self.head_dim ** -0.5)
        out = out.transpose(1, 2).contiguous().reshape(1, q.shape[2], -1)
        return self.o_proj(out, lora_mask)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        size, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = Linear(size, intermediate, bias=False)
        self.up_proj = Linear(size, intermediate, bias=False)
        self.down_proj = Linear(intermediate, size, bias=False)
        self.act = config["hidden_act"]

    def forward(self, x, lora_mask):
        return self.down_proj(activate(self.gate_proj(x, lora_mask), self.act) * self.up_proj(x, lora_mask), lora_mask)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(self, x, cos, sin, history, mask, lora_mask):
        q, k, v = self.self_attn.project(self.input_layernorm(x), cos, sin, lora_mask)
        history.append((k, v))
        all_k = torch.cat([entry[0] for entry in history], dim=2)
        all_v = torch.cat([entry[1] for entry in history], dim=2)
        x = x + self.self_attn.attend(q, all_k, all_v, mask, lora_mask)
        return x + self.mlp(self.post_attention_layernorm(x), lora_mask)


class QwenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Embedding(config["vocab_size"], config["hidden_size"], padding_idx=config["pad_token_id"])
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config["num_hidden_layers"]))
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.rotary_emb = RotaryEmbedding(config)


@dataclass
class TeacherForcedOutput:
    logits: torch.Tensor  # [T, V], model dtype
    iter_count: torch.Tensor  # [T]
    visited: torch.Tensor  # [N, T]
    decision: torch.Tensor
    decision_logits: torch.Tensor
    continue_probability: torch.Tensor
    hidden_positions: torch.Tensor
    hidden_layer_indices: torch.Tensor
    decider_hidden: torch.Tensor


class EagerTaHModel(nn.Module):
    def forward_cached(self, embeddings, position_ids, depths, slots, pool, attention, *, rotary=None):
        """One persistent-cache pass per row; accepts mixed requests/depths."""
        from .cached_model import forward_cached
        return forward_cached(self, embeddings, position_ids, depths, slots, pool, attention, rotary=rotary)

    def forward_chunk(self, input_ids, pool, request, **kwargs):
        """Bounded eager teacher forcing; retains request KV between chunks."""
        from .cached_model import forward_chunk
        return forward_chunk(self, input_ids, pool, request, **kwargs)

    def __init__(self, config, tah_config):
        super().__init__()
        self.config, self.tah_config = config, tah_config
        self.model = QwenModel(config)
        self.lm_head = Linear(config["hidden_size"], config["vocab_size"], bias=False)
        if config["tie_word_embeddings"]:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.max_iter = tah_config.max_iter
        self.input_topk = int(tah_config.input_updater_kwargs.get("topk", 100))
        self.iter_decider = None
        self.eval_iter_decider = None

    @classmethod
    def from_pretrained(cls, model_path, config_override=None, *, device="cpu", dtype=torch.bfloat16):
        from .weights import load_model
        return load_model(model_path, config_override, device=device, dtype=dtype)

    @property
    def device(self):
        return self.model.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.model.embed_tokens.weight.dtype

    @torch.inference_mode()
    def forward(self, input_ids, *, iter_count_labels=None, position_ids=None, hidden_positions=None):
        if self.training:
            raise RuntimeError("EagerTaHModel is inference-only; call eval()")
        ids = torch.as_tensor(input_ids, device=self.device)
        if ids.ndim != 1 or ids.numel() == 0 or ids.dtype != torch.long:
            raise ValueError("teacher forcing requires a nonempty 1D int64 token sequence")
        if bool(((ids < 0) | (ids >= self.config["vocab_size"])).any()):
            raise ValueError("token id outside vocabulary")
        length = ids.numel()
        positions = torch.arange(length, device=self.device) if position_ids is None else torch.as_tensor(position_ids, device=self.device)
        labels = None if iter_count_labels is None else torch.as_tensor(iter_count_labels, device=self.device)
        if positions.shape != ids.shape or positions.dtype != torch.long or bool((positions < 0).any()):
            raise ValueError("position_ids must be nonnegative int64 with input shape")
        if labels is not None and (labels.shape != ids.shape or labels.dtype != torch.long):
            raise ValueError("iter_count_labels must be int64 with input shape")
        selected = torch.empty(0, device=self.device, dtype=torch.long) if hidden_positions is None else torch.as_tensor(hidden_positions, device=self.device)
        if selected.ndim != 1 or selected.dtype != torch.long or bool(((selected < 0) | (selected >= length)).any()):
            raise ValueError("hidden_positions must be 1D int64 sequence indices")
        decider = self.eval_iter_decider if self.eval_iter_decider is not None else self.iter_decider
        layers = decider.selected_layers(len(self.model.layers)) if hasattr(decider, "selected_layers") else []
        hidden_indices = torch.tensor(layers, device=self.device, dtype=torch.long)
        trace_hidden = torch.full((self.max_iter, len(selected), len(layers), self.config["hidden_size"]),
                                  float("nan"), dtype=self.dtype, device=self.device)
        visited = torch.zeros(self.max_iter, length, device=self.device, dtype=torch.bool)
        decision = torch.zeros_like(visited)
        raw = torch.full(visited.shape, float("nan"), dtype=self.dtype, device=self.device)
        probability = torch.full_like(raw, float("nan"))
        cumulative = torch.zeros(length, self.config["vocab_size"], dtype=self.dtype, device=self.device)
        counts = torch.zeros_like(ids)
        embeddings = self.model.embed_tokens(ids, False)  # SPEC: pass 1 LoRA off
        active = torch.arange(length, device=self.device)
        histories = [[] for _ in self.model.layers]
        history_positions = []
        for depth in range(1, self.max_iter + 1):
            pos = positions[active]
            history_positions.append(pos)
            allowed = torch.cat(history_positions)[None, :] <= pos[:, None]
            mask = torch.full(allowed.shape, torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device)
            mask.masked_fill_(allowed, 0.)
            mask = mask[None, None]
            x = embeddings.unsqueeze(0)
            cos, sin = self.model.rotary_emb(x, pos)
            hidden = []
            for layer, history in zip(self.model.layers, histories):
                hidden.append(x)
                x = layer(x, cos, sin, history, mask, depth >= 2)
            x = self.model.norm(x)
            hidden.append(x)
            stacked = torch.stack(hidden, dim=-2)[0]
            cumulative[active] = cumulative[active] + self.lm_head(x, depth >= 2)[0]
            keep, depth_raw, depth_prob = decider(cumulative[active], depth, stacked,
                                                  None if labels is None else labels[active])
            if depth == self.max_iter:
                keep = torch.zeros_like(keep)
                depth_prob = torch.zeros_like(depth_prob)
            visited[depth - 1, active] = True
            decision[depth - 1, active] = keep
            raw[depth - 1, active] = depth_raw
            probability[depth - 1, active] = depth_prob
            counts[active] += 1
            if layers and selected.numel():
                matches = active[:, None] == selected[None, :]
                a, b = matches.nonzero(as_tuple=True)
                trace_hidden[depth - 1, b] = stacked[a].index_select(-2, hidden_indices)
            active = active[keep]
            if not active.numel():
                break
            values, indices = cumulative[active].topk(min(self.input_topk, cumulative.shape[-1]), dim=-1)
            embeddings = (values.softmax(-1)[..., None] * self.model.embed_tokens.weight[indices]).sum(-2).to(self.dtype)
        return TeacherForcedOutput(cumulative, counts, visited, decision, raw, probability,
                                   selected, hidden_indices, trace_hidden)
