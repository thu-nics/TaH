"""Inference-only TaH deciders, owned entirely by the engine."""
from __future__ import annotations

import torch
from torch import nn


class ClassifierBlock(nn.Module):
    def __init__(self, input_dim, output_dim, expansion_factor, dropout_rate, dtype):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim, dtype=dtype)
        self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim * expansion_factor, dtype=dtype),
                                 nn.GELU(), nn.Dropout(dropout_rate),
                                 nn.Linear(input_dim * expansion_factor, output_dim, dtype=dtype),
                                 nn.Dropout(dropout_rate))
        self.dim_change = nn.Linear(input_dim, output_dim, dtype=dtype) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.dim_change(x) + self.mlp(self.layer_norm(x))


class ClassifierBackbone(nn.Module):
    def __init__(self, input_dim, dims, expansion, dropout, dtype):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, dims[0], dtype=dtype)
        block_dims = list(dims) + [dims[-1]]
        self.blocks = nn.ModuleList(ClassifierBlock(a, b, expansion, dropout, dtype)
                                    for a, b in zip(block_dims, block_dims[1:]))
        self.output_layer = nn.Sequential(nn.LayerNorm(dims[-1], dtype=dtype), nn.Linear(dims[-1], 1, dtype=dtype))

    def forward(self, x):
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class IterLabelDecider(nn.Module):
    def __init__(self, max_iter=3):
        super().__init__()
        self.max_iter = max_iter
        self.register_buffer("threshold", torch.tensor(0.5))

    def forward(self, logits, depth, hidden=None, labels=None, *, hidden_indices=None):
        decision = torch.zeros(logits.shape[:-1], dtype=torch.bool, device=logits.device)
        raw = torch.where(torch.as_tensor(depth, device=logits.device) >= self.max_iter, -10., 0.).to(logits.dtype).expand_as(logits[..., 0])
        if labels is not None:
            decision = (labels > depth) & (labels != -100) & (depth < self.max_iter)
        return decision, raw, decision.to(logits.dtype)


class MLPIterDecider(nn.Module):
    def __init__(self, topk=100, hidden_states_size=1024, hidden_states_layer_nums=(16, 20, 24, 28),
                 hidden_dims=(256, 512, 256), expansion_factor=4, dropout_rate=0.3,
                 normalize_input=False, threshold=0.5, max_iter=3, dtype=torch.bfloat16):
        super().__init__()
        self.max_iter, self.topk = max_iter, topk
        self.hidden_states_size = hidden_states_size
        self.hidden_states_layer_nums = list(hidden_states_layer_nums)
        self.normalize_input = normalize_input
        if topk < 1 or not hidden_dims or any(d < 1 for d in hidden_dims):
            raise ValueError("invalid MLP decider dimensions")
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=dtype))
        count = max(1, len(self.hidden_states_layer_nums))
        if normalize_input:
            self.layer_norm_hidden_states = nn.LayerNorm(hidden_states_size * count, dtype=dtype)
        self.logits_projection = nn.Linear(topk, hidden_states_size, dtype=dtype)
        self.combined_projection = nn.Linear(hidden_states_size * (count + 1), hidden_dims[0], dtype=dtype)
        self.backbone = ClassifierBackbone(hidden_dims[0], hidden_dims, expansion_factor, dropout_rate, dtype)

    def selected_layers(self, layer_count):
        return self.hidden_states_layer_nums or [layer_count]

    def forward(self, logits, depth, hidden=None, labels=None, *, hidden_indices=None):
        if isinstance(depth, int) and depth >= self.max_iter:
            return (torch.zeros_like(logits[..., 0], dtype=torch.bool),
                    torch.full_like(logits[..., 0], -10.), torch.zeros_like(logits[..., 0]))
        indices = self.selected_layers(hidden.shape[-2] - 1)
        features = hidden.index_select(-2, torch.tensor(indices, device=hidden.device) if hidden_indices is None else hidden_indices).flatten(-2)
        top = logits.topk(self.topk, dim=-1).values
        if self.normalize_input:
            features = self.layer_norm_hidden_states(features)
            top = top.softmax(-1)
        x = self.combined_projection(torch.cat((features, self.logits_projection(top)), dim=-1))
        raw = self.backbone(x).squeeze(-1)
        probability = raw.sigmoid()  # deliberately model dtype, including bf16 rounding
        if isinstance(depth, int):
            return probability > self.threshold, raw, probability
        capped = depth >= self.max_iter
        raw = torch.where(capped, -10., raw)
        probability = torch.where(capped, 0., probability)
        return (probability > self.threshold) & ~capped, raw, probability


def build_decider(name, kwargs, max_iter):
    classes = {"MLPIterDecider": MLPIterDecider, "IterLabelDecider": IterLabelDecider}
    if name not in classes:
        raise ValueError(f"unsupported iter decider {name!r}")
    args = dict(kwargs)
    args.setdefault("max_iter", max_iter)
    return classes[name](**args)
