"""Unmerged base/adapter weights with a tensor row mask for later decode WPs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter("lora_A", None)
        self.register_parameter("lora_B", None)
        self.register_parameter("lora_bias", None)
        self.scaling = 1.0

    def forward(self, x, lora_mask=False):
        result = F.linear(x, self.weight, self.bias)
        if self.lora_A is not None:
            # Keep PEFT's operation/rounding order, including multiplication
            # of B's result by the alpha/r (or alpha/sqrt(r)) scalar.
            delta = F.linear(F.linear(x, self.lora_A), self.lora_B, self.lora_bias) * self.scaling
            if isinstance(lora_mask, torch.Tensor):
                delta = delta * lora_mask.unsqueeze(-1).to(delta.dtype)
            elif not lora_mask:
                return result
            result = result + delta
        return result


class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter("lora_A", None)
        self.register_parameter("lora_B", None)
        self.scaling = 1.0

    def forward(self, ids, lora_mask=False):
        result = F.embedding(ids, self.weight, self.padding_idx)
        if self.lora_A is not None and (isinstance(lora_mask, torch.Tensor) or lora_mask):
            delta = (F.embedding(ids, self.lora_A.T, self.padding_idx) @ self.lora_B.T) * self.scaling
            if isinstance(lora_mask, torch.Tensor):
                delta = delta * lora_mask.unsqueeze(-1).to(delta.dtype)
            result = result + delta
        return result


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(x.dtype)


def activate(x, name):
    if name == "silu":
        return F.silu(x)
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x)
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu_new":
        import math
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    raise ValueError(name)
