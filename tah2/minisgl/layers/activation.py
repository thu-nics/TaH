from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


def relu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    n = x.shape[-1] // 2
    result = F.relu(x[..., :n]) * x[..., n:]
    if out is None:
        return result
    out.copy_(result)
    return out


__all__ = ["silu_and_mul", "gelu_and_mul", "relu_and_mul"]
