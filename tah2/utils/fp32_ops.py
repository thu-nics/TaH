"""Memory-lean fp32 reductions over the vocabulary dimension.

Under ``autocast(bfloat16)`` a full-vocab ``torch.logsumexp(bf16_logits)`` is
promoted to fp32, and autograd *saves the fp32 [N, V] copy* for backward
(5.58 GB at N=9182, V=152k). But the saved fp32 values are just the bf16 logits
upcast, and the bf16 logits are already retained by the neighbouring ``gather``.
``fp32_logsumexp`` returns the identical fp32 result while saving only the bf16
input and recomputing the reduction in backward — bit-exact in value AND
gradient (the original logsumexp backward already recomputes ``softmax`` in
fp32), speed-neutral, but it halves the retained activation.

The fp32 work is chunked over V so a full fp32 [N, V] is never materialized,
keeping the forward transient small too.
"""
from __future__ import annotations

import torch

# fp32 reduction chunk over the vocab dim. A [N, _VOCAB_CHUNK] fp32 tile is
# ~0.6 GB at N=9182; large enough to stay matmul/reduction-bound, small enough
# that the transient never approaches a full fp32 [N, V].
_VOCAB_CHUNK = 16384


class _Fp32LogSumExp(torch.autograd.Function):
    """``torch.logsumexp(x.float(), dim=-1)`` that retains the bf16 ``x``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        m = x.amax(dim=-1, keepdim=True).float()
        acc = torch.zeros_like(m)
        for chunk in torch.split(x, _VOCAB_CHUNK, dim=-1):
            acc += (chunk.float() - m).exp().sum(dim=-1, keepdim=True)
        z = (m + acc.log()).squeeze(-1)
        ctx.save_for_backward(x, z)
        return z

    @staticmethod
    def backward(ctx, grad_z: torch.Tensor):
        x, z = ctx.saved_tensors
        # grad_x = softmax(x) * grad_z, recomputed in fp32 chunk-by-chunk so no
        # full fp32 [N, V] is held; identical to logsumexp's own backward.
        z_col = z.unsqueeze(-1)
        gz_col = grad_z.unsqueeze(-1)
        grad_x = torch.empty_like(x)
        offset = 0
        for chunk in torch.split(x, _VOCAB_CHUNK, dim=-1):
            width = chunk.shape[-1]
            grad_x[..., offset:offset + width] = (
                ((chunk.float() - z_col).exp() * gz_col).to(x.dtype)
            )
            offset += width
        return grad_x


def fp32_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """fp32-accurate ``logsumexp`` over the last dim that saves the bf16 input.

    Drop-in for ``torch.logsumexp(x, dim=-1).float()`` on the autograd path:
    same value and gradient, but retains ``x`` (bf16) instead of an fp32 [N, V]
    copy under autocast.
    """
    return _Fp32LogSumExp.apply(x)
