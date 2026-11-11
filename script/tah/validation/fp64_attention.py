"""Correct float64 SDPA on PyTorch < 2.7 for the oracle (PyTorch issue #143644).

PyTorch 2.6 ``cunn_SoftMaxForward`` computes wrong (and run-to-run variable)
float64 softmax on CUDA when the softmax width is odd, 513..1023 and not 31 mod 32,
and more than one row is reduced: rows whose start is not 16-byte aligned drive an
``int`` size negative, which is then reduced modulo an unsigned block extent. The
SDPA math kernel (the only CUDA SDPA kernel accepting float64) reaches it through
the attention scores. Fixed upstream in v2.7.0 (PR #144009) and v2.8.0 (PR #154778).

Inside ``fp64_attention_guard()`` every float64 ``F.scaled_dot_product_attention``
call with an odd key/value length gets one extra zero key/value slot that every
query is masked from with ``-inf`` (``False`` for boolean masks). The attention
distribution over the original slots is unchanged; the softmax width becomes
even, and the freshly allocated score rows are therefore 16-byte aligned, which
selects the correct warp/shared-memory kernels. Other dtypes, even lengths and
CPU arithmetic are passed through untouched. Validation tooling only: nothing in
``tah/`` changes, and the guard restores the function on exit.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

POLICY = "odd-kv-masked-pad-v1 (PyTorch #143644)"


class _Guard:
    def __init__(self, native):
        self.native = native
        self.calls = 0       # float64 SDPA calls seen
        self.padded = 0      # of those, calls given an extra masked slot

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                 **kwargs):
        if query.dtype != torch.float64:
            return self.native(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                               is_causal=is_causal, scale=scale, **kwargs)
        self.calls += 1
        length = key.shape[-2]
        if length % 2:
            if is_causal:
                if attn_mask is not None:
                    raise ValueError("scaled_dot_product_attention: attn_mask and is_causal are exclusive")
                # SDPA's is_causal is the top-left aligned lower triangle.
                attn_mask = torch.ones(query.shape[-2], length, dtype=torch.bool, device=query.device).tril()
                is_causal = False
            elif attn_mask is None:
                attn_mask = torch.ones(query.shape[-2], length, dtype=torch.bool, device=query.device)
            if attn_mask.shape[-1] != length:
                attn_mask = attn_mask.expand(*attn_mask.shape[:-1], length)
            fill = False if attn_mask.dtype == torch.bool else float("-inf")
            attn_mask = F.pad(attn_mask, (0, 1), value=fill)
            key = F.pad(key, (0, 0, 0, 1))
            value = F.pad(value, (0, 0, 0, 1))
            self.padded += 1
        return self.native(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                           is_causal=is_causal, scale=scale, **kwargs)


@contextmanager
def fp64_attention_guard():
    """Route float64 SDPA through the padding above; yields the guard (call counters).

    Re-entrant: a nested guard reuses the active one. Callers looking up
    ``torch.nn.functional.scaled_dot_product_attention`` at call time (as
    transformers' SDPA attention does) are covered; references bound earlier are not.
    """
    current = F.scaled_dot_product_attention
    if isinstance(current, _Guard):
        yield current
        return
    guard = _Guard(current)
    F.scaled_dot_product_attention = guard
    try:
        yield guard
    finally:
        F.scaled_dot_product_attention = current
