"""HF attention-interface adapter that dispatches TaH Triton when enabled.

Registers a custom ``tah_sdpa`` attention implementation with HF.  Two
dispatch cases:

* TaH iter>=1 prefill: ``attention_mask`` carries a TaH sentinel
  (``_tah_triton`` attribute); call the fused sparse-predicate Triton
  kernel.
* Autoregressive decode: no sentinel and ``query.shape[-2] == 1`` —
  call the single-token decode kernel.
* Otherwise (iter-0 prefill, 2-D / None mask, Lq > 1): fall through to
  stock SDPA.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from tah2.kernels.tah_attention import tah_attention
from tah2.kernels.tah_attention_decode import tah_attention_decode

_REGISTERED = False


def _make_tah_triton_mask(
    *,
    kv_iter: torch.Tensor,
    kv_pos: torch.Tensor,
    kv_valid: torch.Tensor,
    q_pos: torch.Tensor,
    cur_iter: int,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    kv_is_continue: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a 1x1x1x1 sentinel tensor carrying TaH metadata as attributes.

    HF's masking machinery treats any 4-D tensor as already-prepared and
    passes it through to the attention interface, which reads the
    attributes and dispatches the Triton kernel.

    ``kv_is_continue`` is only set for ``mode="duo_posterior"`` (the posterior
    side pass); it is None for the standard causal / duo paths.
    """
    sentinel = torch.zeros((1, 1, 1, 1), dtype=dtype, device=device)
    sentinel._tah_triton = True
    sentinel._tah_kv_iter = kv_iter
    sentinel._tah_kv_pos = kv_pos
    sentinel._tah_kv_valid = kv_valid
    sentinel._tah_q_pos = q_pos
    sentinel._tah_cur_iter = int(cur_iter)
    sentinel._tah_mode = mode
    sentinel._tah_kv_is_continue = kv_is_continue
    return sentinel


def _tah_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs: Any,
):
    """Drop-in SDPA replacement that dispatches to Triton kernels.

    Dispatch order:
      1. TaH sentinel on ``attention_mask`` → sparse-predicate Triton kernel.
      2. Lq == 1 without sentinel → single-token decode Triton kernel.
      3. Else → stock SDPA fallback.
    """
    sm_scale = scaling if scaling is not None else (1.0 / (query.shape[-1] ** 0.5))

    has_sentinel = (
        isinstance(attention_mask, torch.Tensor)
        and getattr(attention_mask, "_tah_triton", False)
    )

    if has_sentinel:
        # Sparse TaH iter>=1 prefill kernel.  q/k/v are [B, H, L, D].
        out = tah_attention(
            q=query,
            k=key,
            v=value,
            kv_iter=attention_mask._tah_kv_iter,
            kv_pos=attention_mask._tah_kv_pos,
            kv_valid=attention_mask._tah_kv_valid,
            q_pos=attention_mask._tah_q_pos,
            cur_iter=attention_mask._tah_cur_iter,
            sm_scale=sm_scale,
            mode=attention_mask._tah_mode,
            kv_is_continue=getattr(attention_mask, "_tah_kv_is_continue", None),
        )
        return out.transpose(1, 2).contiguous(), None

    # Autoregressive decode path: plain causal, single-token Q.
    # ``attention_mask`` at Lq=1 is either None (no padding) or a 2-D/4-D
    # padding mask.  Our decode kernel assumes all entries are valid
    # (caller's responsibility); routing through it is only safe when
    # ``attention_mask`` is None.
    if query.shape[-2] == 1 and attention_mask is None:
        out = tah_attention_decode(q=query, k=key, v=value, sm_scale=sm_scale)
        return out.transpose(1, 2).contiguous(), None

    # Fall-through: stock SDPA.
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(
        module, query, key, value, attention_mask,
        dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
    )


def register_tah_attention_impl(name: str = "tah_sdpa") -> None:
    """Register the ``tah_sdpa`` interface with HF.  Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS._global_mapping[name] = _tah_attention_forward
    # Mirror sdpa's mask builder so HF still constructs a standard causal
    # mask when the caller passes a 2-D padding mask (iter-0 path).
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        mapping = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
        if name not in mapping and "sdpa" in mapping:
            mapping[name] = mapping["sdpa"]
    except Exception:
        pass
    _REGISTERED = True
