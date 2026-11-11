"""Bounded custom-mask prefill for the pinned FlashInfer 0.2.5 FA2 backend.

FlashInfer may call LogitsMask for query rows beyond the logical Q length in
the last tensor-core tile. Its default custom mask reads those rows without a
bounds check. Keep the default attention math and guard just the mask lookup.
No installed FlashInfer files are modified.
"""
from functools import cache
import hashlib


_VARIANT = r"""
#include <flashinfer/attention/variants.cuh>

template <bool use_custom_mask>
struct TaHBoundedPrefillAttention
    : flashinfer::DefaultAttention<use_custom_mask, false, false, false> {
  using Base = flashinfer::DefaultAttention<use_custom_mask, false, false, false>;

  template <typename Params>
  __device__ __host__ TaHBoundedPrefillAttention(
      const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
      : Base(params, batch_idx, smem_ptr) {}

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(
      const Params& params, uint32_t batch_idx, uint32_t qo_idx,
      uint32_t kv_idx, uint32_t qo_head_idx, uint32_t kv_head_idx) {
    if (qo_idx >= this->qo_len || kv_idx >= this->kv_len) {
      return false;
    }
    return Base::LogitsMask(params, batch_idx, qo_idx, kv_idx,
                            qo_head_idx, kv_head_idx);
  }
};
"""


@cache
def _module(dtype, head_dim):
    import torch
    from flashinfer.jit.attention import gen_customize_batch_prefill_module
    from flashinfer.prefill import get_batch_prefill_jit_module

    digest = hashlib.sha256(_VARIANT.encode()).hexdigest()[:12]
    name = f'tah_bounded_prefill_{str(dtype).split(".")[-1]}_d{head_dim}_{digest}'
    module = gen_customize_batch_prefill_module(
        'fa2', name, dtype, dtype, dtype, torch.int32, head_dim, head_dim,
        ['maybe_custom_mask', 'maybe_mask_indptr', 'maybe_alibi_slopes'],
        ['uint8_t', 'int32_t', 'float'],
        ['logits_soft_cap', 'sm_scale', 'rope_rcp_scale', 'rope_rcp_theta'],
        ['double', 'double', 'double', 'double'],
        'TaHBoundedPrefillAttention<use_custom_mask>', _VARIANT)
    return get_batch_prefill_jit_module(name, module)


def make_prefill_wrapper(workspace, dtype, head_dim):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, 'NHD', backend='fa2')
    # Narrow adapter to the pinned 0.2.5 wrapper internals. Supplying jit_args to
    # every constructor would regenerate/load the module for every prompt pass;
    # cache it once per dtype/head dimension instead, as FlashInfer's default does.
    wrapper._jit_module = _module(dtype, head_dim)
    return wrapper


def run_prefill(wrapper, q, kv_cache):
    # A JIT wrapper expects its additional arguments explicitly. The standard
    # plan() still packs and owns the masks; use its per-group byte offsets.
    # Engine Q/K already include RoPE; no ALiBi, soft cap or sliding window.
    return wrapper.run(q, kv_cache, wrapper._custom_mask_buf,
                       wrapper._mask_indptr_buf, None,
                       0., q.shape[-1] ** -.5, 1., 1.)
