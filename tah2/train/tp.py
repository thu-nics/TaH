"""Torch-native parallelism helpers: DTensor TP (tp mesh) + FSDP2 (dp mesh).

Backbone linears are sharded Megatron-style; norms and TaH side modules stay
replicated across tp so iteration routing is tp-rank-consistent.
"""

from typing import Iterable, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

LAYER_TP_PLAN = {
    "self_attn.q_proj": ColwiseParallel,
    "self_attn.k_proj": ColwiseParallel,
    "self_attn.v_proj": ColwiseParallel,
    "self_attn.o_proj": RowwiseParallel,
    "mlp.gate_proj": ColwiseParallel,
    "mlp.up_proj": ColwiseParallel,
    "mlp.down_proj": RowwiseParallel,
}


def local_tensor(t: torch.Tensor) -> torch.Tensor:
    """The local shard of a DTensor, or the tensor itself."""
    return t.to_local() if isinstance(t, DTensor) else t


def scope_tah_sync_group_solo(model: nn.Module, rank: int, world_size: int) -> None:
    """tp=1 / dp>1: scope TaH's iter-sync + label-occupancy all_reduces to a single-rank
    group so each DP replica iterates independently.

    Companion to ``apply_tensor_parallel`` (which scopes them to the TP group for tp>1).
    With NO tensor parallelism each rank is a full DP replica; left on the default WORLD
    process group the replicas desync — under ``dynamic_batch`` they pack their different
    data into different micro-batch counts and so issue the recurrent iter-sync collective
    (tah_model.py) a *different number of times*, and NCCL times out (observed: tp1/dp2,
    ALLREDUCE SeqNum 280, 600s timeout — rank0 the scalar iter-sync vs rank1 the 1.7B grad
    all-reduce at the same seq id). A single-rank group makes those all_reduces local
    no-ops; DP replicas reconcile only via the gradient all-reduce.
    """
    if world_size <= 1 or not dist.is_initialized():
        return
    solo = None
    for r in range(
        world_size
    ):  # new_group is collective: every rank must call it for each r
        g = dist.new_group([r])
        if r == rank:
            solo = g
    base = getattr(model, "simple_base_model", model)
    for _obj in (model, base, getattr(model, "iter_label_generator", None)):
        if _obj is not None:
            _obj.tah_sync_group = solo


def apply_tensor_parallel(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    vocab_parallel: bool = True,
) -> nn.Module:
    """Shard the Qwen3 backbone across ``tp_mesh``.

    With ``vocab_parallel`` the embedding/lm_head weight is also sharded on the
    vocab dim and vocab-space ops run on local shards via
    ``model.vocab_parallel_ops``. TP=1 returns unchanged so memory-lean
    non-TP vocab reductions remain available.

    """
    src_data_rank = 0
    base = getattr(model, "simple_base_model", model)
    inner = getattr(base, "model", base)

    if tp_mesh.size() <= 1:
        return model

    # TaH's recurrent iter-sync (tah_model.py) and label-occupancy collectives
    # (iter_label.py) all_reduce over ``tah_sync_group``; left unset it defaults to the
    # WORLD process group. That is wrong under dp>1: the DP replicas pack their different
    # data into different micro-batch counts under dynamic_batch, so over the world PG
    # they issue these collectives a *different number of times* and NCCL desyncs -> hang
    # (data-dependent; observed at step 15 of an 8B dp2/tp4 run). Scope them to the TP
    # (model-shard) group instead: it holds exactly the ranks that share data, hence
    # identical micro-batch counts; DP replicas stay independent and reconcile only via
    # the gradient all-reduce on the dp mesh.
    tp_group = tp_mesh.get_group()
    for _obj in (model, base, getattr(model, "iter_label_generator", None)):
        if _obj is not None:
            _obj.tah_sync_group = tp_group

    layers = getattr(inner, "layers", None)
    if not layers:
        raise ValueError(
            "apply_tensor_parallel expects decoder layers at <base>.model.layers"
        )

    num_heads = base.config.num_attention_heads
    num_kv_heads = base.config.num_key_value_heads
    tp_size = tp_mesh.size()
    if num_heads % tp_size or num_kv_heads % tp_size:
        raise ValueError(
            f"num_attention_heads ({num_heads}) and num_key_value_heads ({num_kv_heads}) "
            f"must both be divisible by tp size ({tp_size})."
        )

    for layer in layers:
        parallelize_module(
            layer,
            tp_mesh,
            {name: style() for name, style in LAYER_TP_PLAN.items()},
            src_data_rank=src_data_rank,
        )

    if vocab_parallel:
        _apply_vocab_parallel(model, base, inner, tp_mesh, src_data_rank)
    return model


def _apply_vocab_parallel(
    model, base, inner, tp_mesh: DeviceMesh, src_data_rank: int | None = 0
) -> None:
    from tah2.utils.vocab_parallel import VocabParallelLMHead, VocabParallelOps

    if getattr(model, "memory_lean_fp32_reductions", False):
        # Redundant under vocab parallel (reductions already run on V/tp shards).
        if dist.get_rank() == 0:
            print(
                "[tp] memory_lean_fp32_reductions disabled (superseded by vocab parallel)"
            )
        model.memory_lean_fp32_reductions = False

    embed = inner.embed_tokens
    tied = (
        getattr(base, "lm_head", None) is not None
        and base.lm_head.weight is embed.weight
    )

    if tp_mesh.size() > 1:
        # Weight Shard(0); rowwise embedding masks out-of-shard ids and
        # allreduces the partial output back to replicate.
        parallelize_module(
            inner,
            tp_mesh,
            {
                "embed_tokens": RowwiseParallel(
                    input_layouts=Replicate(), output_layouts=Replicate()
                )
            },
            src_data_rank=src_data_rank,
        )

    ops = VocabParallelOps(tp_mesh.get_group(), int(base.config.vocab_size))

    if tied:
        head_weight = inner.embed_tokens.weight
    elif tp_mesh.size() > 1:
        head_weight = nn.Parameter(
            distribute_tensor(
                base.lm_head.weight.data,
                tp_mesh,
                [Shard(0)],
                src_data_rank=src_data_rank,
            )
        )
    else:
        head_weight = base.lm_head.weight
    base.lm_head = VocabParallelLMHead(head_weight, ops)

    model.vocab_parallel_ops = ops
    for component_name in ("iter_label_generator", "iter_decider", "eval_iter_decider"):
        component = getattr(model, component_name, None)
        if component is not None:
            component.vocab_parallel_ops = ops


def apply_fsdp(model: nn.Module, dp_mesh: DeviceMesh, layer_wrap: bool) -> nn.Module:
    """fully_shard (FSDP2) the possibly-TP-sharded model over the dp mesh.

    ``dp_mesh`` may be 1-D (flat FSDP) or 2-D ``(dp_replicate, dp_shard)`` for
    HSDP: weights shard over dim 1 only, gradients reduce-scatter within the
    shard group then all-reduce across replicate groups. Keep the shard group
    intra-node so cross-node traffic is just the sharded-grad all-reduce.

    Params stay fp32 (the sharded master the optimizer steps); compute is bf16
    via MixedPrecisionPolicy with fp32 reduce. ``layer_wrap=True`` uses
    memory-lean per-layer groups and requires the caller to keep recurrent
    forward/backward graph structure aligned across data-parallel ranks.

    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    if layer_wrap:
        base = getattr(model, "simple_base_model", model)
        inner = getattr(base, "model", base)
        for layer in getattr(inner, "layers", []):
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=False)
    return model


def _replicated_across_tp(grad: torch.Tensor) -> bool:
    """Plain tensors and DTensors without a 'tp' mesh dim (1-D dp shards) are
    tp-replicated; params sharded on tp must not be tp-reduced."""
    if not isinstance(grad, DTensor):
        return True
    names = getattr(grad.device_mesh, "mesh_dim_names", None) or ()
    return "tp" not in names


def _allreduce_flat(
    grads: List[torch.Tensor], group, op, divide_by: float = 1.0
) -> None:
    """Bucketed all-reduce (one flat reduce per dtype). Zero-numel shards are
    skipped: nothing to reduce, and ``_unflatten_dense_tensors`` returns them
    as shape ``(0,)`` which breaks the copy back."""
    by_dtype = {}
    for grad in grads:
        if grad.numel() == 0:
            continue
        by_dtype.setdefault(grad.dtype, []).append(grad)
    for bucket in by_dtype.values():
        flat = torch._utils._flatten_dense_tensors(bucket)
        dist.all_reduce(flat, op=op, group=group)
        if divide_by != 1.0:
            flat.div_(divide_by)
        for grad, synced in zip(
            bucket, torch._utils._unflatten_dense_tensors(flat, bucket)
        ):
            grad.copy_(synced)


def sync_replicated_grads(model: nn.Module, tp_group) -> None:
    """All-reduce (mean) tp-replicated param grads over tp ranks: their compute
    is identical per rank, but nondeterministic kernels drift and the drift
    would otherwise step into the weights."""
    if tp_group is None or dist.get_world_size(tp_group) <= 1:
        return
    grads = []
    for param in model.parameters():
        grad = param.grad
        if grad is None or not _replicated_across_tp(grad):
            continue
        grads.append(grad.to_local() if isinstance(grad, DTensor) else grad)
    if grads:
        _allreduce_flat(
            grads, tp_group, dist.ReduceOp.SUM, divide_by=dist.get_world_size(tp_group)
        )


def full_plain_state_dict(module, cast_floats_to=None) -> dict:
    """CPU state dict with every DTensor gathered full. COLLECTIVE.

    ``cast_floats_to`` casts floating tensors (e.g. the fp32 FSDP masters)
    before gathering — checkpoints are served in bf16. Returns None for a
    None module (so callers can pass optional modules).
    """
    if module is None:
        return None
    out = {}
    for key, value in module.state_dict().items():
        if cast_floats_to is not None and value.is_floating_point():
            value = value.to(cast_floats_to)
        if isinstance(value, DTensor):
            value = value.full_tensor()
        out[key] = value.detach().cpu()
    return out


def full_base_state_dict(model: nn.Module, cast_floats_to=None) -> dict:
    """``full_plain_state_dict`` of the BASE model (base-relative keys, feeds
    ``base.save_pretrained(..., state_dict=...)``). COLLECTIVE."""
    return full_plain_state_dict(
        getattr(model, "simple_base_model", model), cast_floats_to=cast_floats_to
    )


def split_dtensor_param_groups(optimizer: torch.optim.Optimizer) -> None:
    """Split param groups by sharding layout, in place: fused/foreach optimizers
    refuse mixed plain/DTensor operands or mixed meshes/placements."""
    new_groups = []
    for group in optimizer.param_groups:
        buckets: dict = {}
        for p in group["params"]:
            if isinstance(p, DTensor):
                key = (id(p.device_mesh), tuple(str(pl) for pl in p.placements))
            else:
                key = "plain"
            buckets.setdefault(key, []).append(p)
        for params in buckets.values():
            new_group = {k: v for k, v in group.items() if k != "params"}
            new_group["params"] = params
            new_groups.append(new_group)
    optimizer.param_groups = new_groups


def _total_grad_norm(grads: List[torch.Tensor]) -> torch.Tensor:
    """Global L2 norm over mixed plain / 1-D / 2-D DTensor gradients.

    Grads are grouped by (mesh, placements) — foreach norm refuses mixed
    operands — and each group's partial norm is materialized globally with
    ``full_tensor()`` before combining. Collective: every rank must call it.
    """
    dt_groups: dict = {}
    plain_grads = []
    for grad in grads:
        if isinstance(grad, DTensor):
            key = (id(grad.device_mesh), tuple(str(pl) for pl in grad.placements))
            dt_groups.setdefault(key, []).append(grad)
        else:
            plain_grads.append(grad)

    norms = []
    for group in dt_groups.values():
        norm = torch.nn.utils.get_total_norm(group, norm_type=2.0)
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        norms.append(norm.to(torch.float32))
    if plain_grads:
        norms.append(
            torch.nn.utils.get_total_norm(plain_grads, norm_type=2.0).to(torch.float32)
        )
    if not norms:
        return torch.zeros((), dtype=torch.float32, device=torch.cuda.current_device())
    return torch.linalg.vector_norm(torch.stack(norms), 2.0)


def grad_norm_tp(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    """Pre-clip global grad norm of ``parameters`` (e.g. one TaH side module)."""
    return _total_grad_norm([p.grad for p in parameters if p.grad is not None])


def clip_grad_norm_tp(
    parameters: Iterable[nn.Parameter], max_norm: float
) -> torch.Tensor:
    """Clip by the global grad norm (see ``_total_grad_norm``); returns the pre-clip norm."""
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _total_grad_norm(grads)
    if max_norm > 0:
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        for grad in grads:
            grad.mul_(clip_coef.to(grad.device))
    return total_norm
