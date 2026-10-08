"""Vocab-parallel (tensor-parallel lm_head/embedding) ops for TaH.

With vocab parallel the tied embedding/lm_head weight is sharded on the vocab
dim and every logits tensor in the forward is a local ``[..., V/tp]`` shard;
the ops here compute global vocab-space reductions with one or two small
collectives each.

Adjoint rules (Megatron-style; each rank backwards its own replica of the
downstream graph):
  - replicated tensor entering sharded math -> identity fwd / allreduce-SUM bwd
    (``_CopyToVocabParallel``);
  - sharded partials feeding a REPLICATED consumer -> SUM fwd / identity bwd
    (``_ReduceFromVocabParallel``);
  - sharded partials whose consumers are themselves shard-specific (e.g. the
    lse in ``local - lse``) -> SUM fwd / allreduce-SUM bwd
    (``_SumAcrossVocabParallel``).
Using the wrong adjoint keeps the loss correct but scales gradients by ~tp.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class _CopyToVocabParallel(torch.autograd.Function):
    """forward: identity; backward: allreduce-sum over the tp group."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous().clone()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


class _ReduceFromVocabParallel(torch.autograd.Function):
    """forward: allreduce-sum; backward: identity (replicated consumer)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        x = x.contiguous().clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _SumAcrossVocabParallel(torch.autograd.Function):
    """forward: allreduce-sum; backward: allreduce-sum (shard-specific consumers)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        x = x.contiguous().clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous().clone()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


class _GatherFromVocabParallel(torch.autograd.Function):
    """forward: all-gather + concat on the last dim; backward: local slice."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        world = dist.get_world_size(group)
        ctx.rank = dist.get_rank(group)
        ctx.local_dim = x.shape[-1]
        x = x.contiguous()
        chunks = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(chunks, x, group=group)
        return torch.cat(chunks, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.rank * ctx.local_dim
        return grad_output[..., start : start + ctx.local_dim].contiguous(), None


class VocabParallelOps:
    """Global vocab-space reductions over locally-sharded logits."""

    def __init__(self, group, vocab_size: int):
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if vocab_size % self.world != 0:
            raise ValueError(
                f"vocab_size {vocab_size} not divisible by tp={self.world}"
            )
        self.vocab_size = int(vocab_size)
        self.local_size = self.vocab_size // self.world
        self.start = self.rank * self.local_size

    @staticmethod
    def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
        # At least fp32, but never downcast fp64 (numerical grad checks).
        return torch.promote_types(dtype, torch.float32)

    def copy_to_shards(self, x: torch.Tensor) -> torch.Tensor:
        """Identity fwd / allreduce-SUM bwd. Required for any replicated tensor
        entering shard-parallel vocab math (e.g. stop-prob mixture log-weights),
        else upstream replicated params get token-partitioned grads."""
        return _CopyToVocabParallel.apply(x, self.group)

    def logsumexp(self, local_logits: torch.Tensor) -> torch.Tensor:
        """Global logsumexp (>=fp32) over the sharded last dim."""
        return self._logsumexp(local_logits, replicated_consumer=False)

    def logsumexp_lean(self, local_logits: torch.Tensor) -> torch.Tensor:
        """Memory-lean global logsumexp for REPLICATED consumers.

        ``_logsumexp`` materializes and retains a full fp32 ``[N, V/tp]`` exp
        activation per call — called once per TaH iteration that alone exceeds
        GPU memory at max_iter=16. Here each shard runs ``fp32_logsumexp``
        (fp32-exact, saves only the bf16 shard, chunked) and ranks combine on
        [N]-sized tensors. Adjoint convention matches ``gather_values``
        (_ReduceFromVocabParallel): feed the result to replicated math only.
        """
        from tah2.utils.fp32_ops import fp32_logsumexp

        local_lse = fp32_logsumexp(local_logits)  # [N] fp32, lean + exact
        m = local_lse.detach().clone()
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=self.group)
        # m is a detached per-token constant: the usual logsumexp shift trick —
        # gradients through local_lse stay exact.
        s = torch.exp(local_lse - m)
        s = _ReduceFromVocabParallel.apply(s, self.group)
        return m + torch.log(s)

    def _logsumexp(
        self, local_logits: torch.Tensor, replicated_consumer: bool
    ) -> torch.Tensor:
        """Global logsumexp; the inner reduce's adjoint depends on WHO consumes
        the result: shard-specific consumers (``local - lse``) need the
        grad-allreduce adjoint, replicated consumers (``ce = lse - target``)
        need identity — summing again would scale the softmax term by tp."""
        dtype = self._compute_dtype(local_logits.dtype)
        m = local_logits.detach().to(dtype).amax(dim=-1)
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=self.group)
        s = torch.exp(local_logits.to(dtype) - m.unsqueeze(-1)).sum(dim=-1)
        reduce = (
            _ReduceFromVocabParallel if replicated_consumer else _SumAcrossVocabParallel
        )
        s = reduce.apply(s, self.group)
        return m + torch.log(s)

    def log_softmax(self, local_logits: torch.Tensor) -> torch.Tensor:
        """Globally-normalized log-probs (local shard). Returns >=fp32: casting
        the lse to bf16 before subtracting biases the NLL by ~5e-3/token."""
        lse = self.logsumexp(local_logits)
        return local_logits.to(lse.dtype) - lse.unsqueeze(-1)

    def gather_target(
        self, local_values: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """``values[..., target]`` with vocab-sharded values. ``targets`` >= 0."""
        local_t = targets - self.start
        in_shard = (local_t >= 0) & (local_t < self.local_size)
        safe = local_t.clamp(0, self.local_size - 1)
        vals = local_values.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        vals = torch.where(in_shard, vals, torch.zeros_like(vals))
        return _ReduceFromVocabParallel.apply(vals, self.group)

    def gather_values(
        self, local_values: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Gather global-vocab IDs into replicated values."""
        local_ids = token_ids - self.start
        in_shard = (local_ids >= 0) & (local_ids < self.local_size)
        safe_ids = local_ids.clamp(0, self.local_size - 1)
        values = local_values.gather(-1, safe_ids)
        values = torch.where(in_shard, values, torch.zeros_like(values))
        return _ReduceFromVocabParallel.apply(values, self.group)

    def cross_entropy_per_token(
        self,
        local_logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """>=fp32 per-token CE (0 at ignore_index), matching
        ``F.cross_entropy(logits.float(), t, reduction="none")``."""
        valid = targets != ignore_index
        safe_t = torch.where(valid, targets, torch.zeros_like(targets))
        # The CE is replicated on every rank -> identity adjoint for the lse,
        # which is exactly logsumexp_lean's convention. The lean version also
        # drops _logsumexp's fp32 [N, V/tp] exp materialization per call.
        ce = self.logsumexp_lean(local_logits) - self.gather_target(
            local_logits.to(self._compute_dtype(local_logits.dtype)), safe_t
        )
        return torch.where(valid, ce, torch.zeros_like(ce))

    def nll_from_log_probs(
        self,
        local_log_probs: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """>=fp32 per-token NLL when the inputs are already global log-probs."""
        valid = targets != ignore_index
        safe_t = torch.where(valid, targets, torch.zeros_like(targets))
        nll = -self.gather_target(
            local_log_probs.to(self._compute_dtype(local_log_probs.dtype)), safe_t
        )
        return torch.where(valid, nll, torch.zeros_like(nll))

    def softmax_topk_values(self, local_logits: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k of the GLOBAL softmax, values only (differentiable).

        Memory-lean: the previous ``exp(log_softmax(...))`` retained TWO fp32
        ``[N, V/tp]`` activations per call (one per TaH iteration via the
        decider). Same values and gradients: top-k indices are picked under
        no_grad on the logits (monotone in probs), the differentiable path
        only touches ``[N, k]`` gathers + the lean global logsumexp.
        """
        k_local = min(k, local_logits.shape[-1])
        Z = self.logsumexp_lean(local_logits)
        with torch.no_grad():
            idx = local_logits.topk(k_local, dim=-1).indices
        local_top = torch.exp(
            local_logits.gather(-1, idx).float() - Z.unsqueeze(-1)
        )
        gathered = _GatherFromVocabParallel.apply(local_top, self.group)
        k_out = min(k, gathered.shape[-1])
        with torch.no_grad():
            gidx = gathered.topk(k_out, dim=-1).indices
        return gathered.gather(-1, gidx)


class VocabParallelLMHead(nn.Module):
    """lm_head returning the LOCAL logits shard ``[..., V/tp]``.

    ``weight`` is the vocab-sharded (tied) weight; a plain weight is accepted
    for the degenerate tp=1 case so tp=1 runs the same code path.
    """

    def __init__(self, weight: nn.Parameter, ops: VocabParallelOps):
        super().__init__()
        self.weight = weight
        self.ops = ops

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = _CopyToVocabParallel.apply(hidden_states, self.ops.group)
        weight = self.weight
        if hasattr(weight, "to_local"):
            weight = weight.to_local()
        return F.linear(hidden_states, weight)
