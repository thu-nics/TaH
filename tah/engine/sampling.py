"""Static-shape, device-only sampling for depth-mixed decode rows.

Build a bounded sampler bucket when every sampled row has 1 <= top_k <= K.
It selects K raw-logit candidates once, with descending logit and ascending
vocabulary-index ordering inside the shortlist, then performs all filters and
sampling there. At the outer CUDA shortlist boundary, tied membership follows
the pinned PyTorch 2.6 top-K kernel for that fixed shape and dtype.
Rows with top_k=0 or top_k>K use the general full-vocabulary bucket. Bucket
choice is fixed at admission and cannot change during CUDA graph replay.

The sampling call contains no host value reads, ragged shapes or global RNG.
Create/update parameter and request state before capture, then retain buffer
addresses. Logits may contain -inf, but each row needs a finite value and must
not contain NaN/+inf. FP32 filtering is allowed by SPEC.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from .tensors import upload


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    min_p: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = True

    def __post_init__(self):
        for name in ("temperature", "min_p", "top_p"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 <= self.min_p <= 1 or not 0 <= self.top_p <= 1:
            raise ValueError("min_p and top_p must be in [0, 1]")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be bool")

    @property
    def greedy(self) -> bool:
        # temperature=0 is the engine convention requested by SPEC. The raw
        # reference helper instead needs do_sample=False to avoid division by 0.
        return not self.do_sample or self.temperature == 0


class SamplingBatch:
    """Fixed-address parameter buffers and an immutable filter-path bucket.

    Updates retain tensor addresses. A general bucket may accept any later
    parameters; a bounded bucket may only receive greedy rows or 1 <= top_k <= K.
    Recreate a batch (and graph) to move between buckets.
    """

    def __init__(self, params: Sequence[SamplingParams], *, device, max_top_k=None,
                 greedy_only=False):
        if not params:
            raise ValueError("sampling batch must have at least one row")
        self.max_top_k = max_top_k
        self.greedy_only = greedy_only
        self.bounded = self._can_bound(params)
        self.temperature = torch.empty(len(params), dtype=torch.float32, device=device)
        self.min_p = torch.empty_like(self.temperature)
        self.top_p = torch.empty_like(self.temperature)
        self.top_k = torch.empty(len(params), dtype=torch.int64, device=device)
        self.do_sample = torch.empty(len(params), dtype=torch.bool, device=device)
        self.update(params)

    def _can_bound(self, params):
        return self.max_top_k is not None and all(
            p.greedy or 1 <= p.top_k <= self.max_top_k for p in params)

    def update(self, params: Sequence[SamplingParams]) -> None:
        if len(params) != self.temperature.numel():
            raise ValueError("parameter update must preserve the batch shape")
        if any(not isinstance(p, SamplingParams) for p in params):
            raise TypeError("parameters must be SamplingParams")
        if self.greedy_only and any(not p.greedy for p in params):
            raise ValueError("greedy_only sampler requires every row to be greedy")
        if self.bounded and not self._can_bound(params):
            raise ValueError("bounded sampler bucket requires 1 <= top_k <= max_top_k; "
                             "recreate with the general sampler for top_k=0 or top_k>max_top_k")
        for name in ("temperature", "min_p", "top_p", "top_k", "do_sample"):
            buffer = getattr(self, name)
            values = [getattr(p, name) for p in params]
            if name == "top_k":
                # Greedy rows do not need a filter path and can share a bounded bucket.
                values = [1 if p.greedy else p.top_k for p in params]
            buffer.copy_(upload(values, buffer.device, buffer.dtype))


@dataclass
class RequestRNG:
    """Per-request signed-int64 seeds and draw counters, carried with a request.

    SplitMix64(seed + gamma * (counter + 1)) gives an open-interval uniform.
    Greedy/inactive rows do not advance. Reset counters for a new request;
    preserve them when rows are reordered or swapped out.
    """
    seeds: torch.Tensor
    counters: torch.Tensor

    @classmethod
    def from_seeds(cls, seeds: Sequence[int], *, device="cpu") -> RequestRNG:
        if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or
                            not -(1 << 63) <= s < (1 << 63) for s in seeds):
            raise ValueError("seeds must be a nonempty sequence of signed int64 integers")
        tensor = torch.tensor(seeds, dtype=torch.int64, device=device)
        return cls(tensor, torch.zeros_like(tensor))


def _logical_right_shift(value: torch.Tensor, bits: int) -> torch.Tensor:
    return (value >> bits) & ((1 << (64 - bits)) - 1)


def request_uniform(rng: RequestRNG) -> torch.Tensor:
    """Pure device ops, with defined two's-complement int64 wraparound."""
    z = rng.seeds + (-7046029254386353131) * (rng.counters + 1)
    z = (z ^ _logical_right_shift(z, 30)) * (-4658895280553007687)
    z = (z ^ _logical_right_shift(z, 27)) * (-7723592293110705685)
    z = z ^ _logical_right_shift(z, 31)
    # 52 bits plus a half-bin offset are exactly representable in float64.
    return (_logical_right_shift(z, 12).to(torch.float64) + 0.5) * (2.0 ** -52)


class GraphSampler:
    """Per-row sampling with a one-top-K large-vocabulary path.

    `max_top_k=K` selects a static bounded batch when all sampled rows have
    1<=k<=K. That path uses one raw-logit top-K over V, then O(K) filtering
    and inverse-CDF math. At the CUDA outer-K boundary, equal-score membership
    follows the deterministic selection order of the pinned PyTorch 2.6 kernel;
    within the selected shortlist, ties are explicitly ordered by token ID.
    Changing the PyTorch/CUDA execution environment requires rerunning tie
    tests. If any row has k=0 or k>K, admission puts the whole batch in the
    general full-sort bucket. Greedy-only calls argmax alone.
    """

    def __init__(self, vocab_size: int, *, max_top_k: int | None = None,
                 greedy_only: bool = False):
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        if max_top_k is not None and (isinstance(max_top_k, bool) or
                                      not isinstance(max_top_k, int) or max_top_k < 1):
            raise ValueError("max_top_k must be a positive integer or None")
        self.vocab_size = vocab_size
        self.max_top_k = max_top_k
        self.greedy_only = greedy_only
        self.shortlist_size = min(max_top_k or vocab_size, vocab_size)

    def make_batch(self, params: Sequence[SamplingParams], *, device="cpu") -> SamplingBatch:
        return SamplingBatch(params, device=device, max_top_k=self.max_top_k,
                             greedy_only=self.greedy_only)

    def _check(self, logits, batch):
        if logits.ndim != 2 or logits.shape != (batch.temperature.numel(), self.vocab_size):
            raise ValueError("logits must have shape [batch, vocab_size]")
        if logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("logits must have floating dtype fp32/bf16/fp16")
        if (batch.temperature.device != logits.device or batch.max_top_k != self.max_top_k
                or batch.greedy_only != self.greedy_only):
            raise ValueError("parameter buffers must match sampler bound and logits device")

    def _select_shortlist(self, logits):
        """Select K raw scores and order shortlist ties by ascending token ID.

        For CUDA, equal-score membership at the outer K boundary follows the
        pinned PyTorch 2.6 top-K kernel. Tests pin this behavior for uniform and
        quantized ties and check eager/captured replay. Sorting the selected
        shortlist by token ID first and score second makes later filter ties
        explicit. CPU uses a unique composite key for score/token ordering.
        """
        if logits.is_cuda:
            selected, ids = torch.topk(logits, self.shortlist_size, dim=-1,
                                       sorted=False)
            by_token = torch.argsort(ids, dim=-1, stable=True)
            ids = ids.gather(-1, by_token)
            selected = selected.gather(-1, by_token)
            by_score = torch.argsort(selected, dim=-1, descending=True, stable=True)
            ids = ids.gather(-1, by_score)
            selected = selected.gather(-1, by_score)
            return ids, selected.float()

        scores = logits.float().contiguous()
        bits = scores.view(torch.int32).to(torch.int64)
        ordered = torch.where(bits < 0, ~bits, bits ^ 0x80000000)
        ordered = torch.where(bits == -2147483648, 0x80000000, ordered)
        token_ids = torch.arange(self.vocab_size, dtype=torch.int64, device=logits.device)
        key = (ordered << 18) | (self.vocab_size - token_ids)
        top_key = torch.topk(key, self.shortlist_size, dim=-1, sorted=True).values
        suffix = top_key & ((1 << 18) - 1)
        ids = self.vocab_size - suffix
        return ids, scores.gather(-1, ids)

    def _bounded_distribution(self, logits, batch):
        """Return [B,K] IDs/probabilities; no operation scales with V after top-K."""
        ids, selected = self._select_shortlist(logits)
        # Shortlist FP64 avoids backend-specific FP32 reduction drift at exact
        # top-p boundaries; this wide math is O(K), never O(V).
        temperature = torch.where(batch.temperature > 0, batch.temperature, 1.0).double()
        probs = torch.softmax(selected.double() / temperature[:, None], dim=-1)

        min_kept = probs >= batch.min_p[:, None] * probs.amax(-1, keepdim=True)
        after_min = torch.where(min_kept, probs, 0.0)
        after_min = torch.where(batch.min_p[:, None] > 0,
                                after_min / after_min.sum(-1, keepdim=True), probs)

        # The candidate list is already ordered by score, then token ID. The
        # documented top-k cutoff rule keeps the first k members of a tie.
        positions = torch.arange(self.shortlist_size, dtype=torch.int64,
                                 device=logits.device)[None, :]
        k = batch.top_k.clamp(1, self.shortlist_size)
        after_k = torch.where(positions < k[:, None], after_min, 0.0)
        after_k = after_k / after_k.sum(-1, keepdim=True)

        # Softmax is monotone in raw score, so the shortlist's score/token
        # order is also the top-p probability order after prefix filtering.
        ranked = after_k
        # Accumulate only this shortlist in fp64 so an exact top-p boundary
        # does not depend on the CUDA scan partition for a much wider V.
        cumulative = ranked.double().cumsum(-1)
        remove = cumulative > batch.top_p.double()[:, None]
        shifted = torch.cat((torch.zeros_like(remove[:, :1]), remove[:, :-1]), dim=-1)
        remove = torch.where(batch.top_p[:, None] < 1, shifted, False)
        kept = torch.where(remove, 0.0, ranked)
        final_probs = torch.where(batch.top_p[:, None] < 1,
                                  kept / kept.sum(-1, keepdim=True), ranked)
        return ids, final_probs

    def _full_distribution(self, logits, batch):
        """General k=0/k>K path; ties use score then token-ID order."""
        scores = logits.float()
        temperature = torch.where(batch.temperature > 0, batch.temperature, 1.0)
        probs = torch.softmax(scores / temperature[:, None], dim=-1)
        min_kept = probs >= batch.min_p[:, None] * probs.amax(-1, keepdim=True)
        after_min = torch.where(min_kept, probs, 0.0)
        after_min = torch.where(batch.min_p[:, None] > 0,
                                after_min / after_min.sum(-1, keepdim=True), probs)

        # First establish the deterministic secondary order for equal values.
        raw_order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        raw_probs = after_min.gather(-1, raw_order)
        ranked, prob_order = torch.sort(raw_probs, dim=-1, descending=True, stable=True)
        ids = raw_order.gather(-1, prob_order)

        k = batch.top_k.clamp(max=self.vocab_size)
        positions = torch.arange(self.vocab_size, dtype=torch.int64,
                                 device=logits.device)[None, :]
        after_k = torch.where((batch.top_k[:, None] == 0) |
                              (positions < k[:, None]), ranked, 0.0)
        after_k = torch.where(batch.top_k[:, None] > 0,
                             after_k / after_k.sum(-1, keepdim=True), ranked)
        cumulative = after_k.cumsum(-1)
        remove = cumulative > batch.top_p[:, None]
        shifted = torch.cat((torch.zeros_like(remove[:, :1]), remove[:, :-1]), dim=-1)
        remove = torch.where(batch.top_p[:, None] < 1, shifted, False)
        kept = torch.where(remove, 0.0, after_k)
        final_probs = torch.where(batch.top_p[:, None] < 1,
                                  kept / kept.sum(-1, keepdim=True), after_k)
        return ids, final_probs

    def _distribution(self, logits, batch):
        if (logits.is_cuda and self.vocab_size > 4096 and batch.bounded
                and self.max_top_k is not None):
            return self._bounded_distribution(logits, batch)
        return self._full_distribution(logits, batch)

    @torch.no_grad()
    def probabilities(self, logits: torch.Tensor, batch: SamplingBatch) -> torch.Tensor:
        """Diagnostic [B,V] target probabilities; decode sampling stays on [B,K]."""
        self._check(logits, batch)
        if self.greedy_only:
            return torch.zeros_like(logits, dtype=torch.float32).scatter(
                -1, logits.argmax(-1, keepdim=True), 1.0)
        ids, probs = self._distribution(logits, batch)
        target = torch.zeros_like(logits, dtype=torch.float32)
        target.scatter_(-1, ids, probs.float())
        greedy = (~batch.do_sample) | (batch.temperature == 0)
        one_hot = torch.zeros_like(target).scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
        return torch.where(greedy[:, None], one_hot, target)

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, batch: SamplingBatch, rng: RequestRNG,
               active: torch.Tensor | None = None) -> torch.Tensor:
        """Return int64 [B] IDs (-1 inactive), advancing active sampling counters.

        RNG state and active mask are int64/bool [B] on logits.device. No host
        sync, nonzero(), dynamic-shape operation or global RNG is used.
        """
        self._check(logits, batch)
        for state in (rng.seeds, rng.counters):
            if state.shape != logits.shape[:1] or state.device != logits.device or state.dtype != torch.int64:
                raise ValueError("RNG state must be int64 [batch] on logits device")
        if active is None:
            active = torch.ones_like(batch.do_sample)
        if active.shape != logits.shape[:1] or active.device != logits.device or active.dtype != torch.bool:
            raise ValueError("active must be bool [batch] on logits device")
        if self.greedy_only:
            return torch.where(active, logits.argmax(-1), -1)

        ids, probs = self._distribution(logits, batch)
        cdf = probs.cumsum(-1)
        uniform = request_uniform(rng)
        target = uniform[:, None] * cdf[:, -1:].to(torch.float64)
        if probs.dtype == torch.float64:
            below = cdf < target
        else:
            # Resolve FP32 CDF boundaries with only a [B,1] double comparison.
            rounded = target.float()
            below = (cdf < rounded) | ((cdf == rounded) & (rounded.double() < target))
        slot = below.sum(-1).clamp(max=probs.shape[-1] - 1)
        token = ids.gather(-1, slot[:, None]).squeeze(-1)
        greedy = (~batch.do_sample) | (batch.temperature == 0)
        token = torch.where(greedy, logits.argmax(-1), token)
        rng.counters.add_((active & batch.do_sample & (batch.temperature > 0)).to(torch.int64))
        return torch.where(active, token, -1)
