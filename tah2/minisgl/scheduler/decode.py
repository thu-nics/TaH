from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from tah2.minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    # DUO TaH: per-step extra-iter KV slots live in separate page tables outside
    # the prefix radix, so they're invisible to ``available_size``. Reserve a
    # fraction of the remaining decode for projected growth (the factor is
    # pre-scaled by (max_iter-1) extra streams). Set to 0 in non-duo paths.
    duo_iter1_reserve_factor: float = 0.0
    transient_iter_slots: int = 0
    running_reqs: Set[Req] = field(default_factory=set)
    iterating_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """Add new reqs and route them to the correct pool."""
        for req in reqs:
            # Skip reqs freed by an abort earlier in this overlap iter — their page_table
            # entries now point to pages in free_slots/radix and would corrupt shared pages.
            if getattr(req, "_freed", False) or not req.can_decode:
                self.running_reqs.discard(req)
                self.iterating_reqs.discard(req)
                continue
            if req.is_iterating:
                self.iterating_reqs.add(req)
                self.running_reqs.discard(req)
            else:
                self.running_reqs.add(req)
                self.iterating_reqs.discard(req)
        # Prune reqs that finished/were freed while already sitting in running_reqs.
        self.running_reqs = {
            r for r in self.running_reqs if r.can_decode and not getattr(r, "_freed", False)
        }

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)
        self.iterating_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for pool in (self.running_reqs, self.iterating_reqs):
            for req in pool:
                if req.uid == uid:
                    pool.remove(req)
                    return req
        return None

    @property
    def inflight_tokens(self) -> int:
        all_reqs = self.running_reqs | self.iterating_reqs
        tokens_reserved = (self.page_size - 1) * len(all_reqs)
        base = (
            sum(req.remain_len for req in all_reqs)
            + tokens_reserved
            + len(all_reqs) * self.transient_iter_slots
        )
        if self.duo_iter1_reserve_factor > 0.0:
            # Project iter1 KV growth over the remaining decode. Already-held
            # iter1 slots are *not* added here: they're already missing from
            # ``available_size`` (= free + evictable), so summing them again
            # would double-count and starve the admission gate.
            base += sum(int(req.remain_len * self.duo_iter1_reserve_factor) for req in all_reqs)
        return base

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None

        iterating = sorted(self.iterating_reqs, key=lambda req: req.uid)
        decode_ready = sorted(self.running_reqs, key=lambda req: req.uid)

        if iterating:
            # Mixed batch: iterating reqs first (they hold recurrent slots),
            # then decode-ready reqs to fill the batch.
            return Batch(reqs=iterating + decode_ready, phase="mixed_decode")

        return Batch(reqs=decode_ready, phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0 or len(self.iterating_reqs) > 0
