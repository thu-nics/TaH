from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch

from tah2.minisgl.core import Batch, Req
from tah2.minisgl.env import ENV
from tah2.minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from tah2.minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager
from .utils import PendingReq

if TYPE_CHECKING:
    from tah2.minisgl.engine import BatchSamplingArgs, ForwardOutput, MixedForwardOutput
    from tah2.minisgl.engine.engine import PendingMixedForward


logger = init_logger(__name__)

# DUO TaH: extra-iter KV slots are allocated from the same page pool as iter0
# but stored in ``engine._tah_iter_page_tables`` (one table per extra
# iteration) — *outside* the prefix radix.  The scheduler's reservation must
# reflect both currently-held slots (``tah_total_iter_slots``) and projected
# growth across the remaining decode (factor × (max_iter-1) × remain_len). The
# base factor is exposed as ``SchedulerConfig.tah_duo_iter1_reserve_factor``
# because it is a deployment throughput/capacity tradeoff, not a model semantic.

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs | None
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D | None  # (req_mapping, seq_lens or 0)


class PendingForward(NamedTuple):
    """Handle between :meth:`_forward_dispatch` and :meth:`_forward_finalize`.

    TaH mixed path: ``pending_mixed`` is the engine's live
    ``PendingMixedForward`` (sync + state updates happen in finalize).
    Baseline path: ``forward_output`` is already populated, finalize is
    a pass-through.
    """

    is_mixed: bool
    forward_input: ForwardInput
    pending_mixed: PendingMixedForward | None
    forward_output: ForwardOutput | MixedForwardOutput | None


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput | MixedForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from tah2.minisgl.engine import Engine

        if config.tah_duo_iter1_reserve_factor < 0.0:
            raise ValueError("tah_duo_iter1_reserve_factor must be >= 0")

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )

        # Give engine access to cache_manager for TaH iter=1 page allocation
        self.engine._tah_cache_manager = self.cache_manager

        # Detect TaH model for mixed-phase decode dispatch
        from tah2.minisgl.models.tah_qwen3 import TaHQwen3ForCausalLM

        self._is_tah_model = isinstance(self.engine.model, TaHQwen3ForCausalLM)

        # Each iterating token can spawn up to (max_iter-1) extra KV slots over
        # its iterations, so scale the per-token reserve by the stream count.
        duo_factor = (
            config.tah_duo_iter1_reserve_factor * self.engine._tah_num_iter_streams
            if self._is_tah_model and self.engine.model.iter_attention_mode == "duo"
            else 0.0
        )
        # Dynamic preemption replaces the static reserve: admission goes greedy
        # (duo_factor=0) and KV pressure is relieved at runtime by demoting the
        # newest running reqs to the radix cache (see ``_select_preempt_victims``).
        self._dynamic_preempt = (
            bool(config.tah_dynamic_preempt)
            and self._is_tah_model
            and self.engine.model.iter_attention_mode == "duo"
        )
        # Prompt-side iter KV is allocated synchronously during prefill (can't be
        # preempted mid-forward), so even in preempt mode reserve its worst case.
        prefill_iter_factor = 0.0
        if self._dynamic_preempt:
            duo_factor = 0.0
            prefill_iter_factor = float(self.engine._tah_num_iter_streams)
        # Reqs chosen for preemption this step; freed/re-queued in
        # ``_finalize_preemptions`` once the in-flight batch retires.
        self._preempt_pending: List[Req] = []
        self._num_preemptions = 0
        transient_slots = (
            self.engine._tah_num_iter_streams
            if self._is_tah_model and self.engine.model.iter_attention_mode == "causal"
            else 0
        )
        self.decode_manager = DecodeManager(
            config.page_size,
            duo_iter1_reserve_factor=duo_factor,
            transient_iter_slots=transient_slots,
        )
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            duo_iter1_reserve_factor=duo_factor,
            prefill_iter_reserve_factor=prefill_iter_factor,
            transient_iter_slots=transient_slots,
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.num_generated_tokens_since_log = 0
        self.last_stats_time = 0.0
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """Main loop: overlap the current batch's GPU work with the previous
        batch's CPU reply bookkeeping.

        TaH mixed forward splits into ``dispatch`` (queue GPU work) and
        ``finalize`` (sync + commit); ``_process_last_data`` runs between
        them so CPU work hides the graph-replay wait.  Baseline models
        finish inside ``_forward_dispatch`` and finalize is a pass-through.
        """
        blocking = not (
            last_data is not None or self.prefill_manager.runnable or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Prev iter's finalize (on engine.stream) may have rebound
        # ``cache_mgr.free_slots`` via ``_free``; this iter's
        # ``_schedule_next_batch`` and ``_process_last_data`` read and
        # rewrite that same attribute on self.stream, so self.stream must
        # see engine.stream's state before any free_slots access here.
        self.stream.wait_stream(self.engine.stream)

        forward_input = self._schedule_next_batch()
        pending = None
        if forward_input is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                pending = self._forward_dispatch(forward_input)

        self._process_last_data(last_data)
        # Victims selected in this iter's _schedule_next_batch are excluded from
        # the in-flight batch; their last token is now appended, so free them.
        if self._preempt_pending:
            self._finalize_preemptions()

        if pending is None:
            return None
        with self.engine_stream_ctx:
            # _process_last_data (self.stream) may have rebound
            # ``cache_mgr.free_slots`` via lazy_free's final cat; finalize's
            # own ``_free`` reads that same attribute on engine.stream, so
            # engine.stream must see self.stream's state.
            self.engine.stream.wait_stream(self.stream)
            return forward_input, self._forward_finalize(pending)

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        if self._preempt_pending:
            self._finalize_preemptions()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if (
            ENV.DISABLE_OVERLAP_SCHEDULING
            or not self._is_tah_model
            or self.engine.model.iter_attention_mode == "causal"
        ):
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        forward_input, forward_output = last_data

        # Dispatch to mixed-phase handler if applicable
        from tah2.minisgl.engine.engine import MixedForwardOutput

        if isinstance(forward_output, MixedForwardOutput):
            self._process_mixed_data(forward_output)
            return

        batch = forward_input.batch
        _, next_tokens_cpu, _, iter_counts_cpu, copy_done = forward_output
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        finished_reply_idx: int | None = None
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                # Abort in this overlap iter already freed resources; reinserting stale
                # page_indices would double-free pages the radix still references.
                if getattr(req, "_freed", False):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                iter_count = int(iter_counts_cpu[i].item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                self._on_generated_tokens(1)
                prompt_iter_counts = None
                if self._is_tah_model and batch.is_prefill:
                    self.engine.tah_iter_log.setdefault(req.uid, []).append(iter_count)
                    prompt_iter_counts = self.engine.tah_prompt_iter_log.pop(req.uid, None)

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        iter_count=iter_count,
                        prompt_iter_counts=prompt_iter_counts,
                    )
                )
                if finished and req in new_finished_reqs:
                    finished_reply_idx = len(reply) - 1

        self.finished_reqs = new_finished_reqs
        if finished_reply_idx is not None:
            reply[finished_reply_idx].stats = self._make_completion_stats()
        self.send_result(reply)

    def _process_mixed_data(self, output: MixedForwardOutput) -> None:
        """Process results from a mixed-phase TaH batch.

        Rollback and ``complete_one()`` have already been handled inside
        ``forward_mixed_tah_batch`` (engine-side) to maintain correct timing
        for overlap scheduling.  Here we only do host-side bookkeeping.
        """
        _, next_tokens_cpu, _, iter_counts_cpu, copy_done, sampled_reqs = output
        copy_done.synchronize()
        # ``.tolist()`` does the whole pinned→Python conversion in C.
        tokens_list = next_tokens_cpu.tolist()
        iter_counts_list = iter_counts_cpu.tolist()

        reply: List[DetokenizeMsg] = []
        finished_reply_idx: int | None = None
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(sampled_reqs):
                # See ``_process_last_data``: abort-in-flight means append_host/cache_req
                # would touch stale state.
                if getattr(req, "_freed", False):
                    continue
                next_token = tokens_list[i]
                req.append_host(next_tokens_cpu[i : i + 1])
                finished = not req.can_decode or (
                    not req.sampling_params.ignore_eos and next_token == self.eos_token_id
                )
                self._on_generated_tokens(1)
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        iter_count=iter_counts_list[i],
                    )
                )
                if finished and req in new_finished_reqs:
                    finished_reply_idx = len(reply) - 1

        self.finished_reqs = new_finished_reqs
        if finished_reply_idx is not None:
            reply[finished_reply_idx].stats = self._make_completion_stats()
        self.send_result(reply)

    def _on_generated_tokens(self, num_tokens: int) -> None:
        if self.last_stats_time == 0.0:
            self.last_stats_time = time.perf_counter()
        self.num_generated_tokens_since_log += num_tokens

    def _make_completion_stats(self) -> Dict[str, Any]:
        now = time.perf_counter()
        interval = max(now - self.last_stats_time, 1e-6)
        gen_throughput = self.num_generated_tokens_since_log / interval
        self.num_generated_tokens_since_log = 0
        self.last_stats_time = now

        total_tokens = self.cache_manager.num_pages * self.cache_manager.page_size
        used_tokens = total_tokens - self.cache_manager.available_size
        reserved_tokens = self.decode_manager.inflight_tokens
        token_usage = used_tokens / total_tokens if total_tokens > 0 else 0.0
        admission_usage = (
            (used_tokens + reserved_tokens) / total_tokens if total_tokens > 0 else 0.0
        )
        running_reqs = len(self.decode_manager.running_reqs) + len(
            self.decode_manager.iterating_reqs
        )

        return {
            "running_req": running_reqs,
            "queue_req": len(self.prefill_manager.pending_list),
            "used_tokens": used_tokens,
            "available_tokens": self.cache_manager.available_size,
            "reserved_tokens": reserved_tokens,
            "token_usage": token_usage,
            "admission_usage": admission_usage,
            "gen_throughput": gen_throughput,
            "num_preemptions": self._num_preemptions,
        }

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Still ack it: without a terminal reply the frontend's
                # wait_for_ack never returns and the HTTP client hangs forever.
                # A finished msg carrying eos decodes to "", so the caller gets
                # an empty completion instead of a black hole.
                return self.send_result(
                    [DetokenizeMsg(uid=msg.uid, next_token=self.eos_token_id, finished=True)]
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                if req_to_free.is_iterating:
                    req_to_free.finish_iteration()
                    # No forward to run ``complete_one`` for us on the abort path; without
                    # this, iter=0's page at the ``saved_device_len - 1`` boundary leaks.
                    req_to_free.complete_one()
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        # Double-free drives the radix handle's ref_count negative; callers must gate on _freed.
        assert not getattr(req, "_freed", False), f"double-free for uid={req.uid}"
        req._freed = True
        # DUO: release any persisted extra-iter KV slots (all streams) back to
        # the page pool.
        if self.engine._tah_has_iter_tables and req.tah_total_iter_slots > 0:
            iter_pts = self.engine._tah_iter_page_tables
            freed_chunks = [
                iter_pts[j][req.table_idx, :c].clone()
                for j, c in enumerate(req.tah_iter_counts)
                if c > 0
            ]
            if freed_chunks:
                self.cache_manager._free(torch.cat(freed_chunks).to(torch.int32))
            req.tah_iter_counts = []
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        if batch.is_mixed_decode or (batch.is_decode and self._is_tah_model):
            return self._prepare_mixed_batch(batch)
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        sample_args = self.engine.sampler.prepare(batch)
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _prepare_mixed_batch(self, batch: Batch) -> ForwardInput:
        """Allocate root KV for new tokens and side-table KV for recurrent rows.

        DUO retains recurrent slots across tokens. Uniform releases them after
        sampling, while both modes advance the main sequence only on commit.
        """
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged([req for req in batch.reqs if not req.is_iterating])
        iter_reqs = [req for req in batch.reqs if req.is_iterating]
        iter_slots = self._tah_allocate_iter_slots(iter_reqs) if iter_reqs else None

        batch.positions = _make_positions_mixed(batch, self.device)

        reqs = batch.padded_reqs
        table_host = torch.tensor(
            [r.table_idx for r in reqs],
            dtype=torch.int64,
            pin_memory=True,
        )
        table_gpu = table_host.to(self.device, non_blocking=True)
        pos_host = torch.tensor(
            [r.device_len - 1 for r in reqs],
            dtype=torch.int64,
            pin_memory=True,
        )
        pos_gpu = pos_host.to(self.device, non_blocking=True)
        batch.out_loc = self.engine.page_table[table_gpu, pos_gpu].clone()
        if iter_slots is not None:
            rows = torch.tensor(
                [i for i, req in enumerate(reqs) if req.is_iterating],
                dtype=torch.int64,
                device=self.device,
            )
            batch.out_loc.index_copy_(0, rows, iter_slots)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=None,
            input_tuple=(table_gpu, batch.positions.to(torch.int64)),
            write_tuple=None,
        )

    def _tah_allocate_iter_slots(self, iter_reqs: List[Req]) -> torch.Tensor:
        """Allocate a fresh KV slot per iterating req into its current
        iteration's stream.

        A req at ``tah_iter_depth == d`` (cur_iter d) writes this tick's K/V
        into stream ``d`` (page table index ``d-1``).  Slots are grouped by
        target stream so each scatter targets one page-table tensor; the
        returned tensor preserves ``iter_reqs`` order (one slot per req, which
        is that req's out_loc this iteration).
        """
        n = len(iter_reqs)
        slots = self.cache_manager._allocate(n)  # (n,) int32 on GPU, iter_reqs order
        iter_pts = self.engine._tah_iter_page_tables
        # Group req indices by target stream (depth-1). For max_iter=3 there are
        # at most 2 distinct streams active in one batch.
        by_stream: dict[int, List[int]] = {}
        for i, r in enumerate(iter_reqs):
            d = r.tah_iter_depth
            while len(r.tah_iter_counts) < d:
                r.tah_iter_counts.append(0)
            by_stream.setdefault(d - 1, []).append(i)
        for s, idxs in by_stream.items():
            table_gpu = torch.tensor(
                [iter_reqs[i].table_idx for i in idxs],
                dtype=torch.int64,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            count_gpu = torch.tensor(
                [iter_reqs[i].tah_iter_counts[s] for i in idxs],
                dtype=torch.int64,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            idx_gpu = torch.tensor(
                idxs,
                dtype=torch.int64,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            iter_pts[s][table_gpu, count_gpu] = slots.index_select(0, idx_gpu)
        for r in iter_reqs:
            r.tah_iter_counts[r.tah_iter_depth - 1] += 1
        return slots

    def _schedule_next_batch(self) -> ForwardInput | None:
        # Relieve KV pressure before forming the batch so iterating reqs can get
        # their extra-iter slot without being forced to sample early. Victims are
        # removed from running_reqs here (excluded from this batch) and actually
        # freed in _finalize_preemptions after the in-flight batch retires.
        if self._dynamic_preempt:
            self._select_preempt_victims()

        # Finish recurrent tokens before admitting another prefill batch.
        if self.decode_manager.iterating_reqs:
            batch = self.decode_manager.schedule_next_batch()
            return self._prepare_batch(batch) if batch else None

        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _select_preempt_victims(self) -> None:
        """DUO dynamic preemption: when the KV pool nears full, mark the
        newest running reqs for preemption so the upcoming step has room to grow
        iter0 KV *and* allocate one extra-iter slot per iterating req without
        triggering the forced-early-sample fallback (which would silently lower
        avg_iter).

        Only non-iterating ``running_reqs`` are candidates — iterating reqs hold
        recurrent KV mid-iteration and are unsafe to preempt. Victims are removed
        from ``running_reqs`` here (so they are excluded from this step's batch)
        but not freed until ``_finalize_preemptions`` runs after the in-flight
        batch retires and their last sampled token has been appended.
        """
        running = self.decode_manager.running_reqs
        if not running:
            return
        # Headroom to survive one mixed-decode step without forcing early samples:
        # the schedule allocates up to ``active`` slots (decode growth + iter
        # slots), and finalize wants room for the continuing rows on top.
        active = len(running) + len(self.decode_manager.iterating_reqs)
        headroom = 2 * active + 64
        avail = self.cache_manager.available_size
        if avail >= headroom:
            return
        # Preempt newest-first (largest remain_len ≈ least work done) so older,
        # nearly-finished reqs keep running. Freeing a victim demotes its prefix
        # (cached_len) to the evictable radix, frees the decode tail (~1), and
        # releases its persisted extra-iter slots — all of which grow
        # available_size. Crediting the iter slots is essential: under pressure
        # they are the bulk of a victim's footprint, so omitting them would
        # massively over-preempt and erase the concurrency win.
        for req in sorted(running, key=lambda r: r.remain_len, reverse=True):
            if avail >= headroom:
                break
            running.discard(req)
            self._preempt_pending.append(req)
            avail += req.cached_len + 1 + req.tah_total_iter_slots

    def _finalize_preemptions(self) -> None:
        """Free + re-queue the reqs marked in ``_select_preempt_victims``.

        Runs after ``_process_last_data`` so every victim's last sampled token is
        already appended. Freeing demotes the victim's KV to the radix cache and
        releases its iter slots / table row; it is then re-queued internally (same
        uid, not via a UserMsg) so the open client stream resumes seamlessly from
        the next token — no re-tokenization, no duplicate prompt-info reply.
        """
        victims = self._preempt_pending
        self._preempt_pending = []
        for req in victims:
            # Skip reqs that finished or were freed during this step's bookkeeping.
            if getattr(req, "_freed", False) or not req.can_decode:
                continue
            device_len = req.device_len
            resume_ids = req.input_ids[:device_len].clone()
            new_params = replace(req.sampling_params, max_tokens=req.remain_len)
            uid = req.uid
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
            self.prefill_manager.pending_list.insert(
                0, PendingReq(uid=uid, input_ids=resume_ids, sampling_params=new_params)
            )
            self._num_preemptions += 1

    def _forward_dispatch(self, forward_input: ForwardInput) -> "PendingForward":
        """Queue GPU work.  TaH mixed returns a handle that ``_finalize``
        will sync + commit; baseline finishes here and ``_finalize`` is a
        pass-through.
        """
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]

        # TaH routes every decode batch through the mixed path so reqs that
        # need another iteration land in iterating_reqs automatically.
        if batch.is_mixed_decode or (batch.is_decode and self._is_tah_model):
            return PendingForward(
                is_mixed=True,
                forward_input=forward_input,
                pending_mixed=self.engine.forward_mixed_tah_dispatch(batch),
                forward_output=None,
            )
        assert sample_args is not None and output_mapping is not None
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # Baseline forward_batch called complete_one per req → filter now.
        self.decode_manager.filter_reqs(batch.reqs)
        return PendingForward(
            is_mixed=False,
            forward_input=forward_input,
            pending_mixed=None,
            forward_output=forward_output,
        )

    def _forward_finalize(
        self,
        pending: "PendingForward",
    ) -> ForwardOutput | MixedForwardOutput:
        """For TaH mixed: sync, commit iter state, scatter sampled tokens
        into ``token_pool`` in one batched write, then route reqs.  For
        baseline: pass through.
        """
        if not pending.is_mixed:
            assert pending.forward_output is not None
            return pending.forward_output

        assert pending.pending_mixed is not None
        forward_output = self.engine.forward_mixed_tah_finalize(pending.pending_mixed)
        sampled = forward_output.sampled_reqs
        if sampled:
            # complete_one() already advanced device_len inside finalize.
            write_table = torch.tensor(
                [r.table_idx for r in sampled],
                dtype=torch.int64,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            write_pos = torch.tensor(
                [r.device_len - 1 for r in sampled],
                dtype=torch.int64,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            self.token_pool[write_table, write_pos] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(pending.forward_input.batch.reqs)
        return forward_output

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput | MixedForwardOutput:
        pending = self._forward_dispatch(forward_input)
        return self._forward_finalize(pending)


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_positions_mixed(batch: Batch, device: torch.device) -> torch.Tensor:
    """Like ``_make_positions`` but iter>=1 reqs reuse their iter0 RoPE
    position.  Mixed/decode always has ``extend_len == 1``, so a list-comp
    + ``torch.tensor(list, pin_memory=True)`` suffices.
    """
    positions = [
        r.tah_iter0_position if r.is_iterating else r.device_len - 1 for r in batch.padded_reqs
    ]
    return torch.tensor(positions, dtype=torch.int32, pin_memory=True).to(
        device,
        non_blocking=True,
    )


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """Decode and prefill map differently: decode has ``extend_len == 1``
    per req (one table_idx per row, list-comp), prefill can have ``>1``
    (per-req ``fill_`` over a pre-allocated buffer).
    """
    reqs = batch.padded_reqs
    if batch.is_decode or batch.is_mixed_decode:
        mapping_host = torch.tensor(
            [r.table_idx for r in reqs],
            dtype=torch.int64,
            pin_memory=True,
        )
    else:
        mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
        offset = 0
        for req in reqs:
            length = req.extend_len
            mapping_host[offset : offset + length].fill_(req.table_idx)
            offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
