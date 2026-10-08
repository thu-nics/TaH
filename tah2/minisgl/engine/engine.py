from __future__ import annotations

import math
from datetime import timedelta
from typing import Any, Dict, List, NamedTuple, Tuple

import torch

from tah2.minisgl.attention import create_attention_backend
from tah2.minisgl.core import Batch, Context, Req, set_global_ctx
from tah2.minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from tah2.minisgl.env import ENV
from tah2.minisgl.kvcache import create_kvcache_pool
from tah2.minisgl.layers import set_rope_device
from tah2.minisgl.models import create_model, load_tah_weights, load_weight
from tah2.minisgl.utils import div_even, init_logger, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    iter_counts_gpu: torch.Tensor
    iter_counts_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class MixedForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor  # only for sampled reqs
    next_tokens_cpu: torch.Tensor  # async copy
    iter_counts_gpu: torch.Tensor  # only for sampled reqs
    iter_counts_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event
    sampled_reqs: List[Req]  # reqs that got tokens


class PendingMixedForward(NamedTuple):
    """Handle returned by ``forward_mixed_tah_dispatch`` before its sync.
    ``forward_mixed_tah_finalize`` performs the sync, partition, and
    per-req state commit.
    """

    batch: Batch
    bs: int
    needs_iter_cpu: torch.Tensor
    needs_iter_event: torch.cuda.Event
    next_tokens_all_gpu: torch.Tensor


class Engine:
    def __init__(self, config: EngineConfig):
        if torch.cuda.is_initialized():
            logger.warning(
                "CUDA was already initialized before Engine.__init__; "
                "this is expected in offline/eval mode but not in serving."
            )
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(ENV.SAMPLING_SEED.value)
        logger.info(f"sampling seed {ENV.SAMPLING_SEED.value}")
        self.stream = torch.cuda.Stream()
        # TaH: per-request prompt-side iteration counts from the duo prefill
        # cascade {uid: [iter_count_per_prompt_token, ...]}; popped by the
        # scheduler on the first sampled token and shipped in the usage payload.
        self.tah_prompt_iter_log: dict[int, list[int]] = {}
        # TaH: per-request iteration count log  {uid: [iter_count_per_decode_step, ...]}
        # Bounded to _TAH_ITER_LOG_MAX entries to prevent unbounded memory growth.
        self.tah_iter_log: dict[int, list[int]] = {}
        self._tah_iter_log_max: int = 8192
        # TaH: cache manager reference; set by Scheduler for mixed decode page management.
        self._tah_cache_manager = None
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ── TaH-specific weights (input_updater.bin, tah_config.json) ──
        if not config.use_dummy_weight:
            load_tah_weights(config.model_path, self.model, self.device)
        self._apply_tah_runtime_config(config)
        self._init_tah_state_pools(config)

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        # DUO mode: an iter-d query attends iter0 KV + persisted iter1..iter_d
        # KV.  The combined row is bounded by max_iter * max_seq_len; the attn
        # backend's captured page_table buffer is sized below accordingly.
        from tah2.minisgl.models.tah_qwen3 import TaHQwen3ForCausalLM

        # Persisted extra-iter KV side tables (duo mode).
        self._tah_has_iter_tables = isinstance(self.model, TaHQwen3ForCausalLM)
        if self._tah_has_iter_tables and config.page_size != 1:
            raise RuntimeError(
                "DUO TaH currently requires --page-size 1 because iter KV "
                "slots are tracked as token-level entries outside the radix page table."
            )
        # DUO iter>=1 reqs attend iter0 KV (<= L) plus one persisted slot per
        # extra iteration per token (<= (max_iter-1) * L). Size the attn
        # backend's captured page_table buffer for the worst-case combined row.
        self._tah_num_iter_streams = self.model.tah_max_iter - 1 if self._tah_has_iter_tables else 0
        attn_max_seq_len = (
            (self._tah_num_iter_streams + 1) * aligned_max_seq_len
            if self._tah_has_iter_tables and self.model.iter_attention_mode == "duo"
            else aligned_max_seq_len + self._tah_num_iter_streams
        )
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        if self._tah_has_iter_tables:
            # One page table per extra iteration (max_iter-1 streams), each
            # holding up to one KV slot per token position.
            self._tah_iter_page_tables = [
                torch.zeros(
                    (config.max_running_req + 1, aligned_max_seq_len),
                    dtype=torch.int32,
                    device=self.device,
                )
                for _ in range(self._tah_num_iter_streams)
            ]
            self.ctx.tah_iter_page_tables = self._tah_iter_page_tables
            self.ctx.tah_iter_attention_mode = self.model.iter_attention_mode
        else:
            self._tah_iter_page_tables = None

        # ======================= Attention backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        from tah2.minisgl.models.tah_qwen3 import TaHQwen3ForCausalLM

        _tah_hidden_size = (
            config.model_config.hidden_size if isinstance(self.model, TaHQwen3ForCausalLM) else None
        )
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=attn_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            hidden_size=_tah_hidden_size,
            dtype=self.dtype,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _init_tah_state_pools(self, config: EngineConfig) -> None:
        """Allocate per-``table_idx`` pools for TaH iteration state, plus
        persistent scratch buffers sized to the largest req batch.

        One slot per ``table_idx`` (+1 for padding).  Slots are only valid
        for iterating reqs; stale values in other slots are never read.
        The iter=0 embed anchor is NOT pooled — the scheduler keeps
        iterating rows pointed at their frozen ``tah_iter0_position``, so
        ``embed_tokens(batch.input_ids)`` already yields the anchor.
        """
        from tah2.minisgl.models.tah_qwen3 import TaHQwen3ForCausalLM

        self._tah_hidden_pool = None
        self._tah_log_accum_pool = None
        self._tah_rem_pool = None
        if not isinstance(self.model, TaHQwen3ForCausalLM):
            return
        n_slots = config.max_running_req + 1
        hidden_size = config.model_config.hidden_size
        kw = {"dtype": self.dtype, "device": self.device}
        # Stop-prob mixture state dtype (see ENV.TAH_MIX_FP32).
        self._tah_mix_dtype = (
            torch.float32
            if ENV.TAH_MIX_FP32.value and self._tah_weighted_method in ("stop_prob_mix", "even_mix")
            else self.dtype
        )
        mix_kw = {"dtype": self._tah_mix_dtype, "device": self.device}
        self._tah_hidden_pool = torch.empty(n_slots, hidden_size, **kw)
        if self._tah_weighted_method in ("stop_prob_mix", "even_mix"):
            # Per-row log-probability accumulator (size V). Larger than the hidden
            # pool; only allocated in log-prob mix modes.
            vocab_size = config.model_config.vocab_size
            self._tah_log_accum_pool = torch.empty(n_slots, vocab_size, **mix_kw)
        self._tah_rem_pool = torch.empty(n_slots, **mix_kw)

        # Per-step scratch, preallocated once to avoid per-iter
        # ``torch.tensor(list, device=cuda)`` stalls on a busy stream.
        max_bs = config.max_running_req
        pin = {"pin_memory": True}
        dev = {"device": self.device}
        self._tah_host_table_idx = torch.empty(max_bs, dtype=torch.int64, **pin)
        self._tah_device_table_idx = torch.empty(max_bs, dtype=torch.int64, **dev)
        self._tah_host_iter_depth = torch.empty(max_bs, dtype=torch.int32, **pin)
        self._tah_device_iter_depth = torch.empty(max_bs, dtype=torch.int32, **dev)
        self._tah_host_needs_iter = torch.empty(max_bs, dtype=torch.bool, **pin)
        self._tah_host_sample_pos = torch.empty(max_bs, dtype=torch.int64, **pin)
        self._tah_device_sample_pos = torch.empty(max_bs, dtype=torch.int64, **dev)
        self._tah_host_iterN_pos = torch.empty(max_bs, dtype=torch.int64, **pin)
        self._tah_device_iterN_pos = torch.empty(max_bs, dtype=torch.int64, **dev)
        # iter_counts is read CPU-side only; the GPU slot of MixedForwardOutput
        # is a stable empty placeholder.
        self._tah_empty_int32_gpu = torch.empty(0, dtype=torch.int32, **dev)

    def _apply_tah_runtime_config(self, config: EngineConfig) -> None:
        """Apply depth, mixture and gate overrides before CUDA graph capture."""
        from tah2.minisgl.models.tah_qwen3 import Qwen3MLPIterDecider, TaHQwen3ForCausalLM

        # Defaults so non-TaH paths have stable attributes; also used if we early-return.
        self._tah_weighted_method: str | None = None
        self._tah_iter_decision: str = "threshold"

        if not isinstance(self.model, TaHQwen3ForCausalLM):
            return

        if config.tah_max_iter is not None:
            self.model._tah_max_iter = int(config.tah_max_iter)

        effective_method = (
            config.tah_weighted_hidden_method
            if config.tah_weighted_hidden_method is not None
            else self.model.weighted_hidden_method
        )
        if effective_method not in ("stop_prob_mix", "even_mix"):
            raise RuntimeError(
                f"TaH serving requires weighted_hidden_method in "
                f"('stop_prob_mix', 'even_mix'); "
                f"got {effective_method!r}."
            )
        if self.model.iter_attention_mode == "causal" and effective_method != "even_mix":
            raise ValueError("Causal TaH in this build requires uniform even_mix recurrence")
        if effective_method != "even_mix" and not isinstance(
            self.model.tah_decider, Qwen3MLPIterDecider
        ):
            raise ValueError("DUO TaH requires a trained Qwen3MLPIterDecider")
        self._tah_weighted_method = effective_method
        self.model.weighted_hidden_method = effective_method

        if config.tah_iter_decision not in ("threshold", "sample"):
            raise RuntimeError(
                f"--tah-iter-decision must be 'threshold' or 'sample'; "
                f"got {config.tah_iter_decision!r}."
            )
        self._tah_iter_decision = config.tah_iter_decision
        decider = self.model.tah_decider
        if config.tah_iter_threshold is not None and hasattr(decider, "_threshold"):
            decider._threshold = float(config.tah_iter_threshold)
        # Log the effective threshold: the CLI value is a no-op on deciders
        # without `_threshold`, and load_tah_weights printed the pre-override one.
        eff_thr = getattr(decider, "_threshold", None)
        logger.info_rank0(
            f"TaH decider: {type(decider).__name__}; "
            f"max_iter: {self.model.tah_max_iter}; "
            f"weighted hidden: {self._tah_weighted_method}; "
            f"iter decision: {self._tah_iter_decision}"
            f"; effective iter threshold: {eff_thr}"
        )

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        # DUO TaH prefill runs a cascade of extra passes (iter1..iter_{max_iter-1})
        # over prompt tokens to populate per-iteration K/V at decider-selected
        # positions (decode batches go through forward_mixed_tah_*).
        run_duo_prefill = (
            self._tah_has_iter_tables
            and self.model.iter_attention_mode == "duo"
            and batch.is_prefill
            and not batch.is_mixed_decode
        )
        with self.ctx.forward_batch(batch):
            # TaH models only capture from-embeds decode graphs.
            use_graph = self.graph_runner.can_use_cuda_graph(batch) and not self.graph_runner.is_tah
            if use_graph:
                logits = self.graph_runner.replay(batch)
            elif batch.is_prefill and self._tah_weighted_method == "even_mix":
                hidden, logits, base_embeds = self.model.forward_with_hidden()
                from .tah_duo_prefill import run_uniform_prefill

                run_uniform_prefill(self, batch, hidden, logits, base_embeds)
            elif run_duo_prefill:
                hidden, logits, base_embeds = self.model.forward_with_hidden()
                from .tah_duo_prefill import run_duo_iter_prefill

                run_duo_iter_prefill(self, batch, hidden, logits, base_embeds)
            else:
                logits = self.model.forward()

        fixed_depth = self.model.tah_max_iter if self._tah_weighted_method == "even_mix" else 1
        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        iter_counts_gpu = torch.tensor(
            [
                self.tah_prompt_iter_log.get(req.uid, [fixed_depth])[-1]
                if batch.is_prefill
                else fixed_depth
                for req in batch.reqs
            ],
            dtype=torch.int32,
            device=self.device,
        )
        iter_counts_cpu = iter_counts_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(
            next_tokens_gpu, next_tokens_cpu, iter_counts_gpu, iter_counts_cpu, copy_done_event
        )

    # ------------------------------------------------------------------ #
    # Mixed-phase TaH forward
    # ------------------------------------------------------------------ #

    def forward_mixed_tah_dispatch(self, batch: Batch) -> PendingMixedForward:
        """Queue all GPU work (graph replay, decider, weighted-hidden accum,
        pool scatter, full-batch sample) and return a handle.  The scheduler
        calls :meth:`forward_mixed_tah_finalize` after running the previous
        iteration's reply bookkeeping so that CPU work overlaps the
        graph-replay wait.
        """
        from tah2.minisgl.models.tah_qwen3 import TaHQwen3ForCausalLM

        assert isinstance(self.model, TaHQwen3ForCausalLM)
        assert torch.cuda.current_stream() == self.stream

        model: TaHQwen3ForCausalLM = self.model
        bs = batch.size
        max_iter = model._tah_max_iter
        assert self._tah_hidden_pool is not None

        # ── classify reqs; fill persistent host/device index buffers ──
        # List-comps + torch.tensor(list) + copy_ stays on the CPython C
        # fast path; per-req Python loops over pinned tensors were 30-50µs
        # slower at bs=64.
        reqs = batch.reqs
        iterN_indices: List[int] = [i for i, r in enumerate(reqs) if r.is_iterating]
        h_tab = self._tah_host_table_idx[:bs]
        h_dep = self._tah_host_iter_depth[:bs]
        h_tab.copy_(torch.tensor([r.table_idx for r in reqs], dtype=torch.int64))
        h_dep.copy_(torch.tensor([r.tah_iter_depth for r in reqs], dtype=torch.int32))
        batch_table_t = self._tah_device_table_idx[:bs]
        batch_iter_depth_t = self._tah_device_iter_depth[:bs]
        batch_table_t.copy_(h_tab, non_blocking=True)
        batch_iter_depth_t.copy_(h_dep, non_blocking=True)

        # ── fill fused iter-graph inputs: prev_hidden for iterN rows + mask ──
        # The iter_graph takes ``batch.input_ids`` (carries iter=0 token
        # for iterating rows too, via ``tah_iter0_position``) and produces
        # the base anchor internally; we only need the gathered hidden for
        # iterN rows plus a per-row mask.
        assert self.graph_runner.can_use_cuda_graph(batch), (
            "TaH decode requires CUDA graphs; raise --cuda-graph-max-bs for larger batches."
        )
        prev_hidden_buf, mask_buf = self.graph_runner.get_tah_fused_buffers(batch)
        prev_hidden_buf.zero_()
        mask_buf.zero_()
        iterN_pos_t: torch.Tensor | None = None
        iterN_table_t: torch.Tensor | None = None
        if iterN_indices:
            n_iter = len(iterN_indices)
            h_pos = self._tah_host_iterN_pos[:n_iter]
            h_pos.copy_(torch.tensor(iterN_indices, dtype=torch.int64))
            iterN_pos_t = self._tah_device_iterN_pos[:n_iter]
            iterN_pos_t.copy_(h_pos, non_blocking=True)
            iterN_table_t = batch_table_t.index_select(0, iterN_pos_t)
            prev_hidden_buf.index_copy_(
                0,
                iterN_pos_t,
                self._tah_hidden_pool.index_select(0, iterN_table_t),
            )
            mask_buf.index_fill_(0, iterN_pos_t, True)

        # One fused replay: embed_tokens + updater + blend + layers + decider.
        hidden, logits, needs_iter, continue_probs = self.graph_runner.replay_tah_fused(batch)
        # Sample-mode decider decision overrides the threshold mask from the graph.
        # Run bernoulli outside the graph to avoid in-graph RNG gotchas; this is one
        # small extra kernel launch only in sample mode.
        if self._tah_iter_decision == "sample":
            needs_iter = torch.rand_like(continue_probs) < continue_probs
        if self._tah_weighted_method in ("even_mix",):
            # Fixed-iter: every token runs max_iter with weight 1/max_iter; decider ignored.
            needs_iter = batch_iter_depth_t < max_iter - 1
        else:
            needs_iter = needs_iter & (batch_iter_depth_t < max_iter - 1)
        continue_probs = continue_probs.to(dtype=self._tah_mix_dtype)

        # Kick off the decider D2H immediately and record the event: anything
        # queued below runs concurrently with the event.synchronize() wait
        # in ``_finalize``.
        needs_iter_cpu = self._tah_host_needs_iter[:bs]
        needs_iter_cpu.copy_(needs_iter, non_blocking=True)
        needs_iter_event = torch.cuda.Event()
        needs_iter_event.record(self.stream)

        # Weighted-hidden accumulators — full batch.  Rows that end up sampling
        # simply won't have their new pool values read next step (begin_iteration
        # overwrites first).
        sample_args = self.sampler.prepare(batch)
        rem_t = self._tah_rem_pool.new_ones(bs)
        if iterN_pos_t is not None:
            rem_t.index_copy_(
                0,
                iterN_pos_t,
                self._tah_rem_pool.index_select(0, iterN_table_t),
            )
        new_rem_t = rem_t * continue_probs

        # sample_logits_all is log-prob; sampler's softmax gives the mixture.
        log_p = torch.log_softmax(logits, dim=-1, dtype=self._tah_mix_dtype)
        log_accum_t = log_p.new_full(log_p.shape, float("-inf"))
        if iterN_pos_t is not None:
            log_accum_t.index_copy_(
                0,
                iterN_pos_t,
                self._tah_log_accum_pool.index_select(0, iterN_table_t),
            )
        if self._tah_weighted_method == "even_mix":
            # Partial mixtures at earlier iters are discarded.
            log_w_cont = log_p.new_full((bs, 1), -math.log(max_iter))
            log_w_smp = log_w_cont
        else:
            log_w_cont = torch.log((rem_t * (1.0 - continue_probs)).clamp_min(1e-10)).unsqueeze(
                -1
            )
            log_w_smp = torch.log(rem_t.clamp_min(1e-10)).unsqueeze(-1)
        new_log_accum_t = torch.logaddexp(log_accum_t, log_w_cont + log_p)
        sample_logits_all = torch.logaddexp(log_accum_t, log_w_smp + log_p)
        self._tah_log_accum_pool.index_copy_(0, batch_table_t, new_log_accum_t)

        # Scatter pools back for every row (method-agnostic).
        self._tah_hidden_pool.index_copy_(0, batch_table_t, hidden)
        self._tah_rem_pool.index_copy_(0, batch_table_t, new_rem_t)

        # Sample all rows; we'll slice out the sampled subset in _finalize.
        # At iter_ratio ~3% the wasted ~2% of sample GPU time is far less
        # than the CPU work we'd skip by gating post-sync.
        next_tokens_all_gpu = self.sampler.sample(
            sample_logits_all,
            sample_args,
        ).to(torch.int32)

        return PendingMixedForward(
            batch=batch,
            bs=bs,
            needs_iter_cpu=needs_iter_cpu,
            needs_iter_event=needs_iter_event,
            next_tokens_all_gpu=next_tokens_all_gpu,
        )

    def forward_mixed_tah_finalize(self, pending: PendingMixedForward) -> MixedForwardOutput:
        """Sync on the decider event, then partition reqs, commit state, and
        slice out the sampled subset of the full-batch sample result.
        """
        batch, bs = pending.batch, pending.bs
        cache_mgr = self._tah_cache_manager
        assert cache_mgr is not None
        pending.needs_iter_event.synchronize()

        # Partition into continue / sample.  The scatter+sample queued in
        # dispatch kept running concurrently with this event wait.
        needs_iter_list = pending.needs_iter_cpu.tolist()
        continue_indices: List[int] = []
        sample_indices: List[int] = []
        for i, v in enumerate(needs_iter_list):
            (continue_indices if v else sample_indices).append(i)

        # DUO extra-iter KV uses real pages on the *next* scheduler tick.  Each
        # continuing row allocates exactly one new slot (into its current
        # iteration's stream), regardless of depth.  Admission uses a tunable
        # reserve estimate, so under high pressure the decider can still produce
        # more continuing rows than the page pool can satisfy.  Instead of
        # letting the next _allocate() crash, force the overflow rows to sample
        # from the already-computed current-iter logits.
        if self.model.iter_attention_mode == "duo" and continue_indices:
            available_iter_slots = cache_mgr.available_size // cache_mgr.page_size
            if available_iter_slots < len(continue_indices):
                keep_n = max(int(available_iter_slots), 0)
                forced_sample = set(continue_indices[keep_n:])
                continue_indices = continue_indices[:keep_n]
                sample_indices = [
                    i for i in range(bs) if (not needs_iter_list[i]) or i in forced_sample
                ]
                logger.warning_rank0(
                    "DUO TaH iter page budget exhausted; forced %d/%d continuing "
                    "rows to sample early (available_slots=%d).",
                    len(forced_sample),
                    len(forced_sample) + len(continue_indices),
                    available_iter_slots,
                )
        sampled_reqs: List[Req] = [batch.reqs[i] for i in sample_indices]

        # Advance scalar iter state on continuing reqs.
        for i in continue_indices:
            req = batch.reqs[i]
            if req.is_iterating:
                req.tah_iter_depth += 1
            else:
                req.begin_iteration()

        # Uniform causal attention needs only this token's recurrent KV.
        # DUO retains every depth stream until the request finishes.
        freed_chunks = []
        sampled_iter_counts = []
        for req in sampled_reqs:
            sampled_iter_counts.append(req.tah_iter_depth + 1)
            if self.model.iter_attention_mode == "causal":
                for j, count in enumerate(req.tah_iter_counts):
                    if count:
                        freed_chunks.append(
                            self._tah_iter_page_tables[j][req.table_idx, :count].clone()
                        )
                req.tah_iter_counts = []
            req.finish_iteration()
            req.complete_one()
        if freed_chunks:
            cache_mgr._free(torch.cat(freed_chunks))

        # Bounded-FIFO iter log.
        overflow = len(self.tah_iter_log) + len(sampled_reqs) - self._tah_iter_log_max
        if overflow > 0:
            for k in list(self.tah_iter_log.keys())[:overflow]:
                del self.tah_iter_log[k]
        for req, n in zip(sampled_reqs, sampled_iter_counts):
            self.tah_iter_log.setdefault(req.uid, []).append(n)

        # Slice sampled tokens out of the full-batch result.  iter_counts
        # is never read on GPU — skip the round-trip and build the CPU
        # tensor directly.
        if sample_indices:
            h_sp = self._tah_host_sample_pos[: len(sample_indices)]
            h_sp.copy_(torch.tensor(sample_indices, dtype=torch.int64))
            sample_pos_gpu = self._tah_device_sample_pos[: len(sample_indices)]
            sample_pos_gpu.copy_(h_sp, non_blocking=True)
            next_tokens_gpu = pending.next_tokens_all_gpu.index_select(0, sample_pos_gpu)
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            iter_counts_cpu = torch.tensor(sampled_iter_counts, dtype=torch.int32)
        else:
            next_tokens_gpu = torch.empty(0, dtype=torch.int32, device=self.device)
            next_tokens_cpu = torch.empty(0, dtype=torch.int32, device="cpu")
            iter_counts_cpu = torch.empty(0, dtype=torch.int32, device="cpu")

        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return MixedForwardOutput(
            next_tokens_gpu,
            next_tokens_cpu,
            self._tah_empty_int32_gpu,
            iter_counts_cpu,
            copy_done_event,
            sampled_reqs,
        )

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        override("attention_backend", "fi")
