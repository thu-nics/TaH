from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from tqdm import tqdm

from tah2.minisgl.core import Batch, Req, get_global_ctx
from tah2.minisgl.distributed import get_tp_info
from tah2.minisgl.utils import init_logger

if TYPE_CHECKING:
    from tah2.minisgl.attention import BaseAttnBackend
    from tah2.minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    # TaH-only buffers: iter_graph replays ``embed_tokens + updater +
    # where(iterN_mask, updated, base) + layers + lm_head + decider``.
    hidden: torch.Tensor | None = None
    prev_hidden_blob: torch.Tensor | None = None
    iterN_mask: torch.Tensor | None = None
    needs_iter: torch.Tensor | None = None
    continue_probs: torch.Tensor | None = None

    @classmethod
    def init(
        cls,
        bs: int,
        vocab_size: int,
        device: torch.device,
        hidden_size: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> GraphCaptureBuffer:
        tah = hidden_size is not None
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            hidden=torch.empty(bs, hidden_size, dtype=dtype, device=device) if tah else None,
            prev_hidden_blob=torch.empty(bs, hidden_size, dtype=dtype, device=device)
            if tah
            else None,
            iterN_mask=torch.empty(bs, dtype=torch.bool, device=device) if tah else None,
            needs_iter=torch.empty(bs, dtype=torch.bool, device=device) if tah else None,
            continue_probs=torch.empty(bs, dtype=torch.float32, device=device) if tah else None,
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [
        bs for bs in [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8)) if bs <= cuda_graph_max_bs
    ]


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        hidden_size: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self.is_tah = hidden_size is not None
        self._capture_graphs(max_seq_len, vocab_size, model, hidden_size, dtype)

    def _capture_graphs(
        self,
        max_seq_len: int,
        vocab_size: int,
        model: BaseLLMModel,
        hidden_size: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.iter_graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs,
            vocab_size,
            self.device,
            hidden_size,
            dtype,
        )

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph() if not self.is_tah else None
            iter_graph = torch.cuda.CUDAGraph() if self.is_tah else None
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                if self.is_tah:
                    assert self.buffer.prev_hidden_blob is not None
                    assert self.buffer.iterN_mask is not None
                    # Zero inputs so capture sees deterministic intermediates
                    # (prev_hidden=0, mask=False → all rows take base embeds).
                    self.buffer.prev_hidden_blob[:bs].zero_()
                    self.buffer.iterN_mask[:bs].fill_(False)
                    self._run_tah_capture_body(model, bs)  # warm-up
                    assert iter_graph is not None
                    with torch.cuda.graph(iter_graph, pool=pool, stream=self.stream):
                        self._run_tah_capture_body(model, bs)
                    if pool is None:
                        pool = iter_graph.pool()
                else:
                    self.buffer.logits[:bs] = model.forward()
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        self.buffer.logits[:bs] = model.forward()
                    if pool is None:
                        pool = graph.pool()
            if graph is not None:
                self.graph_map[bs] = graph
            if iter_graph is not None:
                self.iter_graph_map[bs] = iter_graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _run_tah_capture_body(self, model: BaseLLMModel, bs: int) -> None:
        """Body of the TaH iter-graph capture (and its warm-up).

        The graph runs, in order:

            base    = embed_tokens(input_ids)          # anchor for all rows
            updated = updater(base, prev_hidden_blob)  # updater on all rows
            embeds  = where(iterN_mask, updated, base) # per-row pick
            h, l    = layers(embeds) + lm_head
            n, c    = decider(l, h, base)              # base is the anchor

        Fusing this sequence collapses ~12 out-of-graph kernel launches
        into one replay.  Running the updater on the full batch is a
        tiny extra cost (memory-bound on weights) and removes all the
        dynamic indexing that the engine used to do.
        """
        assert self.buffer.hidden is not None
        assert self.buffer.prev_hidden_blob is not None
        assert self.buffer.iterN_mask is not None
        assert self.buffer.needs_iter is not None
        assert self.buffer.continue_probs is not None
        base = model.model.embed_tokens.forward(self.buffer.input_ids[:bs])  # type: ignore[attr-defined]
        updated = model.compute_iter_embeds(  # type: ignore[attr-defined]
            base,
            self.buffer.prev_hidden_blob[:bs],
        )
        embeds = torch.where(self.buffer.iterN_mask[:bs].unsqueeze(-1), updated, base)
        h2, l2 = model.forward_from_embeds_with_hidden(embeds)  # type: ignore[attr-defined]
        self.buffer.hidden[:bs] = h2
        self.buffer.logits[:bs] = l2
        ni, cp = model.tah_decider.forward(  # type: ignore[attr-defined]
            l2,
            hidden_states=h2,
            input_embeds=base,
            return_continue_probs=True,
        )
        self.buffer.needs_iter[:bs] = ni
        self.buffer.continue_probs[:bs] = cp

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return not batch.is_prefill and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        assert not self.is_tah, "TaH models use replay_tah_fused() for decode."
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def get_tah_fused_buffers(
        self,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prev_hidden_blob, iterN_mask)`` slices for the caller
        to fill before :meth:`replay_tah_fused`.  Padding rows are forced
        to ``False`` so they always take the base-embed path.
        """
        assert self.is_tah and self.can_use_cuda_graph(batch)
        assert self.buffer.prev_hidden_blob is not None
        assert self.buffer.iterN_mask is not None
        bs, pbs = batch.size, batch.padded_size
        if bs < pbs:
            self.buffer.iterN_mask[bs:pbs].fill_(False)
        return self.buffer.prev_hidden_blob[:bs], self.buffer.iterN_mask[:bs]

    def replay_tah_fused(self, batch: Batch) -> Tuple[torch.Tensor, ...]:
        """Replay the TaH iter graph.  Assumes :meth:`get_tah_fused_buffers`
        was already filled.  Returns ``(hidden, logits, needs_iter, continue_probs)``.
        """
        assert self.is_tah and self.can_use_cuda_graph(batch)
        assert self.buffer.hidden is not None and self.buffer.needs_iter is not None
        assert self.buffer.continue_probs is not None
        self.buffer.copy_from(batch)
        bs = batch.size
        self.attn_backend.prepare_for_replay(batch)
        self.iter_graph_map[batch.padded_size].replay()
        return (
            self.buffer.hidden[:bs],
            self.buffer.logits[:bs],
            self.buffer.needs_iter[:bs],
            self.buffer.continue_probs[:bs],
        )

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        del self.iter_graph_map
        gc.collect()
