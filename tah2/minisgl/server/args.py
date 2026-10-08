from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List

import torch

from tah2.minisgl.distributed import DistributedInfo
from tah2.minisgl.scheduler import SchedulerConfig
from tah2.minisgl.scheduler.config import _ipc_dir


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    served_model_name: str = "minisgl"
    distributed_port: int | None = None
    dp: int = 0
    num_tokenizer: int = 0
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return f"ipc://{_ipc_dir()}/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = f"ipc://{_ipc_dir()}/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        port = (
            self.distributed_port if self.distributed_port is not None else (self.server_port + 1)
        )
        return f"tcp://127.0.0.1:{port}"


def parse_args(args: List[str]) -> ServerArgs:
    from tah2.minisgl.attention import validate_attn_backend
    from tah2.minisgl.kvcache import SUPPORTED_CACHE_MANAGER

    parser = argparse.ArgumentParser(description="MiniSGL recurrent Qwen3 inference")
    parser.add_argument("--model-path", "--model", required=True)
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto"
    )
    parser.add_argument("--tensor-parallel-size", "--tp-size", type=int, default=1)
    parser.add_argument("--host", dest="server_host", default=ServerArgs.server_host)
    parser.add_argument("--port", dest="server_port", type=int, default=ServerArgs.server_port)
    parser.add_argument("--distributed-port", type=int, default=None)
    parser.add_argument("--served-model-name", default=ServerArgs.served_model_name)
    for flag, field_name in [
        ("max-running-requests", "max_running_req"),
        ("max-seq-len-override", "max_seq_len_override"),
        ("cuda-graph-max-bs", "cuda_graph_max_bs"),
        ("max-prefill-length", "max_extend_tokens"),
        ("num-pages", "num_page_override"),
        ("num-tokenizer", "num_tokenizer"),
        ("tah-max-iter", "tah_max_iter"),
        ("dp", "dp"),
    ]:
        parser.add_argument(
            f"--{flag}", dest=field_name, type=int, default=getattr(ServerArgs, field_name)
        )
    parser.add_argument("--page-size", type=int, choices=[1], default=1)
    parser.add_argument("--memory-ratio", type=float, default=ServerArgs.memory_ratio)
    parser.add_argument("--attention-backend", "--attn", type=validate_attn_backend, default="auto")
    parser.add_argument(
        "--cache-type",
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        default=ServerArgs.cache_type,
    )
    parser.add_argument("--disable-pynccl", action="store_false", dest="use_pynccl", default=True)
    parser.add_argument("--dummy-weight", action="store_true", dest="use_dummy_weight")
    parser.add_argument("--tah-iter-threshold", type=float, default=None)
    parser.add_argument(
        "--tah-weighted-hidden-method", choices=["stop_prob_mix", "even_mix"], default=None
    )
    parser.add_argument("--tah-iter-decision", choices=["threshold", "sample"], default="threshold")
    parser.add_argument(
        "--tah-duo-iter1-reserve-factor",
        type=float,
        default=ServerArgs.tah_duo_iter1_reserve_factor,
    )
    parser.add_argument("--tah-dynamic-preempt", action="store_true")
    kwargs = vars(parser.parse_args(args))
    if kwargs["tah_max_iter"] is not None and kwargs["tah_max_iter"] < 1:
        parser.error("--tah-max-iter must be >= 1")
    if kwargs["tah_duo_iter1_reserve_factor"] < 0:
        parser.error("--tah-duo-iter1-reserve-factor must be >= 0")
    if not 0 < kwargs["memory_ratio"] <= 1:
        parser.error("--memory-ratio must be in (0, 1]")
    kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])
    dtype = kwargs["dtype"]
    if dtype == "auto":
        from tah2.minisgl.utils import cached_load_hf_config

        dtype = cached_load_hf_config(kwargs["model_path"]).dtype
    kwargs["dtype"] = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    kwargs["tp_info"] = DistributedInfo(0, kwargs.pop("tensor_parallel_size"))
    return ServerArgs(**kwargs)
