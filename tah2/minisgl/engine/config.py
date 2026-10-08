from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch

from tah2.minisgl.distributed import DistributedInfo
from tah2.minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from tah2.minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # TaH runtime overrides; the checkpoint selects the decider.
    tah_max_iter: int | None = None
    tah_iter_threshold: float | None = None
    tah_weighted_hidden_method: str | None = None
    tah_iter_decision: str = "threshold"  # "threshold" | "sample"

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from tah2.minisgl.models import ModelConfig, is_tah_checkpoint

        config = ModelConfig.from_hf(self.hf_config)
        if is_tah_checkpoint(self.model_path):
            config = replace(config, architectures=["TaHQwen3ForCausalLM"])
        # --max-seq-len-override past max_position must grow the RoPE cos/sin table
        # (layers/rotary.py sizes it from rotary_config.max_position). Plain
        # extrapolation, no YaRN.
        if (
            self.max_seq_len_override is not None
            and self.max_seq_len_override > config.rotary_config.max_position
        ):
            config = replace(
                config,
                rotary_config=replace(config.rotary_config, max_position=self.max_seq_len_override),
            )
        return config

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        if not hasattr(self, "_dist_port"):
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            object.__setattr__(self, "_dist_port", port)
        return f"tcp://127.0.0.1:{self._dist_port}"
