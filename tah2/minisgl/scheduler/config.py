from __future__ import annotations

from dataclasses import dataclass, field

from tah2.minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


def _ipc_dir() -> str:
    """Directory for minisgl's ZMQ ``ipc://`` socket files. Defaults to ``/tmp`` (the
    historic location); override with ``MINISGL_IPC_DIR`` when the shared /tmp overlay is
    full — a full /tmp makes even the tiny ipc socket bind fail with ``ENOSPC`` and the
    server dies at startup. e.g. ``MINISGL_IPC_DIR=/dev/shm``."""
    import os

    return os.environ.get("MINISGL_IPC_DIR", "/tmp")


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    # DUO TaH admission reserve for projected iter1 KV growth. Only applied
    # when the loaded TaH checkpoint uses iter_attention_mode="duo".
    tah_duo_iter1_reserve_factor: float = 0.5
    # DUO TaH: replace the static reserve factor with runtime preemption
    # keyed on real-time KV occupancy. When set, admission becomes greedy
    # (reserve factor forced to 0) and the scheduler demotes the newest running
    # requests to the radix cache when the KV pool approaches full, so iterating
    # requests keep their full iteration budget instead of being forced to
    # sample early. No per-model tuning required.
    tah_dynamic_preempt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return f"ipc://{_ipc_dir()}/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return f"ipc://{_ipc_dir()}/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return f"ipc://{_ipc_dir()}/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
