from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from tah2.minisgl.models import ModelConfig


def validate_attn_backend(backend: str, allow_auto: bool = True) -> str:
    if backend not in (("auto", "fi") if allow_auto else ("fi",)):
        raise ValueError("This build uses FlashInfer attention: choose 'auto' or 'fi'")
    return backend


def create_attention_backend(backend: str, config: ModelConfig) -> BaseAttnBackend:
    from .fi import FlashInferBackend

    validate_attn_backend(backend, allow_auto=False)
    return FlashInferBackend(config)


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "create_attention_backend",
    "validate_attn_backend",
]
