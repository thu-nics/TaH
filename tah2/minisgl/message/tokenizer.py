from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from tah2.minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool
    iter_count: int = 1
    prompt_iter_counts: List[int] | None = None
    stats: Dict[str, Any] | None = None


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams
    chat_template_kwargs: Dict[str, Any] | None = None


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
