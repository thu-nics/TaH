from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

from tah2.minisgl.core import SamplingParams
from tah2.minisgl.distributed import DistributedInfo
from tah2.minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from tah2.minisgl.scheduler import Scheduler, SchedulerConfig
from tah2.minisgl.tokenizer.detokenize import DetokenizeManager


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


StreamCallback = Callable[[DetokenizeMsg, str], None]


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0
        self.pending_sampling_params = {}
        self._stream_callback: StreamCallback | None = None
        self._stream_detokenizer: DetokenizeManager | None = None

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.pending_sampling_params[uid] = sampling_params
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        incremental_outputs: List[str] | None = None
        if self._stream_callback is not None:
            assert self._stream_detokenizer is not None
            incremental_outputs = self._stream_detokenizer.detokenize(reply)

        for idx, msg in enumerate(reply):
            status = self.status_map[msg.uid]
            if self.pending_sampling_params[msg.uid].ignore_eos or not (
                msg.finished and msg.next_token == self.eos_token_id
            ):
                status.output_ids.append(msg.next_token)
            if self._stream_callback is not None:
                assert incremental_outputs is not None
                incremental = incremental_outputs[idx]
                self._stream_callback(msg, incremental)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
        stream_callback: StreamCallback | None = None,
    ) -> List[Dict[str, str | List[int]]]:
        if not isinstance(sampling_params, SamplingParams) and len(sampling_params) != len(prompts):
            raise ValueError("Provide one SamplingParams per prompt or a single shared SamplingParams.")
        self.pending_requests = []
        self.status_map = {}
        self.pending_sampling_params = {}
        self.counter = 0
        self._stream_callback = stream_callback
        self._stream_detokenizer = (
            DetokenizeManager(self.tokenizer) if stream_callback is not None else None
        )
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        finally:
            self._stream_callback = None
            self._stream_detokenizer = None
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
