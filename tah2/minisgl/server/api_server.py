from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tah2.minisgl.core import SamplingParams
from tah2.minisgl.env import ENV
from tah2.minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    PromptInfoReply,
    TokenizeMsg,
    UserReply,
)
from tah2.minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[BaseFrontendMsg]:
    if isinstance(msg, BatchFrontendMsg):
        return list(msg.data)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(ge=1)
    ignore_eos: bool = False
    chat_template_kwargs: Dict[str, Any] | None = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0)

    top_k: int = -1
    top_p: float = Field(default=1.0, gt=0, le=1)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stop: List[str] = Field(default_factory=list, max_length=0)
    presence_penalty: float = Field(default=0.0, ge=0, le=0)
    frequency_penalty: float = Field(default=0.0, ge=0, le=0)

    ignore_eos: bool = False
    chat_template_kwargs: Dict[str, Any] | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)
    # Served sampling seed (MINISGL_SAMPLING_SEED), so eval clients can record
    # the seed that actually produced a run instead of trusting their own env.
    sampling_seed: int | None = None
    # "adaptive" = TaH decider routes per token (per-token iter_counts are real);
    # "fixed" = uniform recurrence or a single pass; fixed_depth counts passes.
    iter_mode: str | None = None
    fixed_depth: float | None = None


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    prompt_token_ids_map: Dict[int, List[int] | None] = field(default_factory=dict)
    prompt_info_event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    tah_total_tokens: int = 0
    tah_iterated_tokens: int = 0  # iter_count > 1
    tah_total_iters: int = 0
    tah_max_iter_observed: int = 1
    tah_iter_hist: Dict[int, int] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self.prompt_token_ids_map[uid] = None
        self.prompt_info_event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                uid = getattr(msg, "uid", None)
                if uid not in self.ack_map:
                    continue
                if isinstance(msg, PromptInfoReply):
                    self.prompt_token_ids_map[msg.uid] = list(msg.prompt_token_ids)
                    if msg.uid in self.prompt_info_event_map:
                        self.prompt_info_event_map[msg.uid].set()
                    continue
                assert isinstance(msg, UserReply)
                iter_count = max(int(msg.iter_count), 1)
                self.tah_total_tokens += 1
                self.tah_total_iters += iter_count
                if iter_count >= 2:
                    self.tah_iterated_tokens += 1
                self.tah_max_iter_observed = max(self.tah_max_iter_observed, iter_count)
                self.tah_iter_hist[iter_count] = self.tah_iter_hist.get(iter_count, 0) + 1
                self.ack_map[msg.uid].append(msg)
                if msg.stats is not None:
                    self.log_completion_stats(msg.stats)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    def log_completion_stats(self, stats: Dict[str, Any]) -> None:
        logger.info(
            f"DP{self.config.dp}: "
            f"#running-req: {stats['running_req']}, "
            f"#queue-req: {stats['queue_req']}, "
            f"#token: {stats['used_tokens']}, "
            f"#available-token: {stats['available_tokens']}, "
            f"#reserved-token: {stats['reserved_tokens']}, "
            f"token usage: {stats['token_usage']:.2f}, "
            f"admission usage: {stats['admission_usage']:.2f}, "
            f"gen throughput (token/s): {stats['gen_throughput']:.2f}, "
            f"#preempt: {stats.get('num_preemptions', 0)}"
        )

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        self.prompt_token_ids_map.pop(uid, None)
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        iter_counts: List[int] = []
        prompt_iter_counts: List[int] | None = None
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output
            iter_counts.append(ack.iter_count)
            if ack.prompt_iter_counts is not None:
                prompt_iter_counts = ack.prompt_iter_counts

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.config.served_model_name,
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.config.served_model_name,
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            # Same TaH per-token stats as the non-stream path, so streaming
            # benchmark clients can compute decode/prompt iter ratios per request.
            "usage": {
                "iter_counts": iter_counts,
                "prompt_iter_counts": prompt_iter_counts or [],
            },
        }
        self.prompt_token_ids_map.pop(uid, None)
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def chat_completion(self, uid: int):
        content_parts = []
        iter_counts = []
        prompt_iter_counts = None
        token_ids = []
        async for ack in self.wait_for_ack(uid):
            if ack.incremental_output:
                content_parts.append(ack.incremental_output)
            iter_counts.append(ack.iter_count)
            if ack.prompt_iter_counts is not None:
                prompt_iter_counts = ack.prompt_iter_counts
            if ack.token_id is not None:
                token_ids.append(int(ack.token_id))
            if ack.finished:
                break
        if uid in self.prompt_info_event_map:
            await self.prompt_info_event_map.pop(uid).wait()
        prompt_token_ids = self.prompt_token_ids_map.pop(uid, None)

        return {
            "id": f"cmpl-{uid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "".join(content_parts),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "iter_counts": iter_counts,
                "prompt_iter_counts": prompt_iter_counts or [],
                "token_ids": token_ids,
                "prompt_token_ids": prompt_token_ids or [],
            },
        }

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        if uid in self.prompt_token_ids_map:
            del self.prompt_token_ids_map[uid]
        if uid in self.prompt_info_event_map:
            del self.prompt_info_event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()

    def get_tah_stats(self) -> Dict[str, float | int]:
        tokens = self.tah_total_tokens
        iterated = self.tah_iterated_tokens
        total_iters = self.tah_total_iters
        total_extra_iters = max(total_iters - tokens, 0)
        avg_iter = (total_iters / tokens) if tokens > 0 else 0.0
        iterated_ratio = (iterated / tokens) if tokens > 0 else 0.0
        avg_extra_iter = (total_extra_iters / tokens) if tokens > 0 else 0.0
        iter_hist_json = {
            str(k): v for k, v in sorted(self.tah_iter_hist.items(), key=lambda x: x[0])
        }
        return {
            "total_tokens": tokens,
            "iterated_tokens": iterated,
            "total_iters": total_iters,
            "total_extra_iters": total_extra_iters,
            "iterated_ratio": iterated_ratio,
            "avg_iter": avg_iter,
            "avg_extra_iter": avg_extra_iter,
            "max_iter_observed": self.tah_max_iter_observed,
            "iter_hist": iter_hist_json,
            # Backward compatibility aliases
            "iter2_tokens": iterated,
            "iter_ratio": iterated_ratio,
        }


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
            chat_template_kwargs=req.chat_template_kwargs,
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        if req.prompt is None:
            raise HTTPException(status_code=400, detail="Either 'messages' or 'prompt' must be provided")
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
            chat_template_kwargs=req.chat_template_kwargs,
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    return await state.chat_completion(uid)


@app.post("/v1/completions")
async def v1_text_completions(req: OpenAICompletionRequest, request: Request):
    """Legacy text-completion shape, for clients that pre-render the prompt.

    BFCL's OSS handlers call `client.completions.create(...)`, so they need this
    route rather than /v1/chat/completions. The body is the same request model
    and the same generation path -- only the response envelope differs, so the
    openai-python client can validate it as a `Completion`.
    """
    if req.prompt is None:
        raise HTTPException(status_code=400, detail="'prompt' must be provided")
    if req.stream:
        raise HTTPException(status_code=400, detail="streaming is not supported on /v1/completions")

    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
            chat_template_kwargs=req.chat_template_kwargs,
        )
    )
    body = await state.chat_completion(uid)
    usage = body["usage"]
    prompt_tokens = len(usage["prompt_token_ids"])
    completion_tokens = len(usage["token_ids"])
    return {
        "id": body["id"],
        "object": "text_completion",
        "created": body["created"],
        "model": body["model"],
        "choices": [
            {
                "index": 0,
                "text": body["choices"][0]["message"]["content"],
                "logprobs": None,
                "finish_reason": body["choices"][0]["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # kept so TaH iter stats survive this route too
            "iter_counts": usage["iter_counts"],
            "prompt_iter_counts": usage.get("prompt_iter_counts", []),
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    mc = state.config.model_config
    # A TaH* arch means the checkpoint's max_iter > 1; --tah-max-iter 1 still
    # serves it single-pass.
    tah_cfg = {}
    from pathlib import Path

    cfg_path = Path(state.config.model_path) / "tah_config.json"
    if cfg_path.exists():
        tah_cfg = json.loads(cfg_path.read_text())
    depth = state.config.tah_max_iter or int(tah_cfg.get("max_iter", 1))
    uniform = (
        state.config.tah_weighted_hidden_method or tah_cfg.get("weighted_hidden_method")
    ) == "even_mix"
    adaptive = mc.architectures[0].startswith("TaH") and not uniform and depth > 1
    return ModelList(
        data=[ModelCard(id=state.config.served_model_name, root=state.config.served_model_name)],
        sampling_seed=int(ENV.SAMPLING_SEED.value),
        iter_mode="adaptive" if adaptive else "fixed",
        fixed_depth=None if adaptive else (float(depth) if uniform else 1.0),
    )


@app.get("/v1/tah/stats")
async def tah_stats():
    state = get_global_state()
    return state.get_tah_stats()


def run_api_server(
    config: ServerArgs,
    start_backend: Callable[[], None],
) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
    """

    global _GLOBAL_STATE

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    uvicorn.run(app, host=host, port=port, access_log=False)
