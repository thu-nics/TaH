"""Full-graph Inductor decode step with an opaque paged-attention operator."""
from __future__ import annotations

import os
from pathlib import Path
import time
import weakref

import torch
from torch import Tensor, nn

from .sampling import GraphSampler, RequestRNG


_ATTENTION_PLANS = {}


@torch.library.custom_op("tah_engine::decode_attention",
                         mutates_args={"attended"})
def _decode_attention(q: Tensor, k: Tensor, v: Tensor, slots: Tensor,
                      attended: Tensor,
                      plan_key: Tensor, layer: int) -> None:
    """Write this layer's K/V and run its preplanned FlashInfer attention."""
    attention_ref = _ATTENTION_PLANS.get(plan_key.data_ptr())
    attention = attention_ref() if attention_ref is not None else None
    if attention is None:
        raise RuntimeError("compiled attention plan is no longer alive")
    kv_values = attention.pool.values
    kv_values[layer, 0, :, 0].index_copy_(0, slots, k)
    kv_values[layer, 1, :, 0].index_copy_(0, slots, v)
    attention.wrapper.run(q, (kv_values[layer, 0], kv_values[layer, 1]), out=attended)


@_decode_attention.register_fake
def _decode_attention_fake(q, k, v, slots, attended, plan_key, layer):
    return None


def register_attention_plan(attention):
    key = torch.empty((), dtype=torch.uint8, device=attention.pool.device)
    _ATTENTION_PLANS[key.data_ptr()] = weakref.ref(attention)
    return key


def configure_compile_cache(directory=None):
    """Choose a stable TorchInductor cache location before the first compile."""
    configured = directory or os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if configured is None:
        configured = Path.home() / ".cache" / "torch" / "inductor"
    path = Path(configured).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(path)
    return str(path)


class CompiledDecodeStep(nn.Module):
    """One dynamic-shape full graph; the caller keeps exact-B CUDA graphs."""

    def __init__(self, engine, cache_dir=None):
        super().__init__()
        self.model = engine.model
        self.decider = engine.decider
        self.decider_indices = engine.decider_indices
        self.bounded_sampler = GraphSampler(self.model.config["vocab_size"],
                                            max_top_k=engine.sampling_topk)
        self.general_sampler = GraphSampler(self.model.config["vocab_size"])
        self.input_topk = self.model.input_topk
        self.max_iter = self.model.max_iter
        self.cache_dir = configure_compile_cache(cache_dir)
        self.compile_seconds = 0.0
        self.compile_graphs = 0
        self._bounded = torch.compile(self._bounded_step, fullgraph=True, dynamic=True)
        self._general = torch.compile(self._general_step, fullgraph=True, dynamic=True)

    def _pass(self, embeddings, positions, depths, slots, rotary_cos, rotary_sin,
              attended, attention_key):
        model = self.model
        cfg = model.config
        x = embeddings[None]
        lora_mask = (depths >= 2)[None]
        hidden = []
        for index, layer in enumerate(model.model.layers):
            hidden.append(x)
            q, k, v = layer.self_attn.project(layer.input_layernorm(x),
                                               rotary_cos, rotary_sin, lora_mask)
            q = q[0].transpose(0, 1).contiguous()
            k = k[0].transpose(0, 1).contiguous()
            v = v[0].transpose(0, 1).contiguous()
            torch.ops.tah_engine.decode_attention.default(
                q, k, v, slots, attended, attention_key, index)
            x = x + layer.self_attn.o_proj(attended.reshape(1, embeddings.shape[0], -1), lora_mask)
            x = x + layer.mlp(layer.post_attention_layernorm(x), lora_mask)
        x = model.model.norm(x)
        hidden.append(x)
        return model.lm_head(x, lora_mask)[0], torch.stack(hidden, dim=-2)[0]

    def _step(self, sampler, embeddings, positions, depths, slots, logits, rows,
              engine_logits, engine_embeddings, engine_counters, seeds, counters,
              rotary_cos, rotary_sin, attended, attention_key, params):
        model = self.model
        increment, hidden = self._pass(embeddings, positions, depths, slots,
                                       rotary_cos, rotary_sin, attended, attention_key)
        logits.add_(increment)  # preserve model-dtype cumulative accumulation
        keep, raw, probability = self.decider(logits, depths, hidden,
                                              hidden_indices=self.decider_indices)
        keep = keep & (depths < self.max_iter)
        probability = torch.where(depths < self.max_iter, probability, 0.)
        rng = RequestRNG(seeds, counters)
        sampled = sampler.sample(logits, params, rng, active=~keep)
        if self.max_iter > 1:
            values, indices = logits.topk(min(self.input_topk, logits.shape[-1]), dim=-1)
            soft = (values.softmax(-1)[..., None] * model.model.embed_tokens.weight[indices]).sum(-2).to(model.dtype)
            next_input = torch.where(keep[:, None], soft,
                model.model.embed_tokens(sampled.clamp_min(0), False))
        else:
            next_input = model.model.embed_tokens(sampled.clamp_min(0), False)
        engine_logits.index_copy_(0, rows, torch.where(keep[:, None], logits, 0.))
        engine_embeddings.index_copy_(0, rows, next_input)
        engine_counters.index_copy_(0, rows, counters)
        status = torch.stack((keep.double(), sampled.double(), raw.double(), probability.double()), -1)
        return status

    def _bounded_step(self, embeddings, positions, depths, slots, logits, rows,
                      engine_logits, engine_embeddings, engine_counters, seeds, counters,
                      rotary_cos, rotary_sin, attended, attention_key, params):
        return self._step(self.bounded_sampler, embeddings, positions, depths, slots, logits,
            rows, engine_logits, engine_embeddings, engine_counters, seeds, counters,
            rotary_cos, rotary_sin, attended, attention_key, params)

    def _general_step(self, embeddings, positions, depths, slots, logits, rows,
                      engine_logits, engine_embeddings, engine_counters, seeds, counters,
                      rotary_cos, rotary_sin, attended, attention_key, params):
        return self._step(self.general_sampler, embeddings, positions, depths, slots, logits,
            rows, engine_logits, engine_embeddings, engine_counters, seeds, counters,
            rotary_cos, rotary_sin, attended, attention_key, params)

    @torch.inference_mode()
    def run(self, bucket, engine, rotary):
        function = self._bounded if bucket.sampling_key[1] else self._general
        before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        started = time.perf_counter()
        status = function(bucket.embeddings, bucket.positions, bucket.depths, bucket.slots,
            bucket.logits, bucket.rows, engine.logits, engine.embeddings, engine.counters,
            bucket.rng.seeds, bucket.rng.counters, rotary[0], rotary[1],
            bucket.attention.output, bucket.attention_key, bucket.params)
        elapsed = time.perf_counter() - started
        after = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        if after > before:
            self.compile_graphs += after - before
            self.compile_seconds += elapsed
        return status
