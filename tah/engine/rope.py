"""Qwen3 RoPE, including the scaling schemes in the frozen reference.

Formulas follow Transformers 4.52.4 modeling_rope_utils (Apache-2.0,
Copyright 2024 The HuggingFace Team); this module imports only torch.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def frequencies(config, device=None, seq_len=None):
    dim, base = config["head_dim"], config["rope_theta"]
    scaling = config["rope_scaling"] or {}
    kind = scaling.get("rope_type", scaling.get("type", "default"))
    factor = scaling.get("factor", 1.)
    length = config["max_position_embeddings"]
    attention = 1.
    exponent = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    if kind == "dynamic":
        if dim <= 2:
            raise ValueError("dynamic RoPE requires head_dim > 2")
        current = max(seq_len or length, length)
        base = base * ((factor * current / length) - (factor - 1)) ** (dim / (dim - 2))
    inv = 1. / (base ** exponent)
    if kind in ("default", "dynamic"):
        return inv, attention
    if kind == "linear":
        return inv / factor, attention
    if kind == "llama3":
        low, high, original = (scaling[k] for k in (
            "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"))
        wavelength = 2 * math.pi / inv
        adjusted = torch.where(wavelength > original / low, inv / factor, inv)
        smooth = (original / wavelength - low) / (high - low)
        interpolated = (1 - smooth) * adjusted / factor + smooth * adjusted
        medium = ~(wavelength < original / high) & ~(wavelength > original / low)
        return torch.where(medium, interpolated, adjusted), attention
    if kind == "longrope":
        original = config.get("original_max_position_embeddings", length)
        if "original_max_position_embeddings" in config:
            factor = length / original
        attention = scaling.get("attention_factor")
        if attention is None:
            attention = 1. if factor <= 1 else math.sqrt(1 + math.log(factor) / math.log(original))
        ext = scaling["long_factor"] if seq_len and seq_len > original else scaling["short_factor"]
        if len(ext) != dim // 2:
            raise ValueError("longrope factors must have head_dim/2 entries")
        return 1. / (torch.tensor(ext, dtype=torch.float32, device=device) * base ** exponent), attention
    if kind == "yarn":
        original = scaling.get("original_max_position_embeddings", length)
        if "original_max_position_embeddings" in scaling:
            factor = length / original
        def mscale(scale, multiplier=1):
            return 1. if scale <= 1 else 0.1 * multiplier * math.log(scale) + 1.
        attention = scaling.get("attention_factor")
        if attention is None:
            a, b = scaling.get("mscale"), scaling.get("mscale_all_dim")
            attention = mscale(factor, a) / mscale(factor, b) if a and b else mscale(factor)
        fast, slow = scaling.get("beta_fast") or 32, scaling.get("beta_slow") or 1
        def correction(rotations):
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(math.floor(correction(fast)), 0), min(math.ceil(correction(slow)), dim - 1)
        if low == high:
            high += .001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low)).clamp(0, 1)
        extrapolation = 1 - ramp
        interpolation = 1. / (factor * base ** exponent)
        return interpolation * (1 - extrapolation) + inv * extrapolation, attention
    raise ValueError(f"unsupported rope_type {kind!r}")


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kind = (config["rope_scaling"] or {}).get("rope_type", (config["rope_scaling"] or {}).get("type", "default"))
        inv, self.attention_scaling = frequencies(config)
        self.register_buffer("inv_freq", inv, persistent=False)
        # HF keeps the original frequencies as an ordinary fp32 tensor:
        # model.to(bf16) casts inv_freq, but reset restores the fp32 original.
        self.original_inv_freq = inv
        self.max_seq_len_cached = config["max_position_embeddings"]

    @torch.no_grad()
    def forward(self, x, positions, *, sequence_length=None):
        if self.kind in ("dynamic", "longrope"):
            length = int(positions.max()) + 1 if sequence_length is None else sequence_length
            original = self.config["max_position_embeddings"]
            if self.kind == "dynamic":
                if length > self.max_seq_len_cached:
                    # HF reinitializes grown frequencies in fp32 even when
                    # the original buffer has been cast to bf16 by eval.
                    self.inv_freq, self.attention_scaling = frequencies(self.config, x.device, length)
                    self.max_seq_len_cached = length
                if length < original and self.max_seq_len_cached > original:
                    self.original_inv_freq = self.original_inv_freq.to(x.device)
                    self.inv_freq = self.original_inv_freq
                    self.max_seq_len_cached = original
            elif length > self.config.get("original_max_position_embeddings", original):
                self.inv_freq, _ = frequencies(self.config, x.device, length)
            else:
                self.original_inv_freq = self.original_inv_freq.to(x.device)
                self.inv_freq = self.original_inv_freq
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (self.inv_freq[None, :, None].float() @ positions[None, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.unsqueeze(-2) + rotated * sin.unsqueeze(-2)
