"""Reference precision after RoPE growth, short reset, and device/dtype cast."""
import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from tah.engine.config import qwen_config
from tah.engine.rope import RotaryEmbedding


@pytest.mark.parametrize("scaling", [
    {"rope_type": "dynamic", "factor": 2.},
    {"rope_type": "longrope", "factor": 2., "short_factor": [1.] * 4, "long_factor": [2.] * 4},
])
def test_scaled_rope_growth_and_reset_match_reference(scaling, monkeypatch):
    initialized_before = torch.cuda.is_initialized()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    def forbidden(*args, **kwargs):
        raise AssertionError("RoPE development test must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, head_dim=8, num_attention_heads=4,
                      num_key_value_heads=2, num_hidden_layers=2, max_position_embeddings=16,
                      rope_scaling=scaling)
    reference = Qwen3RotaryEmbedding(cfg).to(dtype=torch.bfloat16)
    engine = RotaryEmbedding(qwen_config(cfg.to_dict())).to(dtype=torch.bfloat16)
    assert engine.original_inv_freq.dtype == reference.original_inv_freq.dtype == torch.float32
    for length in (7, 21, 7, 31, 3):
        positions = torch.arange(length)
        hidden = torch.zeros(1, length, 32, dtype=torch.bfloat16)
        expected = reference(hidden, positions[None])
        actual = engine(hidden, positions)
        torch.testing.assert_close(engine.inv_freq, reference.inv_freq, rtol=0, atol=0)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.cuda.is_initialized() == initialized_before
