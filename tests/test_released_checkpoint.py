"""End-to-end smoke test against the real released TaH-plus-1.7B checkpoint.

Marked ``slow``: requires ~3 GB checkpoint cached + a CUDA device with ~5 GB
free. Skipped when CUDA is unavailable. Validates the load → forward →
generate → save → reload chain that downstream consumers depend on.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch


CHECKPOINT = os.environ.get("TAH_CHECKPOINT", "nics-efc/TaH-plus-1.7B")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="released checkpoint smoke test needs CUDA"
)


@pytest.fixture(scope="module")
def model_and_tok():
    from transformers import AutoTokenizer
    from tah.model.tah_model import TaHForCausalLM

    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = TaHForCausalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    return model, tok


def test_forward_basic(model_and_tok):
    """A single forward should populate logits + iter_count without raising."""
    from tah.model.causal_cache import TaHCache

    model, tok = model_and_tok
    text = "What is 2 + 2? Answer with just the number."
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    assert out.logits.shape[0] == 1
    assert out.logits.shape[1] == inputs["input_ids"].shape[1]
    assert out.iter_count.shape == inputs["input_ids"].shape
    # iter_count is in [1, max_iter]; max_iter is 2 for this checkpoint.
    assert int(out.iter_count.min()) >= 1
    assert int(out.iter_count.max()) <= 2


def test_generate_short(model_and_tok):
    """Greedy 16-token generation must produce non-empty text without OOMing."""
    from tah.model.utils import TaHForCasualLM_generate

    model, tok = model_and_tok
    msg = [{"role": "user", "content": "Compute 17 + 25. Reply with a single integer."}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors="pt", padding=True, padding_side="left").to(model.device)

    output_tokens, output_texts = TaHForCasualLM_generate(
        tah_model=model,
        tokenizer=tok,
        model_inputs=dict(inputs),
        max_new_tokens=16,
        do_sample=False,
        verbose=False,
    )
    assert len(output_texts) == 1
    assert len(output_tokens[0]) > 0
    assert isinstance(output_texts[0], str)


def test_save_load_roundtrip_real(model_and_tok):
    """Save the loaded released model to a temp dir, reload it, forward must match."""
    from tah.model.tah_model import TaHForCausalLM

    model, tok = model_and_tok
    inputs = tok("The quick brown fox", return_tensors="pt").to(model.device)
    with torch.no_grad():
        ref_out = model(**inputs, use_cache=False)

    with tempfile.TemporaryDirectory(prefix="tah_release_smoke_") as tmp:
        model.save_pretrained(tmp)

        # Layout sanity: tah_config + iter_decider + lora + base must be on disk.
        assert os.path.isfile(os.path.join(tmp, "tah_config.json"))
        assert os.path.isfile(os.path.join(tmp, "iter_decider.bin"))
        assert os.path.isdir(os.path.join(tmp, "lora"))
        assert os.path.isfile(os.path.join(tmp, "lora", "adapter_config.json"))
        # Base safetensors (sharded or single-file).
        st_files = [f for f in os.listdir(tmp) if f.startswith("model") and f.endswith((".safetensors", ".json"))]
        assert any("safetensors" in f for f in st_files), f"no base safetensors in {tmp}"

        # Read back tah_config.json and verify shape.
        with open(os.path.join(tmp, "tah_config.json")) as f:
            cfg_dict = json.load(f)
        assert cfg_dict["iter_decider"] == "MLPIterDecider"
        assert cfg_dict["adapter"] == "lora"
        assert cfg_dict["max_iter"] == 2

        reloaded = TaHForCausalLM.from_pretrained(
            tmp, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
        )
        with torch.no_grad():
            new_out = reloaded(**inputs, use_cache=False)

        # bf16 + scattered ops give ~1e-2 drift; iter_count must match exactly.
        max_logit_diff = (new_out.logits.float() - ref_out.logits.float()).abs().max().item()
        assert max_logit_diff < 5e-2, f"reload drift {max_logit_diff:.4e}"
        assert torch.equal(new_out.iter_count, ref_out.iter_count), "iter_count drift after reload"

        del reloaded
        torch.cuda.empty_cache()
