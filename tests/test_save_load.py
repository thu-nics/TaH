"""Save → load roundtrip for ``TaHForCausalLM``.

Pins the on-disk layout that downstream consumers depend on:

* ``tah_config.json`` — config (component names + kwargs).
* ``lora/`` — PEFT adapter directory (``adapter_model.safetensors``,
  ``adapter_config.json``).
* ``iter_decider.bin`` — pickled state dict + class name + init args.
* ``model.safetensors`` — base-model weights with cleaned state-dict keys
  (no ``.base_layer`` PEFT prefix; no ``lora_*`` weights).

A reload of the saved directory must produce a forward pass that matches the
original within fp tolerance.
"""
from __future__ import annotations

import os
import json
import shutil
import tempfile

import pytest
import torch

from tests._harness import assert_close


BASE_MODEL = os.environ.get("TAH_TEST_BASE_MODEL", "Qwen/Qwen3-0.6B")


def _build_inputs(device):
    g = torch.Generator(device=device).manual_seed(151)
    B, T = 1, 8
    input_ids = torch.randint(10, 1000, (B, T), generator=g, device=device, dtype=torch.long)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    return input_ids, attention_mask


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def initial_model(device):
    from transformers import AutoModelForCausalLM
    from tah.model.tah_config import TaHConfig
    from tah.model.tah_model import TaHForCausalLM

    cfg = TaHConfig(
        embedding_key="model.embed_tokens",
        max_iter=2,
        input_updater_kwargs={"topk": 8},
        iter_decider="MLPIterDecider",
        iter_decider_kwargs={
            "topk": 8,
            "hidden_states_size": 1024,  # matches Qwen3-0.6B hidden_size
            "hidden_states_layer_nums": [0, 4, 8, 12],
            "hidden_dims": [16, 16, 16, 16, 16, 16],
            "expansion_factor": 2,
            "dropout_rate": 0.0,
            "normalize_input": False,
            "threshold": 0.5,
            "max_iter": 2,
            "dtype": torch.float32,
        },
        eval_iter_decider=None,
        adapter="lora",
        adapter_kwargs={
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": "all-linear",
            "bias": "none",
        },
        train_loss="NextTokenPredLoss",
        train_loss_kwargs={},
        eval_loss="NextTokenPredLoss",
        eval_loss_kwargs={},
    )
    torch.manual_seed(13)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device).eval()
    torch.manual_seed(13)
    model = TaHForCausalLM(base_model=base, config=cfg).to(device).eval()
    return model, cfg


def test_save_load_roundtrip(initial_model, device):
    from tah.model.tah_model import TaHForCausalLM

    model, _cfg = initial_model
    input_ids, attn = _build_inputs(device)
    with torch.no_grad():
        ref_out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)

    with tempfile.TemporaryDirectory(prefix="tah_release_save_") as tmp:
        model.save_pretrained(tmp)
        # On-disk layout assertions
        assert os.path.isfile(os.path.join(tmp, "tah_config.json")), "tah_config.json missing"
        assert os.path.isfile(os.path.join(tmp, "iter_decider.bin")), "iter_decider.bin missing"
        assert os.path.isdir(os.path.join(tmp, "lora")), "lora/ missing"
        assert os.path.isfile(os.path.join(tmp, "lora", "adapter_config.json")), "adapter_config.json missing"
        # Base model files
        st = [f for f in os.listdir(tmp) if f.startswith("model") and (f.endswith(".safetensors") or f == "model.safetensors.index.json")]
        assert st, f"no base-model safetensors in {tmp}: {os.listdir(tmp)}"

        # tah_config.json shape
        with open(os.path.join(tmp, "tah_config.json")) as f:
            cfg_json = json.load(f)
        assert cfg_json["iter_decider"] == "MLPIterDecider"
        assert cfg_json["adapter"] == "lora"

        # Reload
        torch.manual_seed(0)
        reloaded = TaHForCausalLM.from_pretrained(
            tmp, torch_dtype=torch.float32, attn_implementation="sdpa"
        ).to(device).eval()

        with torch.no_grad():
            new_out = reloaded(input_ids=input_ids, attention_mask=attn, use_cache=False)

    assert_close("logits", new_out.logits, ref_out.logits, atol=1e-4, rtol=1e-3)
    assert_close("iter_count", new_out.iter_count, ref_out.iter_count, atol=0)
