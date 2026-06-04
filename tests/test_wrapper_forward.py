"""End-to-end acc test for ``TaHForCausalLM.forward``.

Drops a small real Qwen3-0.6B-Base into the wrapper, runs one forward in
training mode (labels + iter_count_labels supplied), and asserts the cleaned
package produces the same loss / logits / iter_count tensors as public TaH.

This test is the single most important regression gate — every refactor
inside the wrapper has to leave this green.

Cost: ~3-5s per run on a B200 once Qwen3-0.6B is in HF cache.
"""
from __future__ import annotations

import os

import pytest
import torch

from tests._harness import (
    assert_close,
    capture,
    have_baseline,
    load_baseline,
)


BASE_MODEL = os.environ.get("TAH_TEST_BASE_MODEL", "Qwen/Qwen3-0.6B")
DTYPE_STR = "float32"  # CPU-friendly + numerically tight


def _wrapper_inputs(device):
    g = torch.Generator(device=device).manual_seed(131)
    # Small batch to keep the test fast.
    B, T = 2, 16
    input_ids = torch.randint(10, 1000, (B, T), generator=g, device=device, dtype=torch.long)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    attention_mask[1, :2] = 0  # left pad on row 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    iter_count_labels = torch.randint(0, 3, (B, T), generator=g, device=device, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "iter_count_labels": iter_count_labels,
    }


CONFIG_DICT = {
    "embedding_key": "model.embed_tokens",
    "max_iter": 2,
    "input_updater_kwargs": {"topk": 8},
    "iter_decider": "IterLabelDecider",
    "iter_decider_kwargs": {"max_iter": 2},
    "eval_iter_decider": None,
    "eval_iter_decider_kwargs": {},
    "adapter": "lora",
    "adapter_kwargs": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "bias": "none",
    },
    "train_loss": "NextTokenPredLoss",
    "train_loss_kwargs": {},
    "eval_loss": "NextTokenPredLoss",
    "eval_loss_kwargs": {},
}


# Public-TaH config: superset of the cleaned config dict above. Includes the
# inert fields (output_updater, iter_label_generator, iter_attention_mode,
# input_updater) that the public dataclass requires. Behavior must match the
# cleaned package since the dropped slots have only one used implementation.
_PUBLIC_CONFIG_DICT = {
    **CONFIG_DICT,
    "input_updater": "TrivialUpdater",
    "output_updater": "AdditiveLogitsUpdater",
    "output_updater_kwargs": {},
    "iter_label_generator": "FixedIterLabelGenerator",
    "iter_label_generator_kwargs": {},
    "iter_attention_mode": "duo",
}

WRAPPER_RUNNER = (
    "def run(payload):\n"
    "    import torch\n"
    "    from transformers import AutoModelForCausalLM\n"
    "    from tah.model.tah_config import TaHConfig\n"
    "    from tah.model.recurrent_transformer import TaHForCausalLM\n"
    "    device = payload['device']\n"
    "    torch.manual_seed(11)\n"
    f"    base = AutoModelForCausalLM.from_pretrained({BASE_MODEL!r}, torch_dtype=torch.{DTYPE_STR}, attn_implementation='sdpa')\n"
    "    base = base.to(device).eval()\n"
    f"    cfg = TaHConfig(**{_PUBLIC_CONFIG_DICT!r})\n"
    "    torch.manual_seed(11)\n"
    "    model = TaHForCausalLM(base_model=base, config=cfg).to(device).eval()\n"
    "    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in payload.items()}\n"
    "    out = model(\n"
    "        input_ids=inputs['input_ids'],\n"
    "        attention_mask=inputs['attention_mask'],\n"
    "        labels=inputs['labels'],\n"
    "        iter_count_labels=inputs['iter_count_labels'],\n"
    "        use_cache=False,\n"
    "    )\n"
    "    return {\n"
    "        'loss': out.loss.detach() if out.loss is not None else None,\n"
    "        'logits': out.logits.detach(),\n"
    "        'iter_count': out.iter_count,\n"
    "    }\n"
)


@pytest.fixture(scope="module")
def baseline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = _wrapper_inputs(device)
    inputs = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()}
    inputs["device"] = device
    name = "wrapper_forward_qwen3_0.6b"
    if not have_baseline(name):
        capture(name, WRAPPER_RUNNER, payload=inputs)
    return load_baseline(name)


def test_wrapper_forward_acc(baseline, device):
    from transformers import AutoModelForCausalLM
    from tah.model.tah_config import TaHConfig
    from tah.model.tah_model import TaHForCausalLM

    args = {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in baseline["args"].items()}

    torch.manual_seed(11)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=getattr(torch, DTYPE_STR), attn_implementation="sdpa"
    )
    base = base.to(device).eval()
    cfg = TaHConfig(**CONFIG_DICT)
    torch.manual_seed(11)
    model = TaHForCausalLM(base_model=base, config=cfg).to(device).eval()
    out = model(
        input_ids=args["input_ids"],
        attention_mask=args["attention_mask"],
        labels=args["labels"],
        iter_count_labels=args["iter_count_labels"],
        use_cache=False,
    )

    # Tolerances: cleaned version may reorder mathematically equivalent ops
    # (e.g., gather/scatter) which can introduce ~1e-4 drift in fp32. Tighten
    # later if specific paths warrant it.
    if baseline["out"]["loss"] is not None:
        assert_close("loss", out.loss, baseline["out"]["loss"], atol=1e-4, rtol=1e-3)
    assert_close("logits", out.logits, baseline["out"]["logits"], atol=2e-3, rtol=1e-3)
    assert_close("iter_count", out.iter_count, baseline["out"]["iter_count"], atol=0)
