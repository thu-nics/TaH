"""Smoke tests for the SFT pipeline.

Exercises ``CustomTaHDataCollator`` on a synthetic batch and runs 2 training
steps of ``CustomTaHTrainer`` against a 4-example synthetic dataset. Doesn't
load a real labelled dataset (avoids the multi-GB download); every other
production code path through ``tah/train`` and the wrapper's training-mode
forward IS exercised.
"""
from __future__ import annotations

import os
import tempfile

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SFT smoke needs CUDA")
BASE_MODEL = os.environ.get("TAH_TEST_BASE_MODEL", "Qwen/Qwen3-0.6B")


def test_data_collator_pads_iter_count_labels():
    """Pad iter_count_labels to the same length as input_ids, on the same side."""
    from transformers import AutoTokenizer
    from tah.train import CustomTaHDataCollator

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    features = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "iter_count_labels": [1, 2, 1]},
        {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5], "iter_count_labels": [1, 1, 2, 2, 1]},
    ]
    coll = CustomTaHDataCollator(tokenizer=tok, padding=True)
    batch = coll(features)
    assert batch["input_ids"].shape == batch["iter_count_labels"].shape, "iter_count_labels not aligned to input_ids"
    # First row should be padded with -100 in iter_count_labels
    assert int(batch["iter_count_labels"][0, -1].item()) == -100, "first row didn't pad with ignore index"
    # Second row should be unchanged
    assert int(batch["iter_count_labels"][1, 0].item()) == 1


def test_trainer_runs_two_steps_on_synthetic_dataset(tmp_path):
    """Build a 4-example synthetic dataset, run 2 SFT steps, verify a loss is logged."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from tah.model.tah_config import TaHConfig
    from tah.model.tah_model import TaHForCausalLM
    from tah.train import CustomTaHDataCollator, CustomTaHTrainer, LoggerCallback

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    rng = torch.Generator().manual_seed(311)
    examples = []
    for _ in range(4):
        T = int(torch.randint(8, 24, (1,), generator=rng).item())
        ids = torch.randint(10, 1000, (T,), generator=rng).tolist()
        labels = list(ids)
        labels[0] = -100  # mimic prompt-mask
        iter_labels = torch.randint(1, 3, (T,), generator=rng).tolist()
        examples.append({"input_ids": ids, "labels": labels, "iter_count_labels": iter_labels})
    ds = Dataset.from_list(examples)

    cfg = TaHConfig(
        embedding_key="model.embed_tokens",
        max_iter=2,
        input_updater_kwargs={"topk": 8},
        iter_decider="IterLabelDecider",
        iter_decider_kwargs={"max_iter": 2},
        eval_iter_decider=None,
        adapter="lora",
        adapter_kwargs={"r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
                        "target_modules": "all-linear", "bias": "none"},
        train_loss="NextTokenPredLoss",
        eval_loss="NextTokenPredLoss",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, attn_implementation="sdpa",
    ).to("cuda:0")
    model = TaHForCausalLM(base_model=base, config=cfg).to("cuda:0")

    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        bf16=False,
    )
    trainer = CustomTaHTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
    )
    trainer.data_collator = CustomTaHDataCollator(tokenizer=tok, padding=True)
    callback = LoggerCallback()
    model.logger_callback = callback
    trainer.callback_handler.callbacks.insert(0, callback)

    trainer.train()

    # Verify a loss appeared and is finite.
    losses = [e.get("loss") for e in trainer.state.log_history if "loss" in e]
    assert losses, f"no train loss in log_history: {trainer.state.log_history}"
    assert all(loss is not None and not (loss != loss) and loss < float("inf") for loss in losses), losses


def test_trainer_save_load_checkpoint(tmp_path):
    """After a step, _save must write a TaH-layout checkpoint we can reload."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from tah.model.tah_config import TaHConfig
    from tah.model.tah_model import TaHForCausalLM
    from tah.train import CustomTaHDataCollator, CustomTaHTrainer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    cfg = TaHConfig(
        embedding_key="model.embed_tokens",
        max_iter=2,
        input_updater_kwargs={"topk": 8},
        iter_decider="IterLabelDecider",
        iter_decider_kwargs={"max_iter": 2},
        adapter="lora",
        adapter_kwargs={"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                        "target_modules": "all-linear", "bias": "none"},
        train_loss="NextTokenPredLoss",
        eval_loss="NextTokenPredLoss",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda:0")
    model = TaHForCausalLM(base_model=base, config=cfg).to("cuda:0")

    ds = Dataset.from_list([
        {"input_ids": [10, 20, 30, 40], "labels": [-100, 20, 30, 40], "iter_count_labels": [1, 2, 1, 1]},
    ])
    args = TrainingArguments(
        output_dir=str(tmp_path / "run"),
        per_device_train_batch_size=1, max_steps=1, learning_rate=1e-5,
        save_strategy="steps", save_steps=1, save_total_limit=1,
        logging_steps=1, report_to="none", remove_unused_columns=False, bf16=False,
    )
    trainer = CustomTaHTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
    )
    trainer.data_collator = CustomTaHDataCollator(tokenizer=tok, padding=True)
    trainer.train()

    # find the checkpoint dir
    ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
    assert ckpts, f"no checkpoint dir under {args.output_dir}: {os.listdir(args.output_dir)}"
    ckpt = os.path.join(args.output_dir, ckpts[0])
    assert os.path.isfile(os.path.join(ckpt, "tah_config.json")), "tah_config.json missing"
    assert os.path.isfile(os.path.join(ckpt, "iter_decider.bin")), "iter_decider.bin missing"
    assert os.path.isdir(os.path.join(ckpt, "lora")), "lora/ missing"

    # Reload and forward — proves the saved layout is valid.
    reloaded = TaHForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda:0")
    inputs = tok("hello world", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = reloaded(**inputs, use_cache=False)
    assert out.logits.shape[0] == 1
