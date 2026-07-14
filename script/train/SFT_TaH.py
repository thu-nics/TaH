"""TaH SFT entrypoint.

Loads a YAML recipe (see ``script/recipes/qwen3_1.7/sft_tah_step{1,2}.yaml``),
constructs the wrapper, runs HuggingFace Trainer.

Step 1 — train the LoRA adapter against oracle iteration labels
         (iter_decider=IterLabelDecider, train_loss=NextTokenPredLoss).

Step 2 — train the iter decider on top of the frozen base+adapter
         (iter_decider=MLPIterDecider, train_loss=IterDeciderLoss,
          freeze_component=[model.simple_base_model]).

Run via ``accelerate launch`` so DeepSpeed / DDP wrappers are in place:

    python -m accelerate.commands.launch \
        --config_file ./script/recipes/accelerate_configs/zero2.yaml \
        --num_processes 8 \
        ./script/train/SFT_TaH.py \
        --config ./script/recipes/qwen3_1.7/sft_tah_step1.yaml
"""
from __future__ import annotations

import argparse
import os
from dataclasses import fields
from datetime import datetime
from typing import Dict

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from tah.model.iter_decider import load_iter_decider
from tah.model.tah_config import TaHConfig
from tah.model.tah_model import TaHForCausalLM
from tah.model.utils import compute_trainable_param_size_gb, freeze_components, set_all_seeds
from tah.train import CustomTaHDataCollator, CustomTaHTrainer, LoggerCallback
from tah.utils.data_prepare import preprocess_dataset

set_all_seeds(420)


_DTYPE_BY_STR = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _load_yaml(path: str) -> Dict:
    print(f"Loading configuration from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _configure_special_tokens(model, tokenizer, model_config: Dict) -> None:
    """Keep tokenizer and saved checkpoint special-token metadata aligned."""
    eos_token = model_config.get("eos_token") or tokenizer.eos_token
    pad_token = model_config.get("pad_token") or tokenizer.pad_token or eos_token

    def require_existing_single_token(name: str, token: str) -> int:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if (
            len(token_ids) != 1
            or tokenizer.convert_ids_to_tokens(token_ids[0]) != token
        ):
            raise ValueError(
                f"{name} must be an existing single token: {token!r} -> {token_ids}"
            )
        return token_ids[0]

    eos_token_id = require_existing_single_token("eos_token", eos_token)
    pad_token_id = require_existing_single_token("pad_token", pad_token)

    tokenizer.eos_token = eos_token
    tokenizer.pad_token = pad_token

    # TaHForCausalLM.save_pretrained() saves this underlying Qwen model at
    # checkpoint root, so its config and generation_config must be updated in
    # addition to the wrapper's copies.
    checkpoint_model = model.simple_base_model.base_model.model
    for target in (model, checkpoint_model):
        target.config.eos_token_id = eos_token_id
        target.config.pad_token_id = pad_token_id

        generation_config = getattr(target, "generation_config", None)
        if generation_config is not None:
            generation_config.eos_token_id = eos_token_id
            generation_config.pad_token_id = pad_token_id


def _build_model_and_tokenizer(model_config: Dict, accelerator: Accelerator):
    """Either resume an existing TaH checkpoint or build a fresh wrapper."""
    accelerator.print("Loading model and tokenizer…")
    torch_dtype = _DTYPE_BY_STR.get(model_config["torch_dtype"], torch.bfloat16)
    # accelerate handles device placement; never pass device_map="auto" here.
    device_map = None

    # Tokenizer location: use ``name`` if given, else fall back to the TaH ckpt.
    tok_path = model_config.get("name", model_config.get("tah_model_path"))
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, trust_remote_code=model_config.get("trust_remote_code", True), padding_side="right",
    )

    valid = {f.name for f in fields(TaHConfig)}
    if "tah_model_path" in model_config:
        accelerator.print(f"Resuming from TaH checkpoint: {model_config['tah_model_path']}")
        # Override only the fields explicitly set in the new YAML.
        override = TaHConfig(**{k: v for k, v in model_config.items() if k in valid})
        model = TaHForCausalLM.from_pretrained(
            model_config["tah_model_path"], tah_config=override,
        ).to(dtype=torch_dtype)
    else:
        # Construct fresh from a base model + recipe-specified components.
        cfg = TaHConfig(**{k: v for k, v in model_config.items() if k in valid})
        base = AutoModelForCausalLM.from_pretrained(
            model_config["name"], torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=model_config.get("trust_remote_code", True),
            attn_implementation=model_config.get("attn_implementation", "sdpa"),
        )
        if "load_path" in (cfg.iter_decider_kwargs or {}):
            iter_decider_path = cfg.iter_decider_kwargs.pop("load_path")
            model = TaHForCausalLM(base_model=base, config=cfg)
            model.iter_decider = load_iter_decider(iter_decider_path)
        else:
            model = TaHForCausalLM(base_model=base, config=cfg)

    _configure_special_tokens(model, tokenizer, model_config)
    accelerator.print(
        f"EOS: {tokenizer.eos_token!r} ({tokenizer.eos_token_id}); "
        f"PAD: {tokenizer.pad_token!r} ({tokenizer.pad_token_id})"
    )
    return model, tokenizer


def _build_training_args(training_config: Dict, data_config: Dict, output_dir: str, timestamp: str) -> TrainingArguments:
    args = {
        "output_dir": output_dir,
        "num_train_epochs": training_config["num_train_epochs"],
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "gradient_checkpointing": training_config.get("gradient_checkpointing", False),
        "learning_rate": training_config["learning_rate"],
        "warmup_ratio": training_config["warmup_ratio"],
        "weight_decay": training_config["weight_decay"],
        "max_grad_norm": training_config["max_grad_norm"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "lr_scheduler_kwargs": training_config["lr_scheduler_kwargs"],
        "logging_steps": training_config["logging_steps"],
        "save_strategy": training_config["save_strategy"],
        "save_steps": training_config.get("save_steps", 100),
        "save_only_model": training_config["save_only_model"],
        "save_total_limit": training_config["save_total_limit"],
        "report_to": training_config["report_to"],
        "bf16": training_config["bf16"],
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
    }

    if data_config.get("eval_data_path") or data_config.get("eval_data_ratio", 0.0) > 0:
        args.update({
            "eval_strategy": training_config.get("eval_strategy"),
            "eval_steps": training_config.get("eval_steps"),
            "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size"),
            "eval_on_start": training_config.get("eval_on_start"),
        })

    if training_config["report_to"] == "wandb":
        for env_key, cfg_key in (("WANDB_PROJECT", "wandb_project"), ("WANDB_NAME", "wandb_name"), ("WANDB_ENTITY", "wandb_entity")):
            if cfg_key in training_config:
                os.environ[env_key] = training_config[cfg_key]
        args["run_name"] = training_config.get("wandb_name", f"training_{timestamp}")

    return TrainingArguments(**args)


def main(config: Dict):
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb" if os.environ.get("WANDB_MODE") != "disabled" else None,
    )

    # Single timestamp shared across all ranks.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if accelerator.is_main_process else None
    ts_holder = [timestamp]
    broadcast_object_list(ts_holder)
    timestamp = ts_holder[0]

    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]

    # Output dir: continue-training puts everything under continue_training/<base>/<ts>;
    # from-scratch puts under <base>_<iter_decider>_<adapter>/<ts>.
    base_name = (model_config.get("name") or model_config["tah_model_path"]).split("/")[-1]
    if "tah_model_path" in model_config:
        output_dir = os.path.join(data_config["output_dir"], "continue_training", base_name, timestamp)
    else:
        decider = (model_config.get("iter_decider") or "decider").rsplit("Decider", 1)[0] or "decider"
        output_dir = os.path.join(
            data_config["output_dir"], f"{base_name}_{decider}_{model_config.get('adapter', 'lora')}", timestamp,
        )

    model, tokenizer = _build_model_and_tokenizer(model_config, accelerator)

    freeze_list = training_config.get("freeze_component") or []
    if isinstance(freeze_list, str):
        freeze_list = [freeze_list]
    if freeze_list:
        accelerator.print(f"Freezing components: {freeze_list}")
        freeze_components(model, freeze_list, accelerator)
    accelerator.print(f"Trainable parameter size: {compute_trainable_param_size_gb(model):.3f} GB")

    train_ds, eval_ds = preprocess_dataset(training_config, data_config, model_config, accelerator)

    training_args = _build_training_args(training_config, data_config, output_dir, timestamp)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "training_config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        accelerator.print(f"Configuration saved to: {output_dir}/training_config.yaml")

    trainer = CustomTaHTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
    )
    trainer.data_collator = CustomTaHDataCollator(tokenizer=tokenizer, padding=True)

    callback = LoggerCallback()
    model.logger_callback = callback
    trainer.callback_handler.callbacks.insert(0, callback)

    accelerator.print("\n--- Starting Training ---")
    resume_path = model_config["tah_model_path"] if training_config.get("resume_from_ckpt") and "tah_model_path" in model_config else None
    if resume_path:
        accelerator.print(f"Resuming optimizer state from: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()
    accelerator.print("--- Training Complete ---")

    final_dir = os.path.join(output_dir, "final_model")
    accelerator.print(f"Saving final model to: {final_dir}")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TaH SFT entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML recipe")
    args = parser.parse_args()
    main(_load_yaml(args.config))
