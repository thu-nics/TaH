from dataclasses import fields
from typing import Dict

import torch
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tah2.model.tah_config import TaHConfig
from tah2.model.tah_model import TaHForCausalLM
from tah2.utils.model_io import load_input_updater, load_iter_decider


def build_tah_config(model_config: Dict) -> TaHConfig:
    """Build TaHConfig from the recipe's model block."""
    tah_config = TaHConfig(
        embedding_key=model_config.get("embedding_key", "model.embed_tokens")
    )
    for f in fields(TaHConfig):
        if f.name == "embedding_key":
            continue
        if f.name in model_config:
            setattr(tah_config, f.name, model_config[f.name])
    return tah_config


def load_model_and_tokenizer(
    training_config: Dict, model_config: Dict, accelerator: Accelerator
):
    """Load model/tokenizer from config while preserving original behavior."""
    del training_config
    accelerator.print("Loading model and tokenizer...")

    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping.get(model_config["torch_dtype"], torch.bfloat16)
    # torchrun/FSDP2 places each model on its rank's device after loading.
    device_map = None

    if "tah_model_path" in model_config:
        accelerator.print(
            f"Loading pretrained TaH model from: {model_config['tah_model_path']}"
        )

        tah_config = None
        tah_config_fields = [field.name for field in fields(TaHConfig)]
        if any(key in model_config for key in tah_config_fields):
            override_config_dict = {}
            for field in tah_config_fields:
                if field in model_config:
                    override_config_dict[field] = model_config[field]
            tah_config = TaHConfig(**override_config_dict)
            accelerator.print("Using TaH config from YAML to override saved config:")
            accelerator.print(f"TaH config override: {override_config_dict}")
        else:
            accelerator.print("Using saved TaH config from pretrained model")

        model = TaHForCausalLM.from_pretrained(
            model_config["tah_model_path"],
            tah_config=tah_config,
        ).to(dtype=torch_dtype)

        # Tokenizer from the ORIGINAL model dir, not the checkpoint: a
        # save_pretrained round-trip re-serializes tokenizer.json with the
        # transformers-4.57 fixed pre-tokenizer regex, which tokenizes some
        # strings differently -> resume would re-tokenize the dataset with
        # slightly different labels and miss the preprocessing cache.
        tokenizer_path = model_config.get("name", model_config.get("tah_model_path"))
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_config.get("trust_remote_code", True),
            padding_side="right",
            fix_mistral_regex=True,
        )
        accelerator.print("Successfully loaded pretrained TaH model")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"],
            trust_remote_code=model_config["trust_remote_code"],
            padding_side="right",
            fix_mistral_regex=True,
        )

        tah_config = build_tah_config(model_config)

        base_config = AutoConfig.from_pretrained(
            model_config["name"],
            trust_remote_code=model_config.get("trust_remote_code", True),
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            config=base_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=model_config["trust_remote_code"],
            attn_implementation=model_config["attn_implementation"],
        )

        if "load_path" in tah_config.iter_decider_kwargs:
            iter_decider_path = tah_config.iter_decider_kwargs.pop("load_path")
            model = TaHForCausalLM(
                base_model=base_model, config=tah_config, device_map=device_map
            )
            model.iter_decider = load_iter_decider(iter_decider_path)
        elif "load_path" in tah_config.input_updater_kwargs:
            input_updater_path = tah_config.input_updater_kwargs.pop("load_path")
            model = TaHForCausalLM(
                base_model=base_model, config=tah_config, device_map=device_map
            )
            model.input_updater = load_input_updater(input_updater_path)
        else:
            model = TaHForCausalLM(
                base_model=base_model, config=tah_config, device_map=device_map
            )

        # TaH components may create params in their own dtype (e.g. the MLP
        # decider defaults to bf16); FSDP requires a uniform original dtype.
        model = model.to(dtype=torch_dtype)

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def maybe_enable_gradient_checkpointing(
    model, training_config: Dict, accelerator: Accelerator
):
    if not training_config.get("gradient_checkpointing", False):
        return

    kwargs = training_config.get("gradient_checkpointing_kwargs", {}) or {}
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(kwargs)
        except TypeError:
            model.gradient_checkpointing_enable()
        accelerator.print(f"Enabled gradient checkpointing. kwargs={kwargs}")

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
