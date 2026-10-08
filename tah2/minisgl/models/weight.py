from __future__ import annotations

import glob
import json
import os
from typing import Dict

import safetensors
import torch
from tqdm import tqdm

from tah2.minisgl.distributed import get_tp_info
from tah2.minisgl.utils import div_ceil, download_hf_weight


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]
    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = div_ceil(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value
    return shard_state_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if key.count(".q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif key.count(".gate_proj"):
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            continue
        else:
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def load_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}

    tp_info = get_tp_info()
    disable_tqdm = (tp_info.rank != 0) if tp_info.size > 1 else False
    device_str = "cpu" if tp_info.size > 1 else str(device)

    for file in tqdm(sorted(files), desc="Loading weights", disable=disable_tqdm):
        with safetensors.safe_open(file, framework="pt", device=device_str) as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    if tp_info.size > 1:
        state_dict = _shard_state_dict(state_dict)

    out = _merge_state_dict(state_dict)
    if tp_info.size > 1:
        # Move only this rank's merged shards to the device (see above).
        out = {k: v.to(device) for k, v in out.items()}
    return out


def is_tah_checkpoint(model_path: str) -> bool:
    """True iff ``model_path`` is a TaH checkpoint (``tah_config.json`` present
    with ``max_iter > 1``). Returns False for ordinary HF checkpoints.
    """
    model_folder = download_hf_weight(model_path, extra_patterns=["tah_config.json"])
    cfg_path = os.path.join(model_folder, "tah_config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if int(cfg.get("max_iter", 2)) <= 1:
        return False
    if cfg.get("iter_attention_mode", "causal") not in ("duo", "causal"):
        raise ValueError("Only TaH DUO and causal uniform attention are supported")
    if cfg.get("input_updater") != "Qwen3MLPUpdater":
        raise ValueError("TaH checkpoints require Qwen3MLPUpdater")
    return True


def load_tah_weights(model_path: str, model, device: torch.device) -> None:
    """Load TaH-specific weights if the model supports them.

    Called after ``model.load_state_dict()`` to load ``input_updater.bin``
    and ``tah_config.json`` from the checkpoint directory.
    """
    from .tah_qwen3 import TaHQwen3ForCausalLM

    if not isinstance(model, TaHQwen3ForCausalLM):
        return
    model_folder = download_hf_weight(
        model_path,
        extra_patterns=["tah_config.json", "input_updater.bin", "iter_decider.bin"],
    )
    model.load_tah_weights(model_folder, device)
