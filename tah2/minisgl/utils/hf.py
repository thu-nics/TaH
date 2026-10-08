import functools
import os
from typing import Any

from huggingface_hub import snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
        # Backward compatibility for older transformers versions.
        return AutoTokenizer.from_pretrained(model_path)


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except TypeError:
        return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str, extra_patterns: list[str] | None = None) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", *(extra_patterns or [])],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
