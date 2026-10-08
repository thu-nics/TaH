import importlib

from .config import ModelConfig

_MODEL_REGISTRY = {
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "TaHQwen3ForCausalLM": (".tah_qwen3", "TaHQwen3ForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
