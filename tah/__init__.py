"""TaH: Selective Latent Iterations to Improve Reasoning Language Models.

Reference exports load on access so engine imports need no HF/PEFT model
code. Existing top-level names and their identities are preserved.
"""
from importlib import import_module

_EXPORTS = {
    "TaHCache": "tah.model.causal_cache",
    "IterLabelDecider": "tah.model.iter_decider",
    "MLPIterDecider": "tah.model.iter_decider",
    "IterDeciderLoss": "tah.model.loss",
    "NextTokenPredLoss": "tah.model.loss",
    "TaHConfig": "tah.model.tah_config",
    "TaHCausalLMOutputWithPast": "tah.model.tah_model",
    "TaHForCausalLM": "tah.model.tah_model",
}

__all__ = [
    "IterDeciderLoss",
    "IterLabelDecider",
    "MLPIterDecider",
    "NextTokenPredLoss",
    "TaHCache",
    "TaHCausalLMOutputWithPast",
    "TaHConfig",
    "TaHForCausalLM",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
