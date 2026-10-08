from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class TaHConfig:
    """Configuration for TaH model components."""
    # Overidable configs
    embedding_key: str = "model.embed_tokens"
    max_iter: int = None
    iter_decider: str = None
    input_updater: str = None
    train_loss: str = None
    eval_loss: str = None
    # Optional: use a different iter_decider for evaluation/inference.
    # Supports either a single spec or a list of specs (evaluated in order).
    eval_iter_decider: Any = None
    iter_label_generator: str = None
    # "causal": earlier tokens expose iter-0 only; "duo": all iters 0..cur;
    # "same_iter": each query attends ONLY to same-iteration KV, causally (earlier
    #   tokens expose their iter==cur entry, nothing else); cross-iter coupling is
    #   carried solely by the input_updater recurrent state. Intended for always-iter
    #   (fixed-depth, Universal-Transformer-style) models. triton MODE==5.
    iter_attention_mode: str = "causal"
    attn_implementation: str = "sdpa"
    # iter_attention_impl — which attention kernel runs at each iter depth:
    #   "sdpa":          SDPA + 4-D mask at every iter (baseline).
    #   "fa2_hybrid":    FA2 at iter=0, SDPA at iter>=1.
    #   "triton":        SDPA at iter=0, fused Triton kernel at iter>=1.
    #   "triton_hybrid": FA2 at iter=0, fused Triton kernel at iter>=1.
    iter_attention_impl: str = "sdpa"
    # Mix iteration probabilities using stopping probabilities or uniform weights.
    # Both methods return mixture log-probabilities in outputs.logits.
    weighted_hidden_method: str = "stop_prob_mix"
    # Use custom fp32 reductions that recompute logsumexp backward while saving
    # bf16 logits instead of a full fp32 vocabulary activation.
    memory_lean_fp32_reductions: bool = False

    # Non-overidable configs
    iter_decider_kwargs: Dict[str, Any] = field(default_factory=dict)
    input_updater_kwargs: Dict[str, Any] = field(default_factory=dict)
    train_loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_iter_decider_kwargs: Dict[str, Any] = field(default_factory=dict)
    iter_label_generator_kwargs: Dict[str, Any] = field(default_factory=dict)
