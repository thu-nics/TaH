import torch
from typing import Any, Dict, List
import pandas as pd


class TaHTracker:
    """Utility to track TaH model internal states."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.records: List[Dict[str, Any]] = []
        self._orig_fn = None
        self._model = None
        self._call_idx = 0

    def attach(self, model: Any) -> None:
        if self._model is not None:
            raise RuntimeError("Tracker already attached to a model")

        self._model = model
        self._orig_fn = model._process_sparse_iteration

        def wrapper(*args, **kwargs):
            outputs = self._orig_fn(*args, **kwargs)
            iter_depth = kwargs.get("iter_depth")
            valid_mask = kwargs.get("valid_mask")
            cache = kwargs.get("past_key_values")
            if iter_depth is None and len(args) > 5:
                iter_depth = args[5]
            if valid_mask is None and len(args) > 2:
                valid_mask = args[2]
            if cache is None and len(args) > 6:
                cache = args[6]

            logits = outputs.logits
            if logits is not None:
                k = min(self.top_k, logits.size(-1))
                last_token_logits = logits[:, -1, :]
                values, indices = torch.topk(last_token_logits, k=k, dim=-1)
                perplexity, entropy = self.logits_to_perplexity_entropy(last_token_logits)

                for batch_idx in range(logits.size(0)):
                    if valid_mask is not None and valid_mask[batch_idx, -1] == 0:
                        continue
                    self.records.append(
                        {
                            "batch_idx": batch_idx,
                            "call_index": self._call_idx,
                            "iter_depth": iter_depth,
                            "step_index": (cache.get_seq_length() if cache is not None else None),
                            "perplexity": perplexity[batch_idx].item(),
                            "entropy": entropy[batch_idx].item(),
                            "topk_values": values.detach().cpu()[batch_idx, :].tolist(),
                            "topk_indices": indices.detach().cpu()[batch_idx, :].tolist(),
                        }
                    )
                self._call_idx += 1
            return outputs

        model._process_sparse_iteration = wrapper

    def detach(self) -> None:
        if self._model is not None and self._orig_fn is not None:
            self._model._process_sparse_iteration = self._orig_fn
        self._model = None
        self._orig_fn = None

    def clear(self) -> None:
        self.records.clear()
        self._call_idx = 0

    @staticmethod
    def logits_to_perplexity_entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return torch.exp(entropy).detach().cpu(), entropy.detach().cpu()

    def to_pandas(self, selected_keys: List[str] | None = None):
        if not self.records:
            return pd.DataFrame()
        keys = list(self.records[0].keys()) if selected_keys is None else list(selected_keys)
        return pd.DataFrame([{k: rec.get(k) for k in keys} for rec in self.records])[keys]
