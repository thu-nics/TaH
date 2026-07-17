"""Iter deciders: per-token "should we iterate again?" classifiers.

Two implementations are kept — both used in the canonical training/eval
recipes. Public TaH also shipped ``TrivialIterDecider``,
``AlwaysWrapperIterDecider``, and ``OracleDynamicIterDecider``; none of those
are used by the released checkpoints and have been removed.

* :class:`IterLabelDecider` — step-1 SFT. Continues iff the dataset's oracle
  ``iter_count_labels`` say so, ignoring model logits entirely. Used to teach
  the LoRA adapter on tokens that the labeller marked "hard".
* :class:`MLPIterDecider` — step-2 SFT and eval/serving. Small MLP over a
  selected subset of per-layer hidden states plus the projected top-k logits;
  outputs a continue-probability per token.

Each decider returns ``(decision_bool, decision_logits)``:

* ``decision_bool`` — ``(N,)`` BoolTensor, True ⇒ continue iterating.
* ``decision_logits`` — ``(N,)`` float tensor of pre-sigmoid logits, used by
  ``IterDeciderLoss`` for BCE supervision. May be a constant (e.g.
  ``IterLabelDecider`` produces NEUTRAL_LOGITS) when the decider doesn't
  itself produce a learnable score.

Persistence: :func:`save_iter_decider` pickles ``(class_name, init_args,
state_dict)`` to ``iter_decider.bin``. :func:`load_iter_decider` restores via
``ITER_DECIDER_BY_NAME``.
"""
from __future__ import annotations

import inspect
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Per-class threshold semantics for IterLabelDecider's auxiliary logits.
NEUTRAL_LOGITS = 0.0
MINUS_INFINITY_LOGITS = -10.0


class IterDecider(nn.Module):
    """Base class. Subclasses must override :meth:`forward`."""

    def __init__(self, threshold: float = 0.5, max_iter: int = 3):
        super().__init__()
        # Buffer (not parameter) so subclasses can also override as a learnable
        # nn.Parameter without registry conflicts.
        self.register_buffer("threshold", torch.tensor(float(threshold), dtype=torch.float32))
        self.max_iter = max_iter
        self._init_args: dict = {}

    def forward(
        self,
        logits: torch.Tensor,
        iter_depth: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _stop_decision(self, logits: torch.Tensor, fill: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(all-False, full-of-`fill`)`` shaped to match logits' leading dims.

        Used by subclasses when ``iter_depth >= max_iter`` (cap exhausted) or
        when there's no useful input to decide on.
        """
        shape = logits.shape[:-1]
        decision = torch.zeros(shape, dtype=torch.bool, device=logits.device)
        return decision, torch.full(shape, fill, dtype=logits.dtype, device=logits.device)


def _capture_init_args(cls):
    """Decorator: store positional + keyword __init__ args on ``self._init_args``.

    Used by :func:`save_iter_decider` so that loaders can re-instantiate the
    class with the same constructor arguments before applying the state dict.
    """
    original_init = cls.__init__
    sig = inspect.signature(original_init)
    param_names = [p for p in sig.parameters.keys() if p != "self"]

    def new_init(self, *args, **kwargs):
        captured: dict = {}
        for i, arg in enumerate(args):
            if i < len(param_names):
                captured[param_names[i]] = arg
        captured.update(kwargs)
        original_init(self, *args, **kwargs)
        self._init_args = captured

    cls.__init__ = new_init
    return cls


# ────────────────────────────────────────────────────────────────────────────
# IterLabelDecider — oracle-supervised, used in step-1 SFT.
# ────────────────────────────────────────────────────────────────────────────


@_capture_init_args
class IterLabelDecider(IterDecider):
    """Continues iff ``iter_count_labels > iter_depth`` (and not ignored)."""

    def __init__(self, max_iter: int = 3):
        super().__init__(max_iter=max_iter)

    def forward(
        self,
        logits: torch.Tensor,
        iter_depth: int,
        iter_count_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if iter_depth >= self.max_iter:
            return self._stop_decision(logits, fill=MINUS_INFINITY_LOGITS)
        if iter_count_labels is None:
            return self._stop_decision(logits, fill=NEUTRAL_LOGITS)
        decision = (iter_count_labels > iter_depth) & (iter_count_labels != -100)
        _, neutral = self._stop_decision(logits, fill=NEUTRAL_LOGITS)
        return decision, neutral


# ────────────────────────────────────────────────────────────────────────────
# MLPIterDecider — learned classifier, used in step-2 SFT and eval/serving.
# ────────────────────────────────────────────────────────────────────────────


class _ClassifierBlock(nn.Module):
    """LayerNorm + 2-layer MLP with residual; supports dim change."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expansion_factor: int,
        dropout_rate: float,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory = {"dtype": dtype} if dtype is not None else {}
        self.layer_norm = nn.LayerNorm(input_dim, **factory)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * expansion_factor, **factory),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * expansion_factor, output_dim, **factory),
            nn.Dropout(dropout_rate),
        )
        self.dim_change = (
            nn.Linear(input_dim, output_dim, **factory) if input_dim != output_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dim_change(x) + self.mlp(self.layer_norm(x))


class _ClassifierBackbone(nn.Module):
    """Stacked _ClassifierBlocks with kaiming-init linears and a final norm + projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        expansion_factor: int,
        dropout_rate: float,
        output_dim: int = 1,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory = {"dtype": dtype} if dtype is not None else {}
        self.input_projection = nn.Linear(input_dim, hidden_dims[0], **factory)
        block_dims = list(hidden_dims) + [hidden_dims[-1]]
        self.blocks = nn.ModuleList(
            _ClassifierBlock(block_dims[i], block_dims[i + 1], expansion_factor, dropout_rate, **factory)
            for i in range(len(block_dims) - 1)
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1], **factory),
            nn.Linear(hidden_dims[-1], output_dim, **factory),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


@_capture_init_args
class MLPIterDecider(IterDecider):
    """Decides by combining a slice of per-layer hidden states with top-k logits.

    Hidden features: pick layer indices ``hidden_states_layer_nums`` from the
    base model's stacked hidden states (shape ``(..., L, H)``), concat into a
    single ``(..., len(layer_nums) * H)`` vector. Logit features: top-k of the
    per-token logits, projected up to the hidden size. Concat both, project to
    backbone-input width, run the MLP backbone, sigmoid → continue probability.
    """

    def __init__(
        self,
        topk: int = 100,
        hidden_states_size: int = 1024,
        hidden_states_layer_nums = (16, 20, 24, 28),
        hidden_dims = (256, 512, 256),
        expansion_factor: int = 4,
        dropout_rate: float = 0.3,
        normalize_input: bool = False,
        threshold: float = 0.5,
        max_iter: int = 3,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(max_iter=max_iter)
        self.topk = topk
        self.hidden_states_size = hidden_states_size
        self.hidden_states_layer_nums = list(hidden_states_layer_nums)
        self.normalize_input = normalize_input
        self.max_iter = max_iter

        # threshold: replace the buffer with a learnable parameter.
        if hasattr(self.__class__, "threshold"):
            try:
                delattr(self, "threshold")
            except AttributeError:
                pass
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=dtype, requires_grad=True))

        n_layers = max(1, len(self.hidden_states_layer_nums))
        if normalize_input:
            self.layer_norm_hidden_states = nn.LayerNorm(hidden_states_size * n_layers)

        self.logits_projection = nn.Linear(self.topk, hidden_states_size, dtype=dtype)
        self.combined_projection = nn.Linear(hidden_states_size * n_layers + hidden_states_size, hidden_dims[0], dtype=dtype)
        self.backbone = _ClassifierBackbone(
            hidden_dims[0], hidden_dims, expansion_factor, dropout_rate, dtype=dtype,
        )
        self.sigmoid = nn.Sigmoid()

    def _select_hidden_features(
        self,
        all_hidden_states: Optional[torch.Tensor],
        decision_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Pick + flatten the configured hidden-state layers (or zero-fill if missing).

        ``all_hidden_states`` may be ``(..., L, H)`` or ``(..., H)`` (single
        layer); we treat the latter as ``L=1``. Returned shape is
        ``(*decision_shape, n_layers * H)``.
        """
        n_layers = max(1, len(self.hidden_states_layer_nums))
        flat_dim = self.hidden_states_size * n_layers
        if all_hidden_states is None:
            return torch.zeros(*decision_shape, flat_dim, device=device, dtype=dtype)

        hs = all_hidden_states
        if hs.dim() == len(decision_shape) + 1:  # (..., H) — promote to (..., 1, H)
            hs = hs.unsqueeze(-2)
        total_layers = hs.size(-2)
        indices = self.hidden_states_layer_nums or [total_layers - 1]
        idx = torch.as_tensor(indices, device=hs.device, dtype=torch.long)
        if idx.numel() == 0 or int(idx.min()) < 0 or int(idx.max()) >= total_layers:
            raise ValueError(f"hidden_states_layer_nums {indices} out of range for {total_layers} layers")
        return torch.index_select(hs, dim=-2, index=idx).reshape(*decision_shape, flat_dim)

    def forward(
        self,
        logits: torch.Tensor,
        iter_depth: int,
        all_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if iter_depth >= self.max_iter:
            return self._stop_decision(logits, fill=MINUS_INFINITY_LOGITS)

        decision_shape = logits.shape[:-1]
        hidden_concat = self._select_hidden_features(
            all_hidden_states, decision_shape, logits.device, logits.dtype,
        )
        topk_values, _ = torch.topk(logits, k=min(self.topk, logits.size(-1)), dim=-1)
        if self.normalize_input:
            hidden_concat = self.layer_norm_hidden_states(hidden_concat)
            topk_values = torch.softmax(topk_values, dim=-1)

        x = self.combined_projection(
            torch.cat([hidden_concat, self.logits_projection(topk_values)], dim=-1),
        )
        decision_logits = self.backbone(x)
        if decision_logits.dim() == logits.dim():
            decision_logits = decision_logits.squeeze(-1)

        threshold = self.threshold
        if isinstance(threshold, torch.Tensor):
            threshold = float(threshold.detach().item())
        decision_mask = self.sigmoid(decision_logits) > float(threshold)
        return decision_mask, decision_logits


# ────────────────────────────────────────────────────────────────────────────
# Persistence + name-based dispatch.
# ────────────────────────────────────────────────────────────────────────────


ITER_DECIDER_BY_NAME = {
    "IterLabelDecider": IterLabelDecider,
    "MLPIterDecider": MLPIterDecider,
}


def get_iter_decider_class(name: str):
    """Lookup helper kept for backwards-compatible callers."""
    if name not in ITER_DECIDER_BY_NAME:
        raise ValueError(f"Unknown iter_decider {name!r}; have {sorted(ITER_DECIDER_BY_NAME)}")
    return ITER_DECIDER_BY_NAME[name]


def save_iter_decider(decider: IterDecider, save_directory: str) -> None:
    """Pickle ``(class_name, init_args, state_dict)`` to ``iter_decider.bin``."""
    state = {k: v.detach().cpu() for k, v in decider.state_dict().items()}
    payload = {
        "class": decider.__class__.__name__,
        "state_dict": state,
        "init_args": getattr(decider, "_init_args", {}),
    }
    torch.save(payload, os.path.join(save_directory, "iter_decider.bin"))


def load_iter_decider(
    load_directory: str,
    class_name: Optional[str] = None,
    init_args: Optional[dict] = None,
) -> IterDecider:
    path = os.path.join(load_directory, "iter_decider.bin")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no iter_decider.bin at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    cls_name = class_name or data.get("class")
    if not cls_name:
        raise ValueError("iter_decider.bin lacks a class name")
    args = init_args if init_args is not None else data.get("init_args", {})
    decider = get_iter_decider_class(cls_name)(**args)
    sd = data.get("state_dict", {})
    if sd:
        # Drop any state-dict keys that collide with init args (e.g. learnable
        # threshold whose value is also passed in init_args).
        sd = {k: v for k, v in sd.items() if k not in args}
        if sd:
            decider.load_state_dict(sd, strict=False)
    return decider
