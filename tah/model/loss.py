import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from tah.model.registry import register_loss_func, capture_init_args, get_loss_func_class
from tah.train import weighted_cross_entropy, fixed_cross_entropy
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LossFunc:
    _is_intra_iter_loss: bool = False

    def __init__(self, **kwargs):
        self.config = kwargs

    def prepare_loss(self, batch_size, query_len, device, dtype, **kwargs):
        pass

    def intra_iter_loss_func(self, *args, **kwargs):
        raise NotImplementedError(
            "This loss function does not support intra-iteration loss calculation."
        )

    def final_loss_func(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


@register_loss_func
@capture_init_args
class NextTokenPredLoss(LossFunc):
    _is_intra_iter_loss: bool = False

    def __init__(self, hard_token_relative_weight: float = 1.0, weight_hard: float = None, weight_easy: float = None, **kwargs):
        super().__init__()
        self.hard_token_relative_weight = hard_token_relative_weight
        self.weight_hard = weight_hard
        self.weight_easy = weight_easy

    def prepare_loss(self, batch_size, query_len, device, dtype, **kwargs):
        # Update weights if provided in kwargs (allows dynamic weight setting)
        super().prepare_loss(batch_size, query_len, device, dtype, **kwargs)
        if 'weight_hard' in kwargs and kwargs['weight_hard'] is not None:
            self.weight_hard = kwargs['weight_hard']
        if 'weight_easy' in kwargs and kwargs['weight_easy'] is not None:
            self.weight_easy = kwargs['weight_easy']
        if 'hard_token_relative_weight' in kwargs and kwargs['hard_token_relative_weight'] is not None:
            self.hard_token_relative_weight = kwargs['hard_token_relative_weight']

    def final_loss_func(
        self,
        logits: torch.Tensor,
        labels_shifted: torch.Tensor,
        iter_count: torch.Tensor,
        training: bool,
        **kwargs,
    ) -> torch.Tensor:
        num_items_in_batch = kwargs.get("num_items_in_batch", None)

        vocab_size = logits.shape[-1]

        logits = logits.float() # upcast to float to avoid precision issue, following transformers official implementation
        shift_iter = iter_count.contiguous() if not (iter_count == -1).all() else None

        shift_logits = logits.view(-1, vocab_size).float()
        shift_labels = labels_shifted.view(-1)
        shift_iter = shift_iter.view(-1) if shift_iter is not None else None

        shift_labels = shift_labels.to(shift_logits.device)
        ignore_index = -100
        has_custom_weights = (
            self.weight_hard is not None and self.weight_easy is not None
        )

        if self.hard_token_relative_weight == 1.0 or not training:
            return fixed_cross_entropy(
                shift_logits,
                shift_labels,
                num_items_in_batch=num_items_in_batch,
                ignore_index=ignore_index,
            )
        else:
            weight_hard = (
                self.weight_hard
                if has_custom_weights
                else self.hard_token_relative_weight
            )
            weight_easy = self.weight_easy if has_custom_weights else 1.0

            token_weights = torch.full_like(
                shift_labels, weight_easy, dtype=shift_logits.dtype
            )
            if shift_iter is not None:
                token_weights[shift_iter > 1] = weight_hard

            return weighted_cross_entropy(
                shift_logits,
                shift_labels,
                token_weights,
                num_items_in_batch=num_items_in_batch,
                ignore_index=ignore_index,
            )


@register_loss_func
@capture_init_args
class ConsistencyLoss(LossFunc):
    _is_intra_iter_loss: bool = True

    def __init__(self, **kwargs):
        from tah.model.recurrent_transformer import TaHForCausalLM # import like this to avoid circular import
        self.assign_active = TaHForCausalLM.assign_active
        super().__init__(**kwargs)
        
    def prepare_loss(self, batch_size, query_len, device, dtype, **kwargs):
        self.consistency_loss_per_token = torch.zeros(
            batch_size, query_len, device=device, dtype=torch.float32
        ) # noqa: always use float32 for loss

    def intra_iter_loss_func(
        self,
        active_logits: torch.Tensor,
        current_iter_mask: torch.BoolTensor,
        active_labels_shifted: torch.Tensor,
        **kwargs,
    ):
        if self.consistency_loss_per_token is None:
            raise RuntimeError(
                "Consistency loss tensor not initialized. Call `init_consistency_loss` first."
            )

        batch_size, query_len = current_iter_mask.shape
        device = active_logits.device
        active_logits = active_logits.float() # upcast to float to avoid precision issue, following transformers official implementation

        token_losses = torch.zeros(
            batch_size, query_len, device=device, dtype=active_logits.dtype
        )

        if not current_iter_mask.any() or active_labels_shifted is None:
            return torch.tensor(0.0, device=active_logits.device, dtype=active_logits.dtype)

        flat_active_logits = active_logits.view(-1, active_logits.size(-1))
        flat_active_labels = active_labels_shifted.view(-1)

        flat_losses = F.cross_entropy(
            flat_active_logits,
            flat_active_labels,
            reduction="none",
            ignore_index=-100,
        )

        active_losses_reshaped = flat_losses.view(batch_size, -1)
        self.assign_active(current_iter_mask, active_losses_reshaped, token_losses)
        self._update_consistency_loss(token_losses)
        
        return token_losses

    def _update_consistency_loss(self, token_losses):
        self.consistency_loss_per_token = token_losses + self.consistency_loss_per_token

    def final_loss_func(
        self,
        labels_shifted: torch.Tensor,
        iter_count: torch.Tensor,
        training: bool,
        **kwargs,
    ) -> torch.Tensor:
        if self.consistency_loss_per_token is None:
            raise RuntimeError(
                "Consistency loss tensor not initialized or already consumed."
            )

        num_items_in_batch = kwargs.get("num_items_in_batch", None)

        valid_mask = (labels_shifted != -100) & (iter_count > 0)

        consistency_loss = self.consistency_loss_per_token
        self.consistency_loss_per_token = None  # Consume the loss

        if not valid_mask.any():
            return torch.tensor(
                0.0, device=labels_shifted.device, dtype=consistency_loss.dtype
            )

        avg_losses = torch.zeros_like(consistency_loss)
        avg_losses[valid_mask] = (
            consistency_loss[valid_mask] / iter_count[valid_mask].float()
        ).to(dtype=consistency_loss.dtype)

        if num_items_in_batch is not None:
            return avg_losses[valid_mask].sum() / num_items_in_batch
        else:
            return avg_losses[valid_mask].mean()

@register_loss_func
@capture_init_args
class IterDeciderLoss(LossFunc):
    """
    Loss function for iter decider that predicts whether each token should continue iterating.
    Uses BCE loss similar to the router training implementation.
    Calculates loss at each iteration depth.
    """
    _is_intra_iter_loss: bool = True

    def __init__(self, pos_weight: Optional[float] = None, skip_last_iter: bool = True, max_iter: Optional[int] = None, **kwargs):
        """
        Initialize IterDeciderLoss.
        
        Args:
            pos_weight: Positive class weight for BCE loss to handle class imbalance
            skip_last_iter: If True, skip loss at the max iteration because it's always stop
        """
        from tah.model.recurrent_transformer import TaHForCausalLM # import like this to avoid circular import
        self.assign_active = TaHForCausalLM.assign_active
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        self.skip_last_iter = bool(skip_last_iter)
        # Optional explicit max_iter (preferred over reading from model at call time)
        self.max_iter: Optional[int] = int(max_iter) if max_iter is not None else None

        if self.skip_last_iter and self.max_iter is None:
            raise ValueError("max_iter must be provided if skip_last_iter is True")
        
        # Create BCE loss criterion
        if pos_weight is not None:
            self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def prepare_loss(self, batch_size, query_len, device, dtype, **kwargs):
        self.iter_decider_loss_per_token = torch.zeros(
            batch_size, query_len, device=device, dtype=torch.float32
        ) # always use float32 for loss
        # Metric accumulators (float32 scalars on device)
        self._metric_correct_count = torch.zeros(1, device=device, dtype=torch.float32)
        self._metric_total_count = torch.zeros(1, device=device, dtype=torch.float32)

    def intra_iter_loss_func(
        self,
        active_logits: torch.Tensor,
        current_iter_mask: torch.BoolTensor,
        active_labels_shifted: torch.Tensor,
        active_valid_continue_logits: Optional[torch.Tensor],
        active_valid_mask: torch.LongTensor,
        iter_depth: int,
        active_iter_count_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Calculate iter decider loss at each iteration depth.
        
        Args:
            active_logits: Model logits (not used)
            current_iter_mask: Mask for current iteration
            active_labels_shifted: Shifted labels (not used)
            active_valid_continue_prob: Continue probabilities from iter_decider
            active_valid_continue_logits: Continue logits from iter_decider
            active_valid_mask: Valid mask for active tokens
            iter_depth: Current iteration depth
            active_iter_count_labels: Target iteration counts
            **kwargs: Additional arguments
        """
        if active_iter_count_labels is None or active_valid_continue_logits is None:
            return torch.tensor(0.0, dtype=torch.float32)

        if not current_iter_mask.any() or active_valid_mask.sum() == 0:
            return torch.tensor(0.0, dtype=torch.float32)

        # Optionally skip loss for the final iteration depth (always-stop step)
        if self.skip_last_iter:
            if int(iter_depth) >= int(self.max_iter):
                return torch.tensor(0.0, dtype=torch.float32)
        
        # Update metrics using probabilities derived from logits and threshold
        if active_iter_count_labels is not None and active_valid_continue_logits is not None:
            valid_active_mask = (active_valid_mask == 1)
            valid_iter_count_labels = active_iter_count_labels[valid_active_mask]
            non_padding_mask = (valid_iter_count_labels != -100)
            if non_padding_mask.any():
                final_continue_targets = (valid_iter_count_labels[non_padding_mask] > iter_depth).to(torch.float32)
                final_continue_probs = torch.sigmoid(active_valid_continue_logits[non_padding_mask]).to(torch.float32)
                # Resolve threshold value
                iter_decider_threshold = kwargs.get('iter_decider_threshold', 0.5)
                if isinstance(iter_decider_threshold, torch.Tensor):
                    threshold_value = float(iter_decider_threshold.detach().item())
                else:
                    threshold_value = float(iter_decider_threshold)

                with torch.no_grad():
                    pred_positive = (final_continue_probs > threshold_value).to(torch.float32)
                    target_positive = final_continue_targets
                    correct = (pred_positive == target_positive).to(torch.float32).sum()
                    total = torch.tensor(float(pred_positive.numel()), device=final_continue_probs.device, dtype=torch.float32)

                    if hasattr(self, '_metric_correct_count') and self._metric_correct_count is not None:
                        self._metric_correct_count += correct
                        self._metric_total_count += total
        
        active_valid_continue_logits = active_valid_continue_logits.float()
        
        device = active_valid_continue_logits.device
        dtype = active_valid_continue_logits.dtype
        batch_size, query_len = current_iter_mask.shape
        
        if self.iter_decider_loss_per_token is None:
            raise RuntimeError(
                "Iter decider loss tensor not initialized. Call `prepare_loss` first."
            )

        # Initialize token losses for this iteration
        token_losses = torch.zeros(
            batch_size, query_len, device=device, dtype=dtype
        )

        # Calculate target labels: should continue if iter_count_labels > iter_depth
        # Only consider valid active tokens
        valid_active_mask = (active_valid_mask == 1)
        # For valid active tokens, calculate binary targets
        valid_iter_count_labels = active_iter_count_labels[valid_active_mask]
        valid_continue_targets = (valid_iter_count_labels > iter_depth).float()
        
        # Exclude padding tokens (-100)
        non_padding_mask = (valid_iter_count_labels != -100)
        if not non_padding_mask.any():
            return torch.tensor(0.0, device=device, dtype=dtype)

        final_continue_targets = valid_continue_targets[non_padding_mask]
        final_continue_logits = active_valid_continue_logits[non_padding_mask]

        # Move pos_weight to correct device if needed
        if hasattr(self.criterion, 'pos_weight') and self.criterion.pos_weight is not None:
            self.criterion.pos_weight = self.criterion.pos_weight.to(device=device)

        # Calculate BCE loss
        loss = self.criterion(final_continue_logits.unsqueeze(-1), final_continue_targets.unsqueeze(-1))

        # Assign loss back to full tensor structure 
        # This is simplified - we assign the same loss to all valid active tokens
        if valid_active_mask.any() and non_padding_mask.any():
            # Create a tensor to hold loss for active tokens
            active_token_losses = torch.zeros(batch_size, active_valid_mask.shape[1], device=device, dtype=loss.dtype)
            # We'll assign the average loss to all contributing tokens
            num_contributing_tokens = non_padding_mask.sum()
            if num_contributing_tokens > 0:
                per_token_loss = loss / num_contributing_tokens
                # Create a full-size tensor for valid active positions
                valid_positions = torch.zeros_like(active_token_losses, dtype=torch.bool)
                valid_positions[valid_active_mask] = non_padding_mask
                active_token_losses[valid_positions] = per_token_loss
                self.assign_active(current_iter_mask, active_token_losses, token_losses)

        # Update cumulative loss
        self._update_iter_decider_loss(token_losses)
        
        return token_losses

    def _update_iter_decider_loss(self, token_losses):
        self.iter_decider_loss_per_token = token_losses + self.iter_decider_loss_per_token

    def final_loss_func(
        self,
        logits: torch.Tensor,
        labels_shifted: torch.Tensor,
        iter_count: torch.Tensor,
        iter_count_labels: Optional[torch.Tensor] = None,
        training: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculate final iter decider loss from accumulated losses.
        
        Args:
            logits: Model logits (not used)
            labels_shifted: Shifted labels (not used)
            iter_count: Actual iteration counts from model
            iter_count_labels: Target iteration count labels (optional)
            training: Whether in training mode
            **kwargs: Additional arguments
            
        Returns:
            Accumulated iter decider loss
        """
        if self.iter_decider_loss_per_token is None:
            raise RuntimeError(
                "Iter decider loss tensor not initialized or already consumed."
            )

        num_items_in_batch = kwargs.get("num_items_in_batch", None)
        
        # Use iter_count_labels if available, otherwise fall back to simple validation
        if iter_count_labels is not None:
            valid_mask = (iter_count_labels != -100) & (iter_count > 0)
        else:
            valid_mask = (iter_count > 0)

        iter_decider_loss = self.iter_decider_loss_per_token
        self.iter_decider_loss_per_token = None  # Consume the loss

        if not valid_mask.any():
            return torch.tensor(
                0.0, device=logits.device, dtype=iter_decider_loss.dtype
            )

        # Compute and log metrics if requested
        logger_callback = kwargs.get('logger_callback', None)
        with torch.no_grad():
            if hasattr(self, '_metric_total_count') and self._metric_total_count is not None and (
                (self._metric_total_count.item() > 0) or (kwargs.get('num_items_in_batch', None) is not None)
            ):
                # Accuracy logging: follow avg_iter_count pattern → correct_count / num_items_in_batch
                if logger_callback is not None:
                    if not hasattr(logger_callback, 'iter_decider_accuracy'):
                        logger_callback.iter_decider_accuracy = 0.0

                    num_items_in_batch = kwargs.get('num_items_in_batch', None)
                    if num_items_in_batch is not None and num_items_in_batch > 0:
                        acc_step = (self._metric_correct_count / num_items_in_batch)
                    else:
                        # Fallback to total-based accuracy if num_items_in_batch is absent
                        total_safe = torch.clamp(self._metric_total_count, min=1.0)
                        acc_step = (self._metric_correct_count / total_safe)
                    logger_callback.iter_decider_accuracy += float(acc_step)

        # Reset metric accumulators after consumption
        self._metric_correct_count = None
        self._metric_total_count = None

        # Calculate average loss over valid tokens
        if num_items_in_batch is not None:
            return iter_decider_loss[valid_mask].sum() / num_items_in_batch
        else:
            return iter_decider_loss[valid_mask].mean()
        

