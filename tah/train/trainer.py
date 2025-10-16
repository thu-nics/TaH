import os
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase, TrainerCallback, TrainerState, TrainerControl
import torch
import torch.nn as nn
from typing import Optional, Any, Union, Tuple
from torch.utils.data import DataLoader, Dataset
from functools import partial
from transformers.trainer_utils import seed_worker

# from tah.evaluate.eval_unified import allocate_gpus_and_run_jobs

class CustomTaHTrainer(Trainer):
    """
    self-defined Trainer class to ensure that the custom save_pretrained method of the TaH model is called when saving the model.
    """
    def __init__(self, *args, **kwargs):
        # Extract prediction_config from kwargs before passing to parent
        self.prediction_config = kwargs.pop('prediction_config', None)
        super().__init__(*args, **kwargs)
        self.gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
    
    # override _get_dataloader method to add shuffle=True parameter
    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[callable] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        """Override _get_dataloader method to add shuffle=True parameter."""
        
        data_collator = self.data_collator
        if hasattr(self, '_remove_unused_columns'):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": True,  # Add shuffle=True parameter
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
                dataloader_params.pop("shuffle", None)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )
        else:
            # For IterableDataset, remove shuffle parameter to avoid conflict
            dataloader_params.pop("shuffle", None)

        dataloader = DataLoader(dataset, **dataloader_params)

        # Accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version for eval dataloaders.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return self.accelerator.prepare(dataloader)
        
    def _save(self, output_dir=None, state_dict=None):
        """Override _save method to ensure that the custom save_pretrained method of the TaH model is called when saving the model."""
        # use output directory or default output directory
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # save model
        if hasattr(self.model, 'save_pretrained') and hasattr(self.model, 'config'):
            # for TaH model, use custom save_pretrained method
            print(f"use TaH custom save_pretrained method to save model to: {output_dir}")
            print(f"   - save base model and config...")
            self.model.save_pretrained(output_dir)
        else:
            # for normal model, use default method
            print(f"use default method to save model to: {output_dir}")
            super()._save(output_dir, state_dict)
        
        # save tokenizer
        if getattr(self, 'tokenizer', None) is not None:
            self.tokenizer.save_pretrained(output_dir)

    def evaluate(self, *args, **kwargs):
        """Override evaluate method to add predict with generation."""
        base_metrics = super().evaluate()
        
        # Only run generation evaluation if prediction_config is provided and this is the main process
        return base_metrics
    

def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """
    Compute weighted cross-entropy loss where each token can have a different weight.
    
    Args:
        logits: Model predictions of shape (batch_size * seq_len, vocab_size)
        labels: Target labels of shape (batch_size * seq_len,)
        token_weights: Weight for each token of shape (batch_size * seq_len,)
        num_items_in_batch: Number of valid items in batch for averaging
        ignore_index: Label index to ignore (default: -100)
        
    Returns:
        Weighted cross-entropy loss
    """
    # Compute per-token cross-entropy loss (no reduction)
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    per_token_loss = loss_fct(logits, labels)
    
    # Apply mask to ignore specified labels
    valid_mask = labels != ignore_index
    
    # Compute weighted loss
    if valid_mask.any():
        valid_losses = per_token_loss[valid_mask]
        valid_weights = token_weights[valid_mask]
        
        # Weighted sum
        weighted_sum = (valid_losses * valid_weights).sum()
        
        if num_items_in_batch is not None:
            # Average by number of items in batch
            loss = weighted_sum / num_items_in_batch
        else:
            # Average by sum of weights
            loss = weighted_sum / valid_weights.sum()
    else:
        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    return loss

class LoggerCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.avg_iter_count = 0
        self.iter_decider_accuracy = 0.0
        self.iter_decider_precision = 0.0

    def _update_iter_decider_training_state(self, state: TrainerState, args: TrainingArguments):
        """Propagate current step/epoch into model.iter_decider if supported."""
        model = getattr(self.trainer, 'model', None)
        if model is None:
            return
        iter_decider = getattr(model, 'iter_decider', None)
        if iter_decider is None:
            return
        # Initialize scheduling meta once at train begin
        if hasattr(iter_decider, 'num_grow_steps') and (getattr(iter_decider, 'num_grow_steps', None) in [None, 0]):
            # Align grow steps with trainer's planned max_steps
            if hasattr(state, 'max_steps') and state.max_steps is not None:
                iter_decider.num_grow_steps = state.max_steps
        if hasattr(iter_decider, 'num_epochs') and (getattr(iter_decider, 'num_epochs', None) in [None, 0]):
            # Align epochs with args
            if hasattr(args, 'num_train_epochs') and args.num_train_epochs is not None:
                try:
                    iter_decider.num_epochs = int(args.num_train_epochs)
                except Exception:
                    pass

        # Update dynamic training state for this step/epoch
        if hasattr(iter_decider, 'update_training_state') and callable(iter_decider.update_training_state):
            current_step = getattr(state, 'global_step', 0) or 0
            current_epoch = int(state.epoch) if getattr(state, 'epoch', None) is not None else 0
            iter_decider.update_training_state(current_step=current_step, current_epoch=current_epoch)

    def _update_loss_training_state(self, state: TrainerState, args: TrainingArguments):
        """Propagate current step/epoch into InterleavedLoss (or nested losses) if supported."""
        model = getattr(self.trainer, 'model', None)
        if model is None:
            return
        loss_objs = []
        if hasattr(model, 'train_loss') and model.train_loss is not None:
            loss_objs.append(model.train_loss)
        if hasattr(model, 'eval_loss') and model.eval_loss is not None:
            loss_objs.append(model.eval_loss)

        if not loss_objs:
            return

        current_step = getattr(state, 'global_step', 0) or 0
        current_epoch = int(state.epoch) if getattr(state, 'epoch', None) is not None else 0

        def _maybe_update(obj):
            if obj is None:
                return
            # Update if this loss exposes the method
            if hasattr(obj, 'update_training_state') and callable(obj.update_training_state):
                try:
                    obj.update_training_state(current_step=current_step, current_epoch=current_epoch)
                except Exception:
                    pass
            # Recurse into composite losses (e.g., CombinedLoss, InterleavedLoss)
            for attr in ('primary_loss', 'secondary_loss'):
                if hasattr(obj, attr):
                    _maybe_update(getattr(obj, attr))

        for loss_obj in loss_objs:
            _maybe_update(loss_obj)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Ensure iter_decider receives initial scheduling meta and state
        self._update_iter_decider_training_state(state, args)
        self._update_loss_training_state(state, args)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Keep iter_decider in sync each step
        self._update_iter_decider_training_state(state, args)
        self._update_loss_training_state(state, args)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Also refresh at epoch boundaries
        self._update_iter_decider_training_state(state, args)
        self._update_loss_training_state(state, args)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Log the average iteration count after each step.

        Kwargs: 'model', 'processing_class', 'optimizer', 'lr_scheduler', 'train_dataloader', 'eval_dataloader', 'logs'
        """
        kwargs['logs']['avg_iter_count'] = self.avg_iter_count
        if self.iter_decider_accuracy is not None and self.iter_decider_accuracy > 0.0:
            kwargs['logs']['iter_decider_accuracy'] = self.iter_decider_accuracy
        # kwargs['logs']['iter_decider_precision'] = self.iter_decider_precision
        
        # if hasattr(self.trainer.model, 'iter_decider'):
        #     iter_decider = self.trainer.model.iter_decider
        #     if hasattr(iter_decider, 'transition_weight'):
        #         kwargs['logs']['transition_weight'] = iter_decider.transition_weight
        
        self.avg_iter_count = 0
        self.iter_decider_accuracy = 0.0
        self.iter_decider_precision = 0.0