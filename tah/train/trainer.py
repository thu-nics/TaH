"""HF Trainer subclass + iter-aware logging callback for TaH SFT.

The Trainer override exists for two reasons:

1. ``_save`` calls ``model.save_pretrained`` so the TaH-specific layout
   (``tah_config.json``, ``iter_decider.bin``, ``lora/``) is written instead
   of the default HF flat-state-dict save.

2. ``_get_dataloader`` adds ``shuffle=True`` to the DataLoader kwargs (the
   public TaH had this; we preserve the behaviour).

The callback (``LoggerCallback``) just wires the wrapper's running counters
(``avg_iter_count``, ``iter_decider_accuracy``) into the per-step ``logs``
dict that HF Trainer flushes to wandb / tensorboard. Public TaH had extra
plumbing for composite/scheduled losses (``CombinedLoss``, ``InterleavedLoss``)
which never existed in the released checkpoint and is dropped.

``fixed_cross_entropy`` is the loss helper used by ``NextTokenPredLoss``;
``weighted_cross_entropy`` from public TaH is removed (the hard-token weight
plumbing it served has been removed from ``NextTokenPredLoss`` too).
"""
from __future__ import annotations

import os
from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import seed_worker


class CustomTaHTrainer(Trainer):
    """Trainer with TaH-aware ``_save`` and a shuffling dataloader override."""

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[callable] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        data_collator = self.data_collator
        if hasattr(self, "_remove_unused_columns"):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(
                self.data_collator, description=description
            )

        params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": True,
        }
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                params["sampler"] = sampler_fn(dataset)
                params.pop("shuffle", None)
            params["drop_last"] = self.args.dataloader_drop_last
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )
        else:
            params.pop("shuffle", None)

        dataloader = DataLoader(dataset, **params)
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            store = getattr(self, "_eval_dataloaders", None) or {}
            store[dataloader_key] = dataloader
            self._eval_dataloaders = store
        return self.accelerator.prepare(dataloader)

    def _save(self, output_dir=None, state_dict=None):
        """Use TaH's custom save_pretrained when available; default Trainer behavior otherwise."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self.model, "save_pretrained") and hasattr(self.model, "config"):
            print(f"Saving TaH model to: {output_dir}")
            self.model.save_pretrained(output_dir)
        else:
            super()._save(output_dir, state_dict)
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.save_pretrained(output_dir)


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """Cross-entropy with optional sum/N normalisation matching HF's per-item averaging."""
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


class LoggerCallback(TrainerCallback):
    """Push the wrapper's running iter-aware counters into the trainer's log stream.

    The wrapper writes ``self.avg_iter_count`` (every forward) and
    ``self.iter_decider_accuracy`` (intra-iter loss path) onto its
    ``logger_callback`` attribute; we forward and reset them on every
    ``on_log`` event.
    """

    def __init__(self):
        self.avg_iter_count = 0.0
        self.iter_decider_accuracy = 0.0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs["logs"]
        logs["avg_iter_count"] = self.avg_iter_count
        if self.iter_decider_accuracy > 0.0:
            logs["iter_decider_accuracy"] = self.iter_decider_accuracy
        self.avg_iter_count = 0.0
        self.iter_decider_accuracy = 0.0
