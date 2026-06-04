"""Data collator for TaH SFT.

Wraps :class:`~transformers.DataCollatorForSeq2Seq` to handle the additional
``iter_count_labels`` field that the labelling pipeline produces. The field is
padded alongside ``input_ids`` (same length, same padding side) using the
label ignore-index ``-100`` for padding positions, then converted to a
``LongTensor``.

Public TaH had separate code paths for "no padding" inputs and for ``list``
vs ``ndarray`` element types; in practice every dataset feeds tokenised
inputs as ``list[int]`` and asks for padding, so the cleaned collator only
implements that one path. ``iter_count_pad_value`` is removed (the base
collator's ``label_pad_token_id`` controls both labels and iter_count_labels).
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForSeq2Seq, PaddingStrategy


class CustomTaHDataCollator:
    """Pads ``input_ids`` / ``attention_mask`` / ``labels`` (via base collator)
    plus the TaH-specific ``iter_count_labels`` field."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=padding, max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of, label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors,
        )

    def __call__(self, features, return_tensors=None):
        return_tensors = return_tensors or self.return_tensors

        iter_labels_list = []
        if features and "iter_count_labels" in features[0]:
            iter_labels_list = [f.pop("iter_count_labels") for f in features]

        batch = self.base_collator(features, return_tensors=return_tensors)

        if not iter_labels_list:
            return batch

        # Pad iter_count_labels to match input_ids length on the same side as
        # the tokenizer's padding side; -100 marks ignored positions for the loss.
        target_len = batch["input_ids"].shape[1] if "input_ids" in batch else max(len(v) for v in iter_labels_list)
        if self.pad_to_multiple_of is not None:
            target_len = (target_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of

        right_pad = self.tokenizer.padding_side == "right"
        pad_val = self.label_pad_token_id
        padded = []
        for v in iter_labels_list:
            v = list(v) if not isinstance(v, list) else v
            n_pad = target_len - len(v)
            row = (v + [pad_val] * n_pad) if right_pad else ([pad_val] * n_pad + v)
            padded.append(row)

        if return_tensors == "pt":
            batch["iter_count_labels"] = torch.tensor(padded, dtype=torch.long)
        else:
            batch["iter_count_labels"] = np.asarray(padded, dtype=np.int64)
        return batch
