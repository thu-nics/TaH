from typing import Optional, Any, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy, DataCollatorForSeq2Seq
import numpy as np


class CustomTaHDataCollator:
    """
    Custom data collator for TaH that handles iter_count field along with standard fields.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[Any] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        """
        Initialize custom data collator for TaH.

        Args:
            tokenizer: Tokenizer instance
            model: Optional model instance
            padding: Padding strategy
            max_length: Maximum length for padding
            pad_to_multiple_of: Pad to multiple of this value
            label_pad_token_id: Padding token ID for labels and iter_count_labels (default: -100)
            return_tensors: Type of tensors to return (default: "pt")
        """
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

        # Create base data collator for handling standard fields
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors,
        )

    @staticmethod
    def _pop_field(features, key):
        if not features or key not in features[0]:
            return None
        return [feature.pop(key) for feature in features]

    def _pad_2d(self, sequences, pad_value, target_rows, dtype):
        padding_side = self.tokenizer.padding_side
        max_cols = 0
        for seq in sequences:
            seq_arr = np.asarray(seq, dtype=dtype)
            if seq_arr.ndim == 2:
                max_cols = max(max_cols, seq_arr.shape[1])

        padded = np.full(
            (len(sequences), target_rows, max_cols), pad_value, dtype=dtype
        )
        for i, seq in enumerate(sequences):
            seq_arr = np.asarray(seq, dtype=dtype)
            if seq_arr.size == 0:
                continue
            if seq_arr.ndim != 2:
                raise ValueError(
                    f"Expected a 2D sequence for padding, got shape={seq_arr.shape}"
                )
            if padding_side != "right":
                row_offset = target_rows - seq_arr.shape[0]
            else:
                row_offset = 0
            padded[
                i, row_offset : row_offset + seq_arr.shape[0], : seq_arr.shape[1]
            ] = seq_arr
        return padded

    def _pad_1d(self, sequences, pad_value, target_length, dtype):
        padded = np.full((len(sequences), target_length), pad_value, dtype=dtype)
        for i, sequence in enumerate(sequences):
            sequence = np.asarray(sequence, dtype=dtype)
            if self.tokenizer.padding_side == "right":
                padded[i, : len(sequence)] = sequence
            else:
                padded[i, target_length - len(sequence) :] = sequence
        return padded

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Extract custom variable-length fields before tokenizer padding.
        iter_count_list = self._pop_field(features, "iter_count")
        iter_count_labels_list = self._pop_field(features, "iter_count_labels")
        topk_token_ids_list = self._pop_field(features, "topk_token_ids")
        topk_probs_list = self._pop_field(features, "topk_probs")
        _ = self._pop_field(features, "positions")
        _ = self._pop_field(features, "data_id")
        _ = self._pop_field(features, "raw_idx")
        _ = self._pop_field(features, "cross_entropy")
        # Sample-length column materialized by data_prepare for the dynamic
        # batcher; must not reach model(**batch).
        _ = self._pop_field(features, "length")

        # Use base collator for standard fields (input_ids, attention_mask, labels)
        batch = self.base_collator(features, return_tensors=return_tensors)

        if iter_count_list:
            target_length = batch["input_ids"].shape[1]
            batch["iter_count"] = self._pad_1d(
                iter_count_list,
                pad_value=0,
                target_length=target_length,
                dtype=np.int64,
            )
            if return_tensors == "pt":
                import torch

                batch["iter_count"] = torch.tensor(
                    batch["iter_count"], dtype=torch.long
                )

        # Handle iter_count_labels field if present
        if iter_count_labels_list:
            # Get padding configuration
            no_padding = (
                self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
            )

            if no_padding:
                # No padding case
                batch["iter_count_labels"] = list(iter_count_labels_list)
            else:
                # Padding case - strictly align with input_ids padding length
                if "input_ids" in batch:
                    max_iter_length = batch["input_ids"].shape[1]
                else:
                    # Fallback: infer from current list
                    max_iter_length = max(len(v) for v in iter_count_labels_list)

                # Apply pad_to_multiple_of if specified
                if self.pad_to_multiple_of is not None:
                    max_iter_length = (
                        (max_iter_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                # Determine padding side
                padding_side = self.tokenizer.padding_side
                pad_value = self.label_pad_token_id

                # Pad iter_count_labels sequences
                if isinstance(iter_count_labels_list[0], list):
                    batch["iter_count_labels"] = [
                        (
                            iter_count_labels
                            + [pad_value] * (max_iter_length - len(iter_count_labels))
                            if padding_side == "right"
                            else [pad_value]
                            * (max_iter_length - len(iter_count_labels))
                            + iter_count_labels
                        )
                        for iter_count_labels in iter_count_labels_list
                    ]
                else:
                    batch["iter_count_labels"] = [
                        (
                            np.concatenate(
                                [
                                    iter_count_labels,
                                    np.array(
                                        [pad_value]
                                        * (max_iter_length - len(iter_count_labels)),
                                        dtype=np.int64,
                                    ),
                                ]
                            )
                            if padding_side == "right"
                            else np.concatenate(
                                [
                                    np.array(
                                        [pad_value]
                                        * (max_iter_length - len(iter_count_labels)),
                                        dtype=np.int64,
                                    ),
                                    iter_count_labels,
                                ]
                            )
                        )
                        for iter_count_labels in iter_count_labels_list
                    ]

        # Convert iter_count_labels to tensors if needed
        if iter_count_labels_list and batch.get("iter_count_labels") is not None:
            if return_tensors == "pt":
                import torch

                batch["iter_count_labels"] = torch.tensor(
                    batch["iter_count_labels"], dtype=torch.long
                )
            else:
                batch["iter_count_labels"] = np.array(
                    batch["iter_count_labels"], dtype=np.int64
                )

        if topk_token_ids_list:
            max_positions = (
                batch["input_ids"].shape[1]
                if "input_ids" in batch
                else max(len(v) for v in topk_token_ids_list)
            )
            batch["topk_token_ids"] = self._pad_2d(
                topk_token_ids_list,
                pad_value=0,
                target_rows=max_positions,
                dtype=np.int64,
            )
            if return_tensors == "pt":
                import torch

                batch["topk_token_ids"] = torch.tensor(
                    batch["topk_token_ids"], dtype=torch.long
                )
            else:
                batch["topk_token_ids"] = np.array(
                    batch["topk_token_ids"], dtype=np.int64
                )

        if topk_probs_list:
            max_positions = (
                batch["input_ids"].shape[1]
                if "input_ids" in batch
                else max(len(v) for v in topk_probs_list)
            )
            batch["topk_probs"] = self._pad_2d(
                topk_probs_list,
                pad_value=0.0,
                target_rows=max_positions,
                dtype=np.float32,
            )
            if return_tensors == "pt":
                import torch

                batch["topk_probs"] = torch.tensor(
                    batch["topk_probs"], dtype=torch.float32
                )
            else:
                batch["topk_probs"] = np.array(batch["topk_probs"], dtype=np.float32)

        return batch
