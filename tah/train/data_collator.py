from typing import Optional, Any, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy, DataCollatorForSeq2Seq
import numpy as np

class CustomTaHDataCollator:
    """
    Custom data collator for TaH that handles iter_count field along with standard fields.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, 
                 model: Optional[Any] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 label_pad_token_id: int = -100,
                 iter_count_pad_value: int = -1,
                 return_tensors: str = "pt"):
        """
        Initialize custom data collator for TaH.
        
        Args:
            tokenizer: Tokenizer instance
            model: Optional model instance
            padding: Padding strategy
            max_length: Maximum length for padding
            pad_to_multiple_of: Pad to multiple of this value
            label_pad_token_id: Padding token ID for labels (default: -100)
            iter_count_pad_value: Padding value for iter_count (default: -1)
            return_tensors: Type of tensors to return (default: "pt")
        """
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.iter_count_pad_value = iter_count_pad_value
        self.return_tensors = return_tensors
        
        # Create base data collator for handling standard fields
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors
        )
    
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        
        # Extract iter_count_labels from features if present
        iter_count_labels_list = []
        has_iter_count_labels = False
        if features and 'iter_count_labels' in features[0]:
            has_iter_count_labels = True
            iter_count_labels_list = [feature.pop('iter_count_labels') for feature in features]
        
        # Use base collator for standard fields (input_ids, attention_mask, labels)
        batch = self.base_collator(features, return_tensors=return_tensors)
        
        # Handle iter_count_labels field if present
        if has_iter_count_labels and iter_count_labels_list:
            # Get padding configuration
            no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
            
            if no_padding:
                # No padding case
                if isinstance(iter_count_labels_list[0], list):
                    batch["iter_count_labels"] = list(iter_count_labels_list)
                else:
                    batch["iter_count_labels"] = [
                        np.concatenate([iter_count_labels, []])
                        for iter_count_labels in iter_count_labels_list
                    ]
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
                        iter_count_labels + [pad_value] * (max_iter_length - len(iter_count_labels))
                        if padding_side == "right"
                        else [pad_value] * (max_iter_length - len(iter_count_labels)) + iter_count_labels
                        for iter_count_labels in iter_count_labels_list
                    ]
                else:
                    batch["iter_count_labels"] = [
                        np.concatenate([
                            iter_count_labels,
                            np.array([pad_value] * (max_iter_length - len(iter_count_labels)), dtype=np.int64)
                        ]) if padding_side == "right"
                        else np.concatenate([
                            np.array([pad_value] * (max_iter_length - len(iter_count_labels)), dtype=np.int64),
                            iter_count_labels
                        ])
                        for iter_count_labels in iter_count_labels_list
                    ]
        
        # Convert iter_count_labels to tensors if needed
        if has_iter_count_labels and batch.get("iter_count_labels", None) is not None:
            if return_tensors == "pt":
                import torch
                batch["iter_count_labels"] = torch.tensor(batch["iter_count_labels"], dtype=torch.long)
            else:
                batch["iter_count_labels"] = np.array(batch["iter_count_labels"], dtype=np.int64)

        return batch
