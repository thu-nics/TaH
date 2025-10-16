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
        
        # Extract iter_count from features if present
        iter_counts = []
        has_iter_count = False
        if features and 'iter_count' in features[0]:
            has_iter_count = True
            iter_counts = [feature.pop('iter_count') for feature in features]
        
        # Use base collator for standard fields (input_ids, attention_mask, labels)
        batch = self.base_collator(features, return_tensors=return_tensors)
        
        # Handle iter_count field if present
        if has_iter_count and iter_counts:
            # Get padding configuration
            no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
            
            if no_padding:
                # No padding case
                if isinstance(iter_counts[0], list):
                    batch["iter_count"] = list(iter_counts)
                else:
                    batch["iter_count"] = [np.concatenate([iter_count, []]) for iter_count in iter_counts]
            else:
                # Padding case - align with input_ids padding
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                if max_padding:
                    max_iter_length = self.max_length
                else:
                    max_iter_length = max(len(iter_count) for iter_count in iter_counts)
                
                # Apply pad_to_multiple_of if specified
                if self.pad_to_multiple_of is not None:
                    max_iter_length = (
                        (max_iter_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )
                
                # Determine padding side
                padding_side = self.tokenizer.padding_side
                
                # Pad iter_count sequences
                if isinstance(iter_counts[0], list):
                    batch["iter_count"] = [
                        iter_count + [self.iter_count_pad_value] * (max_iter_length - len(iter_count))
                        if padding_side == "right"
                        else [self.iter_count_pad_value] * (max_iter_length - len(iter_count)) + iter_count
                        for iter_count in iter_counts
                    ]
                else:
                    batch["iter_count"] = [
                        np.concatenate([
                            iter_count,
                            np.array([self.iter_count_pad_value] * (max_iter_length - len(iter_count)), dtype=np.int64)
                        ]) if padding_side == "right"
                        else np.concatenate([
                            np.array([self.iter_count_pad_value] * (max_iter_length - len(iter_count)), dtype=np.int64),
                            iter_count
                        ])
                        for iter_count in iter_counts
                    ]
        
        # Convert iter_count to tensors if needed
        if has_iter_count and batch.get("iter_count", None) is not None:
            if return_tensors == "pt":
                import torch
                batch["iter_count"] = torch.tensor(batch["iter_count"], dtype=torch.long)
            else:
                batch["iter_count"] = np.array(batch["iter_count"], dtype=np.int64)

        return batch
