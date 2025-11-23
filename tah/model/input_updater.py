from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from tah.model.recurrent_transformer import TaHForCausalLM

from tah.model.registry import register_input_updater, get_input_updater_class, capture_init_args


class InputUpdater(nn.Module, ABC):
    """
    Base class for updating input embeddings between iterations.
    
    This class is designed to efficiently handle tensors of arbitrary shape (..., x),
    where the leading dimensions can be any combination of batch, sequence, or other
    dimensions. All operations preserve the leading dimensions and only operate on
    the last dimension for vocabulary/embedding operations.
    """

    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        prev_inputs: torch.Tensor,
        embedding_weight: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return updated inputs for the next iteration.
        
        This method efficiently handles tensors of arbitrary shape, preserving all
        leading dimensions while operating only on the embedding dimension.
        
        Args:
            logits: The logits from the token, shape (..., vocab_size)
            prev_inputs: The previous inputs, shape (..., embed_dim)
            embedding_weight: The embedding weight tensor, shape (vocab_size, embed_dim)
            hidden_states: The hidden states, shape (..., hidden_dim)

        Returns:
            The updated inputs, shape (..., embed_dim)
            
        Note:
            All leading dimensions (...) are preserved exactly. The implementation
            is optimized for efficient processing regardless of the number or size
            of leading dimensions (e.g., batch size, sequence length, etc.).
        """
    

@register_input_updater
@capture_init_args
class TrivialUpdater(InputUpdater):
    """
    Trivial update that directly returns logits-weighted embeddings.
    
    Efficiently handles tensors of arbitrary shape (..., vocab_size), preserving
    all leading dimensions while computing weighted embeddings.
    """

    def __init__(self, use_hidden_states: bool = False, topk: Optional[int] = None):
        super().__init__()
        self.use_hidden_states = use_hidden_states
        self.topk = topk

    def forward(
        self,
        logits: torch.Tensor,
        prev_inputs: torch.Tensor,
        embedding_weight: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Direct matrix multiplication preserves all leading dimensions: (..., vocab_size) @ (vocab_size, embed_dim) -> (..., embed_dim)
        if self.use_hidden_states:
            return hidden_states[...,-1,:] # shape: seq_len, num_layer, embed_dim
        else:
            if self.topk is not None:
                topk_values, topk_indices = torch.topk(logits, k=min(self.topk, logits.size(-1)), dim=-1)
                topk_probs = torch.softmax(topk_values, dim=-1)
                topk_embeddings = embedding_weight[topk_indices]
                return torch.sum(topk_probs.unsqueeze(-1) * topk_embeddings, dim=-2)
            else:
                return torch.softmax(logits, dim=-1) @ embedding_weight


