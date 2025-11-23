"""
Custom Causal Cache implementation for TaH that supports hierarchical iteration access.
"""

import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.cache_utils import Cache, DynamicCache


class TaHCache(DynamicCache):
    """
    A cache that supports hierarchical iteration access where deeper iterations
    can see cache from all previous iterations, but previous iterations cannot
    see cache from future iterations.

    This enables parallel prefilling at each iteration depth while maintaining
    causal constraints across iterations.
    """

    def __init__(self):
        super().__init__()
        # Structure: {layer_idx: {iter_depth: Tensor_data}}
        # Use internal names to avoid clashing with transformers.Cache properties
        self._tah_key_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        self._tah_value_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        self._tah_position_id_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        self._tah_valid_mask_cache: Dict[int, Dict[int, torch.Tensor]] = {}

        self.current_iter_depth = 0
        self.batch_size: Optional[int] = None  # Track current batch size

        self._device = None
        self._dtype = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the given key_states and value_states for the layer layer_idx.

        Args:
            key_states: The key states to cache (batch_size, num_heads, seq_len, head_dim)
            value_states: The value states to cache (batch_size, num_heads, seq_len, head_dim)
            layer_idx: The index of the layer to cache the states for
            cache_kwargs: Additional arguments, should include cache_position to indicate the position ids of the current iteration

        Returns:
            Tuple containing the concatenated key and value states from all accessible iterations
        """
        # Update batch size from input tensors
        self.batch_size = (
            key_states.shape[0] if self.batch_size is None else self.batch_size
        )
        assert (
            self.batch_size == key_states.shape[0]
            and self.batch_size == value_states.shape[0]
        ), "Batch size mismatch, expected {}, got {} and {}".format(
            self.batch_size, key_states.shape[0], value_states.shape[0]
        )

        # Get iteration depth, position, and token mask, all set outside of this function
        iter_depth = self.current_iter_depth
        new_position_ids = self.position_ids_to_cache
        new_valid_mask = self.valid_mask_to_cache

        # Initialize layer cache if needed
        if layer_idx not in self._tah_key_cache:
            self._tah_key_cache[layer_idx] = {}
            self._tah_value_cache[layer_idx] = {}
            self._tah_position_id_cache[layer_idx] = {}
            self._tah_valid_mask_cache[layer_idx] = {}

        # Update cache for this iteration depth
        if iter_depth in self._tah_key_cache[layer_idx]:
            # Concatenate with existing cache for this iteration depth
            self._tah_key_cache[layer_idx][iter_depth] = torch.cat(
                [self._tah_key_cache[layer_idx][iter_depth], key_states], dim=-2
            )
            self._tah_value_cache[layer_idx][iter_depth] = torch.cat(
                [self._tah_value_cache[layer_idx][iter_depth], value_states], dim=-2
            )
            self._tah_position_id_cache[layer_idx][iter_depth] = torch.cat(
                [self._tah_position_id_cache[layer_idx][iter_depth], new_position_ids],
                dim=-1,
            )
            self._tah_valid_mask_cache[layer_idx][iter_depth] = torch.cat(
                [self._tah_valid_mask_cache[layer_idx][iter_depth], new_valid_mask], dim=-1
            )
        else:
            # First entry for this iteration depth
            self._tah_key_cache[layer_idx][iter_depth] = key_states
            self._tah_value_cache[layer_idx][iter_depth] = value_states
            self._tah_position_id_cache[layer_idx][iter_depth] = new_position_ids
            self._tah_valid_mask_cache[layer_idx][iter_depth] = new_valid_mask

        # Return concatenated cache from all accessible iterations (0 to iter_depth)
        return self.get_cache_upto_iter(layer_idx, iter_depth)

    @property
    def current_iter_depth(self) -> int:
        """
        Get the current iteration depth.
        """
        return self._current_iter_depth

    @current_iter_depth.setter
    def current_iter_depth(self, iter_depth: int):
        self._current_iter_depth = iter_depth

    @property
    def position_ids_to_cache(self) -> torch.Tensor:
        """
        Get the position ids to cache for the current iteration depth, shape: (batch_size, seq_len)
        """
        # Default position ids
        # batch_size = key_states.shape[0]
        # seq_length = key_states.shape[-2]
        # kv_cache_length_this_iter = self.get_cache_length(layer_idx, iter_depth)
        # position_ids = torch.arange((batch_size, seq_length + kv_cache_length_this_iter), device=key_states.device, dtype=torch.long)
        return self._position_ids_to_cache

    @position_ids_to_cache.setter
    def position_ids_to_cache(self, position_ids: torch.Tensor):
        self._position_ids_to_cache = position_ids

    @property
    def valid_mask_to_cache(self) -> torch.Tensor:
        """
        Get the token mask to cache for the current iteration depth, shape: (batch_size, seq_len)
        """
        # torch.ones_like(new_position_ids)
        return self._valid_mask_to_cache

    @valid_mask_to_cache.setter
    def valid_mask_to_cache(self, valid_mask: torch.Tensor):
        self._valid_mask_to_cache = valid_mask

    def get_position_id_upto_iter(
        self, layer_idx: int, upto_iter_idx: int, init_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Get the position id upto a given layer and iteration depth.

        Args:
            layer_idx: Layer index
            upto_iter_idx: Maximum iteration depth to include
            batch_size: Batch size if the position id is not cached for the given layer and iteration depth

        Returns:
            Position id of shape (batch_size, total sequence length until current iteration depth)
        """

        def _get_position_id_of_iter(
            self, layer_idx: int, iter_idx: int, batch_size: int = 1
        ) -> torch.Tensor:
            """
            Get the position id for a given layer and iteration depth.
            """
            if (layer_idx not in self._tah_position_id_cache) or (
                iter_idx not in self._tah_position_id_cache[layer_idx]
            ):
                return torch.empty(
                    size=(batch_size, 0), device=self.device, dtype=torch.long
                )
            else:
                return self._tah_position_id_cache[layer_idx][iter_idx]

        all_position_ids = []
        batch_size = init_batch_size
        for iter_depth in range(upto_iter_idx + 1):
            position_id = _get_position_id_of_iter(
                self, layer_idx, iter_depth, batch_size
            )
            batch_size = position_id.shape[0]
            all_position_ids.append(position_id)

        return torch.cat(all_position_ids, dim=-1)

    def get_valid_mask_upto_iter(
        self, layer_idx: int, upto_iter_idx: int, init_batch_size: int = 1
    ) -> torch.Tensor:
        """
        Get the token mask upto a given layer and iteration depth.
        upto_iter_idx=0 means getting the valid mask of the first iteration
        """

        def _get_valid_mask_of_iter(
            self, layer_idx: int, iter_idx: int, batch_size: int = 1
        ) -> torch.Tensor:
            """
            Get the token mask for a given layer and iteration depth.
            """
            if (layer_idx not in self._tah_valid_mask_cache) or (
                iter_idx not in self._tah_valid_mask_cache[layer_idx]
            ):
                return torch.empty(
                    size=(batch_size, 0), device=self.device, dtype=torch.long
                )
            else:
                return self._tah_valid_mask_cache[layer_idx][iter_idx]

        all_valid_masks = []
        batch_size = init_batch_size
        for iter_depth in range(upto_iter_idx + 1):
            valid_mask = _get_valid_mask_of_iter(
                self, layer_idx, iter_depth, batch_size
            )
            batch_size = valid_mask.shape[0]
            all_valid_masks.append(valid_mask)

        return torch.cat(all_valid_masks, dim=-1)

    def get_cache_iter_index_upto_iter(
        self, layer_idx: int, upto_iter_idx: int
    ) -> torch.Tensor:
        """
        Get the iter index of each KV Cache value upto a certain iter depth
        Args:
            layer_idx: Layer index
            iter_depth: Iteration depth
        Returns:
            iter id of shape (total cache length until current iteration depth, )
        """

        def _update_iter_index_of_iter(
            self, layer_idx: int, iter_idx: int
        ) -> torch.Tensor:
            """
            Get the iter id for a given layer and iteration depth.
            """
            if (layer_idx not in self._tah_position_id_cache) or (
                iter_idx not in self._tah_position_id_cache[layer_idx]
            ):
                return torch.empty(size=(0,), device=self.device, dtype=torch.long)
            else:
                return iter_idx

        cache_length_upto_iter = self.get_cache_length_upto_iter(layer_idx, upto_iter_idx)

        if cache_length_upto_iter == 0:  # the first position id does not exist
            return torch.empty(size=(0,), device=self.device, dtype=torch.long)
        else:
            iter_id_tensor = torch.zeros(
                size=(cache_length_upto_iter,), device=self.device, dtype=torch.long
            )
            cache_length_upto_current_iter = 0
            for iter_idx in range(upto_iter_idx):
                cache_length_upto_current_iter += self.get_cache_length(
                    layer_idx, iter_idx
                )
                iter_id_tensor[cache_length_upto_current_iter:] += 1
            return iter_id_tensor

    def get_cache_upto_iter(
        self, layer_idx: int, upto_iter_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get concatenated cache from iterations 0 to upto_iter_idx (inclusive).

        Args:
            layer_idx: Layer index
            upto_iter_idx: Maximum iteration depth to include

        Returns:
            Concatenated key and value states with consistent batch dimensions
        """
        if layer_idx not in self._tah_key_cache:
            return None, None

        all_keys = []
        all_values = []

        # Collect cache from all iterations up to current depth
        for iter_depth in range(upto_iter_idx + 1):
            if iter_depth in self._tah_key_cache[layer_idx]:
                all_keys.append(self._tah_key_cache[layer_idx][iter_depth])
                all_values.append(self._tah_value_cache[layer_idx][iter_depth])

        if not all_keys:
            return None, None

        concatenated_keys = torch.cat(all_keys, dim=-2)
        concatenated_values = torch.cat(all_values, dim=-2)

        return concatenated_keys, concatenated_values

    def get_cache_length(
        self, layer_idx: Optional[int] = 0, iter_idx: Optional[int] = None
    ) -> int:
        """
        Get the cache length for a given layer and iteration depth. If iter_idx is not provided, return the total cache length across all iterations.
        Args:
            layer_idx: Layer index
            iter_idx: Iteration depth
        Returns:
            Cache length
        """
        if layer_idx not in self._tah_key_cache or not self._tah_key_cache[layer_idx]:
            return 0

        total_length = 0
        if iter_idx is None:
            # Return total cache length across ALL stored iterations
            for iter_depth in self._tah_key_cache[layer_idx]:
                key_states = self._tah_key_cache[layer_idx][iter_depth]
                total_length += key_states.shape[-2]
        else:
            # Return cache length for a given iteration
            key_states = self._tah_key_cache[layer_idx][iter_idx]
            total_length = key_states.shape[-2]

        return total_length

    def get_cache_length_upto_iter(
        self, layer_idx: Optional[int] = 0, iter_depth: int = 0
    ) -> int:
        """Returns cache length from iterations 0 to before_iter_depth-1."""
        if layer_idx not in self._tah_key_cache or not self._tah_key_cache[layer_idx]:
            return 0

        total_length = 0
        for iter_depth in range(iter_depth + 1):
            if iter_depth in self._tah_key_cache[layer_idx]:
                key_states = self._tah_key_cache[layer_idx][iter_depth]
                total_length += key_states.shape[-2]

        return total_length

    def get_seq_length(
        self, layer_idx: Optional[int] = 0, iter_idx: Optional[int] = 0
    ) -> int:
        """Returns the current sequence length (max position + 1) for a given layer."""
        if layer_idx not in self._tah_position_id_cache:
            return 0

        # Find maximum position across all iterations (sequence grows during generation)
        # max_position = torch.max(self.position_id_cache[layer_idx][iter_idx]).item() # make more sense, but not exactly the same as huggingface
        max_position = self._tah_key_cache[layer_idx][iter_idx].shape[-2]
        return max_position

    # Not used
    def get_max_length(self) -> Optional[int]:
        """Returns the maximum cache length if it exists."""
        return None  # Dynamic cache has no maximum length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object."""
        return None  # Dynamic cache has no maximum length

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        return self.get_cache_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache according to beam_idx for beam search."""
        for layer_idx in self._tah_key_cache:
            for iter_depth in self._tah_key_cache[layer_idx]:
                device = self._tah_key_cache[layer_idx][iter_depth].device
                # Reorder key cache
                self._tah_key_cache[layer_idx][iter_depth] = self._tah_key_cache[layer_idx][
                    iter_depth
                ].index_select(0, beam_idx.to(device))
                # Reorder value cache
                self._tah_value_cache[layer_idx][iter_depth] = self._tah_value_cache[layer_idx][
                    iter_depth
                ].index_select(0, beam_idx.to(device))

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> Tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        """
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_cache_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, 0

    @property
    def device(self) -> torch.device:
        """Returns the device of the cached tensors."""
        # use the device of the first cached tensor
        key_device = None
        for layer_idx in self._tah_key_cache:
            for iter_depth in self._tah_key_cache[layer_idx]:
                if self._tah_key_cache[layer_idx][iter_depth] is not None:
                    return self._tah_key_cache[layer_idx][iter_depth].device

        # Else, use cpu as default
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the cached tensors."""
        # use the dtype of the first cached tensor
        for layer_idx in self._tah_key_cache:
            for iter_depth in self._tah_key_cache[layer_idx]:
                if self._tah_key_cache[layer_idx][iter_depth] is not None:
                    return self._tah_key_cache[layer_idx][iter_depth].dtype
        
        # Else, use bfloat16 as default
        return torch.bfloat16

    def to(self, *args, **kwargs) -> "TaHCache":
        """
        Move all cached tensors to the specified device and/or convert to specified dtype.
        
        Supports the same interface as PyTorch tensor.to():
        - to(device)
        - to(dtype) 
        - to(device, dtype)
        - to(device=..., dtype=...)

        Returns:
            Self for method chaining
        """
        # Parse arguments similar to PyTorch tensor.to()
        device = kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)
        
        # Handle positional arguments
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        # Convert key and value caches (both device and dtype)
        kv_cache_dicts = [self._tah_key_cache, self._tah_value_cache]
        for cache_dict in kv_cache_dicts:
            for layer_idx in cache_dict:
                for iter_depth in cache_dict[layer_idx]:
                    if cache_dict[layer_idx][iter_depth] is not None:
                        cache_dict[layer_idx][iter_depth] = cache_dict[layer_idx][iter_depth].to(
                            *([device] if device is not None else []),
                            *([dtype] if dtype is not None else [])
                        )
        
        # Convert position_id and valid_mask caches (device only, preserve dtype)
        metadata_cache_dicts = [self._tah_position_id_cache, self._tah_valid_mask_cache]
        for cache_dict in metadata_cache_dicts:
            for layer_idx in cache_dict:
                for iter_depth in cache_dict[layer_idx]:
                    if cache_dict[layer_idx][iter_depth] is not None and device is not None:
                        cache_dict[layer_idx][iter_depth] = cache_dict[layer_idx][iter_depth].to(device)
        
        return self
