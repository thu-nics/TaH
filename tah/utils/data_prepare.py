from typing import Dict
import numpy as np
from datasets import load_from_disk
from functools import partial
from accelerate import Accelerator


def _infer_strategy_from_iter_decider(model_config: Dict, is_eval: bool = False) -> str | None:
    """
    Infer hard token strategy from iter_decider configuration in model_config.
    Returns one of {"mismatch", "divergent"} or None when not applicable.
    """
    # Select decider source
    decider_key = 'eval_iter_decider' if is_eval and 'eval_iter_decider' in model_config else 'iter_decider'
    decider_kwargs_key = 'eval_iter_decider_kwargs' if is_eval and 'eval_iter_decider_kwargs' in model_config else 'iter_decider_kwargs'

    iter_decider = model_config.get(decider_key)
    iter_decider_kwargs = model_config.get(decider_kwargs_key, {}) or {}

    if iter_decider is None:
        return None

    # Direct FixedLabelIterDecider
    if iter_decider == 'FixedLabelIterDecider':
        label_type = (iter_decider_kwargs or {}).get('label_type', None)
        return label_type if label_type in {"mismatch", "divergent"} else None

    # SmoothTransitionIterDecider: look at initial decider
    if iter_decider == 'SmoothTransitionIterDecider':
        init_cls = iter_decider_kwargs.get('initial_iter_decider_cls')
        init_kwargs = iter_decider_kwargs.get('initial_iter_decider_kwargs', {}) or {}
        if init_cls == 'FixedLabelIterDecider':
            label_type = init_kwargs.get('label_type', None)
            return label_type if label_type in {"mismatch", "divergent"} else None

    return None

def calculate_hard_token_ratio(dataset, hard_token_strategy: str, accelerator: Accelerator):
    """
    Optimized calculation of hard token ratio using batch processing and vectorized operations.
    """
        
    accelerator.print(f"Calculating hard token ratio for {hard_token_strategy} strategy...")
    
    def calculate_stats_batch(examples):
        """Calculate statistics for a batch of examples."""
        batch_labels = []
        batch_iter_count_labels = []
        
        for i in range(len(examples['labels'])):
            batch_labels.extend(examples['labels'][i])
            batch_iter_count_labels.extend(examples['iter_count_labels'][i])
        
        # Convert to numpy for vectorized operations
        labels_np = np.array(batch_labels)
        iter_count_labels_np = np.array(batch_iter_count_labels)
        
        # Calculate masks
        valid_mask = labels_np != -100
        hard_mask = iter_count_labels_np > 1
        
        valid_tokens = np.sum(valid_mask)
        hard_tokens = np.sum(valid_mask & hard_mask)
        
        return {
            'valid_tokens': [valid_tokens],
            'hard_tokens': [hard_tokens]
        }
    
    # Use map with batched=True for efficient processing
    # Ensure only the main process performs the computation first to populate cache
    with accelerator.main_process_first():
        stats_dataset = dataset.map(
            calculate_stats_batch,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            desc="Calculating token statistics"
        )
    
    # Sum up the results
    total_valid_tokens = sum(stats_dataset['valid_tokens'])
    total_hard_tokens = sum(stats_dataset['hard_tokens'])
    
    if total_valid_tokens > 0:
        avg_hard_ratio = total_hard_tokens / total_valid_tokens
        accelerator.print(f"hard token ratio for {hard_token_strategy}: {avg_hard_ratio:.4f}")
        accelerator.print(f"  - Total hard tokens: {total_hard_tokens}")
        accelerator.print(f"  - Total valid tokens: {total_valid_tokens}")
        return avg_hard_ratio
    
    return None

def preprocess_for_sft_batch(
    examples: Dict,
    max_length: int,
    max_iter: int,
    iter_count_strategy: str,
    iter_count_strategy_kwargs: Dict,
    iter_count_label_strategy: str,
    query_iter_count: int = 1,
    max_length_action: str = "cutoff",
) -> Dict:
    """
    Optimized batch processing function for better performance with large datasets.
    """
    batch_size = len(examples['real_token'])
    batch_input_ids = []
    batch_iter_count_labels = []
    batch_attention_mask = []
    batch_labels = []
    
    for i in range(batch_size):
        input_ids = examples['real_token'][i]
        prompt_mask = examples['mask'][i]
        
        # skip this data if prompt_mask is all zeros
        if not any(prompt_mask):
            continue
            
        mismatch_mask = examples['mismatch'][i] if 'mismatch' in examples else None
        divergent_mask = examples['divergent'][i] if 'divergent' in examples else None
        entropy = examples['entropy'][i] if 'entropy' in examples else None
        cross_entropy = examples['cross_entropy'][i] if 'cross_entropy' in examples else None
        ds_divergence = examples['ds_divergence'][i] if 'ds_divergence' in examples else None
        
        # Apply truncation or filtering if max_length is specified
        if max_length is not None and len(input_ids) > max_length:
            action = (max_length_action or "cutoff").lower()
            if action == "filter":
                # skip this sample entirely when it exceeds max_length
                continue
            # default: cutoff
            input_ids = input_ids[:max_length]
            prompt_mask = prompt_mask[:max_length]
            mismatch_mask = mismatch_mask[:max_length] if mismatch_mask is not None else None
            divergent_mask = divergent_mask[:max_length] if divergent_mask is not None else None
            entropy = entropy[:max_length] if entropy is not None else None
            cross_entropy = cross_entropy[:max_length] if cross_entropy is not None else None
        
        # Vectorized label creation - replace prompt tokens with -100
        labels = np.array(input_ids, dtype=np.int64)
        prompt_mask_np = np.array(prompt_mask)
        labels[prompt_mask_np == 0] = -100
        labels = labels.tolist()
        
        attention_mask = [1] * len(input_ids)
        
        iter_count_labels = np.ones(len(input_ids), dtype=np.int32)

        # Initialize iter_count/iter_count_labels independently based on strategies
        if max_iter > 1:
            def _sample_random_indices(length: int, k: int) -> np.ndarray:
                if k <= 0:
                    return np.array([], dtype=np.int64)
                k = min(k, length)
                return np.random.choice(length, k, replace=False)

            def _select_indices_by_entropy(entropy_list, k: int) -> np.ndarray:
                if k <= 0:
                    return np.array([], dtype=np.int64)
                entropy_np = np.array(entropy_list)
                k = min(k, len(entropy_np))
                return np.argsort(entropy_np)[-k:]

            def _assign_binned_iter_values(sorted_idx: np.ndarray, values: np.ndarray, max_iter: int) -> None:
                num_tokens = len(sorted_idx)
                if num_tokens == 0:
                    return
                
                num_bins = max_iter - 1
                base = num_tokens // num_bins if num_bins > 0 else num_tokens
                rem = num_tokens % num_bins if num_bins > 0 else 0
                start = 0
                for bin_id in range(max(num_bins, 1)):
                    size = base + (1 if bin_id < rem else 0)
                    if size <= 0:
                        continue
                    end = start + size
                    bin_indices = sorted_idx[start:end]
                    iter_value = max_iter - bin_id if num_bins > 0 else 2
                    iter_value = max(2, min(max_iter, iter_value))
                    values[bin_indices] = iter_value
                    start = end

            def compute_iter_values(strategy: str, strategy_kwargs: Dict | None = None, base_counts: np.ndarray | None = None, prompt_mask_np: np.ndarray = None) -> np.ndarray:
                values = np.ones(len(input_ids), dtype=np.int32)
                if strategy == "copy":
                    # copy only makes sense for labels; fall back to ones if base_counts is None
                    return base_counts.copy() if base_counts is not None else values
                if strategy == "random":
                    random_token_ratio = strategy_kwargs.get('random_token_ratio', 0.1)
                    # Only select from valid tokens (prompt_mask==1)
                    valid_idx = np.nonzero(prompt_mask_np)[0]
                    if len(valid_idx) > 0:
                        num = int(len(valid_idx) * random_token_ratio)
                        idx = _sample_random_indices(len(valid_idx), num)
                        if len(idx) > 0:
                            selected_idx = valid_idx[idx]
                            values[selected_idx] = np.random.randint(2, max_iter + 1, len(selected_idx))
                elif strategy == "top_entropy":
                    entropy_token_ratio = strategy_kwargs.get('top_entropy_ratio', 0.1)
                    # Only select from valid tokens (prompt_mask==1)
                    valid_idx = np.nonzero(prompt_mask_np)[0]
                    if len(valid_idx) > 0:
                        valid_entropy = [entropy[i] for i in valid_idx]
                        num = int(len(valid_idx) * entropy_token_ratio)
                        relative_idx = _select_indices_by_entropy(valid_entropy, num)
                        if len(relative_idx) > 0:
                            selected_idx = valid_idx[relative_idx]
                            values[selected_idx] = np.random.randint(2, max_iter + 1, len(selected_idx))
                elif strategy == "mismatch":
                    mismatch_np = np.array(mismatch_mask)
                    if np.any(mismatch_np > 1):
                        values = mismatch_np + 1
                    elif cross_entropy is not None:
                        # If given top_ce_ratio, select the top ce tokens from mismatch tokens
                        ce_np = np.array(cross_entropy)
                        # Only consider valid tokens (prompt_mask==1)
                        mismatch_and_valid = mismatch_np & prompt_mask_np
                        mismatch_idx = np.nonzero(mismatch_and_valid)[0]
                        if len(mismatch_idx) > 0:
                            ce_on_mismatch = ce_np[mismatch_idx]
                            order = np.argsort(-ce_on_mismatch)
                            sorted_idx = mismatch_idx[order]
                            # If top_ce_ratio is specified, only keep top ratio of tokens
                            if 'top_ce_ratio' in strategy_kwargs:
                                top_ce_ratio = strategy_kwargs['top_ce_ratio']
                                num_top_ce = max(1, int(len(sorted_idx) * top_ce_ratio))
                                sorted_idx = sorted_idx[:num_top_ce]
                            
                            # Use the extracted function to assign binned iter values
                            _assign_binned_iter_values(sorted_idx, values, max_iter)
                    else:
                        # Only consider valid tokens (prompt_mask==1)
                        mismatch_and_valid = mismatch_np & prompt_mask_np
                        idx = np.nonzero(mismatch_and_valid)[0]
                        if len(idx) > 0:
                            values[idx] = np.random.randint(2, max_iter + 1, len(idx))
                elif strategy == "top_ce":
                    # Select top ce_ratio tokens with highest cross entropy as hard tokens
                    if cross_entropy is not None and 'top_ce_ratio' in strategy_kwargs:
                        ce_np = np.array(cross_entropy)
                        top_ce_ratio = strategy_kwargs['top_ce_ratio']
                        
                        # Get all valid token indices (excluding prompt tokens)
                        valid_idx = np.nonzero(prompt_mask_np)[0]
                        if len(valid_idx) > 0:
                            # Get cross entropy values for valid tokens
                            ce_on_valid = ce_np[valid_idx]
                            # Sort by cross entropy (highest first)
                            order = np.argsort(-ce_on_valid)
                            sorted_idx = valid_idx[order]
                            
                            # Select top ratio tokens
                            num_top_ce = max(1, int(len(sorted_idx) * top_ce_ratio))
                            sorted_idx = sorted_idx[:num_top_ce]
                            
                            # Use the extracted function to assign binned iter values
                            _assign_binned_iter_values(sorted_idx, values, max_iter)
                elif strategy == "ds_divergence":
                    # Select top ds_ratio tokens with highest ds_divergence as hard tokens
                    if ds_divergence is not None and 'top_ds_ratio' in strategy_kwargs:
                        ds_np = np.array(ds_divergence)
                        top_ds_ratio = strategy_kwargs['top_ds_ratio']
                        # Get all valid token indices (excluding prompt tokens)
                        valid_idx = np.nonzero(prompt_mask_np)[0]
                        if len(valid_idx) > 0:
                            # Get ds_divergence values for valid tokens
                            ds_on_valid = ds_np[valid_idx]
                            # Sort by ds_divergence (highest first)
                            order = np.argsort(-ds_on_valid)
                            sorted_idx = valid_idx[order]
                            # Select top ratio tokens
                            num_top_ds = max(1, int(len(sorted_idx) * top_ds_ratio))
                            sorted_idx = sorted_idx[:num_top_ds]
                            # Use the extracted function to assign binned iter values
                            _assign_binned_iter_values(sorted_idx, values, max_iter)
                elif strategy == "nonmismatch":
                    nonmismatch_np = np.array(mismatch_mask) == 0
                    if np.any(nonmismatch_np > 1):
                        values = nonmismatch_np + 1
                    else:
                        # Only consider valid tokens (prompt_mask==1)
                        nonmismatch_and_valid = nonmismatch_np & prompt_mask_np
                        idx = np.nonzero(nonmismatch_and_valid)[0]
                        if len(idx) > 0:
                            values[idx] = np.random.randint(2, max_iter + 1, len(idx))
                elif strategy == "divergent":
                    # Only consider valid tokens (prompt_mask==1)
                    divergent_and_valid = np.array(divergent_mask) & prompt_mask_np
                    idx = np.nonzero(divergent_and_valid)[0]
                    if len(idx) > 0:
                        values[idx] = np.random.randint(2, max_iter + 1, len(idx))
                elif strategy == "all":
                    # Only assign to valid tokens (prompt_mask==1)
                    valid_idx = np.nonzero(prompt_mask_np)[0]
                    if len(valid_idx) > 0:
                        values[valid_idx] = np.random.randint(2, max_iter + 1, len(valid_idx))
                elif strategy == "maxiter":
                    # Only assign to valid tokens (prompt_mask==1)
                    valid_idx = np.nonzero(prompt_mask_np)[0]
                    if len(valid_idx) > 0:
                        values[valid_idx] = max_iter
                return values

            # Compute arrays (iter_count is local helper, not returned)
            iter_count = compute_iter_values(iter_count_strategy, iter_count_strategy_kwargs, prompt_mask_np=prompt_mask_np)
            iter_count_labels = compute_iter_values(iter_count_label_strategy, base_counts=iter_count, prompt_mask_np=prompt_mask_np)
            
            # Ensure prompt tokens use configured iteration for both arrays
            iter_count[prompt_mask_np == 0] = query_iter_count
            iter_count_labels[prompt_mask_np == 0] = query_iter_count
        
        batch_input_ids.append(input_ids)
        batch_iter_count_labels.append(iter_count_labels.tolist())
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        
    if max_iter > 1:
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels,
            'iter_count_labels': batch_iter_count_labels,
        }
    else:
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels,
        }


def preprocess_dataset(training_config: Dict, data_config: Dict, model_config: Dict, accelerator: Accelerator):
    """Load and preprocess the dataset with optimizations for large datasets."""
    accelerator.print("Loading dataset...")
    # get data config
    train_data_path = data_config['train_data_path']
    eval_data_path = data_config.get('eval_data_path', None)
    train_data_ratio = data_config.get('train_data_ratio', 1.0)
    eval_data_ratio = data_config.get('eval_data_ratio', 0.05)
    
    # get hard token strategy and relative weight
    # Prefer strategy inferred from iter_decider config; fallback to training config
    hard_token_strategy = _infer_strategy_from_iter_decider(model_config, is_eval=False)
    eval_hard_token_strategy = _infer_strategy_from_iter_decider(model_config, is_eval=True)
    hard_relative_weight = training_config.get('hard_token_relative_weight', 1.0)

    # get model config
    max_iter = model_config.get('max_iter', 1)
    
    # Load train dataset
    with accelerator.main_process_first():
        raw_train_dataset = load_from_disk(train_data_path)
    
    if train_data_ratio != 1.0:
        accelerator.print(f"Using {train_data_ratio} of train dataset")
        raw_train_dataset = raw_train_dataset.select(range(int(len(raw_train_dataset) * train_data_ratio)))
    
    # Check if eval_data_path is provided and exists
    raw_eval_dataset = None
    use_separate_eval = False
    if eval_data_path and eval_data_path.strip():
        try:
            with accelerator.main_process_first():
                raw_eval_dataset = load_from_disk(eval_data_path)
            use_separate_eval = True
            accelerator.print(f"Using separate evaluation dataset from: {eval_data_path}")
            accelerator.print(f"Training with full train dataset")
        except Exception as e:
            accelerator.print(f"Warning: Could not load eval dataset from {eval_data_path}: {e}")
            accelerator.print("Will split train dataset instead")
    
    if not use_separate_eval:
        accelerator.print("No separate eval dataset provided, will split train dataset using ratio")
    
    # Get max_length and action from config if available
    max_length = data_config.get('max_length', None)
    max_length_action = (data_config.get('max_length_action', 'cutoff') or 'cutoff').lower()
    if max_length_action not in {"cutoff", "filter"}:
        max_length_action = "cutoff"
    
    # Optimized sequence length analysis using numpy for train dataset
    accelerator.print("Analyzing sequence lengths...")
    # Use map with batched=True for faster processing
    def get_lengths_batch(examples):
        return {"lengths": [len(tokens) for tokens in examples['real_token']]}
    
    with accelerator.main_process_first():
        lengths_dataset = raw_train_dataset.map(
            get_lengths_batch, 
            batched=True, 
            batch_size=1000,
            num_proc=16,  # Use multiple processes
            remove_columns=raw_train_dataset.column_names
        )
    
    raw_lengths = lengths_dataset['lengths']
    
    if max_length:
        accelerator.print(f"Using max_length: {max_length}")
        long_sequences = sum(1 for length in raw_lengths if length > max_length)
        if long_sequences > 0:
            if max_length_action == "filter":
                accelerator.print(f"Warning: {long_sequences} sequences will be filtered out")
            else:
                accelerator.print(f"Warning: {long_sequences} sequences will be truncated")
    
    # Prefilter samples in a single pass to avoid batched map row count mismatch
    def _prefilter_batch(examples):
        masks = examples['mask']
        tokens = examples['real_token']
        mismatches = examples['mismatch'] if 'mismatch' in examples else None
        enforce_len = bool(max_length) and (max_length_action == "filter")
        apply_window = bool(max_length) and (max_length_action == "cutoff")
        keep = []
        for i in range(len(tokens)):
            end = max_length if apply_window else len(tokens[i])
            mask_i = masks[i][:end]
            mismatch_i = mismatches[i][:end] if mismatches is not None else None
            ok = any(mask_i)
            if ok and mismatch_i is not None:
                has_mismatch_on_mask = any((mask_i[j] == 1) and (mismatch_i[j] == 1) for j in range(len(mask_i)))
                ok = ok and has_mismatch_on_mask
            if enforce_len:
                ok = ok and (len(tokens[i]) <= max_length)
            keep.append(ok)
        return keep

    with accelerator.main_process_first():
        raw_train_dataset = raw_train_dataset.filter(
            _prefilter_batch,
            batched=True,
            batch_size=1000,
            num_proc=16,
            desc="Prefiltering train dataset"
        )

    if use_separate_eval and raw_eval_dataset is not None:
        with accelerator.main_process_first():
            raw_eval_dataset = raw_eval_dataset.filter(
                _prefilter_batch,
                batched=True,
                batch_size=1000,
                num_proc=16,
                desc="Prefiltering eval dataset"
            )

    accelerator.print("Preprocessing datasets...")
    num_proc = 16                               # use 16 processes to parallel process
    batch_size = 2000                           # batch size
    
    # New: separate strategies for iter_count and iter_count_labels
    # Backward-compat: fall back to model-inferred hard_token_strategy for iter_count; labels default to copy
    iter_count_strategy = data_config.get('iter_count_strategy', hard_token_strategy or 'auto')
    iter_count_strategy_kwargs = data_config.get('iter_count_strategy_kwargs', {})
    iter_count_label_strategy = data_config.get('iter_count_label_strategy', 'copy')
    query_iter_count = int(data_config.get('query_iter_count', 1))
    preprocess_fn = partial(
        preprocess_for_sft_batch, 
        max_iter=max_iter,
        max_length=max_length, 
        iter_count_strategy=iter_count_strategy, 
        iter_count_strategy_kwargs=iter_count_strategy_kwargs,
        iter_count_label_strategy=iter_count_label_strategy,
        query_iter_count=query_iter_count,
        max_length_action=max_length_action,
    )
    
    # Determine columns to remove
    remove_cols = ['data_id', 'real_text', 'real_token', 'mask']
    if 'divergent' in raw_train_dataset.column_names:
        remove_cols.append('divergent')
    # Remove entropy to avoid tokenizer.pad trying to collate it
    if 'entropy' in raw_train_dataset.column_names:
        remove_cols.append('entropy')
    # Remove ds_divergence for the same reason as entropy
    if 'ds_divergence' in raw_train_dataset.column_names:
        remove_cols.append('ds_divergence')
    if 'mismatch' in raw_train_dataset.column_names:
        remove_cols.append('mismatch')
    
    if use_separate_eval:
        # Process train and eval datasets separately
        accelerator.print("Processing train dataset...")
        with accelerator.main_process_first():
            processed_train_dataset = raw_train_dataset.map(
                preprocess_fn,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                remove_columns=remove_cols,
                desc="Processing train dataset"
            )
        
        accelerator.print("Processing eval dataset...")
        # Determine eval columns to remove
        eval_remove_cols = ['data_id', 'real_text', 'real_token', 'mask']
        if 'divergent' in raw_eval_dataset.column_names:
            eval_remove_cols.append('divergent')
        if 'mismatch' in raw_eval_dataset.column_names:
            eval_remove_cols.append('mismatch')
        if 'problem_idx' in raw_eval_dataset.column_names:
            eval_remove_cols.append('problem_idx')
        if 'entropy' in raw_eval_dataset.column_names:
            eval_remove_cols.append('entropy')
        if 'ds_divergence' in raw_eval_dataset.column_names:
            eval_remove_cols.append('ds_divergence')

        eval_preprocess_fn = partial(
            preprocess_for_sft_batch,
            max_iter=max_iter,
            max_length=max_length,
            iter_count_strategy=iter_count_strategy,
            iter_count_strategy_kwargs=iter_count_strategy_kwargs,
            iter_count_label_strategy=iter_count_label_strategy,
            query_iter_count=query_iter_count,
            max_length_action=max_length_action,
        )
            
        with accelerator.main_process_first():
            processed_eval_dataset = raw_eval_dataset.map(
                eval_preprocess_fn,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                remove_columns=eval_remove_cols,
                desc="Processing eval dataset"
            )

    else:
        # Process the combined dataset first, then split
        accelerator.print("Processing combined dataset...")
        with accelerator.main_process_first():
            processed_dataset = raw_train_dataset.map(
                preprocess_fn,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                remove_columns=remove_cols,
                desc="Processing dataset"
            )
        
        # Split dataset into train and eval after preprocessing
        accelerator.print("Splitting dataset into train and eval...")
        if eval_data_ratio > 0:
            split_dataset = processed_dataset.train_test_split(test_size=eval_data_ratio, seed=42)
            processed_train_dataset = split_dataset['train']
            processed_eval_dataset = split_dataset['test']
        else:
            processed_train_dataset = processed_dataset
            processed_eval_dataset = None
    
    if hard_relative_weight != 1.0:
        # Calculate hard token ratio using processed train dataset
        avg_hard_ratio = calculate_hard_token_ratio(
            processed_train_dataset, hard_token_strategy, accelerator
        )
    else:
        avg_hard_ratio = None
    
    accelerator.print(f"Train dataset size: {len(processed_train_dataset)}")
    if processed_eval_dataset is not None:
        accelerator.print(f"Eval dataset size: {len(processed_eval_dataset)}")
    
    return processed_train_dataset, processed_eval_dataset, avg_hard_ratio
