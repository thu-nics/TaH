from typing import Dict
import numpy as np
from datasets import load_from_disk
from functools import partial
from accelerate import Accelerator


def _infer_strategy_from_iter_decider(model_config: Dict, is_eval: bool = False) -> str | None:
    """
    Infer important token strategy from iter_decider configuration in model_config.
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

def calculate_important_token_ratio(dataset, important_token_strategy: str, accelerator: Accelerator):
    """
    Optimized calculation of important token ratio using batch processing and vectorized operations.
    """
    if important_token_strategy not in ["mismatch", "divergent"]:
        return None
        
    accelerator.print(f"Calculating important token ratio for {important_token_strategy} strategy...")
    
    def calculate_stats_batch(examples):
        """Calculate statistics for a batch of examples."""
        batch_labels = []
        batch_iter_counts = []
        
        for i in range(len(examples['labels'])):
            batch_labels.extend(examples['labels'][i])
            batch_iter_counts.extend(examples['iter_count'][i])
        
        # Convert to numpy for vectorized operations
        labels_np = np.array(batch_labels)
        iter_counts_np = np.array(batch_iter_counts)
        
        # Calculate masks
        valid_mask = labels_np != -100
        important_mask = iter_counts_np > 1
        
        valid_tokens = np.sum(valid_mask)
        important_tokens = np.sum(valid_mask & important_mask)
        
        return {
            'valid_tokens': [valid_tokens],
            'important_tokens': [important_tokens]
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
    total_important_tokens = sum(stats_dataset['important_tokens'])
    
    if total_valid_tokens > 0:
        avg_important_ratio = total_important_tokens / total_valid_tokens
        accelerator.print(f"Important token ratio for {important_token_strategy}: {avg_important_ratio:.4f}")
        accelerator.print(f"  - Total important tokens: {total_important_tokens}")
        accelerator.print(f"  - Total valid tokens: {total_valid_tokens}")
        return avg_important_ratio
    
    return None

def preprocess_for_sft_batch(
    examples: Dict,
    max_length: int,
    max_iter: int,
    iter_count_strategy: str,
    iter_count_label_strategy: str,
    important_token_ratio: float,
    prompt_iter_count: int = 1,
    max_length_action: str = "cutoff",
) -> Dict:
    """
    Optimized batch processing function for better performance with large datasets.
    """
    batch_size = len(examples['real_token'])
    batch_input_ids = []
    batch_iter_count = []
    batch_iter_count_labels = []
    batch_attention_mask = []
    batch_labels = []
    
    for i in range(batch_size):
        input_ids = examples['real_token'][i]
        prompt_mask = examples['mask'][i]
        
        # skip this data if prompt_mask is all zeros
        if not any(prompt_mask):
            continue
            
        if 'mismatch' in examples:
            mismatch_mask = examples['mismatch'][i]
        else:
            mismatch_mask = None
        if 'divergent' in examples:
            divergent_mask = examples['divergent'][i]
        else:
            divergent_mask = None
        if 'entropy' in examples:
            entropy = examples['entropy'][i]
        else:
            entropy = None
        if 'cross_entropy' in examples:
            cross_entropy = examples['cross_entropy'][i]
        else:
            cross_entropy = None
        
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
        
        iter_count = np.ones(len(input_ids), dtype=np.int32)
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

            def compute_iter_values(strategy: str, base_counts: np.ndarray | None = None) -> np.ndarray:
                values = np.ones(len(input_ids), dtype=np.int32)
                ratio = min(max(important_token_ratio, 0.0), 1.0)
                num = int(len(input_ids) * ratio)
                if strategy == "copy":
                    # copy only makes sense for labels; fall back to ones if base_counts is None
                    return base_counts.copy() if base_counts is not None else values
                if strategy == "random":
                    idx = _sample_random_indices(len(input_ids), num)
                    if len(idx) > 0:
                        values[idx] = np.random.randint(2, max_iter + 1, len(idx))
                elif strategy == "entropy":
                    idx = _select_indices_by_entropy(entropy, num)
                    if len(idx) > 0:
                        values[idx] = np.random.randint(2, max_iter + 1, len(idx))
                elif strategy == "mismatch":
                    mismatch_np = np.array(mismatch_mask)
                    if np.any(mismatch_np > 1):
                        values = mismatch_np + 1
                    elif cross_entropy is not None:
                        ce_np = np.array(cross_entropy)
                        mismatch_idx = np.nonzero(mismatch_np)[0]
                        if len(mismatch_idx) > 0:
                            ce_on_mismatch = ce_np[mismatch_idx]
                            order = np.argsort(-ce_on_mismatch)
                            sorted_idx = mismatch_idx[order]
                            num_tokens = len(sorted_idx)
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
                    else:
                        idx = np.nonzero(mismatch_np)[0]
                        if len(idx) > 0:
                            values[idx] = np.random.randint(2, max_iter + 1, len(idx))
                elif strategy == "nonmismatch":
                    nonmismatch_np = np.array(mismatch_mask) == 0
                    if np.any(nonmismatch_np > 1):
                        values = nonmismatch_np + 1
                    else:
                        idx = np.nonzero(nonmismatch_np)[0]
                        if len(idx) > 0:
                            values[idx] = np.random.randint(2, max_iter + 1, len(idx))
                elif strategy == "divergent":
                    idx = np.nonzero(np.array(divergent_mask))[0]
                    if len(idx) > 0:
                        values[idx] = np.random.randint(2, max_iter + 1, len(idx))
                elif strategy == "all":
                    values = np.random.randint(2, max_iter + 1, len(input_ids))
                elif strategy == "maxiter":
                    values = np.ones(len(input_ids), dtype=np.int32) * max_iter
                elif strategy == "auto":
                    # for counts: -1 to indicate auto; for labels: fall back to mismatch-derived targets
                    if base_counts is None:
                        values = np.ones(len(input_ids), dtype=np.int32) * -1
                    else:
                        mismatch_np = np.array(mismatch_mask)
                        if np.any(mismatch_np > 1):
                            values = mismatch_np + 1
                        else:
                            idx = np.nonzero(mismatch_np)[0]
                            if len(idx) > 0:
                                values[idx] = np.random.randint(2, max_iter + 1, len(idx))
                return values

            # Compute arrays
            iter_count = compute_iter_values(iter_count_strategy)
            iter_count_labels = compute_iter_values(iter_count_label_strategy, base_counts=iter_count)

            # Ensure prompt tokens use configured iteration for both arrays
            iter_count[prompt_mask_np==0] = prompt_iter_count
            iter_count_labels[prompt_mask_np==0] = prompt_iter_count
        
        batch_input_ids.append(input_ids)
        batch_iter_count.append(iter_count.tolist())
        batch_iter_count_labels.append(iter_count_labels.tolist())
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        
    if max_iter > 1:
        return{
            'input_ids': batch_input_ids,
            'iter_count': batch_iter_count,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels,
            'iter_count_labels': batch_iter_count_labels
        }
    else:
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels
        }


def preprocess_dataset(training_config: Dict, data_config: Dict, model_config: Dict, accelerator: Accelerator):
    """Load and preprocess the dataset with optimizations for large datasets."""
    accelerator.print("Loading dataset...")
    # get data config
    train_data_path = data_config['train_data_path']
    eval_data_path = data_config.get('eval_data_path', None)
    train_data_ratio = data_config.get('train_data_ratio', 1.0)
    eval_data_ratio = data_config.get('eval_data_ratio', 0.05)
    
    # get important token strategy and relative weight
    # Prefer strategy inferred from iter_decider config; fallback to training config
    important_token_strategy = _infer_strategy_from_iter_decider(model_config, is_eval=False)
    eval_important_token_strategy = _infer_strategy_from_iter_decider(model_config, is_eval=True)
    important_relative_weight = training_config.get('important_token_relative_weight', 1.0)
    important_token_ratio = training_config.get('important_token_ratio', 0.1)

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
        enforce_len = bool(max_length) and (max_length_action == "filter")
        keep = []
        for i in range(len(tokens)):
            ok = any(masks[i])
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
    # Backward-compat: fall back to model-inferred important_token_strategy for iter_count; labels default to copy
    iter_count_strategy = data_config.get('iter_count_strategy', important_token_strategy or 'auto')
    iter_count_label_strategy = data_config.get('iter_count_label_strategy', 'copy')
    prompt_iter_count = int(data_config.get('prompt_iter_count', 1))
    preprocess_fn = partial(
        preprocess_for_sft_batch, 
        max_iter=max_iter,
        max_length=max_length, 
        iter_count_strategy=iter_count_strategy, 
        iter_count_label_strategy=iter_count_label_strategy,
        important_token_ratio=important_token_ratio,
        prompt_iter_count=prompt_iter_count,
        max_length_action=max_length_action,
    )
    
    # Determine columns to remove
    remove_cols = ['data_id', 'real_text', 'real_token', 'mask', 'mismatch']
    if 'divergent' in raw_train_dataset.column_names:
        remove_cols.append('divergent')
    # Remove entropy to avoid tokenizer.pad trying to collate it
    if 'entropy' in raw_train_dataset.column_names:
        remove_cols.append('entropy')
    
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

        eval_preprocess_fn = partial(
            preprocess_for_sft_batch,
            max_iter=max_iter,
            max_length=max_length,
            iter_count_strategy=iter_count_strategy,
            iter_count_label_strategy=iter_count_label_strategy,
            important_token_ratio=important_token_ratio,
            prompt_iter_count=prompt_iter_count,
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
    
    if important_relative_weight != 1.0:
        # Calculate important token ratio using processed train dataset
        avg_important_ratio = calculate_important_token_ratio(
            processed_train_dataset, important_token_strategy, accelerator
        )
    else:
        avg_important_ratio = None
    
    accelerator.print(f"Train dataset size: {len(processed_train_dataset)}")
    if processed_eval_dataset is not None:
        accelerator.print(f"Eval dataset size: {len(processed_eval_dataset)}")
    
    return processed_train_dataset, processed_eval_dataset, avg_important_ratio
