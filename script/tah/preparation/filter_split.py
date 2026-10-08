"""
Filter and convert parquet data to jsonl format.
Input: data/raw_data/data/*.parquet
Output: data/initial_data/openr1_math/*.jsonl

For each problem with N generations:
1. Select generations where correctness_math_verify is True
2. If all math_verify are False, fallback to correctness_llama
3. Among valid generations, select the one with fewest tokens
4. Filter out samples with output length > 8192 tokens
5. Split into train (99%) and eval (1%)
"""

import os
import json
import glob
import random
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count

# Configuration
INPUT_DIR = "data/raw_data/data"
OUTPUT_DIR = "data/initial_data/openr1_math/"
MODEL_PATH = "Qwen/Qwen3-0.6B"
MAX_OUTPUT_TOKENS = 8192
NUM_WORKERS = min(32, cpu_count())
TRAIN_RATIO = 0.99
RANDOM_SEED = 42

# Global tokenizer for multiprocessing
_tokenizer = None

def init_worker(model_path):
    """Initialize tokenizer in worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def process_row(row_data):
    """Process a single row and return result."""
    problem, generations, math_verify_flags, llama_flags = row_data
    
    # First try correctness_math_verify
    verify_flags = math_verify_flags
    used_llama = False
    
    # Check if all math_verify are False, then fallback to llama
    if math_verify_flags is not None and not any(math_verify_flags):
        if llama_flags is not None and any(llama_flags):
            verify_flags = llama_flags
            used_llama = True
    
    # If math_verify is None, try llama directly
    if math_verify_flags is None:
        if llama_flags is not None:
            verify_flags = llama_flags
            used_llama = True
        else:
            return None, "no_valid"
    
    # Find all valid (verify=True) generations with their token counts
    valid_candidates = []
    for i, (gen, verify) in enumerate(zip(generations, verify_flags)):
        if verify:
            num_tokens = len(_tokenizer.encode(gen, add_special_tokens=False))
            valid_candidates.append((gen, num_tokens))
    
    # No valid generation found
    if not valid_candidates:
        return None, "no_valid"
    
    # Select the one with fewest tokens
    best_gen, best_tokens = min(valid_candidates, key=lambda x: x[1])
    
    # Check max tokens filter
    if best_tokens >= MAX_OUTPUT_TOKENS:
        return None, "too_long"
    
    # Determine stats category
    if used_llama:
        stat_type = "llama_single" if len(valid_candidates) == 1 else "llama_multiple"
    else:
        stat_type = "single" if len(valid_candidates) == 1 else "multiple"
    
    return {
        "question": problem,
        "generation": best_gen,
        "num_tokens": best_tokens
    }, stat_type

def process_parquet_file(parquet_file):
    """Process a single parquet file and return samples and stats."""
    df = pd.read_parquet(parquet_file)
    
    # Prepare row data for processing
    row_data_list = []
    for _, row in df.iterrows():
        math_verify = row.get('correctness_math_verify')
        llama_verify = row.get('correctness_llama')
        row_data_list.append((
            row['problem'],
            row['generations'],
            math_verify,
            llama_verify
        ))
    
    return row_data_list, len(df)

def split_data(samples, train_ratio, seed):
    """Split samples into train and eval sets."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_samples = shuffled[:split_idx]
    eval_samples = shuffled[split_idx:]
    
    return train_samples, eval_samples

def save_jsonl(samples, output_file):
    """Save samples to jsonl file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples, desc=f"Writing {os.path.basename(output_file)}"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def process_data(input_dir, output_dir, max_tokens, train_ratio, seed):
    """Process all parquet files and save as jsonl."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Using {NUM_WORKERS} workers")
    
    # Collect all row data from parquet files
    print("Loading parquet files...")
    all_row_data = []
    total_rows = 0
    for parquet_file in tqdm(parquet_files, desc="Loading files"):
        row_data_list, num_rows = process_parquet_file(parquet_file)
        all_row_data.extend(row_data_list)
        total_rows += num_rows
    
    print(f"Total rows to process: {total_rows}")
    
    # Process rows in parallel using multiprocessing
    print("Processing rows with multiprocessing...")
    all_samples = []
    stats = {
        "total_rows": total_rows,
        "no_valid_generation": 0,
        "single_valid": 0,
        "multiple_valid_selected_shortest": 0,
        "llama_single_valid": 0,
        "llama_multiple_valid": 0,
        "filtered_too_long": 0,
        "final_count": 0,
        "train_count": 0,
        "eval_count": 0
    }
    
    with Pool(NUM_WORKERS, initializer=init_worker, initargs=(MODEL_PATH,)) as pool:
        results = list(tqdm(
            pool.imap(process_row, all_row_data, chunksize=100),
            total=len(all_row_data),
            desc="Processing rows"
        ))
    
    # Collect results
    for sample, stat_type in results:
        if stat_type == "no_valid":
            stats["no_valid_generation"] += 1
        elif stat_type == "too_long":
            stats["filtered_too_long"] += 1
        elif stat_type == "single":
            stats["single_valid"] += 1
            all_samples.append(sample)
        elif stat_type == "multiple":
            stats["multiple_valid_selected_shortest"] += 1
            all_samples.append(sample)
        elif stat_type == "llama_single":
            stats["llama_single_valid"] += 1
            all_samples.append(sample)
        elif stat_type == "llama_multiple":
            stats["llama_multiple_valid"] += 1
            all_samples.append(sample)
    
    stats["final_count"] = len(all_samples)
    
    # Split into train and eval
    print(f"\nSplitting data: {train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% eval (seed={seed})")
    train_samples, eval_samples = split_data(all_samples, train_ratio, seed)
    stats["train_count"] = len(train_samples)
    stats["eval_count"] = len(eval_samples)
    
    # Save to jsonl
    train_file = os.path.join(output_dir, "train.jsonl")
    eval_file = os.path.join(output_dir, "eval.jsonl")
    
    print(f"Saving {len(train_samples)} train samples to {train_file}...")
    save_jsonl(train_samples, train_file)
    
    print(f"Saving {len(eval_samples)} eval samples to {eval_file}...")
    save_jsonl(eval_samples, eval_file)
    
    return stats

def main():
    # Process data
    stats = process_data(INPUT_DIR, OUTPUT_DIR, MAX_OUTPUT_TOKENS, TRAIN_RATIO, RANDOM_SEED)
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Processing Statistics:")
    print("=" * 50)
    print(f"Total rows in parquet files: {stats['total_rows']}")
    print(f"No valid generation (all verify=False): {stats['no_valid_generation']}")
    print(f"[math_verify] Single valid: {stats['single_valid']}")
    print(f"[math_verify] Multiple valid (selected shortest): {stats['multiple_valid_selected_shortest']}")
    print(f"[llama] Single valid (fallback): {stats['llama_single_valid']}")
    print(f"[llama] Multiple valid (fallback): {stats['llama_multiple_valid']}")
    print(f"Filtered (tokens >= {MAX_OUTPUT_TOKENS}): {stats['filtered_too_long']}")
    print(f"Final sample count: {stats['final_count']}")
    print("-" * 50)
    print(f"Train samples: {stats['train_count']} ({stats['train_count']/stats['final_count']*100:.1f}%)")
    print(f"Eval samples: {stats['eval_count']} ({stats['eval_count']/stats['final_count']*100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
