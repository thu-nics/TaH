"""
Step1 use SLM to prefill the LLM responses, finding all non-identical SLM next-token predictions.
Multi-GPU version to read JSONL or JSON format with conversations structure.

Inputs:
- A JSONL file (.jsonl extension) with conversations format (one JSON object per line).
    - Each line contains: {"conversations": [{"from": "human", "value": "..."}, {"from": "assistant", "value": "..."}], "system": "..."}
- Or a JSON file (.json extension) with conversations format (JSON array).
    - Contains an array of objects: [{"conversations": [{"from": "human", "value": "..."}, {"from": "assistant", "value": "..."}], "system": "..."}, ...]

Outputs:
- Processed dataset with data grouped by data_id, containing real_text, real_token, mask, and mismatch information
"""

import json
import os
import argparse
import signal
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import multiprocessing as mp
from datasets import Dataset, concatenate_datasets, DatasetDict

from tah.utils.sampling import sample_token

# Global variable to track running processes
running_processes = []
SYSTEM_PROMPT = """
You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
"""

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print(f"\nReceived signal {signum}, cleaning up processes...")
    global running_processes
    
    for p in running_processes:
        if p.is_alive():
            print(f"Terminating process {p.pid}...")
            p.terminate()
            p.join(timeout=60)  # Give 60 seconds for graceful termination (increased from 5 seconds)
            if p.is_alive():
                print(f"Force killing process {p.pid}...")
                p.kill()
                p.join()
    
    print("Cleanup completed, exiting...")
    sys.exit(0)

def load_model(model_name, device_id):
    """Load a model on specific GPU with basic error handling"""
    try:
        model_config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            device_map=f"cuda:{device_id}",
            torch_dtype=torch.bfloat16
        ).eval()
        print(f"Model {model_name} loaded successfully on GPU {device_id}!")
        return model
    except Exception as e:
        print(f"Error loading model on GPU {device_id}: {e}")
        return None


def load_jsonl_json_dataset(file_path, index_range=None, random_num=None):
    """Load dataset from JSONL or JSON file based on file extension"""
    data = []
    
    # Determine file format based on extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.jsonl':
        # Load JSONL format (one JSON object per line)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    elif file_extension == '.json':
        # Load JSON format (single JSON array)
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            # If it's a list, use it directly; if it's a dict, wrap it in a list
            if isinstance(loaded_data, list):
                data = loaded_data
            else:
                data = [loaded_data]
    else:
        # Default to JSONL format for unknown extensions
        print(f"Warning: Unknown file extension '{file_extension}'. Trying to read as JSONL format.")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} samples from {file_extension if file_extension else 'unknown'} format file: {file_path}")
    
    if index_range:
        start_idx, end_idx = index_range
        data = data[start_idx:end_idx]
        print(f"Selected range [{start_idx}:{end_idx}], resulting in {len(data)} samples")
    
    # Apply random sampling if random_num is specified
    if random_num is not None and random_num > 0 and random_num < len(data):
        import random
        random.seed(42)  # Set seed for reproducibility
        data = random.sample(data, random_num)
        print(f"Randomly sampled {random_num} samples from {len(data)} available samples")
    elif random_num is not None and random_num >= len(data):
        print(f"random_num ({random_num}) is >= dataset size ({len(data)}), using all samples")
    
    return data


def split_dataset(dataset, num_splits):
    """Split dataset into num_splits parts"""
    chunk_size = len(dataset) // num_splits
    remainder = len(dataset) % num_splits
    
    splits = []
    start_idx = 0
    
    for i in range(num_splits):
        # Add one extra item to first 'remainder' splits
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        splits.append((start_idx, end_idx))
        start_idx = end_idx
    
    return splits


def parse_conversations(conversations):
    """Parse conversations to extract input_text, model_reasoning, and model_response"""
    input_text = None
    assistant_response = None
    
    for conv in conversations:
        if conv["from"] == "human" or conv["from"] == "user":
            input_text = conv["value"]
        elif conv["from"] == "assistant":
            assistant_response = conv["value"]
    
    if not input_text or not assistant_response:
        return None, None, None
    
    # Split assistant response into reasoning and response parts
    if "<think>" in assistant_response and "</think>" in assistant_response:
        # Extract thinking content
        think_start = assistant_response.find("<think>")
        think_end = assistant_response.find("</think>") + len("</think>")
        
        model_reasoning = assistant_response[think_start + len("<think>"):assistant_response.find("</think>")].strip()
        model_response = assistant_response[think_end:].strip()
    else:
        # No thinking tags, treat entire response as final response
        model_reasoning = None
        model_response = None
    
    return input_text, model_reasoning, model_response


def apply_qwen_r1_chat_template(messages, add_generation_prompt=False):
    """Apply the Qwen R1 chat template to the messages"""
    prompt = "<｜begin▁of▁sentence｜>"
    ns = {
        "is_first": False,
        "is_tool": False,
        "is_output_first": True,
        "system_prompt": "",
    }

    # extract system prompt
    for message in messages:
        if message["role"] == "system":
            ns["system_prompt"] = message["content"]

    prompt += ns["system_prompt"]

    for message in messages:
        if message["role"] == "user":
            ns["is_tool"] = False
            prompt += "<｜User｜>" + message["content"]

        elif message["role"] == "assistant" and message["content"] is not None:
            content = message["content"]
            prompt += "<｜Assistant｜>" + content + "<｜end▁of▁sentence｜>"

    if add_generation_prompt:
        prompt += "<｜Assistant｜><think>\n"

    return prompt

def replace_mobilellm_think(messages):
    """Replace the think tag with the think tag in the messages"""
    for message in messages:
        if message["role"] == "assistant" and message["content"] is not None:
            message["content"] = message["content"].replace("<think>", "<|think|>").replace("</think>", "<|/think|>")
    return messages

def get_formatted_prompt_1(sample, tokenizer, model_name):
    """Format prompt from conversations structure"""
    question = sample.get("problem", "") or sample.get("question", "")
    answer = sample.get("output", "") or sample.get("solution", "") or sample.get("generations", "") or sample.get("answer", "")

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    
    if "qwen3" in model_name.lower():
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    elif "mobilellm" in model_name.lower():
        messages = replace_mobilellm_think(messages)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, use_system_prompt=False)
    else:
        prompt = apply_qwen_r1_chat_template(messages, add_generation_prompt=False)

    return prompt

def get_formatted_prompt(sample, tokenizer, model_name):
    """Format prompt from conversations structure"""
    conversations = sample.get("conversations", [])
    system_prompt = sample.get("system", "")
    
    # Parse conversations
    input_text, model_reasoning, model_response = parse_conversations(conversations)
    
    if not input_text or model_response is None:
        print(f"Invalid conversation format, skipping")
        return None

    # Build messages
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": None},
    ]
    
    # Add system prompt if present
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    # Format assistant response based on model type
    if "qwen3" in model_name.lower():
        if model_reasoning:
            messages[-1]["content"] = f"{model_reasoning}\n</think>\n\n{model_response}"
        else:
            messages[-1]["content"] = model_response
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
    else:
        if model_reasoning:
            messages[-1]["content"] = f"<think>\n{model_reasoning}\n</think>\n\n{model_response}"
        else:
            messages[-1]["content"] = model_response
        prompt = apply_qwen_r1_chat_template(messages, add_generation_prompt=False)

    return prompt


def categorize_masks(input_ids, tokenizer, model_name):
    """Categorize tokens into mask: system and query are 0, others are 1"""
    
    masks = []
    current_mask = 0  # Default to 0 for system and query
    id_qwen3_think = 151667
    id_qwen3_assistant = 77091
    id_dpsk_think = 151648
    id_mobile_llm = 128002
    if "qwen3" in model_name.lower():
        id_think = id_qwen3_think
    elif "mobilellm" in model_name.lower():
        id_think = id_mobile_llm
    else:
        id_think = id_dpsk_think
    
    for i, token_id in enumerate(input_ids[0]):
        token_id = token_id.item()
        
        if token_id == id_think:
            # count_for_think += 1:  # Switch to 1 only on third occurrence
            current_mask = 1
            
        masks.append(current_mask)
    
    return masks


def calculate_mismatch(predictions, real_tokens, data_ids):
    """Calculate mismatch between predictions[k] and real_tokens[k+1] for each sample"""
    device = predictions.device
    
    # create mismatch tensor with the same size as input, initialized to 0
    mismatch = torch.zeros_like(predictions, dtype=torch.int32, device=device)
    
    # find the end position of each sample (the position where the data_id changes)
    # to handle boundary cases, add a different value to the end of data_ids
    padded_data_ids = torch.cat([data_ids, torch.tensor([data_ids[-1] + 1], device=device)])
    
    # find the position where data_id changes
    change_mask = padded_data_ids[1:] != padded_data_ids[:-1]
    sample_end_indices = torch.where(change_mask)[0]
    
    # create mask, mark all positions except the last position of each sample
    valid_mask = torch.ones(len(predictions), dtype=torch.bool, device=device)
    valid_mask[sample_end_indices] = False
    
    # for valid positions, compare predictions[k] and real_tokens[k+1]
    # only compare non-last positions
    if valid_mask.any():
        valid_indices = torch.where(valid_mask)[0]
        pred_tokens = predictions[valid_indices]
        next_real_tokens = real_tokens[valid_indices + 1]
        
        # calculate mismatch: 1 for mismatch, 0 for match
        mismatch_values = (pred_tokens != next_real_tokens).int()
        mismatch[valid_indices] = mismatch_values
    
    return mismatch.cpu()


def process_single_gpu(args, device_id, data_range, model_name):
    """Process dataset on a single GPU"""
    start_idx, end_idx = data_range
    model_path = model_name.split("/")[-1]
    
    print(f"GPU {device_id}: Processing data range {start_idx}-{end_idx} for model {model_name}")
    
    # Load dataset subset
    dataset = load_jsonl_json_dataset(args.dataset_path, (start_idx, end_idx), args.random_num)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad token for batching
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model on specific GPU
    model = load_model(model_name, device_id)
    if model is None:
        return None
    
    # Store results
    predictions_list = []
    real_tokens_list = []
    token_ids_list = []
    data_ids_list = []
    masks_list = []
    entropy_list = [] if args.save_entropy else None
    ce_list = [] if args.save_ce else None
    
    # Process each sample
    pbar = tqdm(total=len(dataset), desc=f"GPU {device_id} - {model_path}", position=device_id)
    with torch.no_grad():
        bs = max(1, int(getattr(args, "batch_size", 1)))
        num_samples = len(dataset)
        processed = 0
        for batch_start in range(0, num_samples, bs):
            batch_end = min(batch_start + bs, num_samples)
            batch_items = []  # tuples: (ids_1d_cpu, length, global_id)
            prompts_meta = []  # store tensors length and global id for later
            for local_offset, sample in enumerate(dataset[batch_start:batch_end]):
                global_data_id = start_idx + (batch_start + local_offset)
                # Build prompt
                if sample.get("conversations") is not None:
                    prompt = get_formatted_prompt(sample, tokenizer, model_name)
                else:
                    prompt = get_formatted_prompt_1(sample, tokenizer, model_name)
                if prompt is None:
                    continue
                ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                if ids.shape[-1] > args.max_input_length:
                    if args.is_cutoff:
                        ids = ids[:args.max_input_length]
                    else:
                        continue
                batch_items.append(ids)
                prompts_meta.append((ids.shape[-1], global_data_id))
            if not batch_items:
                # even if empty due to skips, advance pbar by original batch window size
                pbar.update(batch_end - batch_start)
                continue
            lengths = [x.shape[0] for x in batch_items]
            padded = pad_sequence(batch_items, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = torch.arange(padded.shape[1]).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
            input_ids = padded.to(model.device)
            attention_mask = attention_mask.to(model.device)
            # Forward
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.to(torch.float32)  # [B, Lmax, V]
            # Per-sample handling
            for b_idx, (seq_len, global_data_id) in enumerate(prompts_meta):
                seq_logits = logits[b_idx, :seq_len, :]
                if args.save_entropy or args.save_ce:
                    lp = F.log_softmax(seq_logits, dim=-1)
                if args.save_entropy:
                    probs = lp.exp()
                    entropy = -(probs * lp).sum(dim=-1).cpu()  # [seq_len]
                    entropy_list.append(entropy)
                pred = sample_token(seq_logits, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k).cpu()
                token_id = torch.arange(0, seq_len, 1).cpu()
                data_id_tensor = torch.full((seq_len,), global_data_id, dtype=token_id.dtype).cpu()
                real_token = input_ids[b_idx, :seq_len].detach().cpu()
                masks = categorize_masks(real_token.unsqueeze(0), tokenizer, model_name)
                masks_tensor = torch.tensor(masks, dtype=torch.int32).cpu()
                if args.save_ce:
                    device = seq_logits.device
                    data_id_dev = data_id_tensor.to(device)
                    padded_ids = torch.cat([data_id_dev, torch.tensor([data_id_dev[-1] + 1], device=device)])
                    change_mask = padded_ids[1:] != padded_ids[:-1]
                    sample_end_indices = torch.where(change_mask)[0]
                    valid_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
                    valid_mask[sample_end_indices] = False
                    ce = torch.zeros(seq_len, dtype=torch.float32, device=device)
                    if valid_mask.any():
                        valid_indices = torch.where(valid_mask)[0]
                        targets = input_ids[b_idx, :seq_len].to(device)[valid_indices + 1]
                        ce_values = -lp[valid_indices, targets]
                        ce[valid_indices] = ce_values
                    ce_list.append(ce.cpu())
                predictions_list.append(pred)
                real_tokens_list.append(real_token)
                token_ids_list.append(token_id)
                data_ids_list.append(data_id_tensor)
                masks_list.append(masks_tensor)
            processed += (batch_end - batch_start)
            pbar.update(batch_end - batch_start)
            if (batch_start // bs) % 10 == 0:
                torch.cuda.empty_cache()
    pbar.close()

    if not predictions_list:
        print(f"GPU {device_id}: No valid samples processed")
        return None

    # Concatenate results
    predictions = torch.cat(predictions_list, dim=0)
    real_tokens = torch.cat(real_tokens_list, dim=0)
    token_ids = torch.cat(token_ids_list, dim=0)
    data_ids = torch.cat(data_ids_list, dim=0)
    masks = torch.cat(masks_list, dim=0)
    # Optional tensors
    if args.save_entropy:
        entropies = torch.cat(entropy_list, dim=0)

    # Calculate mismatch
    print(f"GPU {device_id}: Calculating mismatch...")
    mismatch = calculate_mismatch(predictions, real_tokens, data_ids)
    
    # Convert tensors to python lists for Dataset compatibility
    results_dict = {
        "predictions": predictions.tolist(),
        "small_token": token_ids.tolist(),
        "data_id": data_ids.tolist(),
        "mask": masks.tolist(),
        "real_token": real_tokens.tolist(),
        "mismatch": mismatch.tolist(),
    }
    if args.save_entropy:
        results_dict["entropy"] = entropies.tolist()
    if args.save_ce:
        ce_tensor = torch.cat(ce_list, dim=0)
        results_dict["cross_entropy"] = ce_tensor.tolist()
    
    # Create Dataset from dict
    dataset = Dataset.from_dict(results_dict)
    
    # Save as Dataset
    output_file = os.path.join(args.output_path, f"results_gpu_{device_id}_{model_path}")
    dataset.save_to_disk(output_file)
        
    # Clear variables
    del model
    del tokenizer
    if 'predictions_list' in locals(): del predictions_list
    if 'real_tokens_list' in locals(): del real_tokens_list
    if 'token_ids_list' in locals(): del token_ids_list
    if 'data_ids_list' in locals(): del data_ids_list
    if 'masks_list' in locals(): del masks_list
    if 'entropy_list' in locals(): del entropy_list
    if 'ce_list' in locals(): del ce_list
    
    # Clear GPU cache and synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device_id)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print(f"GPU {device_id}: Saved dataset to {output_file}")
    return dataset


def analyze_detailed_statistics(df, tokenizer):
    """Perform detailed statistical analysis on the dataset"""
    analysis_results = {}
    
    # Basic statistics
    total_tokens = len(df)
    total_samples = df['data_id'].nunique()
    total_mismatch_tokens = sum(df['mismatch'])
    
    analysis_results['basic'] = {
        'total_tokens': int(total_tokens),
        'total_samples': int(total_samples),
        'total_mismatch_tokens': int(total_mismatch_tokens),
        'mismatch_ratio': float((total_mismatch_tokens / total_tokens * 100)) if total_tokens > 0 else 0.0
    }
    
    # Mask-based analysis (0=system/user, 1=assistant)
    mask_0_tokens = len(df[df['mask'] == 0])
    mask_1_tokens = len(df[df['mask'] == 1])
    mask_0_mismatch = sum(df[df['mask'] == 0]['mismatch'])
    mask_1_mismatch = sum(df[df['mask'] == 1]['mismatch'])
    
    analysis_results['mask_analysis'] = {
        'system_user_tokens': int(mask_0_tokens),
        'assistant_tokens': int(mask_1_tokens),
        'system_user_mismatch': int(mask_0_mismatch),
        'assistant_mismatch': int(mask_1_mismatch),
        'system_user_mismatch_ratio': float((mask_0_mismatch / mask_0_tokens * 100)) if mask_0_tokens > 0 else 0.0,
        'assistant_mismatch_ratio': float((mask_1_mismatch / mask_1_tokens * 100)) if mask_1_tokens > 0 else 0.0
    }
    
    # Per-sample analysis
    sample_stats = []
    grouped = df.groupby('data_id')
    token_lengths = []
    assistant_token_lengths = []
    mismatch_ratios = []
    
    for data_id, group in grouped:
        sample_total_tokens = len(group)
        sample_assistant_tokens = len(group[group['mask'] == 1])
        sample_mismatch_tokens = sum(group['mismatch'])
        sample_assistant_mismatch = sum(group[group['mask'] == 1]['mismatch'])
        
        token_lengths.append(sample_total_tokens)
        assistant_token_lengths.append(sample_assistant_tokens)
        
        sample_mismatch_ratio = (sample_mismatch_tokens / sample_total_tokens * 100) if sample_total_tokens > 0 else 0
        mismatch_ratios.append(sample_mismatch_ratio)
        
        sample_stats.append({
            'data_id': int(data_id),
            'total_tokens': int(sample_total_tokens),
            'assistant_tokens': int(sample_assistant_tokens),
            'mismatch_tokens': int(sample_mismatch_tokens),
            'assistant_mismatch_tokens': int(sample_assistant_mismatch),
            'mismatch_ratio': float(sample_mismatch_ratio),
            'assistant_mismatch_ratio': float((sample_assistant_mismatch / sample_assistant_tokens * 100)) if sample_assistant_tokens > 0 else 0.0
        })
    
    # Token length statistics
    analysis_results['length_analysis'] = {
        'avg_tokens_per_sample': float(np.mean(token_lengths)),
        'median_tokens_per_sample': float(np.median(token_lengths)),
        'min_tokens_per_sample': int(np.min(token_lengths)),
        'max_tokens_per_sample': int(np.max(token_lengths)),
        'std_tokens_per_sample': float(np.std(token_lengths)),
        'avg_assistant_tokens': float(np.mean(assistant_token_lengths)),
        'median_assistant_tokens': float(np.median(assistant_token_lengths)),
        'min_assistant_tokens': int(np.min(assistant_token_lengths)),
        'max_assistant_tokens': int(np.max(assistant_token_lengths))
    }
    
    # Mismatch ratio distribution
    analysis_results['mismatch_distribution'] = {
        'avg_mismatch_ratio': float(np.mean(mismatch_ratios)),
        'median_mismatch_ratio': float(np.median(mismatch_ratios)),
        'min_mismatch_ratio': float(np.min(mismatch_ratios)),
        'max_mismatch_ratio': float(np.max(mismatch_ratios)),
        'std_mismatch_ratio': float(np.std(mismatch_ratios)),
        'samples_with_no_mismatch': int(sum(1 for ratio in mismatch_ratios if ratio == 0)),
        'samples_with_high_mismatch': int(sum(1 for ratio in mismatch_ratios if ratio > 50))
    }
    
    # Token frequency analysis for real tokens
    real_token_counts = df['real_token'].value_counts()
    most_common_tokens = real_token_counts.head(20).to_dict()
    
    # Decode most common tokens
    decoded_common_tokens = {}
    for token_id, count in most_common_tokens.items():
        try:
            decoded_token = tokenizer.decode([int(token_id)])
            decoded_common_tokens[f"{int(token_id)} ({repr(decoded_token)})"] = int(count)
        except:
            decoded_common_tokens[str(int(token_id))] = int(count)
    
    analysis_results['token_frequency'] = {
        'most_common_tokens': decoded_common_tokens,
        'unique_tokens': int(len(real_token_counts)),
        'total_token_occurrences': int(real_token_counts.sum())
    }
    
    return analysis_results, sample_stats

def process_and_convert_dataset(merged_dataset, model_name, output_path):
    """Convert merged dataset to final processed format"""
    print("Converting dataset to final format...")
    
    # Load tokenizer for text decoding
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Convert to pandas for easier grouping
    df = merged_dataset.to_pandas()
    
    # Perform detailed analysis on raw data
    print("Performing detailed statistical analysis...")
    analysis_results, sample_stats = analyze_detailed_statistics(df, tokenizer)
    
    # 按data_id分组
    grouped = df.groupby('data_id')
    print(f"Found {len(grouped)} unique data_ids.")
    
    # Initialize counters for statistics
    total_tokens = 0
    total_mismatch_tokens = 0
    
    final_data_list = []
    print("Processing groups...")
    for data_id, group in tqdm(grouped):
        # Convert the real_token list to text
        real_tokens = group['real_token'].tolist()
        real_text = tokenizer.decode(real_tokens)
        
        # Get mismatch indices
        mismatch_indices = group['mismatch'].tolist()
        
        # Update statistics
        total_tokens += len(real_tokens)
        total_mismatch_tokens += sum(1 for x in mismatch_indices if x == 1)
        
        processed_item = {
            'data_id': data_id,
            'real_text': real_text,
            'real_token': real_tokens,
            'mask': group['mask'].tolist(),
            'mismatch': mismatch_indices,
        }
        if 'entropy' in group.columns:
            processed_item['entropy'] = group['entropy'].tolist()
        if 'cross_entropy' in group.columns:
            processed_item['cross_entropy'] = group['cross_entropy'].tolist()
        final_data_list.append(processed_item)
    
    # Print statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS SUMMARY")
    print("="*80)
    
    basic = analysis_results['basic']
    print(f"Basic Statistics:")
    print(f"  Total samples: {basic['total_samples']:,}")
    print(f"  Total tokens: {basic['total_tokens']:,}")
    print(f"  Total mismatch tokens: {basic['total_mismatch_tokens']:,}")
    print(f"  Overall mismatch ratio: {basic['mismatch_ratio']:.2f}%")
    
    mask = analysis_results['mask_analysis']
    print(f"\nMask-based Analysis:")
    print(f"  System/User tokens (mask=0): {mask['system_user_tokens']:,}")
    print(f"  Assistant tokens (mask=1): {mask['assistant_tokens']:,}")
    print(f"  System/User mismatch: {mask['system_user_mismatch']:,} ({mask['system_user_mismatch_ratio']:.2f}%)")
    print(f"  Assistant mismatch: {mask['assistant_mismatch']:,} ({mask['assistant_mismatch_ratio']:.2f}%)")
    
    length = analysis_results['length_analysis']
    print(f"\nToken Length Analysis:")
    print(f"  Avg tokens per sample: {length['avg_tokens_per_sample']:.1f}")
    print(f"  Median tokens per sample: {length['median_tokens_per_sample']:.1f}")
    print(f"  Token range: {length['min_tokens_per_sample']:.0f} - {length['max_tokens_per_sample']:.0f}")
    print(f"  Avg assistant tokens: {length['avg_assistant_tokens']:.1f}")
    
    mismatch_dist = analysis_results['mismatch_distribution']
    print(f"\nMismatch Distribution:")
    print(f"  Avg mismatch ratio per sample: {mismatch_dist['avg_mismatch_ratio']:.2f}%")
    print(f"  Median mismatch ratio: {mismatch_dist['median_mismatch_ratio']:.2f}%")
    print(f"  Samples with no mismatch: {mismatch_dist['samples_with_no_mismatch']}")
    print(f"  Samples with >50% mismatch: {mismatch_dist['samples_with_high_mismatch']}")
    
    print("="*80)
    
    processed_dataset = Dataset.from_pandas(pd.DataFrame(final_data_list))
    
    print(f"Processed dataset info:")
    print(processed_dataset)
    
    # Save processed dataset directly to output_path
    final_output_path = output_path
    processed_dataset.save_to_disk(final_output_path)
    print(f"Processed dataset saved to {final_output_path}")
    
    # Save detailed analysis to files
    analysis_dir = final_output_path
    
    # Save detailed statistics
    detailed_stats_file = os.path.join(analysis_dir, "detailed_analysis.json")
    with open(detailed_stats_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    # Save per-sample statistics
    sample_stats_file = os.path.join(analysis_dir, "per_sample_statistics.csv")
    sample_df = pd.DataFrame(sample_stats)
    sample_df.to_csv(sample_stats_file, index=False)
    
    # Save comprehensive text report
    report_file = os.path.join(analysis_dir, "analysis_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE DATA ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. BASIC STATISTICS\n")
        f.write("-"*20 + "\n")
        f.write(f"Total samples: {basic['total_samples']:,}\n")
        f.write(f"Total tokens: {basic['total_tokens']:,}\n")
        f.write(f"Total mismatch tokens: {basic['total_mismatch_tokens']:,}\n")
        f.write(f"Overall mismatch ratio: {basic['mismatch_ratio']:.4f}%\n\n")
        
        f.write("2. MASK-BASED ANALYSIS\n")
        f.write("-"*20 + "\n")
        f.write(f"System/User tokens (mask=0): {mask['system_user_tokens']:,}\n")
        f.write(f"Assistant tokens (mask=1): {mask['assistant_tokens']:,}\n")
        f.write(f"System/User mismatch: {mask['system_user_mismatch']:,} ({mask['system_user_mismatch_ratio']:.4f}%)\n")
        f.write(f"Assistant mismatch: {mask['assistant_mismatch']:,} ({mask['assistant_mismatch_ratio']:.4f}%)\n\n")
        
        f.write("3. TOKEN LENGTH ANALYSIS\n")
        f.write("-"*25 + "\n")
        f.write(f"Average tokens per sample: {length['avg_tokens_per_sample']:.2f}\n")
        f.write(f"Median tokens per sample: {length['median_tokens_per_sample']:.2f}\n")
        f.write(f"Min tokens per sample: {length['min_tokens_per_sample']:.0f}\n")
        f.write(f"Max tokens per sample: {length['max_tokens_per_sample']:.0f}\n")
        f.write(f"Std deviation: {length['std_tokens_per_sample']:.2f}\n")
        f.write(f"Average assistant tokens: {length['avg_assistant_tokens']:.2f}\n")
        f.write(f"Median assistant tokens: {length['median_assistant_tokens']:.2f}\n\n")
        
        f.write("4. MISMATCH DISTRIBUTION\n")
        f.write("-"*25 + "\n")
        f.write(f"Average mismatch ratio per sample: {mismatch_dist['avg_mismatch_ratio']:.4f}%\n")
        f.write(f"Median mismatch ratio: {mismatch_dist['median_mismatch_ratio']:.4f}%\n")
        f.write(f"Min mismatch ratio: {mismatch_dist['min_mismatch_ratio']:.4f}%\n")
        f.write(f"Max mismatch ratio: {mismatch_dist['max_mismatch_ratio']:.4f}%\n")
        f.write(f"Std deviation: {mismatch_dist['std_mismatch_ratio']:.4f}%\n")
        f.write(f"Samples with no mismatch: {mismatch_dist['samples_with_no_mismatch']}\n")
        f.write(f"Samples with >50% mismatch: {mismatch_dist['samples_with_high_mismatch']}\n\n")
        
        f.write("5. TOKEN FREQUENCY ANALYSIS\n")
        f.write("-"*30 + "\n")
        token_freq = analysis_results['token_frequency']
        f.write(f"Unique tokens: {token_freq['unique_tokens']:,}\n")
        f.write(f"Total token occurrences: {token_freq['total_token_occurrences']:,}\n")
        f.write("Most common tokens:\n")
        for token, count in list(token_freq['most_common_tokens'].items())[:10]:
            f.write(f"  {token}: {count:,}\n")
    
    print(f"Detailed analysis saved to:")
    print(f"  - JSON format: {detailed_stats_file}")
    print(f"  - Per-sample CSV: {sample_stats_file}")
    print(f"  - Text report: {report_file}")
    
    return processed_dataset

def merge_gpu_results(args, model_name):
    """Merge results from all GPUs and convert to final format"""
    model_path = model_name.split("/")[-1]
    all_datasets = []
    
    # Load results from all GPUs
    for gpu_id in range(args.num_gpu):
        result_dir = os.path.join(args.output_path, f"results_gpu_{gpu_id}_{model_path}")
        if os.path.exists(result_dir):
            dataset = Dataset.load_from_disk(result_dir)
            all_datasets.append(dataset)
            print(f"Loaded dataset from GPU {gpu_id}")
    
    if not all_datasets:
        print("No GPU datasets found to merge")
        return None
    
    # Concatenate all datasets
    merged_dataset = concatenate_datasets(all_datasets)
    
    # Process and convert to final format
    processed_dataset = process_and_convert_dataset(merged_dataset, model_name, args.output_path)
    
    # Clean up individual GPU files
    for gpu_id in range(args.num_gpu):
        result_dir = os.path.join(args.output_path, f"results_gpu_{gpu_id}_{model_path}")
        if os.path.exists(result_dir):
            import shutil
            shutil.rmtree(result_dir)
    
    return processed_dataset


def process_dataset_multi_gpu(args):
    """Process the JSONL dataset with multiple GPUs"""
    global running_processes
    
    # Create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load full dataset to get length and split
    print(f"Loading dataset from {args.dataset_path}")
    full_dataset = load_jsonl_json_dataset(args.dataset_path, args.index_range, args.random_num)
    print(f"Dataset length: {len(full_dataset)}")
    
    # Split dataset into multiple parts based on num_gpu
    data_splits = split_dataset(full_dataset, args.num_gpu)
    print(f"Dataset split into {args.num_gpu} parts: {data_splits}")

    # Process each model
    for model_name in args.test_model_list:
        model_path = model_name.split("/")[-1]
        
        # # Skip if processed results already exist
        # if os.path.exists(os.path.join(args.output_path, f"processed_data_{model_path}")):
        #     print(f"Processed results for {model_name} already exist, skipping.")
        #     continue
        
        print(f"Processing model: {model_name}")
        
        # Create processes for each GPU
        processes = []
        
        for gpu_id in range(args.num_gpu):
            p = mp.Process(
                target=process_single_gpu,
                args=(args, gpu_id, data_splits[gpu_id], model_name)
            )
            processes.append(p)
            p.start()
        
        # Update global process list for signal handling
        running_processes = processes
        
        # Wait for all processes to complete with timeout
        timeout = 24 * 60 * 60  # 24 hours timeout per process (increased from 1 hour)
        for i, p in enumerate(processes):
            try:
                p.join(timeout=timeout)
                if p.is_alive():
                    print(f"GPU {i}: Process timeout after {timeout} seconds, terminating...")
                    p.terminate()
                    p.join(timeout=30)  # Give 30 seconds for graceful termination (increased from 10 seconds)
                    if p.is_alive():
                        print(f"GPU {i}: Force killing process...")
                        p.kill()
                        p.join()
                elif p.exitcode != 0:
                    print(f"GPU {i}: Process exited with code {p.exitcode}")
                else:
                    print(f"GPU {i}: Process completed successfully")
            except Exception as e:
                print(f"GPU {i}: Error during process join: {e}")
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=30)  # Give 30 seconds for graceful termination (increased from 10 seconds)
                    if p.is_alive():
                        p.kill()
                        p.join()
        
        # Ensure all processes are cleaned up
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=60)  # Give 60 seconds for graceful termination (increased from 5 seconds)
                if p.is_alive():
                    p.kill()
                    p.join()
        
        print(f"All GPU processes completed for {model_name}")
        
        # Clear global process list
        running_processes = []
        
        # Merge results from all GPUs and convert to final format
        processed_results = merge_gpu_results(args, model_name)
        if processed_results is None:
            print(f"Failed to process results for {model_name}")
            continue

    print("Multi-GPU processing completed!")


def main():
    global running_processes
    
    # Register signal handlers for graceful cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Run multi-GPU model inference on JSONL or JSON datasets"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset file (supports .jsonl and .json formats)"
    )
    parser.add_argument(
        "--num_gpu", type=int, default=4, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--test_model_list",
        nargs="+",
        type=str,
        required=True,
        help="List of test models to run",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=32768,
        help="Maximum length of input tokens",
    )
    parser.add_argument(
        "--is_cutoff",
        type=bool,
        default=False,
        help="whether of not cut off the dataset",
    )
    parser.add_argument(
        "--index_range",
        nargs=2,
        type=int,
        default=None,
        help="Range of dataset samples to process [start_idx, end_idx]",
    )
    parser.add_argument(
        "--random_num",
        type=int,
        default=None,
        help="Number of samples to randomly sample from the dataset",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Number of top predictions to include in the output",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature to apply to logits",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p probability threshold for nucleus sampling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for prefill/inference per GPU process",
    )
    parser.add_argument(
        "--save_entropy",
        action="store_true",
        help="Save per-token entropy of model logits",
    )
    parser.add_argument(
        "--save_ce",
        action="store_true",
        help="Save per-token cross-entropy w.r.t. next-token labels",
    )
    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    try:
        # Process dataset with multiple GPUs
        process_dataset_multi_gpu(args)

        # Save args as json
        with open(os.path.join(args.output_path, "args.json"), "w") as f:
            json.dump(args.__dict__, f)

        print("All processing completed!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        # Clean up any remaining processes
        for p in running_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=60)  # Give 60 seconds for graceful termination (increased from 5 seconds)
                if p.is_alive():
                    p.kill()
                    p.join()
        raise e
    finally:
        # Final cleanup
        running_processes = []


if __name__ == "__main__":
    main() 