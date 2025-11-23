import os
import json
import yaml
import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import math
import multiprocessing as mp
from multiprocessing import Process, Queue
mp.set_start_method("spawn", force=True)
import pandas as pd
from transformers.utils import logging as hf_logging
import logging as pylog

# some constants
SYSTEM_PROMPT = """
You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
"""
USER_PROMPT = """ Please reason step by step, and put your final answer within \\boxed{}."""
QWEN3_EOS_TOKEN_ID = 151645
DEEPSEEK_R1_EOS_TOKEN_ID = 151643


def load_datasets_with_config(dataset_names) -> Tuple[object, Dict]:
    """Load single or multiple datasets using their configurations
    
    Args:
        dataset_names: Can be:
            - A single dataset name (string)
            - Multiple dataset names (comma-separated string)
            - A list of dataset names
    
    Returns:
        Tuple of (dataset, field_mapping) where field_mapping uses standard internal format
    """
    from datasets import load_dataset

    # Parse dataset names
    if isinstance(dataset_names, str):
        # Handle comma-separated string or single dataset name
        dataset_names_list = [name.strip() for name in dataset_names.split(',')]
    elif isinstance(dataset_names, list):
        # Handle list of dataset names
        dataset_names_list = dataset_names
    else:
        raise ValueError(f"Invalid dataset_names type: {type(dataset_names)}. Expected str or list.")
    
    # Remove empty strings
    dataset_names_list = [name for name in dataset_names_list if name]
    
    if not dataset_names_list:
        raise ValueError("No dataset names provided")
    
    # Load and combine datasets (works for both single and multiple datasets)
    combined_data = []
    answer_types = []  # Collect answer types from all datasets
    
    print(f"Loading and combining {len(dataset_names_list)} datasets: {dataset_names_list}")
    
    for i, dataset_name in enumerate(dataset_names_list):
        print(f"\n[{i+1}/{len(dataset_names_list)}] Loading dataset: {dataset_name}")

        # Load dataset configuration and data
        script_dir = Path(__file__).parent
        config_file_path = script_dir / "eval_configs" / "dataset_configs.json"
        with open(config_file_path, 'r', encoding='utf-8') as f:
            dataset_configs = json.load(f)
        
        if dataset_name not in dataset_configs:
            available_datasets = list(dataset_configs.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found in dataset configs. Available datasets: {available_datasets}")
        
        dataset_config = dataset_configs[dataset_name]
        
        # Load dataset
        dataset_path = dataset_config['path']
        # Support both 'subset' and 'dataset_config' as HF config name
        subset = dataset_config.get('subset', None)
        version_tag = dataset_config.get('version_tag', None)
        
        # Load dataset from JSON/JSONL local file
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            if dataset_name in ["mbpp", "humaneval"]:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    records = [json.loads(line) for line in f if line.strip()]
                # Mimic the structure of `load_dataset` for local files,
                # which usually returns a dict like {"train": Dataset(...)}.
                dataset_obj = {"train": records}
            else:
                dataset_obj = load_dataset('json', data_files=dataset_path)
        else:
            # For HuggingFace datasets, use dataset_config or subset as configuration name
            if subset:
                dataset_obj = load_dataset(dataset_path, subset)
            elif version_tag:
                dataset_obj = load_dataset(dataset_path, version_tag = version_tag)
            else:
                dataset_obj = load_dataset(dataset_path)
        
        split_name = dataset_config.get('split_name', 'test')
        
        # Try to get train split first, if not available try test split
        if split_name in dataset_obj:
            dataset = dataset_obj[split_name]
            print(f"Using '{split_name}' split with {len(dataset)} samples")
        elif "train" in dataset_obj:
            dataset = dataset_obj["train"]
            print(f"Using 'train' split with {len(dataset)} samples")
        elif "test" in dataset_obj:
            dataset = dataset_obj["test"]
            print(f"Using 'test' split with {len(dataset)} samples")
        else:
            available_splits = list(dataset_obj.keys())
            raise ValueError(f"Split '{split_name}' not found. Available splits: {available_splits}")
        
        # Apply filter if specified
        if 'filter' in dataset_config:
            filter_config = dataset_config['filter']
            filter_key = filter_config['key']
            filter_values = filter_config['value']
            dataset = dataset.filter(lambda x: x.get(filter_key) in filter_values)
        
        # Get field mapping for this dataset
        original_field_mapping = {
            'id_field': dataset_config['id_field'],
            'question_field': dataset_config['question_field'],
            'answer_field': dataset_config['answer_field'],
            'answer_type': dataset_config['answer_type'],
            'prompt_template': dataset_config.get('prompt_template', '{question}')
        }
        # Optional extra fields from config (e.g., entry_point for code datasets)
        entry_point_field = dataset_config.get('entry_point', None)
        
        # print(f"Original field mapping for {dataset_name}: {original_field_mapping}")
        
        # Collect answer type (should be same for all datasets in practice)
        answer_types.append(original_field_mapping['answer_type'])
        
        # Convert dataset to list and standardize field names
        dataset_list = list(dataset)
        for idx, item in enumerate(dataset_list):
            # Create new standardized item
            standardized_item = {}
            
            # Convert ID field to standard format
            id_field = original_field_mapping['id_field']
            if id_field in item and item[id_field] is not None:
                original_id = str(item[id_field])
                # Keep both standardized id (for this pipeline) and original_id (for downstream tools)
                standardized_item['id'] = f"{dataset_name}_{original_id}"
                standardized_item['_original_id'] = original_id
            else:
                # Generate ID if not present
                generated_id = f"{dataset_name}_{idx}"
                standardized_item['id'] = generated_id
                standardized_item['_original_id'] = generated_id
            
            # Convert question field to standard format
            question_field = original_field_mapping['question_field']
            question_text = str(item.get(question_field, '')).strip()
            
            # Apply prompt template if specified
            prompt_template = original_field_mapping['prompt_template']
            if prompt_template and '{question}' in prompt_template:
                question_text = prompt_template.replace('{question}', question_text)
            
            standardized_item['question'] = question_text
            
            # Convert answer field to standard format
            answer_field = original_field_mapping['answer_field']
            standardized_item['answer'] = str(item.get(answer_field, '')).strip()

            # Optionally keep entry_point in standardized format if specified in config
            if entry_point_field:
                standardized_item['entry_point'] = item.get(entry_point_field)
            
            # Add source dataset information
            standardized_item['_source_dataset'] = dataset_name
            
            # Copy any other fields that might be useful
            for key, value in item.items():
                if key not in [id_field, question_field, answer_field] and not key.startswith('_'):
                    standardized_item[f'_original_{key}'] = value
        
            combined_data.append(standardized_item)
        
        print(f"Added {len(dataset_list)} problems from {dataset_name} (converted to standard format)")
    
    print(f"\nTotal combined dataset size: {len(combined_data)} problems")
    
    # Verify all datasets have the same answer_type
    unique_answer_types = set(answer_types)
    if len(unique_answer_types) > 1:
        print(f"Warning: Multiple answer types found: {unique_answer_types}")
        print("Using the first answer type as default")
    
    # Create combined field mapping using standard format
    combined_field_mapping = {
        'id_field': 'id',
        'question_field': 'question',
        'answer_field': 'answer',
        'answer_type': answer_types[0] if answer_types else 'string',
        'prompt_template': '{question}',  # Already applied during conversion
        'dataset_names': dataset_names_list
    }
    
    # print(f"Using standardized field mapping: {combined_field_mapping}")
    
    return combined_data, combined_field_mapping

def combine_job_results(output_dir: Path, job_nums: int, del_job_dir: bool = False):
    """Combine results from all job directories"""
    all_results = []
    problem_stats = {}
    # Map (problem_id, sample_idx) -> output_tokens for truncating trackers
    sample_output_tokens_map = {}
    
    # Initialize iter count distribution tracking
    iter_count_distribution = {i: 0 for i in range(1, 6)}  # iter_count 1 to 5

    # Prepare combined output directory early (for new aggregated artifacts)
    combined_dir = output_dir
    combined_dir.mkdir(parents=True, exist_ok=True)
    # Prepare aggregated samples.jsonl (truncate if exists)
    samples_jsonl_path = combined_dir / "samples.jsonl"
    with open(samples_jsonl_path, 'w', encoding='utf-8'):
        pass
    # Collect all tracker csv files for later concatenation
    all_tracker_files = []
    
    # Collect results from all job directories
    for job_id in range(job_nums):
        job_dir = output_dir / f'job_{job_id}'
        
        # Read detailed results
        results_file = job_dir / "detailed_results.csv"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string values back to appropriate types
                    row['is_correct'] = row['is_correct'] == 'True'
                    # Handle both old and new field names for backward compatibility
                    if 'has_boxed_answer' in row:
                        row['has_answer'] = row['has_boxed_answer'] == 'True'
                        del row['has_boxed_answer']  # Remove old field
                    elif 'has_answer' in row:
                        row['has_answer'] = row['has_answer'] == 'True'
                    row['sample_idx'] = int(row['sample_idx'])
                    row['input_tokens'] = int(row['input_tokens'])
                    row['output_tokens'] = int(row['output_tokens'])
                    row['processing_time'] = float(row['processing_time'])
                    all_results.append(row)
                    # Record output length for this sample for later tracker truncation
                    sample_output_tokens_map[(row['problem_id'], row['sample_idx'])] = row['output_tokens']
        
        # Read problem statistics
        stats_file = job_dir / "evaluation_stats.csv"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Read header to check format
                for row in reader:
                    if row and row[0] != '' and row[0] != 'Total Accuracy':
                        # Handle both old format (without avg_output_length) and new format
                        if len(row) >= 5 and len(header) >= 5:
                            problem_stats[row[0]] = {
                                "accuracy": f"{float(row[1]):.3f}",
                                "correct_count": int(row[2]),
                                "total_samples": int(row[3]),
                                "avg_output_length": float(row[4])
                            }
                        else:
                            problem_stats[row[0]] = {
                                "accuracy": f"{float(row[1]):.3f}",
                                "correct_count": int(row[2]),
                                "total_samples": int(row[3]),
                                "avg_output_length": 0.0  # Default value for old format
                            }
        
        # Process tracker CSV files for iter count distribution and aggregate samples/tracker
        details_dir = job_dir / 'details'
        if details_dir.exists():
            for problem_dir in details_dir.iterdir():
                if problem_dir.is_dir():
                    # Aggregate sample_*.json into combined samples.jsonl with added fields
                    sample_json_files = sorted(problem_dir.glob('sample_*.json'))
                    for sample_json_file in sample_json_files:
                        with open(sample_json_file, 'r', encoding='utf-8') as f_json:
                            sample_obj = json.load(f_json)
                        problem_id = problem_dir.name
                        try:
                            sample_idx = int(sample_json_file.stem.split('_')[-1])
                        except Exception:
                            sample_idx = -1
                        # Add required fields
                        sample_obj['id'] = problem_id
                        sample_obj['sample'] = sample_idx
                        # Append to JSONL
                        with open(samples_jsonl_path, 'a', encoding='utf-8') as out_f:
                            out_f.write(json.dumps(sample_obj, ensure_ascii=False) + "\n")
                    
                    # Look for tracker CSV files
                    for tracker_file in problem_dir.glob('*_tracker.csv'):
                        df = pd.read_csv(tracker_file)
                        # Truncate tracker rows by output length based on iter_depth==0 counts
                        if 'iter_depth' in df.columns:
                            # Parse sample index from filename like sample_{idx}_tracker.csv
                            sample_idx = -1
                            stem_parts = tracker_file.stem.split('_')
                            if len(stem_parts) >= 3 and stem_parts[0] == 'sample' and stem_parts[-1] == 'tracker':
                                sample_idx = int(stem_parts[1])

                            output_len = sample_output_tokens_map.get((problem_dir.name, sample_idx))
                            if isinstance(output_len, int) and output_len >= 0:
                                depth0_cum = (df['iter_depth'] == 0).cumsum()
                                df = df[depth0_cum <= output_len]
                                # Persist truncated tracker back to file so later combination uses it
                                df.to_csv(tracker_file, index=False)

                            # Use (possibly truncated) df to accumulate iter count distribution
                            iter_depth_counts = df['iter_depth'].value_counts().to_dict()
                            for iter_count in range(1, 6):
                                current_depth = iter_count - 1
                                next_depth = iter_count
                                current_count = iter_depth_counts.get(current_depth, 0)
                                next_count = iter_depth_counts.get(next_depth, 0)
                                tokens_with_this_iter_count = current_count - next_count
                                if tokens_with_this_iter_count > 0:
                                    iter_count_distribution[iter_count] += tokens_with_this_iter_count
                        all_tracker_files.append(tracker_file)
    
    # Calculate statistics for each problem from all_results
    problem_output_stats = {}
    for result in all_results:
        problem_id = result['problem_id']
        if problem_id not in problem_output_stats:
            problem_output_stats[problem_id] = []
        problem_output_stats[problem_id].append(result['output_tokens'])
    
    # Calculate average output length for each problem
    for problem_id in problem_stats:
        if problem_id in problem_output_stats:
            avg_output_length = sum(problem_output_stats[problem_id]) / len(problem_output_stats[problem_id])
            problem_stats[problem_id]["avg_output_length"] = avg_output_length
        else:
            problem_stats[problem_id]["avg_output_length"] = 0.0
    
    # Calculate overall statistics
    total_correct = sum(1 for r in all_results if r['is_correct'])
    total_accuracy = total_correct / len(all_results) if all_results else 0
    overall_avg_output_length = sum(r['output_tokens'] for r in all_results) / len(all_results) if all_results else 0
    
    # Save combined statistics
    combined_dir = output_dir
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = combined_dir / "evaluation_stats.csv"
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["problem_id", "accuracy", "correct_count", "total_samples", "avg_output_length"])
        
        for problem_id, stats in sorted(problem_stats.items()):
            writer.writerow([problem_id, stats['accuracy'], stats['correct_count'], stats['total_samples'], f"{stats['avg_output_length']:.2f}"])
        
        writer.writerow([])
        writer.writerow(["Total Accuracy", f"{total_accuracy:.3f}", total_correct, len(all_results), f"{overall_avg_output_length:.2f}"])
    
    # Save combined detailed results
    results_file = combined_dir / "detailed_results.csv"
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["problem_id", "sample_idx", "correct_answer", "predicted_answer", 
                     "has_answer", "is_correct", "input_tokens", "output_tokens", 
                     "processing_time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in sorted(all_results, key=lambda x: (x['problem_id'], x['sample_idx'])):
            writer.writerow(result)
    
    # Save iter count distribution
    iter_count_file = combined_dir / "iter_count_distribution.csv"
    with open(iter_count_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["iter_count", "token_count"])
        
        total_tokens = sum(iter_count_distribution.values())
        for iter_count in sorted(iter_count_distribution.keys()):
            token_count = iter_count_distribution[iter_count]
            writer.writerow([iter_count, token_count])
        
        writer.writerow([])
        writer.writerow(["Total Tokens", total_tokens])
    
    print(f"\nCombined results from {job_nums} jobs")
    print(f"Overall accuracy: {total_accuracy:.4f}")
    print(f"Total problems: {len(problem_stats)}")
    print(f"Combined statistics saved to: {stats_file}")
    print(f"Combined detailed results saved to: {results_file}")
    print(f"Iter count distribution saved to: {iter_count_file}")
    
    # Print iter count distribution
    print("\n=== Iter Count Distribution ===")
    total_tokens = sum(iter_count_distribution.values())
    for iter_count in sorted(iter_count_distribution.keys()):
        token_count = iter_count_distribution[iter_count]
        if token_count <= 0:
            continue
        percentage = (token_count / total_tokens * 100) if total_tokens > 0 else 0
        print(f"Iter count {iter_count}: {token_count} tokens ({percentage:.2f}%)")
    print(f"Total tokens: {total_tokens}")
    print("==============================\n")

    # Concatenate all tracker CSV files into a single CSV if any exist
    if all_tracker_files:
        combined_tracker_path = combined_dir / 'all_trackers.csv'
        # Write header from the first file, then append rows from all files
        with open(combined_tracker_path, 'w', newline='', encoding='utf-8') as out_f:
            writer = None
            header_written = False
            for idx, tf in enumerate(all_tracker_files):
                with open(tf, 'r', encoding='utf-8') as in_f:
                    reader = csv.reader(in_f)
                    rows = list(reader)
                    if not rows:
                        continue
                    # Augment header with data_id on first write
                    if not header_written:
                        writer = csv.writer(out_f)
                        header = rows[0] + ['data_id']
                        writer.writerow(header)
                        header_written = True
                    data_id = tf.parent.name
                    for row in rows[1:]:
                        writer.writerow(row + [data_id])
        print(f"Combined tracker CSV saved to: {combined_tracker_path}")

    # # After combining, remove per-job directories to reduce disk usage
    if del_job_dir:
        import shutil
        for job_id in range(job_nums):
            job_dir = output_dir / f'job_{job_id}'
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        print("Removed per-job directories after combining results")


def _time_inference(func, cuda_available=True):
    """Common timing wrapper for inference"""
    import torch
    if cuda_available:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = func()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        start_time = time.time()
        result = func()
        end_time = time.time()
        elapsed_time = end_time - start_time
    return result, elapsed_time

def _warmup_model(model, tokenizer, backend, tp_size=1):
    """Common warmup function for all backends"""
    import torch
    print(f"Warming up {backend} model...")
    if backend == 'sglang':
        _ = model.generate(["who are you?"], {"temperature": 0.6, "max_new_tokens": 100, "top_p": 0.95, "top_k": 20, "min_p": 0.0})
    elif backend == 'hf' or backend == 'tah':
        warmup_input = tokenizer("who are you?", return_tensors="pt")
        # Move inputs to the same device as the model
        try:
            model_device = next(model.parameters()).device
            warmup_input = {k: v.to(model_device) for k, v in warmup_input.items()}
        except StopIteration:
            # If model has no parameters, try to use first available CUDA device
            if torch.cuda.is_available():
                warmup_input = {k: v.cuda() for k, v in warmup_input.items()}
        with torch.no_grad():
            output = model.generate(**warmup_input, max_new_tokens=100, do_sample=True)
            print(output)

def _cleanup_resources(model, backend):
    """Common cleanup function for all backends"""
    import torch
    if model is not None:
        if backend == 'sglang':
            model.shutdown()
        else:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_single_job(config: Dict, combined_dataset_name: str, output_dir: str, timestamp: str, model_path: str, job_id: int, job_nums: int, start_idx: int, end_idx: int, tp_size: int, backend: str, data_range=None, problems_data=None, field_mapping=None, unified_code_solutions_file=None):
    """Run inference for a single job"""
    # Lazy import of torch and related libraries to ensure CUDA_VISIBLE_DEVICES is respected.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    import tah.evaluate.matheval as matheval
    import tah.evaluate.codeeval as codeeval
    from tah.model.tah_config import TaHConfig

    if backend == 'sglang':
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError("sglang backend requires sglang to be installed.")
    elif backend == 'tah':
        try:
            from tah.model.recurrent_transformer import TaHForCausalLM
            from tah.model.utils import TaHForCasualLM_generate
            from tah.model.tracker import TaHTracker
        except ImportError:
            raise ImportError("tah backend requires TaH components to be installed.")
    
    # Update output directory for this job (include task suffix if data_range is provided)
    task_suffix = ""
    if data_range:
        range_start, range_end = parse_data_range(data_range)
        task_suffix = f"TASK_{range_start}_{range_end}"

    output_dir = Path(output_dir) / (combined_dataset_name + "_" + backend) / timestamp
    if task_suffix:
        output_dir = output_dir / task_suffix
    output_dir = output_dir / f'job_{job_id}'
    detail_dir = output_dir / 'details'
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)
    
    problems = list(problems_data)
    print(f"Job {job_id+1}/{job_nums}: Processing {len(problems)} problems")
    print(f"Backend: {backend}")
    print(f"Combined datasets: {combined_dataset_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.eos_token_id = QWEN3_EOS_TOKEN_ID
    # if "r1" in model_path.lower():
    #     tokenizer.eos_token_id = DEEPSEEK_R1_EOS_TOKEN_ID
    # elif "qwen3" in model_path.lower():
    #     tokenizer.eos_token_id = QWEN3_EOS_TOKEN_ID
    # else:
    #     tokenizer.eos_token_id = tokenizer.eos_token_id
    
    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model and inference function based on backend
    model = None
    if backend == 'sglang':
        # SGLang backend
        print(f"Loading SGLang engine from: {model_path}")
        
        try:
            # Set sampling parameters for SGLang
            sampling_params = {
                "temperature": config['temperature'],
                "max_new_tokens": config['max_new_tokens'],
                "top_p": config['top_p'],
            }
            if config.get('top_k', None) is not None:
                sampling_params["top_k"] = config['top_k']
            if config.get('min_p', None) is not None:
                sampling_params["min_p"] = config['min_p']
            
            # Create LLM engine with specified tp_size
            model = sgl.Engine(
                model_path=model_path,
                dtype=config.get('dtype', 'bfloat16'),
                tp_size=tp_size,
                mem_fraction_static=config.get('mem_fraction_static', 0.90),
                host="127.0.0.1",
                port=int(os.getenv("SGLANG_NCCL_PORT", 30000)),
                attention_backend=config.get('attention_backend', 'triton'),
            )
            
            _warmup_model(model, tokenizer, backend, tp_size)
        
        except Exception as e:
            print(f"Error loading SGLang engine: {e}")
            raise
        
        def inference_function(inputs):
            """SGLang inference function"""
            batch_outputs = []
            for i in range(0, len(inputs), config['batch_size']):
                batch = inputs[i:i + config['batch_size']]
                
                def generate_batch():
                    return model.generate(batch, sampling_params)
                
                outputs, elapsed_time = _time_inference(generate_batch)
                batch_outputs.extend([(out['text'], elapsed_time) for out in outputs])
            
            return batch_outputs
    
    elif backend == 'hf':
        # Hugging Face backend
        print(f"Loading Hugging Face model from: {model_path}")
        print(f"Process CUDA devices available: {torch.cuda.device_count()}")
        
        try:
            # In multiprocess environment, always use device_map="auto" to properly handle GPU allocation
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, config.get('dtype', 'bfloat16')),
                device_map="auto",  # Let transformers handle device allocation based on visible GPUs
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if config.get('use_flash_attention', False) else None,
                low_cpu_mem_usage=True  # Important for proper weight initialization
            )
        
            # Set sampling parameters for Hugging Face
            generation_config = {
                "temperature": config['temperature'],
                "max_new_tokens": config['max_new_tokens'],
                "top_p": config['top_p'],
                "do_sample": True if config['temperature'] > 0.0 else False,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            # Add min_p if supported
            if config.get('min_p', None) is not None:
                generation_config["min_p"] = config['min_p']
            
            # Add top_k if supported
            if config.get('top_k', None) is not None:
                generation_config["top_k"] = config['top_k']
            
            _warmup_model(model, tokenizer, backend, tp_size)
        
        except Exception as e:
            print(f"Error loading Hugging Face model: {e}")
            raise
        
        def inference_function(inputs):
            """Hugging Face inference function"""
            batch_outputs = []
            for i in range(0, len(inputs), config['batch_size']):
                batch = inputs[i:i + config['batch_size']]
                
                # Tokenize batch
                batch_inputs = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left", truncation=True)
                # Move inputs to the same device as the model
                try:
                    model_device = next(model.parameters()).device
                    batch_inputs = {k: v.to(model_device) for k, v in batch_inputs.items()}
                except StopIteration:
                    # If model has no parameters, try to use first available CUDA device
                    if torch.cuda.is_available():
                        batch_inputs = {k: v.cuda() for k, v in batch_inputs.items()}
                
                def generate_batch():
                    with torch.no_grad():
                        return model.generate(**batch_inputs, **generation_config)
                
                outputs, elapsed_time = _time_inference(generate_batch, torch.cuda.is_available())
                
                # Decode outputs
                for j, output in enumerate(outputs):
                    # Remove input tokens from output
                    input_length = batch_inputs['input_ids'][j].shape[0]
                    generated_tokens = output[input_length:]
                    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    batch_outputs.append((output_text, elapsed_time / len(outputs)))
            
            return batch_outputs
    
    elif backend == 'tah':
        # TaH backend
        print(f"Loading TaH model from: {model_path}")
        print(f"Process CUDA devices available: {torch.cuda.device_count()}")

        max_iter = config.get('max_iter', 3)
        embedding_key = config.get('embedding_key', None)
        iter_decider = config.get('iter_decider', None)
        iter_decider_kwargs = config.get('iter_decider_kwargs', None)
        eval_iter_decider = config.get('eval_iter_decider', None)
        eval_iter_decider_kwargs = config.get('eval_iter_decider_kwargs', None)
        use_tracker = config.get('use_tracker', False)
        tracker_kwargs = config.get('tracker_kwargs', None)
        prompt_iter_count = config.get('prompt_iter_count', None)

        override_config = TaHConfig(
            embedding_key=embedding_key,
            max_iter=max_iter,
            iter_decider=iter_decider,
            iter_decider_kwargs=iter_decider_kwargs,
            eval_iter_decider=eval_iter_decider,
            eval_iter_decider_kwargs=eval_iter_decider_kwargs,
        )

        try:
            # In multiprocess environment, always use device_map="auto"
            model = TaHForCausalLM.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, config.get('dtype', 'bfloat16')),
                device_map="auto",  # Let the library handle device allocation
                attn_implementation="sdpa",
                # low_cpu_mem_usage=True,
                tah_config=override_config,
            )
            
        
        except Exception as e:
            print(f"Error loading TaH model: {e}")
            raise

        model = model.to(dtype=model.dtype)
        
        tracker = None
        if use_tracker:
            tracker = TaHTracker(top_k=tracker_kwargs.get('top_k', 5))
            tracker.attach(model)
        
        def inference_function(inputs):
            """TaH inference function"""
            batch_outputs = []
            for i in range(0, len(inputs), config['batch_size']):
                batch = inputs[i:i + config['batch_size']]
                
                # Tokenize all inputs in the batch at once
                input_tokens = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left")
                
                # Move inputs to the same device as the model
                model_device = model.device
                input_tokens = {k: v.to(model_device) for k, v in input_tokens.items()}
                
                if prompt_iter_count is not None:
                    input_ids = input_tokens["input_ids"]
                    batch_size, seq_len = input_ids.shape
                    iter_count = prompt_iter_count * torch.ones(
                        batch_size,
                        seq_len,
                        dtype=torch.long,
                        device=model_device,
                    )
                else:
                    iter_count = None
                
                # Record the number of tracker records before generation if tracker is enabled
                prev_record_len = len(tracker.records) if tracker else 0

                def generate_batch():
                    with torch.no_grad():
                        return TaHForCasualLM_generate(
                            tah_model=model,
                            tokenizer=tokenizer,
                            model_inputs=input_tokens,
                            iter_count=iter_count,
                            max_new_tokens=config['max_new_tokens'],
                            do_sample=True if config['temperature'] > 0.0 else False,
                            temperature=config['temperature'],
                            top_p=config['top_p'],
                            top_k=config.get('top_k', 0),
                            min_p=config.get('min_p', 0.0),
                            verbose=False
                        )
                
                (_, output_texts), elapsed_time = _time_inference(generate_batch, torch.cuda.is_available())
                
                # Process tracker records if enabled
                if tracker:
                    new_records = tracker.records[prev_record_len:]
                    records_by_batch = {}
                    for rec in new_records:
                        bidx = rec.get("batch_idx", 0)
                        records_by_batch.setdefault(bidx, []).append(rec)
                    
                    for j, output_text in enumerate(output_texts):
                        sample_records = records_by_batch.get(j, [])
                        batch_outputs.append((output_text, elapsed_time / len(output_texts), sample_records))
                else:
                    for output_text in output_texts:
                        batch_outputs.append((output_text, elapsed_time / len(output_texts)))
            
            return batch_outputs
    
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'sglang', 'hf', or 'tah'.")
    
    def cleanup_function():
        """Common cleanup function"""
        _cleanup_resources(model, backend)
    
    # No longer require batch_size to be a multiple of repeat_size
    print(f"Processing with batch_size={config['batch_size']}, repeat_size={config['repeat_size']}")
    
    # Store all results
    all_results = []
    problem_stats = {}
    
    # Create intermediate results file
    intermediate_stats_file = output_dir / "intermediate_stats.csv"

    # Prepare detailed results CSV file and write header
    results_file = output_dir / "detailed_results.csv"
    fieldnames = ["problem_id", "sample_idx", "correct_answer", "predicted_answer", 
                     "has_answer", "is_correct", "input_tokens", "output_tokens", 
                     "processing_time"]
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # For code datasets, use unified file or create job-specific file
    answer_type = field_mapping.get('answer_type', 'boxed')
    is_code_dataset = answer_type in ['livecodebench', 'humaneval', 'mbpp']
    code_solutions_file = unified_code_solutions_file  # Use unified file if provided
    
    # Prepare all problem data first
    problem_data = []
    for idx, item in enumerate(problems):
        # Prefer original index if provided (for interleaved assignment), fallback to sequential
        actual_idx = item.get('_original_index', (start_idx + idx))
        
        # Get problem ID using dynamic field mapping
        id_field = field_mapping['id_field']
        if id_field in item and item[id_field] is not None:
            problem_id = str(item[id_field])
        else:
            problem_id = f"problem_{actual_idx}"
        
        # Get problem text using dynamic field mapping
        question_field = field_mapping['question_field']
        problem_text = str(item.get(question_field, '')).strip()
        
        # Apply prompt template if specified
        prompt_template = field_mapping['prompt_template']
        if prompt_template and '{question}' in prompt_template:
            problem_text = prompt_template.replace('{question}', problem_text)
        
        # Get correct answer using dynamic field mapping
        answer_field = field_mapping['answer_field']
        correct_answer = str(item.get(answer_field, '')).strip()
        
        # Create problem-specific directory
        problem_dir = detail_dir / problem_id
        problem_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare problem data dict
        prob_dict = {
            'problem_id': problem_id,
            # Preserve the original task id for downstream evaluators (e.g., evalplus)
            'original_problem_id': item.get('_original_id', problem_id),
            'problem_text': problem_text,
            'correct_answer': correct_answer,
            'problem_dir': problem_dir,
            'actual_idx': actual_idx,
        }
        
        if is_code_dataset:
            prob_dict['entry_point'] = item['entry_point']
        
        problem_data.append(prob_dict)
    
    # Prepare all inputs upfront (each problem repeated repeat_size times)
    all_inputs = []
    input_to_problem_mapping = []  # Track which input belongs to which problem and sample
    
    for prob_idx, prob_data in enumerate(problem_data):
        problem_text = prob_data['problem_text']
        # Create repeat_size copies of this problem
        for sample_idx in range(config['repeat_size']):
            if is_code_dataset:
                input_text = codeeval.make_raw_chat_prompt_for_code_evaluation(task_prompt=problem_text, reasoning=False, tokenizer=tokenizer)
            else:
                messages = [{"role": "user", "content": problem_text}]
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_inputs.append(input_text)
            input_to_problem_mapping.append((prob_idx, sample_idx))
    
    # Process all inputs in batches using the specified batch_size
    total_batches = math.ceil(len(all_inputs) / config['batch_size'])
    
    for batch_idx in tqdm(range(total_batches), desc=f"Job {job_id} processing inference batches", position=job_id, leave=True):
        batch_start = batch_idx * config['batch_size']
        batch_end = min(batch_start + config['batch_size'], len(all_inputs))
        batch_inputs = all_inputs[batch_start:batch_end]
        
        # Run inference for this batch
        batch_outputs = inference_function(batch_inputs)
        
        # Process and save results for this batch immediately
        with open(results_file, 'a', newline='', encoding='utf-8') as f_results:
            writer = csv.DictWriter(f_results, fieldnames=fieldnames)
            for i, inference_output in enumerate(batch_outputs):
                input_idx = batch_start + i
                prob_idx, sample_idx = input_to_problem_mapping[input_idx]
                
                prob_data = problem_data[prob_idx]
                problem_id = prob_data['problem_id']
                # original_problem_id = prob_data['original_problem_id']
                problem_text = prob_data['problem_text']
                correct_answer = prob_data['correct_answer']
                problem_dir = prob_data['problem_dir']

                # Unpack output data
                if isinstance(inference_output, tuple) and len(inference_output) == 3:
                    output_text, proc_time, sample_tracker_records = inference_output
                else:
                    output_text, proc_time = inference_output
                    sample_tracker_records = None
                
                # Extract answer based on dataset type
                # Check if this is a code evaluation dataset
                answer_type = field_mapping.get('answer_type', 'boxed')
                is_code_dataset = answer_type in ['livecodebench', 'humaneval', 'mbpp']
                
                if is_code_dataset:
                    # For code datasets, skip evaluation during generation
                    # Save to jsonl for later batch evaluation
                    predicted_answer = "pending_code_eval"
                    has_answer = False
                    is_correct = False
                else:
                    # Math evaluation path (original logic)
                    result_eval = matheval.evaluator_map[combined_dataset_name].rule_judge(output_text, correct_answer)
                    if result_eval[1] == "No extracted answer":
                        predicted_answer = ""
                        has_answer = False
                    else:
                        predicted_answer = result_eval[1]
                        has_answer = True
                    is_correct = result_eval[0]
                
                # Calculate token counts
                input_tokens = len(tokenizer.encode(problem_text))
                output_tokens = len(tokenizer.encode(output_text))
                
                result_dict = {
                    "problem_id": problem_id,
                    "sample_idx": sample_idx,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "has_answer": has_answer,
                    "is_correct": is_correct,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "processing_time": proc_time,
                    "full_output": output_text
                }
                
                all_results.append(result_dict)

                # Save detailed output in problem-specific directory
                detail_file = problem_dir / f"sample_{sample_idx}.json"
                detail_data = {
                    "problem": problem_text,
                    "output": output_text,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                }
                
                # For code datasets, also save extracted code
                if is_code_dataset:
                    entry_point = prob_data['entry_point']
                    extracted_code = codeeval.sanitize(output_text, entry_point)
                    detail_data["extracted_code"] = extracted_code
                    detail_data["entry_point"] = entry_point
                
                with open(detail_file, 'w', encoding='utf-8') as f_detail:
                    json.dump(detail_data, f_detail, ensure_ascii=False, indent=2)

                # Save tracker records if available
                if sample_tracker_records:
                    tracker_file = problem_dir / f"sample_{sample_idx}_tracker.csv"
                    pd.DataFrame(sample_tracker_records).to_csv(tracker_file, index=False)

                # Write to detailed_results.csv
                row_to_write = {k: v for k, v in result_dict.items() if k != 'full_output'}
                writer.writerow(row_to_write)
                
                # For code datasets, save solution to JSONL for batch evaluation
                if is_code_dataset and code_solutions_file:
                    import fcntl
                    # Use the original problem id so that it matches external evaluators' expectations
                    original_problem_id = prob_data.get('original_problem_id', problem_id)
                    solution_entry = {
                        "task_id": original_problem_id,
                        "solution": str(extracted_code)
                    }
                    # Use file lock to avoid conflicts when multiple jobs write simultaneously
                    with open(code_solutions_file, 'a', encoding='utf-8') as f_code:
                        fcntl.flock(f_code.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                        try:
                            f_code.write(json.dumps(solution_entry, ensure_ascii=False) + '\n')
                        finally:
                            fcntl.flock(f_code.fileno(), fcntl.LOCK_UN)  # Unlock

    # Group results by problem_id to calculate statistics
    results_by_problem = {}
    for r in all_results:
        pid = r['problem_id']
        if pid not in results_by_problem:
            results_by_problem[pid] = []
        results_by_problem[pid].append(r)

    # Calculate stats for each problem
    for problem_id, results in results_by_problem.items():
        correct_count = sum(1 for r in results if r['is_correct'])
        total_samples = len(results)
        accuracy = correct_count / total_samples if total_samples > 0 else 0
        avg_output_length = sum(r['output_tokens'] for r in results) / total_samples if total_samples > 0 else 0
        problem_stats[problem_id] = {
            "accuracy": f"{accuracy:.3f}",
            "correct_count": correct_count,
            "total_samples": total_samples,
            "avg_output_length": avg_output_length
        }
    
    # Save intermediate statistics after processing all problems
    with open(intermediate_stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["problem_id", "accuracy", "correct_count", "total_samples", "avg_output_length"])
        
        for pid, stats in problem_stats.items():
            writer.writerow([pid, stats['accuracy'], stats['correct_count'], stats['total_samples'], f"{stats['avg_output_length']:.2f}"])
    
    # Calculate overall statistics
    total_correct = sum(r['is_correct'] for r in all_results)
    total_accuracy = total_correct / len(all_results) if all_results else 0
    overall_avg_output_length = sum(r['output_tokens'] for r in all_results) / len(all_results) if all_results else 0
    
    # Save statistics to CSV
    stats_file = output_dir / "evaluation_stats.csv"
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["problem_id", "accuracy", "correct_count", "total_samples", "avg_output_length"])
        
        for problem_id, stats in problem_stats.items():
            writer.writerow([problem_id, stats['accuracy'], stats['correct_count'], stats['total_samples'], f"{stats['avg_output_length']:.2f}"])
        
        writer.writerow([])
        writer.writerow(["Total Accuracy", f"{total_accuracy:.3f}", total_correct, len(all_results), f"{overall_avg_output_length:.2f}"])
    
    print(f"\nJob {job_id} completed!")
    print(f"Job accuracy: {total_accuracy:.4f}")
    
    # Clean up resources
    cleanup_function()


def _is_port_available(port: int) -> bool:
    """Check if a port is available for binding"""
    import socket
    # Create a socket and try to bind to the port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', port))
        return True

def _run_job_process(job_args: Tuple, result_queue: Queue):
    """Run a single job in a separate process with isolated GPU environment"""
    (job_id, config, combined_dataset_name, output_dir, timestamp, model_path, 
     job_nums, start_idx, end_idx, tp_size, backend, data_range, gpu_devices, problems_data, field_mapping, unified_code_solutions_file) = job_args
    
    # initialize logger in generated process
    lvl_name = (config.get("_logger_level") or "WARNING").upper()
    hf_level = getattr(hf_logging, lvl_name, hf_logging.WARNING)
    std_level = getattr(pylog, lvl_name, pylog.WARNING)
    hf_logging.set_verbosity(hf_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_propagation()
    pylog.basicConfig(
        level=std_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Set GPU environment for this process - this is isolated per process
    gpu_str = ','.join(map(str, gpu_devices))
    
    # Set unique NCCL port for each job to avoid conflicts
    # Use a base port and add job_id to ensure uniqueness
    base_port = 29555
    max_retries = 100
    unique_port = None
    
    # Try to find an available port
    for retry in range(max_retries):
        port_candidate = base_port + job_id * max_retries + retry
        if port_candidate > 65535:
            port_candidate = 30514 + ((job_id + retry) % 100)
        
        if _is_port_available(port_candidate):
            unique_port = port_candidate
            break
    
    if unique_port is None:
        raise RuntimeError(f"Job {job_id}: Could not find an available port after {max_retries} attempts")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    os.environ['MASTER_PORT'] = str(unique_port)
    os.environ['SGLANG_NCCL_PORT'] = str(unique_port)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    
    # Set random seed for this process
    import torch
    from tah.model.utils import set_all_seeds
    seed = config.get("_random_seed", 420)
    set_all_seeds(seed)
    # Force CUDA to reinitialize in this process
    if torch.cuda.is_available():
        torch.cuda.init()
    
    try:
        print(f"\nJob {job_id}: Starting with GPUs {gpu_devices} (CUDA_VISIBLE_DEVICES={gpu_str})")
        print(f"Job {job_id}: Using NCCL port {unique_port}")
        print(f"Job {job_id}: Processing indices {start_idx} to {end_idx-1}")
        print(f"Job {job_id}: Available CUDA devices in process: {torch.cuda.device_count()}")
        
        # Run the actual job
        run_single_job(
            config=config,
            combined_dataset_name=combined_dataset_name,
            output_dir=output_dir,
            timestamp=timestamp,
            model_path=model_path,
            job_id=job_id,
            job_nums=job_nums,
            start_idx=start_idx,
            end_idx=end_idx,
            tp_size=tp_size,
            backend=backend,
            data_range=data_range,
            problems_data=problems_data,
            field_mapping=field_mapping,
            unified_code_solutions_file=unified_code_solutions_file
        )
        
        print(f"Job {job_id}: Completed successfully")
        result_queue.put((job_id, True, f"Job {job_id} completed successfully"))
        
    except Exception as e:
        import traceback
        error_msg = f"Job {job_id} failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\nError in {error_msg}")
        result_queue.put((job_id, False, error_msg))
    finally:
        # Clean up environment variables
        if 'MASTER_PORT' in os.environ:
            del os.environ['MASTER_PORT']
        if 'MASTER_ADDR' in os.environ:
            del os.environ['MASTER_ADDR']
        if 'SGLANG_NCCL_PORT' in os.environ:
            del os.environ['SGLANG_NCCL_PORT']
        if 'NCCL_SOCKET_IFNAME' in os.environ:
            del os.environ['NCCL_SOCKET_IFNAME']

def allocate_gpus_and_run_jobs(args):
    """Allocate GPUs to jobs and run them using multiprocessing"""
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    # Create a unified timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Calculate GPU allocation
    gpus_per_job = args.tp_size_per_job
    
    print(f"Running {args.job_nums} jobs with {gpus_per_job} GPUs per job")
    
    # Get the current CUDA_VISIBLE_DEVICES setting
    current_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if current_cuda_devices:
        available_gpus = [int(x.strip()) for x in current_cuda_devices.split(',') if x.strip()]
    else:
        available_gpus = [0,1,2,3,4,5,6,7]
    
    print(f"Available GPU devices: {available_gpus}")
    
    # Load dataset to get total problem count and apply optional data_range / data_ids
    dataset_names = [name.strip() for name in args.dataset_name.split(',')]
    dataset, field_mapping = load_datasets_with_config(dataset_names)
    combined_dataset_name = "_".join(dataset_names)
    
    total_problems = len(dataset)

    # Unified problem selection: get selected problem indices regardless of selection method
    if getattr(args, 'data_ids', None):
        # Select by specific problem IDs
        valid_ids = set(str(item.get('id')) for item in dataset)
        raw_ids = [s.strip() for s in str(args.data_ids).split(',') if s.strip() != '']
        seen = set()
        selected_problems = []
        for pid in raw_ids:
            if pid in valid_ids and pid not in seen:
                seen.add(pid)
                selected_problems.append(pid)
        if not selected_problems:
            raise ValueError("--data_ids did not match any problem IDs. Use standardized IDs like <dataset>_<original_id>.")
    else:
        # Select by data range - convert to list of indices
        range_start, range_end = parse_data_range(args.data_range, total_problems)
        selected_problems = list(range(range_start, range_end))
    
    problems_per_job = math.ceil(len(selected_problems) / args.job_nums)
    
    # Load configuration once for all jobs
    eval_config = load_config(args.eval_config)
    eval_config["_logger_level"] = args.logger_level
    # Pass random seed from CLI args into config so worker processes can read it
    eval_config["_random_seed"] = getattr(args, "random_seed", 420)

    # Save the yaml configuration file to output directory
    # Create output directory structure first
    task_suffix = ""
    if args.data_range and not getattr(args, 'data_ids', None):
        range_start, range_end = parse_data_range(args.data_range, total_problems)
        task_suffix = f"TASK_{range_start}_{range_end}"
    
    default_output_dir = Path(args.model_path) / "eval_results"
    if args.output_dir is None:
        args.output_dir = default_output_dir
    
    combined_output_dir = Path(args.output_dir) / (combined_dataset_name + "_" + args.backend) / timestamp
    if task_suffix:
        combined_output_dir = combined_output_dir / task_suffix
    combined_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the original eval config file to output directory
    import shutil
    config_filename = Path(args.eval_config).name
    saved_config_path = combined_output_dir / config_filename
    shutil.copy2(args.eval_config, saved_config_path)
    print(f"Saved evaluation config to: {saved_config_path}")
    
    # For code datasets, prepare unified code_solutions.jsonl file
    answer_type = field_mapping.get('answer_type', 'boxed')
    is_code_dataset = answer_type in ['livecodebench', 'humaneval', 'mbpp']
    unified_code_solutions_file = None
    
    if is_code_dataset:
        unified_code_solutions_file = combined_output_dir / "code_solutions.jsonl"
        # Create empty file
        with open(unified_code_solutions_file, 'w', encoding='utf-8') as f:
            pass
        print(f"Created unified code solutions file: {unified_code_solutions_file}")
    
    # Prepare job arguments by splitting selected problems across jobs
    job_args_list = []
    for job_id in range(args.job_nums):
        # Split selected problems across jobs using interleaving
        job_problems_indices = selected_problems[job_id::args.job_nums]
        
        if not job_problems_indices:
            continue
            
        gpu_start = job_id * gpus_per_job
        job_gpus = available_gpus[gpu_start:gpu_start + gpus_per_job]
        
        # Extract actual problem data for this job
        if getattr(args, 'data_ids', None):
            # For data_ids mode, filter by problem IDs
            id_to_item = {str(item.get('id')): item for item in dataset}
            job_problems_data = [id_to_item[str(pid)] for pid in job_problems_indices if str(pid) in id_to_item]
            start_idx = 0
            end_idx = len(job_problems_indices)
        else:
            # For data_range mode, use indices directly and attach original index for bookkeeping
            job_problems_data = []
            for i in job_problems_indices:
                item = dict(dataset[i])
                item['_original_index'] = i
                job_problems_data.append(item)
            start_idx = 0
            end_idx = len(job_problems_indices)
        
        # Create unified job_args with pre-processed data
        job_args = (
            job_id, eval_config, combined_dataset_name, args.output_dir, timestamp, args.model_path,
            args.job_nums, start_idx, end_idx, gpus_per_job, args.backend, args.data_range,
            job_gpus, job_problems_data, field_mapping, unified_code_solutions_file
        )
        
        job_args_list.append(job_args)
    
    print(f"\nPrepared {len(job_args_list)} jobs for execution")
    
    # Create result queue for inter-process communication
    result_queue = mp.Queue()
    
    # Execute jobs using multiprocessing with limited concurrency
    completed_jobs = 0
    failed_jobs = 0
    active_processes = []
    
    # Start processes in batches based on max_concurrent_jobs
    job_idx = 0
    while job_idx < len(job_args_list) or active_processes:
        # Start new processes if we have capacity
        while job_idx < len(job_args_list):
            job_args = job_args_list[job_idx]
            p = Process(target=_run_job_process, args=(job_args, result_queue), name=f"sgl-{job_idx}")
            p.start()
            active_processes.append((p, job_args[0]))  # Store process with job_id
            print(f"Started job {job_args[0]}")
            job_idx += 1
        
        # Check for completed processes
        still_active = []
        for p, job_id in active_processes:
            if p.is_alive():
                still_active.append((p, job_id))
            else:
                p.join(timeout=1)  # Ensure process is cleaned up
                if p.exitcode != 0 and p.exitcode is not None:
                    print(f"Process for job {job_id} exited with code {p.exitcode}")
        
        active_processes = still_active
        
        # Process results from queue (non-blocking)
        while not result_queue.empty():
            job_id_result, success, message = result_queue.get_nowait()
            if success:
                completed_jobs += 1
                print(f"\n✓ Job {job_id_result} finished successfully")
            else:
                failed_jobs += 1
                print(f"\n✗ Job {job_id_result} failed: {message}")
        
        # Small sleep to prevent busy waiting
        if active_processes:
            time.sleep(0.1)
    
    # Final check for any remaining results
    while not result_queue.empty():
        job_id_result, success, message = result_queue.get_nowait()
        if success:
            completed_jobs += 1
            print(f"\n✓ Job {job_id_result} finished successfully")
        else:
            failed_jobs += 1
            print(f"\n✗ Job {job_id_result} failed: {message}")
    
    print(f"\nAll jobs completed!")
    print(f"Successful jobs: {completed_jobs}")
    print(f"Failed jobs: {failed_jobs}")
    
    if failed_jobs > 0:
        print(f"Warning: {failed_jobs} jobs failed. Results may be incomplete.")
    
    # Combine results
    print("\nCombining results from all jobs...")
    combined_output_dir = Path(args.output_dir) / (combined_dataset_name + "_" + args.backend) / timestamp
    if args.data_range and not getattr(args, 'data_ids', None):
        range_start, range_end = parse_data_range(args.data_range, total_problems)
        combined_output_dir = combined_output_dir / f"TASK_{range_start}_{range_end}"
    combine_job_results(combined_output_dir, len(job_args_list), args.del_job_dir)
    
    # For code datasets, run batch evaluation using the unified file
    if is_code_dataset and unified_code_solutions_file and unified_code_solutions_file.exists():
        print(f"\n{'='*60}")
        print(f"Starting code evaluation for {combined_dataset_name}...")
        print(f"Solutions file: {unified_code_solutions_file}")
        print(f"Total lines: {sum(1 for _ in open(unified_code_solutions_file))}")
        print(f"{'='*60}\n")

        # Import codeeval
        from tah.evaluate.codeeval import evaluate as code_evaluate
        
        # Determine dataset name for evalplus (humaneval or mbpp)
        answer_type = field_mapping.get('answer_type', 'boxed')
        evalplus_dataset = answer_type if answer_type in ['humaneval', 'mbpp'] else 'humaneval'
        
        # Call codeeval.evaluate
        code_evaluate(
            dataset=evalplus_dataset,
            samples=str(unified_code_solutions_file),
        )
        
        print(f"\n{'='*60}")
        print(f"Code evaluation completed!")
        print(f"Results saved to: {str(unified_code_solutions_file).replace('.jsonl', '.eval_results.json')}")
        print(f"{'='*60}\n")


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Utility to parse data_range argument

def parse_data_range(data_range_list, total_problems: int = None) -> Tuple[int, int]:
    """Parse data_range list and return (start_idx, end_idx).

    data_range_list format examples:
    - [200]      -> start=0,   end=200
    - [100, 200] -> start=100, end=200

    If total_problems is provided, end_idx will be clipped to this value.
    The returned end_idx is exclusive (i.e. slice compatible).
    """
    if not data_range_list:
        return 0, total_problems if total_problems is not None else 0

    # Handle single value (treated as end index)
    if len(data_range_list) == 1:
        start_idx, end_idx = 0, data_range_list[0]
    elif len(data_range_list) == 2:
        start_idx, end_idx = data_range_list[0], data_range_list[1]
    else:
        raise ValueError(f"Invalid data_range: expected 1 or 2 values, got {len(data_range_list)}")

    if total_problems is not None:
        end_idx = min(end_idx, total_problems)

    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError(f"Invalid data_range: start={start_idx}, end={end_idx}")

    return start_idx, end_idx