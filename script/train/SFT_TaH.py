import os
import torch
import yaml
import argparse
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from typing import Dict

from tah.model.recurrent_transformer import TaHForCausalLM
from tah.model.tah_config import TaHConfig
from tah.model.iter_decider import load_iter_decider
from tah.train import CustomTaHTrainer, CustomTaHDataCollator, LoggerCallback
from tah.utils.data_prepare import preprocess_dataset

try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
except ImportError:
    AutoLigerKernelForCausalLM = None

from tah.model.utils import set_all_seeds, freeze_components, compute_trainable_param_size_gb
from dataclasses import fields

set_all_seeds(420)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(training_config: Dict, model_config: Dict, accelerator: Accelerator):
    """Load model and tokenizer based on configuration."""
    accelerator.print("Loading model and tokenizer...")
    
    # Convert torch dtype string to actual torch dtype
    dtype_mapping = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    
    torch_dtype = dtype_mapping.get(model_config['torch_dtype'], torch.bfloat16)
    
    # use accelerator's device setting
    device_map = None if accelerator.num_processes >= 1 else model_config.get('device_map', 'auto')

    # Check if we should load a pretrained TaH model
    if 'tah_model_path' in model_config:
        accelerator.print(f"Loading pretrained TaH model from: {model_config['tah_model_path']}")
        
        # Create TaH config for overriding if specified in model_config
        tah_config = None
        tah_config_fields = [field.name for field in fields(TaHConfig)]
        
        if any(key in model_config for key in tah_config_fields):
            # Only set fields that are present in the YAML config
            overide_config_dict = {}
            for field in tah_config_fields:
                if field in model_config:
                    overide_config_dict[field] = model_config[field]

            tah_config = TaHConfig(**overide_config_dict)
            
            accelerator.print("Using TaH config from YAML to override saved config:")
            accelerator.print(f"TaH config override: {overide_config_dict}")
        else:
            accelerator.print("Using saved TaH config from pretrained model")
        
        # Load pretrained TaH model with optional config override
        model = TaHForCausalLM.from_pretrained(
            model_config['tah_model_path'],
            tah_config=tah_config
        ).to(dtype=torch_dtype)

        # Load tokenizer from the original base model path or from the tah model path
        tokenizer_path = model_config.get('name', model_config['tah_model_path'])
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=model_config.get('trust_remote_code', True),
            padding_side="right"
        )

        accelerator.print("Successfully loaded pretrained TaH model")
        accelerator.print(f"Model architecture: {model}")
        
    else:
        # Original logic for loading base model and creating TaH model
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'], 
            trust_remote_code=model_config['trust_remote_code'],
            padding_side="right"
        )

        # Create TaH config (populate only from provided keys)
        tah_config = TaHConfig(embedding_key=model_config.get('embedding_key', "model.embed_tokens"))
        for f in fields(TaHConfig):
            if f.name == 'embedding_key':
                continue
            if f.name in model_config:
                setattr(tah_config, f.name, model_config[f.name])
        use_base_model_only = (tah_config.max_iter == 1)
        
        # load base model
        if training_config.get('enable_liger_kernel', False):
            if AutoLigerKernelForCausalLM is None:
                raise ImportError("liger_kernel is not installed. Please install it using `pip install liger_kernel`.")
            base_model = AutoLigerKernelForCausalLM.from_pretrained(
                model_config['name'],
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=model_config['trust_remote_code'],
                attn_implementation=model_config['attn_implementation']
            )
            accelerator.print("Using Liger Kernel")
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config['name'],
                torch_dtype=torch_dtype,
                device_map=device_map if not use_base_model_only else None, # cannot use device_map=auto for base model if using TaH
                trust_remote_code=model_config['trust_remote_code'],
                attn_implementation=model_config['attn_implementation']
            )

        if tah_config.max_iter == 1:
            model = base_model
        else:
            if "load_path" in tah_config.iter_decider_kwargs:
                iter_decider_path = tah_config.iter_decider_kwargs.pop("load_path")
                model = TaHForCausalLM(base_model=base_model, config=tah_config, device_map=device_map)
                model.iter_decider = load_iter_decider(iter_decider_path)
            else:
                # regular init
                model = TaHForCausalLM(base_model=base_model, config=tah_config, device_map=device_map)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def create_training_args(training_config: Dict, data_config: Dict, output_dir: str, accelerator: Accelerator, timestamp: str = None) -> TrainingArguments:
    """Create training arguments from configuration."""
    accelerator.print("Configuring training arguments...")
    
    training_args_dict = {
        'output_dir': output_dir,
        'num_train_epochs': training_config['num_train_epochs'],
        'per_device_train_batch_size': training_config['per_device_train_batch_size'],
        'gradient_accumulation_steps': training_config['gradient_accumulation_steps'],
        'gradient_checkpointing': training_config['gradient_checkpointing'],
        'learning_rate': training_config['learning_rate'],
        'warmup_ratio': training_config['warmup_ratio'],
        'weight_decay': training_config['weight_decay'],
        'max_grad_norm': training_config['max_grad_norm'],
        'lr_scheduler_type': training_config['lr_scheduler_type'],
        'lr_scheduler_kwargs': training_config['lr_scheduler_kwargs'],
        'logging_steps': training_config['logging_steps'],
        'save_strategy': training_config['save_strategy'],
        'save_steps': training_config.get('save_steps', 100),
        'save_only_model': training_config['save_only_model'],
        'save_total_limit': training_config['save_total_limit'],
        'report_to': training_config['report_to'],
        'bf16': training_config['bf16'],
        # accelerate
        'remove_unused_columns': False,  
        'ddp_find_unused_parameters': False,  
    }
    
    # set evaluation dataset ratio
    if data_config.get('eval_data_ratio', 0.0) > 0 or data_config.get('eval_data_path', None) is not None:
        training_args_dict['eval_strategy'] = training_config.get('eval_strategy')
        training_args_dict['eval_steps'] = training_config.get('eval_steps')
        training_args_dict['per_device_eval_batch_size'] = training_config.get('per_device_eval_batch_size')
        training_args_dict['eval_on_start'] = training_config.get('eval_on_start')
    
    # set wandb related environment variables and run_name parameter
    if training_config['report_to'] == "wandb":
        # set environment variables
        if 'wandb_project' in training_config:
            os.environ['WANDB_PROJECT'] = training_config['wandb_project']
        if 'wandb_name' in training_config:
            os.environ['WANDB_NAME'] = training_config['wandb_name']
        if 'wandb_entity' in training_config:
            os.environ['WANDB_ENTITY'] = training_config['wandb_entity']
        
        # use run_name parameter (transformers 4.52.4 supported)
        default_run_name = f"training_{timestamp}" if timestamp else f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_args_dict['run_name'] = training_config.get('wandb_name', default_run_name)
    
    return TrainingArguments(**training_args_dict)


def train_model(model, tokenizer, processed_train_dataset, processed_eval_dataset, training_args, accelerator: Accelerator, training_config: Dict, resume_from_checkpoint_path: str = None):
    """Initialize trainer and start training."""
    # Use custom data collator that handles iter_count field
    # Create trainer first (without data_collator)
    trainer = CustomTaHTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        processing_class=tokenizer,  
        prediction_config=None,
    )
    
    # Create data collator (noise logic removed from collator)
    data_collator = CustomTaHDataCollator(
        tokenizer=tokenizer,
        padding=True,
    )
    
    # Set data collator on trainer
    trainer.data_collator = data_collator

    # Instantiate the LoggerCallback to track iter count
    iter_count_callback = LoggerCallback(
        trainer=trainer
    )
    model.logger_callback = iter_count_callback
    trainer.callback_handler.callbacks.insert(0, iter_count_callback)
    
    accelerator.print("\n--- Starting Training ---")
    if resume_from_checkpoint_path is not None:
        accelerator.print(f"Resuming training from checkpoint: {resume_from_checkpoint_path}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint_path)
    else:
        trainer.train()
    accelerator.print("--- Training Complete ---")
    
    # get training history
    training_history = {
        'train_loss': [],
        'eval_loss': [],
    }
    
    # get training history from trainer's log_history
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        accelerator.print(f"Found {len(trainer.state.log_history)} log entries")
        for log_entry in trainer.state.log_history:
            if 'train_loss' in log_entry:
                training_history['train_loss'].append(log_entry['train_loss'])
            if 'eval_loss' in log_entry:
                training_history['eval_loss'].append(log_entry['eval_loss'])
    
    accelerator.print(f"Collected {len(training_history['train_loss'])} training loss entries")
    accelerator.print(f"Collected {len(training_history['eval_loss'])} evaluation loss entries")
    
    return trainer, training_history


def save_final_model(trainer, tokenizer, output_dir: str, config: Dict, accelerator: Accelerator):
    """Save the final model and configuration."""
    # 
    final_model_path = os.path.join(output_dir, "final_model")
    accelerator.print(f"Saving final model to: {final_model_path}")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    accelerator.print("Model and configuration saved successfully!")


def main(config):
    """Main training function."""
    # initialize accelerator first
    accelerator = Accelerator(
        mixed_precision='bf16',
        log_with="wandb" if os.environ.get("WANDB_MODE") != "disabled" else None,
    )
    
    # generate timestamp only on main process and broadcast to all processes
    timestamp = None
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # broadcast timestamp to all processes
    timestamp_list = [timestamp]
    broadcast_object_list(timestamp_list)
    timestamp = timestamp_list[0]

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    
    # Extract key paths
    output_dir = data_config['output_dir']
    
    if 'tah_model_path' in model_config:
        # For continued training from pretrained TaH model
        base_model_name = model_config['name'].split('/')[-1]
        output_dir = os.path.join(output_dir, "continue_training", base_model_name, timestamp)
        accelerator.print(f"Continue training mode - using output directory: {output_dir}")
    else:
        # For training from scratch - use detailed naming
        output_dir = os.path.join(output_dir, (model_config['name'].split('/')[-1] + "_" + model_config['input_updater'][:-7]))
        output_dir = output_dir + "_" + str(model_config['input_updater_kwargs'].get('num_layers', ''))
        output_dir = output_dir + "_" + model_config['iter_decider'][:-11]
        if model_config['adapter'] != 'none':
            output_dir = output_dir + "_" + model_config['adapter']
        output_dir = os.path.join(output_dir, timestamp)
    
    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(training_config, model_config, accelerator)
    
    # Optionally freeze specified components and report trainable size
    freeze_list = training_config.get('freeze_component', [])
    if isinstance(freeze_list, str):
        freeze_list = [freeze_list]
    if freeze_list:
        accelerator.print(f"Freezing components: {freeze_list}")
        freeze_components(model, freeze_list, accelerator)

    trainable_gb = compute_trainable_param_size_gb(model)
    accelerator.print(f"Trainable parameter size: {trainable_gb:.3f} GB")
    
    # Preprocess Dataset
    processed_train_dataset, processed_eval_dataset, avg_hard_ratio = preprocess_dataset(training_config, data_config, model_config, accelerator)
    
    # Calculate and set balanced weights if hard_token_relative_weight is not 1.0
    hard_token_relative_weight = training_config.get('hard_token_relative_weight', 1.0)
    if hard_token_relative_weight != 1.0: 
        # Calculate weights such that:
        # 1. p * weight_hard + (1 - p) * weight_easy = 1.0
        # 2. weight_hard / weight_easy = r
        weight_easy = 1.0 / (avg_hard_ratio * hard_token_relative_weight + (1 - avg_hard_ratio))
        weight_hard = hard_token_relative_weight * weight_easy
        model.weight_hard = weight_hard
        model.weight_easy = weight_easy
        model.hard_token_relative_weight = hard_token_relative_weight
        accelerator.print(f"Calculated balanced weights:")
        accelerator.print(f"  - Hard token ratio: {avg_hard_ratio:.4f}")
        accelerator.print(f"  - Weight for hard tokens: {weight_hard:.4f}")
        accelerator.print(f"  - Weight for easy tokens: {weight_easy:.4f}")
    else:
        accelerator.print(f"Skipping balanced weights calculation because hard_token_relative_weight is 1.0")

    # Create Training Arguments
    training_args = create_training_args(training_config, data_config, output_dir, accelerator, timestamp)
    
    # Print training infos
    if accelerator.is_main_process:
        print(f"Model: {model_config['name']}")
        print(f"Output directory: {output_dir}")
        print(f"Training epochs: {training_config['num_train_epochs']}")
        print(f"Batch size: {training_config['per_device_train_batch_size']}")
        print(f"Learning rate: {training_config['learning_rate']}")
        print(f"Max length: {data_config.get('max_length', None)}")
        print(f"Max length action: {data_config.get('max_length_action', 'cutoff')}")
        print("--- Training begins ---\n")
    
    # Also save the configuration file for reference
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    accelerator.print(f"Configuration saved to: {config_save_path}")
    
    # Determine resume checkpoint path (if requested)
    resume_from_ckpt = training_config.get('resume_from_ckpt', False)
    resume_from_checkpoint_path = None
    if resume_from_ckpt and ('tah_model_path' in model_config):
        resume_from_checkpoint_path = model_config['tah_model_path']
        accelerator.print(f"Resume-from-ckpt enabled. Using checkpoint path: {resume_from_checkpoint_path}")

    # Train Model
    trainer, training_history = train_model(
        model,
        tokenizer,
        processed_train_dataset,
        processed_eval_dataset,
        training_args,
        accelerator,
        training_config,
        resume_from_checkpoint_path=resume_from_checkpoint_path,
    )
    
    # Save Final Model
    save_final_model(trainer, tokenizer, output_dir, config, accelerator)
    
    return training_history

if __name__ == "__main__":
    # Load Configuration
    parser = argparse.ArgumentParser(description='Train a causal language model with configuration file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    
    training_history = main(config)
    
    # Print training history summary
    if training_history:
        print("\n--- Training History Summary ---")
        print(f"Number of training loss entries: {len(training_history['train_loss'])}")
        print(f"Number of evaluation loss entries: {len(training_history['eval_loss'])}")
        if training_history['train_loss']:
            print(f"Final training loss: {training_history['train_loss'][-1]:.6f}")
        if training_history['eval_loss']:
            print(f"Final evaluation loss: {training_history['eval_loss'][-1]:.6f}")
        print("Full training history returned in training_history variable")