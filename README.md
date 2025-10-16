<p align="center">
  <img src="resource/logo.png" alt="Project TaH Logo" width="150"/>
</p>

# Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models

Feel free to star the repo if you find it interesting. The paper is coming soon.

## Environment Setup
Create a new environment:

```bash
conda create -n tah python=3.10
conda activate tah
```

Install the package:

```bash
pip install -e .
```

For training and evaluation, install additional dependencies:

```bash
pip install -e ".[training,evaluation]"
```


## Run an example for TaH

```bash
python script/playground/inference_example.py
```

This script demonstrates TaH's selective latent iteration mechanism, with color-coded output showing the iteration count for each token.

## Run evaluation


### Evaluate TaH model
```bash
python script/evaluation/eval.py \
    --eval_config ./script/recipes/qwen3_1.7/eval_tah.yaml \
    --model_path /path/to/TaH-1.7B \
    --dataset_name gsm8k \
    --backend tah \
    --job_nums 8 \
    --tp_size_per_job 1
```

Key parameters:
- `--eval_config`: Path to evaluation config file
- `--model_path`: Path to the model
- `--dataset_name`: Dataset name (supports gsm8k, math500, aime24, etc. Detailed configs can be found in `tah/evaluate/eval_configs/dataset_configs.json`)
- `--backend`: Inference backend (`tah` for TaH)
- `--job_nums`: Number of parallel jobs
- `--tp_size_per_job`: Tensor parallel size per job

### Evaluate standard baseline model
```bash
python script/evaluation/eval.py \
    --eval_config ./script/recipes/qwen3_1.7/eval_base.yaml \
    --model_path /path/to/standard-1.7B \
    --dataset_name gsm8k \
    --backend hf \
    --job_nums 8 \
    --tp_size_per_job 1
```

Similar to TaH evaluation, but using:
- `--backend hf` or `--backend sglang`

## Train your own TaH model

Training a TaH model consists of three stages:

### Step0: Prepare model and data

**1. Prepare training data**

Use a reference model to generate hard token labels for the training and validation data:

```bash
### step 0
python script/preparation/label.py \
    --num_gpu 8 \
    --dataset_path ./data/initial_data/openr1-math/train.jsonl \
    --test_model_list Qwen/Qwen3-1.7B \
    --output_path ./data/processed_data/openr1-math/1_7/train \
    --max_input_length 10000
python script/preparation/label.py \
    --num_gpu 8 \
    --dataset_path ./data/initial_data/openr1-math/eval.jsonl \
    --test_model_list Qwen/Qwen3-1.7B \
    --output_path ./data/processed_data/openr1-math/1_7/eval \
    --max_input_length 10000 \
```

**2. (Optional) Prepare pruned model**

For the TaH version, prune one layer from the base model to match the parameter count of the standard baseline (skip this step for TaH+ version):

```bash
### step 0
python script/preparation/prune.py \
    --model Qwen/Qwen3-1.7B-Base \
    --dataset ./data/processed_data/openr1-math/1_7/eval \
    --output ./model/qwen3_1.7_base_pruned \
    --num_prune 1
```

### Step1: Train with Fixed Iteration Labels

The first stage uses fixed iteration labels for training:

```bash
### step 1
python -m accelerate.commands.launch \
    --config_file ./script/recipes/accelerate_configs/zero2.yaml \
    --num_processes 8 \
    ./script/train/SFT_TaH.py \
    --config ./script/recipes/qwen3_1.7/sft_tah_step1.yaml
```

Key configurations in Step1 (`sft_tah_step1.yaml`):
- `max_iter: 2`: Maximum number of iterations
- `iter_decider: "FixedLabelIterDecider"`: Use fixed labels to decide iterations
- `iter_label_generator: "FixedIterLabelGenerator"`: Generate labels from mismatch field in data
- `input_updater: "AdditiveUpdater"`: Use additive updater for input updates
- `adapter: "lora"`: Use LoRA adapter for deeper iteration
- `train_loss: "NextTokenPredLoss"`: Next token prediction loss

### Step2: Train Iteration Decider

The second stage trains the iteration decider:


```bash
### step 2
python -m accelerate.commands.launch \
    --config_file ./script/recipes/accelerate_configs/zero2.yaml \
    --num_processes 8 \
    ./script/train/SFT_TaH.py \
    --config ./script/recipes/qwen3_1.7/sft_tah_step2.yaml
```

Key configurations in Step2 (`sft_tah_step2.yaml`):
- `tah_model_path`: Load the model trained in Step1
- `iter_decider: "MLPIterDecider"`: Use MLP decider to automatically determine iterations
- `train_loss: "IterDeciderLoss"`: Iteration decider loss function
- `freeze_component: [model.simple_base_model]`: Freeze model backbone

After two-stage training, the model can automatically decide when to perform latent reasoning iterations.

## Understand the Code

### Code Structure

```
TaH/
├── tah/                           # Core package
│   ├── model/                     # Core model components
│   ├── train/                     # Training components
│   ├── evaluate/                  # Evaluation utilities
│   └── utils/                     # General utilities
├── bash/                          # Bash scripts for training and evaluation
├── script/                        # Execution scripts
│   ├── analysis/                  # Analysis scripts
│   ├── evaluation/                # Evaluation scripts
│   ├── preparation/               # Preparation for training
│   │   ├── label.py               # Data labeling (generate mismatch labels)
│   │   └── prune.py               # Model pruning
│   ├── playground/                # Some examples
│   └── recipes/                   # Configuration files
│       ├── qwen3_0.6/             # Qwen3-0.6B-Base configs
│       ├── qwen3_1.7/             # Qwen3-1.7B-Base configs
│       └── accelerate_configs/    # Distributed training configs
└── pyproject.toml                 # Project configuration
```

## Future Work

- [ ] Support more inference backends (e.g., SGLang)
- [ ] Optimize iteration decision strategies
- [ ] Integrate TaH with online distillation or RL
- [ ] Support training for larger models

