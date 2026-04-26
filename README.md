<div align="center">

  <img src="resource/logo.png" alt="Project TaH Logo" width="100"/>

  

  <h1>Think-at-Hard</h1>

  <h3>Selective Latent Iterations to Improve Reasoning Language Models</h3>

  <p>
    <a href="https://fuvty.github.io/TaH_project_page/">🌐 <b>Project Page</b></a> •
    <a href="https://arxiv.org/abs/2511.08577">📑 <b>Paper</b></a> •
    <a href="https://huggingface.co/collections/nics-efc/tah">🤗 <b>HuggingFace</b></a>
  </p>

</div>
Think-at-Hard (TaH) improves LLM reasoning by running extra latent iterations only on hard tokens instead of all tokens. A lightweight decider and duo-causal attention enable targeted refinement while keeping full parallelism. TaH outperforms fixed two-iteration baselines by 8–11% while skipping 94% of second iterations, and also beats strong single-iteration Qwen3 models by 4–5%.

Feel free to star the repo or cite the paper if you find it interesting.

```bibtex
@article{fu2025tah,
    title={Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models}, 
    author={Tianyu Fu and Yichen You and Zekai Chen and Guohao Dai and Huazhong Yang and Yu Wang},
    journal={arXiv preprint arXiv:2510.08577},
    year={2025},
}
```
## News

* [2025/11] We released the [TaH-plus-1.7B](https://huggingface.co/nics-efc/TaH-plus-1.7B) checkpoint. The model is finetuned from [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) using 100K samples from the [OpenR1](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) dataset, capable of QA, math, and coding. 

* [2025/11] Our paper was featured as the #2 Paper of the Day on [Huggingface Daily Papers](https://huggingface.co/papers/date/2025-11-19)

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

For code generation evaluation, install [evalplus](https://github.com/evalplus/evalplus)

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
    --model_path nics-efc/TaH-plus-1.7B \
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

### Evaluate with a different backend

The same `script/evaluation/eval.py` accepts `--backend hf` (vanilla
`AutoModelForCausalLM.generate` — useful for non-TaH baselines) or
`--backend sglang` (sgl Engine for high-throughput serving). All three
backends share the same job-sharded driver under
`tah/evaluate/jobs.py:allocate_gpus_and_run_jobs`.

## Train your own TaH model

Training a TaH model consists of three stages:

### Step0: Prepare model and data

**1. Prepare training data**

Use a reference model to generate hard token labels for the training and validation data:

```bash
### step 0
# download the default subset of OpenR1-Math-220k
python script/preparation/download.py
# filter and split
python script/preparation/filter_split.py
# label the hard tokens
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
- `max_iter: 2` — maximum number of iterations.
- `iter_decider: "IterLabelDecider"` — continue iff the per-token oracle
  ``iter_count_labels`` (derived from ``mismatch``) say so. Used to teach
  the LoRA adapter on tokens marked "hard" by the labeller.
- `adapter: "lora"` — only LoRA is supported in tah-release.
- `train_loss: "NextTokenPredLoss"` — standard causal-LM cross-entropy.

Note: the input updater (top-k softmax over logits → embedding mix), the
output updater (residual additive accumulation), the iter-label generator
(dense max-merge of dataset labels), and the adapter setup are all inlined
into the wrapper, so there's no separate config field for them anymore.

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
├── tah/
│   ├── model/                     # core model
│   │   ├── tah_model.py           # TaHForCausalLM wrapper + inlined slot helpers
│   │   ├── iter_decider.py        # IterLabelDecider, MLPIterDecider, _BY_NAME
│   │   ├── loss.py                # NextTokenPredLoss, IterDeciderLoss, _BY_NAME
│   │   ├── causal_cache.py        # TaHCache: per-(layer, iter) KV
│   │   ├── tah_config.py          # @dataclass TaHConfig
│   │   └── utils.py               # generation helper, IterCountColors
│   ├── train/                     # HF Trainer subclass + collator + iter-aware callback
│   ├── evaluate/                  # multi-backend eval driver
│   │   ├── datasets.py            # benchmark loading + standardisation
│   │   ├── backends.py            # sglang / hf / tah model + inference fn
│   │   ├── jobs.py                # job-sharded runner + result aggregation
│   │   ├── matheval.py            # math benchmark graders (math_verify)
│   │   ├── codeeval.py            # humaneval / mbpp via evalplus
│   │   └── eval_unified.py        # backwards-compat shim
│   └── utils/                     # SFT preprocessing
├── script/
│   ├── preparation/               # download.py, label.py, prune.py, filter_split.py
│   ├── train/SFT_TaH.py           # SFT entrypoint
│   ├── evaluation/eval.py         # eval CLI entrypoint
│   ├── playground/                # inference demo
│   └── recipes/qwen3_{0.6,1.7}/   # training + eval YAML recipes
├── tests/                         # per-component acc/speed tests
└── pyproject.toml
```

## Tests + benchmarks

```bash
pytest tests/ -q                  # 21 acc + roundtrip tests in ~30s on a B200
python tests/bench.py components  # microbenchmarks for the wrapper's hot helpers
python tests/bench.py e2e         # forward + 32-token generate on TaH-plus-1.7B
python tests/bench_compile.py     # one-off torch.compile vs eager experiment
```

Component baselines (single B200, torch 2.11+cu128, bf16):

| helper | ms |
|---|---|
| topk_softmax_input_update | 0.48 |
| additive_logits_update | 0.03 |
| gather_active | 0.19 |
| scatter_back | 0.12 |
| MLPIterDecider.forward | 0.86 |
| NextTokenPredLoss.final | 0.23 |
| IterDeciderLoss.intra | 0.55 |
| **TaHForCausalLM.forward** (TaH-plus-1.7B, T=15) | **18.0** |
| **TaHForCasualLM_generate(32)** | **691** (~21.6 ms / token) |

## Future Work

- [ ] Optimize iteration decision strategies
- [ ] Integrate TaH with online distillation or RL
- [ ] Support training for larger models

## Related Projects

Explore more efficient LLM projects from us:

<table style="border: none; border-collapse: collapse;" align="center">
<tr>
<td align="center" valign="top" width="25%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/R2R">
<img src="https://raw.githubusercontent.com/thu-nics/R2R/main/resource/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/R2R"><b>R2R</b></a>
<br/><sub>Token-level routing for reasoning LLMs</sub>
</td>
<td align="center" valign="top" width="25%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/C2C">
<img src="https://raw.githubusercontent.com/thu-nics/C2C/main/resource/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/C2C"><b>C2C</b></a>
<br/><sub>Communicate through KV-Cache between LLMs</sub>
</td>
<td align="center" valign="top" width="25%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/FrameFusion">
<img src="https://raw.githubusercontent.com/thu-nics/FrameFusion/main/example/image/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/FrameFusion"><b>FrF</b></a>
<br/><sub>Efficient video token reduction for LVLMs</sub>
</td>
<td align="center" valign="top" width="25%" style="border: none; padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/MoA">
<img src="https://raw.githubusercontent.com/thu-nics/MoA/master/resource/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/MoA"><b>MoA</b></a>
<br/><sub>Mixture of sparse attention for LLMs</sub>
</td>
</tr>
</table>
