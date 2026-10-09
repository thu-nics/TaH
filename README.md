<div align="center">

  <p>
    <img src="resource/logos.svg" alt="TaH2 and TaH logos" width="230" height="100"/>
  </p>

  <h1>Think-at-Hard</h1>

  <h4>Adaptive Looped Large Language Models</h4>

</div>

<h3 align="center">
  TaH2 &nbsp;
  <a href="https://fuvty.github.io/thinking_yard_project_page/projects/tah2/"><strong>🌐 Project</strong></a> ·
  <a href="https://arxiv.org/abs/2609.35748"><strong>📑 Paper</strong></a> ·
  <a href="https://huggingface.co/collections/nics-efc/tah2"><strong>🤗 HuggingFace</strong></a>
</h3>

TaH2 improves test-time scaling by allocating extra latent iterations to tokens that benefit from deeper computation. It jointly trains the backbone and an iteration decider with lookahead depth supervision, using online labels that indicate whether another iteration improves prediction. On challenging AIME benchmarks, TaH2 improves the accuracy-compute slope by 53% over the non-looped baseline and raises peak accuracy by about 3.4 points at matched test-time compute.

```bibtex
@article{you2026tah2,
    title={Improving Test-Time Scaling with Adaptive Looped Transformers},
    author={You, Yichen and Fu, Tianyu and Feng, Aosong and Lv, Xingtai and Ning, Xuefei and Ding, Ning and Wang, Yu},
    journal={arXiv preprint arXiv:2609.35748},
    year={2026},
}
```

<h3 align="center">
  TaH &nbsp;
  <a href="https://fuvty.github.io/thinking_yard_project_page/projects/tah/"><strong>🌐 Project</strong></a> ·
  <a href="https://arxiv.org/abs/2511.08577"><strong>📑 Paper</strong></a> ·
  <a href="https://huggingface.co/collections/nics-efc/tah"><strong>🤗 HuggingFace</strong></a>
</h3>

Think-at-Hard (TaH) improves LLM reasoning by running extra latent iterations only on hard tokens instead of all tokens. A lightweight decider and duo-causal attention enable targeted refinement while keeping full parallelism. TaH outperforms fixed two-iteration baselines by 8–11% while skipping 94% of second iterations, and also beats strong single-iteration Qwen3 models by 4–5%.

```bibtex
@article{fu2025tah,
    title={Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models}, 
    author={Tianyu Fu and Yichen You and Zekai Chen and Guohao Dai and Huazhong Yang and Yu Wang},
    journal={arXiv preprint arXiv:2511.08577},
    year={2025},
}
```

## News

* [2026/10] We released the TaH2 [code](https://github.com/thu-nics/TaH/tree/main), [models](https://huggingface.co/collections/nics-efc/tah2), and [training data](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool).

* [2026/09] We introduced TaH2 in [Improving Test-Time Scaling with Adaptive Looped Transformers](https://arxiv.org/abs/2609.35748).

* [2025/11] We released the [TaH-plus-1.7B](https://huggingface.co/nics-efc/TaH-plus-1.7B) checkpoint. The model is finetuned from [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) using 100K samples from the [OpenR1](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) dataset, capable of QA, math, and coding. 

* [2025/11] Our paper was featured as the #2 Paper of the Day on [Huggingface Daily Papers](https://huggingface.co/papers/date/2025-11-19)

TaH and TaH2 require different dependency versions. Activate their separate environments before running the corresponding scripts.

## TaH2 Usage

This repository includes training recipes, evaluation tools, and an inference engine adapted from [mini-SGLang](https://github.com/sgl-project/mini-sglang), integrated under `tah2/minisgl/`.

### Environment Setup

Use Linux, Python 3.12, and CUDA GPUs. TaH2 uses `tah2/`, `script/tah2/`, and `bash/`. From the repository root:

```bash
python3.12 -m venv .venv-tah2
source .venv-tah2/bin/activate
pip install -e '.[tah2]'
```

### Run an Example

Download a checkpoint:

```bash
hf download nics-efc/TaH2-1.7B-max2 --local-dir models/TaH2-1.7B-max2
```

Generate with the bundled inference engine:

```python
from transformers import AutoTokenizer
from tah2.minisgl.core import SamplingParams
from tah2.minisgl.llm import LLM

model_path = "models/TaH2-1.7B-max2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 1 + 1?"}],
    tokenize=False, add_generation_prompt=True,
)
llm = LLM(model_path=model_path)
try:
    result = llm.generate([prompt], SamplingParams(max_tokens=1024, temperature=0.6))
    print(result[0]["text"])
finally:
    llm.shutdown()
```

| Checkpoint | Model |
| --- | --- |
| [TaH2-1.7B-max2](https://huggingface.co/nics-efc/TaH2-1.7B-max2) | Adaptive iteration, maximum depth 2 |
| [TaH2-1.7B-Standard](https://huggingface.co/nics-efc/TaH2-1.7B-Standard) | Single-iteration baseline; also loads with Transformers |

### Run Evaluation

Start a server, then run evaluation in another terminal:

```bash
MODEL_PATH=models/TaH2-1.7B-max2 bash bash/launch_server.sh \
  --tah_iter_threshold 0.5
```

```bash
MODEL_PATH=models/TaH2-1.7B-max2 bash bash/eval_online.sh \
  --datasets math500 amc23 olympiadbench aime25 aime26
```

The server defaults to GPU 0 and `http://127.0.0.1:30080`. Set `GPU`, `SERVER_HOST`, `PORT`, and `DISTRIBUTED_PORT` to change its placement; set `BASE_URL` for the evaluation client. The server reads the iteration limit from the checkpoint; `--tah_max_iter` overrides it. Set the client's `--tah_iter_threshold` to the server threshold when labeling evaluation outputs. The [engine README](tah2/minisgl/README.md) also describes its Python API and OpenAI-compatible endpoints.

### Train Your Own TaH2 Model

<details>
<summary><strong>Step 0: Prepare Data and Base Model</strong></summary>

Download the prepared data from [nics-efc/TaH2-amteam-tool](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool):

```bash
hf download nics-efc/TaH2-amteam-tool --repo-type dataset --local-dir data
```

The math, code, science, and tool-calling mixture provides `train/` and `eval/` splits for each model.

| Student | Directory | Train samples | Train tokens | Eval samples |
| --- | --- | ---: | ---: | ---: |
| Qwen3-1.7B | `data/1.7b/` | 273,195 | 1,099,413,406 | 955 |
| Qwen3-4B | `data/4b/` | 638,817 | 2,586,857,725 | 1,000 |
| Qwen3-8B | `data/8b/` | 1,282,124 | 5,192,783,145 | 1,000 |

Load with `datasets.load_from_disk("data/1.7b/train")`; `mask=1` marks assistant tokens for training. Recipes load `Qwen/Qwen3-{1.7B,4B,8B}-Base` automatically, or use a local path in `model.name`.

**Data Sources and Regeneration**

| Original dataset | Files used |
| --- | --- |
| [a-m-team/AM-Qwen3-Distilled](https://huggingface.co/datasets/a-m-team/AM-Qwen3-Distilled) | `math.jsonl`, `code.jsonl`, `science.jsonl` |
| [nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | `data/tool_calling.jsonl` |

For 1.7B, **Qwen3-8B** regenerates assistant responses; tool results use reference replay or **Qwen3-32B** simulation. Key code: [generation and tokenization](script/tah2/data/regenerate.py), [tool rollout](script/tah2/data/tool_rollout.py). Input prompts are available at [`1.7b-prompts/`](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool/tree/main/1.7b-prompts).

Serve Qwen3-8B at port 30080 and Qwen3-32B at port 30081, then run:

```bash
python script/tah2/data/regenerate.py generate --kind am \
  --prompts data/1.7b-prompts/am/train.jsonl \
  --urls http://127.0.0.1:30080 --output data/regenerated/am_train.jsonl

python script/tah2/data/regenerate.py generate --kind tool_calling \
  --prompts data/1.7b-prompts/tool_calling/train.jsonl \
  --urls http://127.0.0.1:30080 --sim-urls http://127.0.0.1:30081 \
  --output data/regenerated/tool_train.jsonl

python script/tah2/data/regenerate.py build \
  --inputs data/regenerated/am_train.jsonl data/regenerated/tool_train.jsonl \
  --output data/regenerated/train
```

Prepared data: [`1.7b/`](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool/tree/main/1.7b), [`4b/`](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool/tree/main/4b), [`8b/`](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool/tree/main/8b). The 4B/8B mixtures retain the original teacher responses: 8B uses the full pool; 4B uses a source-stratified subset with the same eval split.

To regenerate both splits, build eval first, then pass `--exclude data/regenerated/eval` when building train to remove duplicate eval samples.

</details>

#### Step 1: Train

| Recipe directory | Standard | Fixed loop-2 | TaH2 |
| --- | --- | --- | --- |
| [`qwen3_1.7/`](script/tah2/recipes/qwen3_1.7/) | `sft_base.yaml` | `sft_fixed.yaml` | `sft_tah.yaml`, `sft_tah_max4.yaml`, `sft_tah_max8.yaml` |
| [`qwen3_4b/`](script/tah2/recipes/qwen3_4b/) | `sft_base.yaml` | — | `sft_tah.yaml` |
| [`qwen3_8b/`](script/tah2/recipes/qwen3_8b/) | `sft_base.yaml` | — | `sft_tah.yaml` |

TaH2 jointly optimizes the backbone, input updater, and decider. Posterior labels are generated online from next-token cross-entropy improvements; separate offline token labeling is unnecessary. The main recipes use DUO attention, Triton kernels, and `stop_prob_mix`. The fixed loop-2 recipe uses `even_mix` without decider supervision.

The 1.7B TaH2 recipes use global batch size 128, three epochs, learning rate `4e-5`, and a 16,384-token packing budget.

For one node with eight GPUs, run:

```bash
NPROC=8 TP=1 CONFIG=script/tah2/recipes/qwen3_1.7/sft_tah.yaml bash bash/sft_tah.sh
```

Change `CONFIG` to select a recipe and adjust `TP` to fit the actual GPU memory usage. Update `data.dp` in the recipe accordingly (`data.dp = NPROC / TP` for one node).

Suggested starting points for one node with 8 H200 GPUs, BF16, and gradient checkpointing:

| Model | `max_iter` | `max_length` | Suggested `TP` | GPU (memory per GPU) |
| --- | ---: | --- | ---: | --- |
| Qwen3-1.7B | 1 | 16K | 1 | H200 (141GB) |
| Qwen3-1.7B | 2 | 16K | 1 | H200 (141GB) |
| Qwen3-1.7B | 4 | 16K | 1 | H200 (141GB) |
| Qwen3-1.7B | 8 | 16K | 2 | H200 (141GB) |
| Qwen3-4B | 1 | 16K | 2 | H200 (141GB) |
| Qwen3-4B | 2 | 16K | 2 | H200 (141GB) |
| Qwen3-8B | 1 | 16K | 2 | H200 (141GB) |
| Qwen3-8B | 2 | 16K | 2 | H200 (141GB) |

Checkpoints include model weights, tokenizer, and the recurrent components when enabled. Recipes use `save_only_model: true`; set it to `false` before training to save optimizer state for continuation.

## TaH Usage

### Environment Setup
Use Python 3.10 and activate a separate environment for `tah/` and `script/tah/`:

```bash
python3.10 -m venv .venv-tah
source .venv-tah/bin/activate
pip install -e '.[tah]'
```

For training and evaluation, install additional dependencies:

```bash
pip install -e '.[tah,training,evaluation]'
```

For code generation evaluation, install [evalplus](https://github.com/evalplus/evalplus)

> **Note** if you ``git pull`` and the top-level package layout changes
> (e.g. ``__init__.py`` is added or removed), re-run ``pip install -e '.[tah]'``
> — the editable install caches the layout in
> ``site-packages/__editable___tah_*_finder.py`` and stale state will
> silently drop ``tah/__init__.py``'s re-exports.

### Run an example for TaH

```bash
python script/tah/playground/inference_example.py                       # quick demo (~1 min)
python script/tah/playground/inference_example.py --max-new-tokens 16384 # full reasoning chain
```

This script demonstrates TaH's selective latent iteration mechanism, with color-coded output showing the iteration count for each token.

### Run evaluation


#### Evaluate TaH model
```bash
python script/tah/evaluation/eval.py \
    --eval_config ./script/tah/recipes/qwen3_1.7/eval_tah.yaml \
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
- `--job_nums`: Number of parallel jobs (one job pins `tp_size_per_job` GPUs)
- `--tp_size_per_job`: Tensor parallel size per job
- `--data_range N` / `--data_range start end`: subset slice — handy for smoke tests
- `--data_ids gsm8k_0,gsm8k_5`: run only specific problem ids

##### Single-GPU smoke
The default recipe targets 8 GPUs (`--job_nums 8`). To sanity-check the pipeline on
one GPU in a couple of minutes, slice the dataset and shrink `max_new_tokens`:
```bash
# clone the recipe and shrink generation length
sed 's/max_new_tokens: 4096/max_new_tokens: 512/' \
    script/tah/recipes/qwen3_1.7/eval_tah.yaml > /tmp/eval_tah_smoke.yaml

CUDA_VISIBLE_DEVICES=0 python script/tah/evaluation/eval.py \
    --eval_config /tmp/eval_tah_smoke.yaml \
    --model_path nics-efc/TaH-plus-1.7B \
    --dataset_name gsm8k --backend tah \
    --job_nums 1 --tp_size_per_job 1 \
    --data_range 5 \
    --output_dir /tmp/tah_eval_smoke
```
The TaH backend is a token-by-token Python loop intended for research; for serving
throughput, use `--backend sglang` or the dedicated `minisgl-tah` server.

#### Evaluate with a different backend

The same `script/tah/evaluation/eval.py` accepts `--backend hf` (vanilla
`AutoModelForCausalLM.generate` — useful for non-TaH baselines) or
`--backend sglang` (sgl Engine for high-throughput serving). All three
backends share the same job-sharded driver under
`tah/evaluate/jobs.py:allocate_gpus_and_run_jobs`.

### Train your own TaH model

Training a TaH model consists of three stages:

#### Step0: Prepare model and data

**1. Prepare training data**

Use a reference model to generate hard token labels for the training and validation data:

```bash
# download the default subset of OpenR1-Math-220k
python script/tah/preparation/download.py
# filter and split
python script/tah/preparation/filter_split.py
# label the hard tokens
python script/tah/preparation/label.py \
    --num_gpu 8 \
    --dataset_path ./data/initial_data/openr1-math/train.jsonl \
    --test_model_list Qwen/Qwen3-1.7B \
    --output_path ./data/processed_data/openr1-math/1_7/train \
    --max_input_length 10000
python script/tah/preparation/label.py \
    --num_gpu 8 \
    --dataset_path ./data/initial_data/openr1-math/eval.jsonl \
    --test_model_list Qwen/Qwen3-1.7B \
    --output_path ./data/processed_data/openr1-math/1_7/eval \
    --max_input_length 10000 \
```

**2. (Optional) Prepare pruned model**

For the TaH version, prune one layer from the base model to match the parameter count of the standard baseline (skip this step for TaH+ version):

```bash
python script/tah/preparation/prune.py \
    --model Qwen/Qwen3-1.7B-Base \
    --dataset ./data/processed_data/openr1-math/1_7/eval \
    --output ./model/qwen3_1.7_base_pruned \
    --num_prune 1
```

#### Step1: Train with Fixed Iteration Labels

The first stage uses fixed iteration labels for training:

```bash
python -m accelerate.commands.launch \
    --config_file ./script/tah/recipes/accelerate_configs/zero2.yaml \
    --num_processes 8 \
    ./script/tah/train/SFT_TaH.py \
    --config ./script/tah/recipes/qwen3_1.7/sft_tah_step1.yaml
```

Key configurations in Step1 (`sft_tah_step1.yaml`):
- `max_iter: 2` — maximum number of iterations.
- `iter_decider: "IterLabelDecider"` — continue iff the per-token oracle
  ``iter_count_labels`` (derived from ``mismatch``) say so. Used to teach
  the LoRA adapter on tokens marked "hard" by the labeller.
- `adapter: "lora"` — only LoRA is supported in tah-release.
- `train_loss: "NextTokenPredLoss"` — standard causal-LM cross-entropy.

Single-implementation hooks (input/output updaters, iter labels, adapter) are inlined into the wrapper — only `iter_decider` and `train_loss` are config-selectable.

#### Step2: Train Iteration Decider

The second stage trains the iteration decider:


```bash
python -m accelerate.commands.launch \
    --config_file ./script/tah/recipes/accelerate_configs/zero2.yaml \
    --num_processes 8 \
    ./script/tah/train/SFT_TaH.py \
    --config ./script/tah/recipes/qwen3_1.7/sft_tah_step2.yaml
```

Key configurations in Step2 (`sft_tah_step2.yaml`):
- `tah_model_path`: Load the model trained in Step1
- `iter_decider: "MLPIterDecider"`: Use MLP decider to automatically determine iterations
- `train_loss: "IterDeciderLoss"`: Iteration decider loss function
- `freeze_component: [model.simple_base_model]`: Freeze model backbone

After two-stage training, the model can automatically decide when to perform latent reasoning iterations.

## Understand the Code

```text
TaH/
├── tah/                # TaH model, LoRA training, and evaluation
├── tah2/
│   ├── model/          # recurrent model, decider, posterior labels, losses
│   ├── kernels/        # Triton recurrent attention
│   ├── train/          # FSDP2/TP training and checkpoint saving
│   ├── evaluate/       # inference backends and benchmark grading
│   ├── minisgl/        # bundled mini-SGLang engine and native kernels
│   └── utils/          # data preparation and serialization
├── script/
│   ├── tah/            # TaH preparation, training, evaluation, and recipes
│   └── tah2/           # TaH2 data, training, evaluation, and recipes
├── bash/               # TaH2 training, evaluation, and server launchers
└── pyproject.toml      # separate tah/tah2 dependency selections
```

## License

TaH and TaH2 are released under [Apache-2.0](LICENSE). The bundled
[mini-SGLang](https://github.com/sgl-project/mini-sglang) engine retains its
[MIT license](tah2/minisgl/LICENSE) and the bundled NCCL header retains its
[NVIDIA license](tah2/minisgl/kernel/csrc/include/minisgl/LICENSE.txt).

## Related Projects

Explore more efficient LLM projects from us:

<table style="border: none; border-collapse: collapse;" align="center">
<tr>
<td align="center" valign="top" width="20%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/R2R">
<img src="https://raw.githubusercontent.com/thu-nics/R2R/main/resource/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/R2R"><b>R2R</b></a>
<br/><sub>Token-level routing for reasoning LLMs</sub>
</td>
<td align="center" valign="top" width="20%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/TokenRouter">
<img src="https://raw.githubusercontent.com/thu-nics/TokenRouter/main/resource/logo.png" alt="TokenRouter Logo" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/TokenRouter"><b>TkR</b></a>
<br/><sub>Efficient serving for token-level LLM routing</sub>
</td>
<td align="center" valign="top" width="20%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/C2C">
<img src="https://raw.githubusercontent.com/thu-nics/C2C/main/resource/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/C2C"><b>C2C</b></a>
<br/><sub>Communicate through KV-Cache between LLMs</sub>
</td>
<td align="center" valign="top" width="20%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/FrameFusion">
<img src="https://raw.githubusercontent.com/thu-nics/FrameFusion/main/example/image/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/FrameFusion"><b>FrF</b></a>
<br/><sub>Efficient video token reduction for LVLMs</sub>
</td>
<td align="center" valign="top" width="20%" style="border: none; padding: 10px; min-width: 50px;">
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
