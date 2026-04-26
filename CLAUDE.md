# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this directory.

## What this is

`tah-release` is the cleaned, single-target version of public TaH (forked from
[thu-nics/TaH](https://github.com/thu-nics/TaH)). Its only supported model is
the released [`nics-efc/TaH-plus-1.7B`](https://huggingface.co/nics-efc/TaH-plus-1.7B).
The user's *other* HR2R fork (a research branch with `Qwen3MLPIterDecider`,
`Qwen3MLPUpdater`, `CombinedLoss`, `iter_attention_mode='causal'`,
`weighted_hidden_method='stop_prob'`) is a separate codebase and is
**not** loadable here — those classes don't exist in this package.

## Architecture (two-sentence version)

`TaHForCausalLM` (`tah/model/tah_model.py`) wraps a HF causal LM (Qwen3) so
that, on each forward pass, every token first runs the base model
(iter_depth=0); tokens for which the iter decider votes "continue" then
re-run with LoRA enabled (iter_depth >= 1), accumulating their logits via a
residual additive update. Per-(layer, iteration) KV is stored in a single
`TaHCache` (`tah/model/causal_cache.py`) so future iterations can causally
see prior ones without disturbing iter-0.

## Key invariants

* **Single-impl interfaces are inlined.** input_updater / output_updater /
  iter_label_generator / adapter / iter_attention_mode all have exactly one
  implementation, called directly inside `TaHForCausalLM.forward`. Don't
  reintroduce a registry for these — the simplification was deliberate.

* **Multi-impl interfaces stay modular.** `iter_decider` (`IterLabelDecider`
  for step-1 SFT, `MLPIterDecider` for step-2 + eval) and `loss`
  (`NextTokenPredLoss`, `IterDeciderLoss`) keep their own modules with
  `_BY_NAME` dicts for dispatch. No registries.

* **The wrapper exposes a minimal contract.** Forward signature is
  `(input_ids, attention_mask?, position_ids?, past_key_values?, labels?,
  iter_count_labels?, use_cache?, new_sequence?)`. Public TaH had several
  more args (`iter_count`, `output_attentions`, `output_hidden_states`) that
  were either unused or unsupported; assertions or removals.

* **Persistence layout.** A saved TaH checkpoint must contain:
  - `tah_config.json` (config; type/dtype objects round-tripped via
    `type_to_dict_string` / `dict_string_to_type`)
  - `iter_decider.bin` (pickled `{class, init_args, state_dict}`)
  - `lora/` (PEFT adapter dir)
  - `model.safetensors` (base model with cleaned keys: no `.base_layer`
    PEFT prefix, no `lora_*` weights — those live in `lora/`)
  Downstream consumers (this repo's eval driver, `minisgl-tah` server's
  `TaHQwen3ForCausalLM`) all rely on this layout. **Don't change without
  also updating those consumers.**

## Common Commands

### Install
```bash
conda activate release
uv pip install -e ".[dev,training,evaluation]"
```

### Tests + benchmarks
```bash
pytest tests/                     # 21 component + wrapper + roundtrip tests
pytest tests/test_<name>.py -v    # one file
TAH_TEST_DEVICE=cpu pytest tests/ # run on CPU (skip needs no flag)
python tests/bench.py components  # per-helper microbench (B200 baseline in README)
python tests/bench.py e2e         # forward + generate on TaH-plus-1.7B
```
See `tests/README.md` for the snapshot/baseline harness details.

### Training (3-stage)
```bash
# Step 0
python script/preparation/label.py --num_gpu 8 \
  --dataset_path <jsonl> --test_model_list <hf-id> --output_path <out>

# Steps 1 + 2
python -m accelerate.commands.launch \
  --config_file ./script/recipes/accelerate_configs/zero2.yaml \
  --num_processes 8 \
  ./script/train/SFT_TaH.py \
  --config ./script/recipes/qwen3_1.7/sft_tah_step{1,2}.yaml
```

### Evaluation (3 backends)
```bash
python script/evaluation/eval.py \
  --eval_config ./script/recipes/qwen3_1.7/eval_tah.yaml \
  --model_path nics-efc/TaH-plus-1.7B \
  --dataset_name gsm8k --backend {tah,hf,sglang} \
  --job_nums 8 --tp_size_per_job 1
```

### Quick inference demo
```bash
python script/playground/inference_example.py
```

## Layout

```
tah/
├── __init__.py                # re-exports TaHForCausalLM, TaHConfig, TaHCache, …
├── model/
│   ├── tah_model.py           # TaHForCausalLM + inlined slot helpers
│   ├── iter_decider.py        # IterLabelDecider, MLPIterDecider, ITER_DECIDER_BY_NAME
│   ├── loss.py                # NextTokenPredLoss, IterDeciderLoss, LOSS_BY_NAME
│   ├── causal_cache.py        # TaHCache: per-(layer, iter) KV with up-to-iter views
│   ├── tah_config.py          # @dataclass TaHConfig
│   └── utils.py               # generation helper + IterCountColors + freeze/seed/sampling helpers
├── train/                     # HF Trainer subclass + collator + iter-aware callback
├── evaluate/
│   ├── datasets.py            # benchmark loading + standardisation
│   ├── backends.py            # sglang / hf / tah model + inference fn
│   ├── jobs.py                # job-sharded runner + result aggregation
│   ├── matheval.py            # math benchmark graders (math_verify)
│   └── codeeval.py            # humaneval / mbpp via evalplus
└── utils/data_prepare.py      # SFT preprocessing
script/
├── preparation/               # download.py, label.py, prune.py, filter_split.py
├── train/SFT_TaH.py           # YAML → wrapper → HF Trainer.train()
├── evaluation/eval.py         # CLI wrapper over tah.evaluate.allocate_gpus_and_run_jobs
├── playground/inference_example.py
└── recipes/                   # qwen3_{0.6,1.7}/sft_tah_step{1,2}.yaml + eval_tah.yaml
tests/                         # _harness.py + per-component test_*.py + baselines/ (gitignored)
```

## Conventions

- All `tah.model.*` modules are designed to be importable in isolation with
  small synthetic shapes — that's what `tests/conftest.py` exercises.
- `tah/model/tah_model.py` is the only place that mutates the wrapper's
  internal state. New iter-loop behaviour goes there, not into the
  iter_decider or loss classes.
- `iter_decider_kwargs.dtype` may be a `torch.dtype` — round-trip through the
  ``_config_to_serialisable`` / ``_config_from_serialisable`` helpers in
  `tah/model/tah_model.py` (called from save_pretrained / from_pretrained).
- The wrapper assumes the base model's hidden size matches
  `iter_decider_kwargs.hidden_states_size`. Recipes set this to 1024 for
  Qwen3-0.6B and 2048 for Qwen3-1.7B; mismatches show up as a Linear
  shape error.
