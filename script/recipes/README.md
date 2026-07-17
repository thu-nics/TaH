# Recipes

YAML configs for training and evaluating TaH models. Pick the directory that
matches your base-model size, then point the relevant entrypoint at the YAML.

## Layout

```
script/recipes/
├── accelerate_configs/        # multi-GPU launch configs for `accelerate launch`
│   ├── zero2.yaml             # DeepSpeed ZeRO stage 2
│   └── zero3.yaml             # DeepSpeed ZeRO stage 3
├── qwen3_0.6/                 # recipes for Qwen3-0.6B-based TaH (hidden_size=1024)
│   ├── sft_tah_step1.yaml
│   ├── sft_tah_step2.yaml
│   └── eval_tah.yaml
├── qwen3_1.7/                 # recipes for Qwen3-1.7B-based TaH (hidden_size=2048)
│   ├── sft_tah_step1.yaml
│   ├── sft_tah_step2.yaml
│   └── eval_tah.yaml
└── qwen3_1.7_1gpu/            # 1-GPU variants of the qwen3_1.7 SFT recipes
    ├── sft_tah_step1.yaml
    └── sft_tah_step2.yaml
```

The `qwen3_1.7_1gpu/` recipes shrink `gradient_accumulation_steps` to 4 (vs
16 in the 8-GPU originals), drop `max_length` to 4096, set `report_to: none`,
and write checkpoints under `/tmp/tah_run/` so a 3-stage reproduction fits
on a single B200. Use them with plain `python script/train/SFT_TaH.py` (no
`accelerate launch` needed); the step-2 recipe expects you to fill in
`tah_model_path` with the step-1 `final_model` path before launching.

## What each file does

| File | Purpose |
|---|---|
| `accelerate_configs/zero2.yaml` | DeepSpeed ZeRO-2 launch config; lower memory savings, less comm overhead. |
| `accelerate_configs/zero3.yaml` | DeepSpeed ZeRO-3 launch config; max memory savings, more comm. Pick whichever fits your GPU/memory budget. |
| `qwen3_*/sft_tah_step1.yaml` | Step 1 SFT — `iter_decider: IterLabelDecider` (oracle hard-token labels) + `train_loss: NextTokenPredLoss`. Teaches the LoRA adapter on tokens marked "hard" by the labeller. |
| `qwen3_*/sft_tah_step2.yaml` | Step 2 SFT — loads the Step 1 checkpoint, switches to `iter_decider: MLPIterDecider` + `train_loss: IterDeciderLoss`. Trains the iter-decider so the model predicts its own hard tokens at inference. |
| `qwen3_*/eval_tah.yaml` | Eval config consumed by `script/evaluation/eval.py`; controls dataset list, generation params, max-new-tokens. |

## Model sizes supported

This release ships recipes for two base-model sizes — Qwen3-0.6B and
Qwen3-1.7B. The hidden size differs (1024 for 0.6B, 2048 for 1.7B), and the
`iter_decider_kwargs.hidden_states_size` field in each YAML is set
accordingly. If you adapt these recipes to a new base model, match its
hidden size.

## How they're used

- `script/train/SFT_TaH.py --config <path>` consumes the Step 1 / Step 2 YAMLs.
- `script/evaluation/eval.py --eval_config <path>` consumes the eval YAML.
- Accelerate configs are passed via `accelerate launch --config_file <path>`.

See the project README's *Train your own TaH model* and *Run evaluation*
sections for full command examples.
