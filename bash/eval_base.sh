#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python script/evaluation/eval.py \
    --eval_config ./script/recipes/qwen3_1.7/eval_base.yaml \
    --model_path nics-efc/Standard-1.7B \
    --dataset_name gsm8k \
    --backend hf \
    --job_nums 8 \
    --tp_size_per_job 1