#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python script/evaluation/eval.py \
    --eval_config ./script/recipes/qwen3_1.7/eval_tah.yaml \
    --model_path nics-efc/TaH-plus-1.7B \
    --dataset_name gsm8k \
    --backend tah \
    --job_nums 8 \
    --tp_size_per_job 1