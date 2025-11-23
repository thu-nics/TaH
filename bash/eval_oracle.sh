#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python script/evaluation/eval.py \
    --eval_config script/recipes/qwen3_1.7/eval_tah_oracle.yaml \
    --model_path nics-efc/TaH-plus-1.7B \
    --output_dir output/evaluation/ \
    --dataset_name math500 \
    --backend tah \
    --job_nums 4 \
    --tp_size_per_job 2 \
    --logger_level WARNING \
    --data_range 10