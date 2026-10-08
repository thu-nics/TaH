#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${MODEL_PATH:?Set MODEL_PATH to a saved checkpoint}"
python script2/eval/eval_offline.py \
  --model_path "$MODEL_PATH" \
  --dtype bfloat16 --batch_size 1 --repeat_size 1 \
  --temperature 0.6 --top_p 0.95 --top_k 20 --max_new_tokens 16384 \
  --use_tracker \
  --backend "${BACKEND:-hf}" --job_nums "${JOBS:-1}" --tp_size_per_job 1 \
  --datasets math500 --thresholds 0.5 "$@"
