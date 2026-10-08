#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${MODEL_PATH:?Set MODEL_PATH to a saved checkpoint}"
python script/tah2/eval/eval_online.py \
  --model_path "$MODEL_PATH" --base_urls "${BASE_URL:-http://127.0.0.1:30080}" \
  --datasets math500 --temperature 0.6 --top_p 0.95 --top_k 20 \
  --max_new_tokens 16384 --repeat_size 1 --tah_iter_threshold 0.5 \
  --wait_for_server "$@"
