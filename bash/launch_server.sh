#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${MODEL_PATH:?Set MODEL_PATH to a saved checkpoint}"
exec python script/tah2/eval/launch_minisgl_server.py \
  --model_path "$MODEL_PATH" --host "${SERVER_HOST:-127.0.0.1}" \
  --ports "${PORT:-30080}" --base_gpu "${GPU:-0}" \
  --distributed_port_base "${DISTRIBUTED_PORT:-31080}" \
  --dtype bfloat16 --wait "$@"
