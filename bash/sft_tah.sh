#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-script2/recipes/qwen3_1.7/sft_tah.yaml}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
TP="${TP:-1}"
HSDP_REPLICATE="${HSDP_REPLICATE:-1}"
if [[ "${ENABLE_WANDB:-0}" != 1 ]]; then export WANDB_MODE=disabled; fi

# Check data-parallel topology and global batch size before launching.
python - "$CONFIG" "$NNODES" "$NPROC" "$TP" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
world = int(sys.argv[2]) * int(sys.argv[3])
tp = int(sys.argv[4])
if tp < 1 or world % tp:
    raise SystemExit("NNODES * NPROC must be divisible by TP.")
dp = world // tp
if config["data"]["dp"] != dp:
    raise SystemExit(f"Set data.dp to {dp} in {sys.argv[1]} for this launch.")
if config["training"]["dynamic_batch"]["global_batch_samples"] % dp:
    raise SystemExit("global_batch_samples must be divisible by data.dp.")
PY

if [[ "$NNODES" == 1 ]]; then
  RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
  LAUNCH=(--standalone --nproc_per_node "$NPROC")
else
  : "${MASTER_ADDR:?Set MASTER_ADDR to the address of node 0}"
  : "${RUN_TS:?Set the same RUN_TS on all nodes}"
  LAUNCH=(--nnodes "$NNODES" --nproc_per_node "$NPROC" --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-29500}")
fi
OUTPUT_DIR="${OUTPUT_DIR:-$(python script2/train/prepare_output_dir.py --config "$CONFIG" --run-ts "$RUN_TS")}"
mkdir -p "$OUTPUT_DIR"
python -m torch.distributed.run "${LAUNCH[@]}" script2/train/SFT_TaH.py \
  --config "$CONFIG" --tp "$TP" --hsdp_replicate "$HSDP_REPLICATE" \
  --output_dir "$OUTPUT_DIR" "$@" 2>&1 | tee "$OUTPUT_DIR/train-node${NODE_RANK}.log"
