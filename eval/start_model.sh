#!/bin/bash
# Start llama.cpp server with a GGUF model
#
# Usage:
#   ./start_model.sh                                 # default: finetuned model, port 8080
#   ./start_model.sh <model_path>                    # custom model, port 8080
#   ./start_model.sh <model_path> --port 8100        # custom model + port
#
# Examples:
#   ./start_model.sh                                                        # finetuned (default)
#   ./start_model.sh ../models/Qwen3-0.6B-Q4_K_M.gguf                      # base model
#   ./start_model.sh ../models/finetuned/finetuned-qwen3-0.6b.Q4_K_M.gguf  # explicit finetuned

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_MODEL="$SCRIPT_DIR/../models/finetuned/finetuned-qwen3-0.6b.Q4_K_M.gguf"

if [ -n "$1" ] && [[ "$1" != --* ]]; then
  MODEL_PATH="$1"
  shift
else
  MODEL_PATH="$DEFAULT_MODEL"
fi

# Parse optional flags (override env vars)
while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    --ctx-size) CTX_SIZE="$2"; shift 2 ;;
    --n-gpu-layers) N_GPU_LAYERS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-32768}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model file not found: $MODEL_PATH"
  exit 1
fi

echo ""
echo "========================================================================"
echo "Starting llama.cpp server"
echo "========================================================================"
echo "Model:      $MODEL_PATH"
echo "Port:       $PORT"
echo "Context:    $CTX_SIZE"
echo "GPU layers: $N_GPU_LAYERS"
echo "========================================================================"
echo ""

llama-server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --threads 8 \
  --parallel 1 \
  --cont-batching \
  --verbose
