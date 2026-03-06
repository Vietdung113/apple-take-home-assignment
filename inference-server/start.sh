#!/usr/bin/env bash
# Start llama.cpp server with continuous batching on Apple Silicon.
#
# Usage:
#   ./start.sh                     # defaults from .env
#   MODEL_PATH=/path/to.gguf ./start.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

# Defaults
PORT="${PORT:-8100}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3-0.6B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3-0.6B-Q4_K_M.gguf}"
CTX_SIZE="${CTX_SIZE:-4096}"
PARALLEL="${PARALLEL:-4}"
GPU_LAYERS="${GPU_LAYERS:--1}"
THREADS="${THREADS:-$(sysctl -n hw.performancecores 2>/dev/null || echo 4)}"

# Download model if MODEL_PATH not set
if [ -z "$MODEL_PATH" ]; then
    echo "Resolving model from HuggingFace: $MODEL_REPO / $MODEL_FILE ..."
    MODEL_PATH=$(python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('$MODEL_REPO', '$MODEL_FILE'))
" 2>/dev/null || uv run python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('$MODEL_REPO', '$MODEL_FILE'))
")
fi

echo "============================================"
echo "  llama.cpp inference server"
echo "============================================"
echo "  Model:    $MODEL_PATH"
echo "  Port:     $PORT"
echo "  Context:  $CTX_SIZE tokens"
echo "  Parallel: $PARALLEL slots"
echo "  GPU:      $GPU_LAYERS layers"
echo "  Threads:  $THREADS"
echo "============================================"

exec llama-server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --n-gpu-layers "$GPU_LAYERS" \
    --threads "$THREADS" \
    --cont-batching
