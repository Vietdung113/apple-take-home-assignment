#!/usr/bin/env bash
# setup_servers.sh — Start 3 llama.cpp servers for evaluation on vast.ai
#
# Servers:
#   Port 8100: Base model GGUF (Qwen3-0.6B)
#   Port 8200: Fine-tuned model GGUF (Qwen3-0.6B + merged adapter)
#   Port 8001: Judge model GGUF (Qwen2.5-32B-Instruct Q4_K_M)
#
# Usage:
#   bash setup_servers.sh                # start all 3 servers
#   bash setup_servers.sh --stop         # stop all servers
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"

# Model paths (update these after downloading GGUFs)
BASE_GGUF="${BASE_GGUF:-$MODELS_DIR/Qwen3-0.6B-Q4_K_M.gguf}"
FINETUNED_GGUF="${FINETUNED_GGUF:-$MODELS_DIR/Qwen3-0.6B-sft-8k-Q4_K_M.gguf}"
JUDGE_GGUF="${JUDGE_GGUF:-$MODELS_DIR/Qwen2.5-32B-Instruct-Q4_K_M.gguf}"

# Ports
BASE_PORT=8100
FINETUNED_PORT=8200
JUDGE_PORT=8001

# Server configs
BASE_CTX=8192
BASE_THREADS=8
BASE_GPU_LAYERS=35

FINETUNED_CTX=8192
FINETUNED_THREADS=8
FINETUNED_GPU_LAYERS=35

JUDGE_CTX=32768
JUDGE_THREADS=8
JUDGE_GPU_LAYERS=60

# ── Stop servers ─────────────────────────────────────────────────────────

if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping all llama.cpp servers ..."
    pkill -f "llama-server.*--port $BASE_PORT" || true
    pkill -f "llama-server.*--port $FINETUNED_PORT" || true
    pkill -f "llama-server.*--port $JUDGE_PORT" || true
    echo "Stopped."
    exit 0
fi

# ── Validate GGUFs ───────────────────────────────────────────────────────

for gguf in "$BASE_GGUF" "$FINETUNED_GGUF" "$JUDGE_GGUF"; do
    if [[ ! -f "$gguf" ]]; then
        echo "ERROR: GGUF not found: $gguf"
        echo ""
        echo "Download GGUFs first:"
        echo "  bash download_models.sh"
        exit 1
    fi
done

# ── Check llama-server ───────────────────────────────────────────────────

if ! command -v llama-server &>/dev/null; then
    echo "ERROR: llama-server not found."
    echo "Install llama.cpp first:"
    echo "  git clone https://github.com/ggerganov/llama.cpp"
    echo "  cd llama.cpp && make LLAMA_CUDA=1"
    echo "  sudo cp llama-server /usr/local/bin/"
    exit 1
fi

# ── Start servers ────────────────────────────────────────────────────────

echo "Starting llama.cpp servers ..."
echo ""

# Base model (port 8100)
echo "[1/3] Base model: $BASE_GGUF → :$BASE_PORT"
nohup llama-server \
    --model "$BASE_GGUF" \
    --host 0.0.0.0 \
    --port $BASE_PORT \
    --ctx-size $BASE_CTX \
    --threads $BASE_THREADS \
    --n-gpu-layers $BASE_GPU_LAYERS \
    --flash-attn \
    > /tmp/llama_base.log 2>&1 &
echo "  PID: $! | Log: /tmp/llama_base.log"

sleep 2

# Fine-tuned model (port 8200)
echo "[2/3] Fine-tuned model: $FINETUNED_GGUF → :$FINETUNED_PORT"
nohup llama-server \
    --model "$FINETUNED_GGUF" \
    --host 0.0.0.0 \
    --port $FINETUNED_PORT \
    --ctx-size $FINETUNED_CTX \
    --threads $FINETUNED_THREADS \
    --n-gpu-layers $FINETUNED_GPU_LAYERS \
    --flash-attn \
    > /tmp/llama_finetuned.log 2>&1 &
echo "  PID: $! | Log: /tmp/llama_finetuned.log"

sleep 2

# Judge model (port 8001)
echo "[3/3] Judge model: $JUDGE_GGUF → :$JUDGE_PORT"
nohup llama-server \
    --model "$JUDGE_GGUF" \
    --host 0.0.0.0 \
    --port $JUDGE_PORT \
    --ctx-size $JUDGE_CTX \
    --threads $JUDGE_THREADS \
    --n-gpu-layers $JUDGE_GPU_LAYERS \
    --flash-attn \
    > /tmp/llama_judge.log 2>&1 &
echo "  PID: $! | Log: /tmp/llama_judge.log"

echo ""
echo "Waiting for servers to start (30s) ..."
sleep 30

echo ""
echo "Testing servers ..."
for port in $BASE_PORT $FINETUNED_PORT $JUDGE_PORT; do
    if curl -s http://localhost:$port/health >/dev/null 2>&1; then
        echo "  :$port ✓"
    else
        echo "  :$port ✗ (check log: /tmp/llama_*.log)"
    fi
done

echo ""
echo "Ready! Monitor logs:"
echo "  tail -f /tmp/llama_base.log"
echo "  tail -f /tmp/llama_finetuned.log"
echo "  tail -f /tmp/llama_judge.log"
echo ""
echo "To stop:"
echo "  bash setup_servers.sh --stop"
