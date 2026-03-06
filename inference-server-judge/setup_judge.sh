#!/usr/bin/env bash
# setup_judge.sh — Deploy Qwen2.5-32B-Instruct as judge via vLLM on vast.ai
#
# Usage:
#   bash setup_judge.sh                    # install + start server
#   bash setup_judge.sh --port 8001        # custom port
#   bash setup_judge.sh --stop             # stop server
set -euo pipefail

PORT="${PORT:-8001}"
MODEL="Qwen/Qwen2.5-32B-Instruct"
TENSOR_PARALLEL=1
GPU_MEM=0.9
MAX_LEN=32768

# ── Parse args ───────────────────────────────────────────────────────────
STOP=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --stop)
            STOP=true
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

# ── Stop server ──────────────────────────────────────────────────────────
if $STOP; then
    echo "Stopping vLLM judge server ..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    echo "Stopped."
    exit 0
fi

# ── Install vLLM ─────────────────────────────────────────────────────────
if ! command -v vllm &>/dev/null; then
    echo "Installing vLLM ..."
    pip install -q vllm
fi

# ── Start vLLM server ────────────────────────────────────────────────────
echo "Starting vLLM judge server ..."
echo "  Model:  $MODEL"
echo "  Port:   $PORT"
echo "  URL:    http://0.0.0.0:$PORT/v1"

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    > /tmp/vllm_judge.log 2>&1 &

echo "  PID:    $!"
echo "  Log:    /tmp/vllm_judge.log"
echo ""
echo "Waiting for server to start (may take 2-3 minutes to load model) ..."
echo "Monitor: tail -f /tmp/vllm_judge.log"
echo ""
echo "Test readiness:"
echo "  curl http://localhost:$PORT/v1/models"
echo ""
echo "To stop:"
echo "  bash setup_judge.sh --stop"
