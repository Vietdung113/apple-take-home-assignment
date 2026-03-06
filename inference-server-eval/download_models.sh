#!/usr/bin/env bash
# download_models.sh — Download GGUF models for evaluation
#
# Downloads:
#   1. Qwen3-0.6B-Q4_K_M.gguf (base model)
#   2. Qwen2.5-32B-Instruct-Q4_K_M.gguf (judge)
#
# Fine-tuned GGUF must be exported separately via train_sft.py --export-gguf
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
mkdir -p "$MODELS_DIR"

echo "Downloading GGUFs to $MODELS_DIR ..."
echo ""

# ── Base model: Qwen3-0.6B ───────────────────────────────────────────────

BASE_URL="https://huggingface.co/unsloth/Qwen3-0.6B-gguf/resolve/main"
BASE_FILE="Qwen3-0.6B-Q4_K_M.gguf"

if [[ -f "$MODELS_DIR/$BASE_FILE" ]]; then
    echo "[1/2] Base model: $BASE_FILE (already exists)"
else
    echo "[1/2] Downloading base model: $BASE_FILE ..."
    wget -q --show-progress -O "$MODELS_DIR/$BASE_FILE" "$BASE_URL/$BASE_FILE"
    echo "  → $MODELS_DIR/$BASE_FILE"
fi

# ── Judge model: Qwen2.5-32B-Instruct ────────────────────────────────────

JUDGE_URL="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main"
JUDGE_FILE="Qwen2.5-32B-Instruct-Q4_K_M.gguf"

if [[ -f "$MODELS_DIR/$JUDGE_FILE" ]]; then
    echo "[2/2] Judge model: $JUDGE_FILE (already exists)"
else
    echo "[2/2] Downloading judge model: $JUDGE_FILE (15GB, may take 10-20 min) ..."
    wget -q --show-progress -O "$MODELS_DIR/$JUDGE_FILE" "$JUDGE_URL/$JUDGE_FILE"
    echo "  → $MODELS_DIR/$JUDGE_FILE"
fi

echo ""
echo "Downloads complete!"
echo ""
echo "Next steps:"
echo "1. Export fine-tuned GGUF:"
echo "   cd finetuning"
echo "   bash setup.sh sft 8k --export-gguf"
echo "   cp output/sft_8k/gguf/*.gguf $MODELS_DIR/Qwen3-0.6B-sft-8k-Q4_K_M.gguf"
echo ""
echo "2. Start servers:"
echo "   cd inference-server-eval"
echo "   bash setup_servers.sh"
