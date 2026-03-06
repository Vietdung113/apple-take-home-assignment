#!/usr/bin/env bash
# setup.sh — Bootstrap a vast.ai instance for SFT/GRPO fine-tuning.
#
# SSH into the instance then run:
#   bash setup.sh                                         # install deps only
#   bash setup.sh sft 8k                                  # install + train SFT 8K
#   bash setup.sh sft 16k                                 # install + train SFT 16K
#   bash setup.sh sft 32k                                 # install + train SFT 32K
#   bash setup.sh sft 8k --max-samples 50 --export-gguf   # smoke test + GGUF
#   bash setup.sh grpo                                    # install + train GRPO
#   bash setup.sh grpo --max-samples 50 --export-gguf     # GRPO smoke test
set -euo pipefail

REPO_URL="${REPO_URL:-}"          # set before running, or clone manually
REPO_DIR="/workspace/finetuning"
WANDB_KEY="${WANDB_API_KEY:-}"

# ── Colors ───────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
err()   { echo -e "\033[1;31m[ERR]\033[0m $*"; exit 1; }

# ── 1. Install dependencies ─────────────────────────────────────────────
info "Installing Python dependencies ..."
pip install -q --upgrade pip
# Unsloth auto-installer detects CUDA version and installs matching torch+unsloth
wget -qO /tmp/_auto_install.py \
    https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py
python /tmp/_auto_install.py
pip install -q pyyaml rouge-score wandb trl
rm -f /tmp/_auto_install.py
ok "Dependencies installed"

# ── 2. Clone repo (if not already present) ──────────────────────────────
if [ ! -f "$REPO_DIR/train_sft.py" ]; then
    if [ -n "$REPO_URL" ]; then
        info "Cloning repo ..."
        git clone "$REPO_URL" /workspace/repo
        ln -sfn /workspace/repo/finetuning "$REPO_DIR"
        ok "Repo cloned"
    else
        info "No REPO_URL set. Upload finetuning/ to $REPO_DIR manually."
        info "Then re-run: bash setup.sh <mode> <args>"
    fi
fi

cd "$REPO_DIR"

# ── 3. wandb login ──────────────────────────────────────────────────────
if [ -n "$WANDB_KEY" ]; then
    wandb login "$WANDB_KEY" 2>/dev/null || true
    ok "wandb configured"
fi

# ── 4. Run training (if mode specified) ─────────────────────────────────
MODE="${1:-}"
shift || true

if [ "$MODE" = "sft" ]; then
    VARIANT="${1:-8k}"
    shift || true
    CONFIG="configs/sft_${VARIANT}.yaml"
    [ -f "$CONFIG" ] || err "Config not found: $CONFIG"
    info "Starting SFT training ($VARIANT) ..."
    python train_sft.py --config "$CONFIG" "$@"

elif [ "$MODE" = "grpo" ]; then
    info "Starting GRPO training ..."
    python train_grpo.py "$@"

elif [ -n "$MODE" ]; then
    err "Unknown mode: $MODE. Use 'sft' or 'grpo'."

else
    ok "Setup complete. Run training with:"
    echo "  bash setup.sh sft 8k                          # SFT 8K full"
    echo "  bash setup.sh sft 16k                         # SFT 16K full"
    echo "  bash setup.sh sft 32k                         # SFT 32K full"
    echo "  bash setup.sh sft 8k --max-samples 50         # smoke test"
    echo "  bash setup.sh grpo                            # GRPO full"
    echo "  bash setup.sh grpo --max-samples 50           # GRPO smoke test"
fi
