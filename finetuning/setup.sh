#!/usr/bin/env bash
# setup.sh — Bootstrap a vast.ai instance for SFT/GRPO fine-tuning.
#
# Usage:
#   bash setup.sh                                         # install deps only
#   bash setup.sh sft 8k                                  # install + train SFT 8K
#   bash setup.sh sft 16k                                 # install + train SFT 16K
#   bash setup.sh sft 32k                                 # install + train SFT 32K
#   bash setup.sh sft 8k --max-samples 50 --export-gguf   # smoke test + GGUF
#   bash setup.sh grpo                                    # install + train GRPO
#   bash setup.sh grpo --max-samples 50 --export-gguf     # GRPO smoke test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WANDB_KEY="${WANDB_API_KEY:-}"

# ── Colors ───────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
err()   { echo -e "\033[1;31m[ERR]\033[0m $*"; exit 1; }

# ── 1. Install uv if missing ────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version)"

# ── 2. Sync dependencies ────────────────────────────────────────────────
cd "$SCRIPT_DIR"
info "Syncing dependencies (uv sync) ..."
uv sync
ok "Dependencies installed"

# ── 3. wandb login ──────────────────────────────────────────────────────
if [ -n "$WANDB_KEY" ]; then
    uv run wandb login "$WANDB_KEY" 2>/dev/null || true
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
    uv run python train_sft.py --config "$CONFIG" "$@"

elif [ "$MODE" = "grpo" ]; then
    info "Starting GRPO training ..."
    uv run python train_grpo.py "$@"

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
