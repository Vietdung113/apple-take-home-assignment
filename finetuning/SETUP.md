# Fine-tuning Setup

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

## 2. Install dependencies

```bash
cd finetuning
uv sync
```

## 3. Prepare data

```bash
uv run python data_analysis/convert_govreport_to_base_format.py \
  --output data/govreport_full \
  --val-split 0.1 \
  --seed 42
```

**Output:**
- `data/govreport_full/train.jsonl` (16,641 samples)
- `data/govreport_full/val.jsonl` (1,849 samples)

**Params:**
- `--output`: Output directory
- `--val-split`: Validation split ratio (default: 0.1)
- `--seed`: Random seed for splitting (default: 42)
- `--max-samples`: Limit samples for testing

## 4. Train

**Quick test:**
```bash
uv run python train_sft_base.py --max-samples 10 --epochs 1
```

**Full training:**
```bash
# Without WandB logging
export WANDB_DISABLED=true
uv run python train_sft_base.py --config config/training.yaml --export-gguf

# With WandB logging (get API key from https://wandb.ai/authorize)
wandb login
uv run python train_sft_base.py --config config/training.yaml --export-gguf
```

**Resume from checkpoint:**
```bash
uv run python train_sft_base.py --resume-from output/sft_base_*/checkpoint-1000
```

**Output:** `output/sft_base_*/`
- `adapter/` - QLoRA weights
- `gguf_gguf/qwen3-0.6b.Q4_K_M.gguf` - Quantized model

---

## Configuration

Edit `config/training.yaml` to modify:
- Model, batch size, learning rate, epochs
- LoRA rank, sequence length
