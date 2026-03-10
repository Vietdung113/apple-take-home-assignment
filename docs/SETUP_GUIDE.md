# Setup and Running Guide

> **Design overview & results:** [README.md](../README.md)

## Table of Contents

- [System Requirements](#system-requirements)
- [1. Fine-tuning](#1-fine-tuning)
  - [1.1 Environment Setup](#11-environment-setup)
  - [1.2 Generate Training Data](#12-generate-training-data)
  - [1.3 QLoRA SFT Training](#13-qlora-sft-training)
  - [1.4 Export to GGUF](#14-export-to-gguf)
- [2. Models](#2-models)
  - [2.1 Download Model](#21-download-model)
  - [2.2 Model List](#22-model-list)
- [3. Evaluation](#3-evaluation)
  - [3.1 Environment Setup](#31-environment-setup)
  - [3.2 Start Model Server](#32-start-model-server)
  - [3.3 Prepare Test Set](#33-prepare-test-set)
  - [3.4 Run Evaluation](#34-run-evaluation)
  - [3.5 View Results](#35-view-results)
- [4. Serving](#4-serving)
  - [4.1 Start llama.cpp](#41-start-llamacpp)
  - [4.2 Docker Compose](#42-docker-compose)
  - [4.3 Run Locally (without Docker)](#43-run-locally-without-docker)
  - [4.4 Test API](#44-test-api)

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | >= 3.11 |
| Package manager | [`uv`](https://docs.astral.sh/uv/) |
| GPU (training) | NVIDIA A100 40GB+ (or RTX 4090 24GB) |
| GPU (inference) | Apple Silicon (Metal) or NVIDIA GPU |
| llama.cpp | `brew install llama.cpp` (Mac) |
| Docker | Docker Desktop (for serving) |
| NVIDIA API Key | Free at https://build.nvidia.com/ (for LLM-as-judge in eval) |

```
apple-take-home-assignment/
├── finetuning/        # QLoRA SFT training
├── serving/           # API + Frontend
│   ├── api_service/   # FastAPI backend
│   └── fe/            # Gradio frontend
├── eval/              # Evaluation pipeline
├── models/            # GGUF model files
└── docs/              # Documentation
```

---

## 1. Fine-tuning

### 1.1 Environment Setup

```bash
cd finetuning
uv sync
```

If you need to generate synthetic data or run LLM-as-judge (optional):

```bash
cp .env.example .env
# Edit .env → add NVIDIA_API_KEY=nvapi-...
```

### 1.2 Generate Training Data

Download GovReport from HuggingFace and convert to training format:

```bash
uv run python data_analysis/convert_govreport_to_base_format.py
```

**Output:**
```
data/govreport_full/
├── train.jsonl   # 16,641 samples (filtered >32K tokens)
└── val.jsonl     # 1,849 samples
```

Each line in the JSONL is a `{"document": "...", "summary": "..."}` pair formatted according to the prompt template.

### 1.3 QLoRA SFT Training

**Training configuration:** `config/training.yaml`

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | `unsloth/Qwen3-0.6B` | Base model |
| **Max seq length** | 32,768 | 32K context |
| **LoRA rank** | 32 | `alpha=64` (2x rank) |
| **LoRA target modules** | `q/k/v/o/gate/up/down_proj` | 7 modules, all attention + MLP |
| **Learning rate** | 2e-4 | Cosine scheduler, warmup 3% |
| **Batch size** | 4 x 8 grad_accum = **32 effective** | Reduce `batch_size` if OOM |
| **Epochs** | 4 | ~2,080 steps |
| **Optimizer** | AdamW 8-bit | Saves VRAM |
| **Checkpoint** | Every 200 steps | Keeps up to 5 checkpoints |

Adjust for your hardware:
- **GPU 24GB (RTX 4090):** `batch_size: 2`, `grad_accum_steps: 16` (keeps effective batch = 32)
- **GPU 40GB+ (A100):** Use default config
- **Multi-GPU:** Add `torchrun --nproc_per_node=N` before the training command

```bash
# Quick test (verify setup)
uv run python train_sft_base.py --max-samples 10 --epochs 1

# Full training
uv run python train_sft_base.py \
  --config config/training.yaml \
  --data data/govreport_full/train.jsonl \
  --val-data data/govreport_full/val.jsonl \
  --epochs 4 \
  --export-gguf
```

**Output:**
```
output/sft_YYYYMMDD_HHMMSS/
├── checkpoint-200/    # Checkpoint every 200 steps
├── checkpoint-400/
├── ...
└── adapter/           # Final adapter weights (~34MB)
    ├── adapter_model.safetensors
    └── adapter_config.json
```

### 1.4 Export to GGUF

Merge adapter into base model → quantize → GGUF file for llama.cpp:

```bash
# Option 1: Export with Unsloth (simple)
uv run python export_gguf.py --checkpoint output/sft_xxx/checkpoint-200

# Option 2: Export with llama.cpp (more quantization options)
uv run python export_gguf_llamacpp.py \
  --adapter output/sft_xxx/adapter/ \
  --quant Q4_K_M
```

**Output:** `.gguf` file (~400MB for Qwen3-0.6B Q4_K_M). Copy to `models/finetuned/`.

---

## 2. Models

### 2.1 Download Model

Requirements: `gdown` (recommended) or `curl`.

```bash
pip install gdown   # install once

cd models
bash download_models.sh
```

The script downloads the fine-tuned GGUF file from Google Drive:

| File | Size | Quantization |
|------|------|-------------|
| `finetuned-qwen3-0.6b.Q4_K_M.gguf` | ~378MB | Q4_K_M |

If the file already exists, the script will prompt before re-downloading.

---

## 3. Evaluation

### 3.1 Environment Setup

```bash
cd eval
uv sync

# Configure NVIDIA API key (required for LLM-as-judge)
cp .env.example .env
# Edit .env → add NVIDIA_API_KEY=nvapi-...
```

### 3.2 Start Model Server

Eval requires a running llama.cpp server:

```bash
./start_model.sh                          # default: finetuned model, port 8080
./start_model.sh <model.gguf>             # specify a different model
./start_model.sh <model.gguf> --port 8100 # change port
```

Verify the server is ready:

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

### 3.3 Prepare Test Set

Create a stratified test set from GovReport:

```bash
uv run python prepare_test_set.py --num-samples 100
```

**Output:** `test_set.jsonl` — 100 samples distributed: 50 short (<=8K) / 30 medium (8K-16K) / 20 long (16K-32K).

### 3.4 Run Evaluation

**Step 1 — Evaluate base model:**

```bash
# Make sure the base model is running on port 8080
./start_model.sh ../models/Qwen3-0.6B-Q4_K_M.gguf

# Run eval (ROUGE + Embedding + LLM-as-judge)
uv run python eval_and_analysis.py \
  --mode base \
  --test-set test_set.jsonl \
  --output results/base_results.csv \
  --all
```

**Step 2 — Evaluate fine-tuned model:**

```bash
# Stop the base model, start the fine-tuned model on the same port
./start_model.sh ../models/finetuned/finetuned-qwen3-0.6b.Q4_K_M.gguf

uv run python eval_and_analysis.py \
  --mode base \
  --test-set test_set.jsonl \
  --output results/finetune_results.csv \
  --all
```

**Step 3 — (Optional) Evaluate agent pipeline:**

```bash
# Requires both llama.cpp + FastAPI API running
./start_model.sh ../models/finetuned/finetuned-qwen3-0.6b.Q4_K_M.gguf
cd ../serving && uv run uvicorn api_service.main:app --port 8001 &

cd ../eval
uv run python eval_and_analysis.py \
  --mode agent \
  --test-set test_set.jsonl \
  --output results/agent_results.csv \
  --all
```

**Metric flags:**

| Flag | Metrics | Speed |
|------|---------|-------|
| *(no flag)* | ROUGE + Embedding | Fast |
| `--judge` | LLM-as-judge only | Medium |
| `--all` | ROUGE + Embedding + LLM-as-judge | Slowest |

### 3.5 View Results

**Output files:**

```
results/
├── base_results.csv                  # Summary table by category
├── base_results_detailed.jsonl       # Per-sample details (summary + scores)
├── base_results_base_low_scores.json # Low-scoring samples (for debugging)
├── finetune_results.csv
├── finetune_results_detailed.jsonl
└── finetune_results_base_low_scores.json
```

View summary:

```bash
# Summary table (CSV: category, rouge, embedding, judge scores)
cat results/base_results.csv

# Per-sample details
cat results/base_results_detailed.jsonl | python3 -m json.tool | head -50

# Analyze low-scoring samples
cat results/base_results_base_low_scores.json | python3 -m json.tool | head -50
```

Deep analysis with notebook:

```bash
# Open error analysis notebook
jupyter notebook finetuning/data_analysis/error_analysis.ipynb
```

---

## 4. Serving

### 4.1 Start llama.cpp

llama.cpp **must run natively** (not in Docker) to leverage Metal GPU on Mac:

```bash
cd eval

# Base model
./start_model.sh ../models/Qwen3-0.6B-Q4_K_M.gguf

# Or fine-tuned model
./start_model.sh ../models/finetuned/finetuned-qwen3-0.6b.Q4_K_M.gguf
```

Verify:

```bash
curl http://localhost:8080/health
```

### 4.2 Docker Compose

After llama.cpp is running, start API + Frontend + Phoenix:

```bash
cd serving
docker compose up --build
```

**Services:**

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Phoenix** | 6006 | http://localhost:6006 | LLM observability — trace pipeline, view prompt/response |
| **API** (FastAPI) | 8001 | http://localhost:8001 | Backend — agentic summarization pipeline |
| **Frontend** (Gradio) | 7860 | http://localhost:7860 | Demo UI — paste text, click Summarize |

```
Browser (:7860)
    → Gradio Frontend (Docker)
        → FastAPI + LangGraph (Docker, :8001)
            → llama.cpp (Native, :8080, Metal GPU)
        → Phoenix (Docker, :6006) — tracing
```

> **Note:** Docker Desktop on Mac runs a Linux VM and cannot access the Metal GPU. Therefore llama.cpp must run natively, and the API in Docker connects via `host.docker.internal:8080`.

### 4.3 Run Locally (without Docker)

If not using Docker, open 3 terminals:

```bash
# Terminal 1: llama.cpp
cd eval && ./start_model.sh ../models/finetuned/finetuned-qwen3-0.6b.Q4_K_M.gguf

# Terminal 2: FastAPI API
cd serving && uv sync
uv run uvicorn api_service.main:app --host 0.0.0.0 --port 8001

# Terminal 3: Gradio UI
cd serving/fe && uv sync
uv run python app.py
```

Access:
- API: http://localhost:8001/summarize
- Gradio UI: http://localhost:7860
- Health check: http://localhost:8001/health

### 4.4 Test API

```bash
# Health check
curl http://localhost:8001/health

# Test with sample document
curl -X POST http://localhost:8001/summarize \
  -H "Content-Type: application/json" \
  -d @serving/tests/data/sample_8k.json

# Unit test (routing logic)
cd serving && uv run python tests/test_routing.py

# E2E test (requires running API)
cd serving && uv run pytest tests/test_e2e.py -v
```
