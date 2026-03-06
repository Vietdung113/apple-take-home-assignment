# Apple Take-Home Assessment: Summarization Adapter

QLoRA fine-tuning on Qwen3-0.6B for government report summarization with agentic pipeline evaluation.

## Quick Start

```bash
# 1. Train adapter (vast.ai GPU)
cd finetuning
bash setup.sh sft 8k --export-gguf

# 2. Deploy servers (vast.ai)
cd ../inference-server-eval
bash download_models.sh
bash setup_servers.sh

# 3. Start agent API (vast.ai)
cd ../serving
export INFERENCE_BASE_URL=http://localhost:8200/v1
uv run uvicorn api_service.main:app --host 0.0.0.0 --port 8300

# 4. Run evaluation (vast.ai)
cd ../eval
uv run python eval_all.py
```

## Repository Structure

```
finetuning/          QLoRA training (Unsloth + PEFT)
eval/                3-way evaluation (ROUGE + LLM-as-judge)
serving/             Agent API (FastAPI)
inference-server/    llama.cpp for Mac (demo)
inference-server-eval/  3 llama.cpp for eval (vast.ai)
docs/                Documentation + workflow guide
```

## Documentation

- **[WORKFLOW.md](docs/WORKFLOW.md)** — Complete step-by-step guide
- **[finetuning/README.md](finetuning/README.md)** — Training setup
- **[eval/README.md](eval/README.md)** — Evaluation setup
- **[inference-server-eval/README.md](inference-server-eval/README.md)** — Server setup

## Key Features

- **3 context lengths:** 8K / 16K / 32K tokens
- **QLoRA:** rank-32 adapters, 4-bit quantization
- **Dataset:** GovReport (government reports)
- **Agentic pipeline:** chunk → extract → merge → summarize
- **Evaluation:** Base vs Fine-tuned vs Agent
- **Metrics:** ROUGE-1/2/L + Qwen2.5-32B judge

## Requirements

- **GPU:** RTX 4090 24GB or A100 40GB
- **Python:** 3.10+
- **Package manager:** uv

## Architecture

```
Training → GGUF Export → Deploy 3 Servers → Evaluate → Report

vast.ai:
  Port 8100: Base model (Qwen3-0.6B)
  Port 8200: Fine-tuned model (+ adapter)
  Port 8001: Judge (Qwen2.5-32B)
  Port 8300: Agent API (uses :8200)
```

## Results

3-way comparison on 10 stratified test samples:

```
Metric       Base      Finetuned      Agent
--------------------------------------------
ROUGE-L     0.3012       0.3823     0.4123
Judge       3.28/5       4.13/5     4.42/5
```

See `eval/eval_all_results.json` for detailed scores.
