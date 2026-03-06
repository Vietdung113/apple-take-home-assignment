# Evaluation Scripts

## Overview

- `eval_adapter.py` — Compare base model vs fine-tuned adapter (ROUGE + LLM-as-judge)
- `eval_judge.py` — Compare baseline vs agentic pipeline (legacy)
- `eval_rouge.py` — ROUGE-only evaluation
- `prepare_dataset.py` — Create stratified test set from GovReport

## Setup

1. **Prepare eval dataset**
```bash
cd serving
uv run python eval/prepare_dataset.py
# → eval/eval_dataset.json (10 samples: 3 short, 3 medium, 4 long)
```

2. **Configure eval/.env**
```bash
cp eval/.env.example eval/.env
# Edit eval/.env with your server URLs
```

## Adapter Evaluation

Compares **base model vs fine-tuned adapter** on 10 stratified samples.

### 1. Start Judge Server

```bash
# On vast.ai or local with GPU
cd inference-server-judge
bash setup_judge.sh
# → http://localhost:8001 (Qwen2.5-32B-Instruct)
```

### 2. Start Base Model Server

```bash
# Option A: Docker (llama.cpp)
cd inference-server
docker compose up
# → http://localhost:8100

# Option B: vLLM
vllm serve Qwen/Qwen3-0.6B --port 8100 --dtype bfloat16
```

### 3. Start Adapter Model Server

```bash
# vLLM with LoRA adapter
vllm serve Qwen/Qwen3-0.6B --port 8200 --dtype bfloat16 \
    --enable-lora \
    --lora-modules adapter=../finetuning/output/sft_8k/adapter
# → http://localhost:8200
```

### 4. Run Evaluation

```bash
cd serving
uv run python eval/eval_adapter.py
```

### Output

```
  ADAPTER EVALUATION  (10 examples)
  ════════════════════════════════════════════════════════

  ── OVERALL (10 examples) ──

  Metric         Base      Adapter      Delta
  ----------------------------------------------
  rouge1       0.3421      0.4123    +0.0702
  rouge2       0.1821      0.2341    +0.0520
  rougeL       0.3012      0.3823    +0.0811

  Dimension    Base      Adapter      Delta
  ----------------------------------------------
  coverage       3.2        4.1       +0.9
  specificity    3.0        4.3       +1.3
  consistency    3.5        4.0       +0.5
  conciseness    3.8        3.9       +0.1
  ----------------------------------------------
  Weighted       3.28       4.13      +0.85

  Avg time (s)   1.2        1.3

  Results saved to eval_adapter_results.json
```

## Evaluation Metrics

### ROUGE (n-gram overlap)
- **rouge1**: Unigram overlap
- **rouge2**: Bigram overlap
- **rougeL**: Longest common subsequence

### LLM-as-Judge (Qwen2.5-32B)
- **Coverage** (30%): Answers who/what/where/when/why?
- **Specificity** (30%): Uses concrete names/numbers/dates?
- **Consistency** (25%): All facts accurate vs source?
- **Conciseness** (15%): No filler/repetition?

Scores: 1-5 scale, weighted average reported.

## Notes

- vLLM `--enable-lora` allows runtime adapter loading
- Judge server can be reused across multiple evals
- Eval dataset is stratified by doc length (short/medium/long)
