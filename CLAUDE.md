# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Apple Language Engineer take-home assessment: Design and implement a **summarization adapter** (QLoRA) on top of Qwen3-0.6B base model, trained on GovReport dataset. Includes fine-tuning, serving API with agentic pipeline, and evaluation suite.

## Repository Structure

Four independent sub-projects, each managed by `uv` with its own `pyproject.toml`:

- **`finetuning/`** — QLoRA SFT training scripts (Unsloth + PEFT)
- **`serving/`** — FastAPI agentic summarization pipeline
- **`eval/`** — 3-way evaluation (base/finetuned/agent) with ROUGE + LLM-as-judge
- **`inference-server/`** — llama.cpp server for local Mac serving
- **`inference-server-eval/`** — 3 llama.cpp servers for eval on vast.ai

## Build & Development

Each sub-project uses `uv`:

```bash
cd <sub-project> && uv sync
uv run python <script.py>
uv run pytest                    # run all tests
```

## Architecture

- **Base model:** Qwen3-0.6B (Unsloth)
- **Adapter:** QLoRA (rank 32, 4-bit quantization)
- **Dataset:** GovReport (government report summarization)
- **Training:** SFT on 3 context lengths (8K/16K/32K tokens)
- **Serving:** Agentic pipeline (chunk → extract facts → merge → summarize)
- **Eval:** ROUGE + Qwen2.5-32B judge (coverage/specificity/consistency/conciseness)

## Evaluation

3-way comparison:
1. **Base model** (Qwen3-0.6B, no fine-tuning)
2. **Fine-tuned model** (Qwen3-0.6B + QLoRA adapter)
3. **Agent pipeline** (using fine-tuned model)

Metrics: ROUGE-1/2/L, LLM-as-judge scores on stratified test set (short/medium/long docs).
