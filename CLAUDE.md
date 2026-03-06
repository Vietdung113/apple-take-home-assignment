# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a take-home assessment for an Apple Language Engineer position. The goal is to design and implement a **summarization adapter** (QLoRA) on top of the Qwen2.5-3B-Instruct base model, trained on the CNN/DailyMail v3.0.0 dataset. The project includes fine-tuning, a serving API with an agentic summarization pipeline, and a frontend demo.

## Repository Structure

Three independent sub-projects, each managed by `uv` with its own `pyproject.toml`:

- **`finetuning/`** — QLoRA adapter training and evaluation scripts (PEFT-based)
- **`serving/`** — FastAPI inference service with an agentic summarization pipeline (`serving/api-service/app/agents/`)
- **`fe/`** — Frontend demo application

## Build & Development Commands

Each sub-project uses `uv` for dependency management:

```bash
# Install dependencies for a sub-project
cd <sub-project> && uv sync

# Run a sub-project's scripts (uv manages the virtualenv)
uv run python <script.py>
uv run pytest                    # run all tests
uv run pytest path/to/test.py    # run a single test file
uv run pytest -k "test_name"     # run a single test by name
```

## Architecture

- **Base model:** Qwen2.5-3B-Instruct
- **Adapter method:** QLoRA via Hugging Face PEFT — low-rank adapters applied to the base model's attention layers with 4-bit quantization
- **Dataset:** CNN/DailyMail v3.0.0 (news article summarization)
- **Serving:** The API service uses an agentic pipeline (in `serving/api-service/app/agents/`) that orchestrates summarization — not a single model call but a multi-step agent workflow
- **Input/Output format:** JSON with `"document"` field in, `"summary"` field out (plain text values)

## Evaluation

Evaluation compares the fine-tuned adapter against the base Qwen2.5-3B-Instruct model using metrics like ROUGE and BLEU scores on the CNN/DailyMail test split.
