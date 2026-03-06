"""Shared utilities for evaluation scripts.

Provides:
- .env loader
- ROUGE scoring
- OpenAI-compatible inference client
- Agent pipeline invocation
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from rouge_score import rouge_scorer


# ── Load .env ────────────────────────────────────────────────────────────


def load_dotenv(path: Path | None = None):
    """Parse a .env file into os.environ (no external dependency)."""
    if path is None:
        path = Path(__file__).parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        os.environ.setdefault(key, value)


# ── ROUGE scoring ────────────────────────────────────────────────────────

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


@dataclass
class RougeScores:
    rouge1: float = 0.0  # F1
    rouge2: float = 0.0  # F1
    rougeL: float = 0.0  # F1


def compute_rouge(prediction: str, reference: str) -> RougeScores:
    """Compute ROUGE F1 scores."""
    scores = _rouge.score(reference, prediction)
    return RougeScores(
        rouge1=scores["rouge1"].fmeasure,
        rouge2=scores["rouge2"].fmeasure,
        rougeL=scores["rougeL"].fmeasure,
    )


# ── Inference client ─────────────────────────────────────────────────────


async def generate_summary(
    server_url: str, document: str, max_tokens: int = 512
) -> str:
    """Generate summary via OpenAI-compatible API (llama.cpp/vLLM)."""
    prompt = f"Summarize the following document:\n\n{document}\n\nSummary:"
    async with httpx.AsyncClient(base_url=server_url, timeout=120.0) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()


# ── Agent pipeline ───────────────────────────────────────────────────────


async def generate_agent_summary(document: str, api_url: str) -> str:
    """Call agentic summarization pipeline via POST /summarize."""
    async with httpx.AsyncClient(base_url=api_url, timeout=300.0) as client:
        resp = await client.post(
            "/summarize",
            json={"document": document},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["summary"].strip()
