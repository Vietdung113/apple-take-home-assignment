"""Shared utilities for evaluation scripts.

Provides:
- .env loader
- RougeScores dataclass + compute_rouge()
- Summary generators (baseline single-pass + agent pipeline)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rouge_score import rouge_scorer

from api_service.agents.graph import pipeline
from api_service.model_loader import generate


# ── Load .env file ──────────────────────────────────────────


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


# ── ROUGE scoring ──────────────────────────────────────────

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


@dataclass
class RougeScores:
    rouge1: float = 0.0  # F1
    rouge2: float = 0.0  # F1
    rougeL: float = 0.0  # F1


def compute_rouge(summary: str, reference: str) -> RougeScores:
    """Compute ROUGE-1/2/L F1 scores between summary and reference."""
    scores = _rouge.score(reference, summary)
    return RougeScores(
        rouge1=scores["rouge1"].fmeasure,
        rouge2=scores["rouge2"].fmeasure,
        rougeL=scores["rougeL"].fmeasure,
    )


# ── Summary generators ─────────────────────────────────────


async def generate_baseline(doc: str, max_doc_chars: int = 28_000) -> str:
    """Single-pass: feed document directly to the model (truncated to fit context)."""
    truncated = doc[:max_doc_chars]
    prompt = (
        "Summarize the following document concisely, "
        "covering all key points.\n\n"
        f"Document:\n{truncated}\n\n"
        "Summary: /no_think"
    )
    return await generate(prompt, max_new_tokens=1024)


async def generate_agent(doc: str) -> str:
    """Two-pass: run the agent pipeline (extract facts -> summarize from facts)."""
    result = await pipeline.ainvoke({"document": doc})
    return result["final_summary"]
