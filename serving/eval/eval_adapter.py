"""Evaluate fine-tuned adapter vs base model with LLM-as-judge.

Compares:
- Base model (no adapter)
- Base model + fine-tuned LoRA adapter

Config (via eval/.env):
    JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL   — Judge model config
    BASE_MODEL_URL                                — Base model inference server
    ADAPTER_MODEL_URL                             — Adapter model inference server
    ADAPTER_PATH                                  — Path to LoRA adapter (for display)

Usage:
    # Start base model server (port 8100)
    cd inference-server && docker compose up

    # Start adapter model server (port 8200) - load base + adapter via vLLM
    vllm serve Qwen/Qwen3-0.6B --port 8200 \
        --enable-lora --lora-modules adapter=../finetuning/output/sft_8k/adapter

    # Run eval
    cd serving && uv run python eval/eval_adapter.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from eval.common import RougeScores, compute_rouge, load_dotenv

load_dotenv()

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

# ── Config ───────────────────────────────────────────────────────────────

BASE_MODEL_URL = os.environ.get("BASE_MODEL_URL", "http://localhost:8100/v1")
ADAPTER_MODEL_URL = os.environ.get("ADAPTER_MODEL_URL", "http://localhost:8200/v1")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "output/sft_8k/adapter")


# ── Scoring dimensions ───────────────────────────────────────────────────

DIMENSIONS: list[str] = [
    "coverage",
    "specificity",
    "consistency",
    "conciseness",
]

WEIGHTS: dict[str, float] = {
    "coverage": 0.30,
    "specificity": 0.30,
    "consistency": 0.25,
    "conciseness": 0.15,
}


# ── Judge prompt ─────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are a busy news reader evaluating two machine-generated summaries.
Your goal: after reading only the summary (not the source), could you \
confidently tell someone what happened?

Below is a source document, a human-written reference summary, and two \
machine-generated summaries. Score each machine summary on 4 dimensions \
(1 = worst, 5 = best).

## Source Document
{source_document}

## Reference Summary (human-written gold standard)
{reference}

## Summary A (Base Model)
{summary_a}

## Summary B (Fine-tuned Model)
{summary_b}

## Scoring Criteria
1. Coverage     — Does it answer who, what, where, when, why? Are all key \
events and outcomes from the reference present?
2. Specificity  — Does it use concrete names, numbers, dates, and places \
from the source instead of vague language like "the company" or "recently"? \
Count how many key entities/numbers from the reference appear in the summary.
3. Consistency  — Are all stated facts accurate vs the source? Any invented \
names, wrong numbers, or events that didn't happen? A single hallucination = score 1.
4. Conciseness  — Does it get to the point without filler, repetition, or \
generic preamble like "This article discusses..."?

Respond ONLY with valid JSON (no markdown, no explanation):
{{"summary_a": {{"coverage": <int>, "specificity": <int>, "consistency": <int>, \
"conciseness": <int>}}, \
"summary_b": {{"coverage": <int>, "specificity": <int>, "consistency": <int>, \
"conciseness": <int>}}}}"""


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class Scores:
    coverage: int = 3
    specificity: int = 3
    consistency: int = 3
    conciseness: int = 3

    def weighted(self) -> float:
        return sum(getattr(self, d) * WEIGHTS[d] for d in DIMENSIONS)

    def mean(self) -> float:
        return sum(getattr(self, d) for d in DIMENSIONS) / len(DIMENSIONS)


@dataclass
class EvalResult:
    idx: int
    bucket: str
    doc_chars: int
    base_summary: str
    adapter_summary: str
    base_time: float
    adapter_time: float
    base_scores: Scores | None = None
    adapter_scores: Scores | None = None
    base_rouge: RougeScores | None = None
    adapter_rouge: RougeScores | None = None


# ── Inference clients ────────────────────────────────────────────────────


async def generate_summary(url: str, document: str) -> str:
    """Generate summary via OpenAI-compatible API."""
    prompt = f"Summarize the following document:\n\n{document}\n\nSummary:"
    async with httpx.AsyncClient(base_url=url, timeout=120.0) as client:
        resp = await client.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()


# ── Judge client ─────────────────────────────────────────────────────────


class JudgeClient:
    """Calls an external LLM via OpenAI-compatible chat completions API."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.url = f"{base_url}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = model

    def score(
        self,
        source_document: str,
        reference: str,
        summary_a: str,
        summary_b: str,
        retries: int = 3,
    ) -> tuple[Scores, Scores] | None:
        """Ask the judge to score both summaries. Returns (scores_a, scores_b)."""
        prompt = JUDGE_PROMPT.format(
            source_document=self._truncate_for_judge(source_document),
            reference=reference,
            summary_a=summary_a,
            summary_b=summary_b,
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise evaluator. Respond with valid JSON only, no markdown fences, no explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        for attempt in range(retries):
            try:
                resp = httpx.post(
                    self.url, json=payload, headers=self.headers, timeout=300
                )
                resp.raise_for_status()
                rjson = resp.json()
                msg = rjson["choices"][0]["message"]
                content = msg.get("content", "").strip()

                if not content:
                    raise ValueError("Judge returned empty content")

                # Strip markdown code fences if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                # Try to find JSON object in the text
                start = content.find("{")
                end = content.rfind("}") + 1
                if start == -1 or end == 0:
                    raise ValueError(f"No JSON object in response: {content[:200]}")

                data = json.loads(content[start:end])
                return (
                    Scores(**{d: data["summary_a"][d] for d in DIMENSIONS}),
                    Scores(**{d: data["summary_b"][d] for d in DIMENSIONS}),
                )
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"    Judge error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2**attempt)

        return None

    def _truncate_for_judge(self, doc: str, max_chars: int = 4000) -> str:
        """Truncate source document to save judge tokens."""
        if len(doc) <= max_chars:
            return doc
        half = max_chars // 2
        return doc[:half] + "\n\n[... truncated ...]\n\n" + doc[-half:]


# ── Report printing ──────────────────────────────────────────────────────

ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]


def _print_bucket(results: list[EvalResult], label: str):
    """Print ROUGE + Judge scores for a set of results."""
    n = len(results)
    if not n:
        return

    print(f"\n  ── {label} ({n} examples) ──")

    # ROUGE
    rouge_scored = [r for r in results if r.base_rouge and r.adapter_rouge]
    if rouge_scored:
        print(f"\n  {'Metric':<14} {'Base':>10} {'Adapter':>10} {'Delta':>10}")
        print(f"  {'-'*46}")
        for metric in ROUGE_METRICS:
            b = sum(getattr(r.base_rouge, metric) for r in rouge_scored) / len(
                rouge_scored
            )
            a = sum(getattr(r.adapter_rouge, metric) for r in rouge_scored) / len(
                rouge_scored
            )
            print(f"  {metric:<14} {b:>10.4f} {a:>10.4f} {a - b:>+10.4f}")

    # Judge
    judge_scored = [r for r in results if r.base_scores and r.adapter_scores]
    if judge_scored:
        print(f"\n  {'Dimension':<14} {'Base':>10} {'Adapter':>10} {'Delta':>10}")
        print(f"  {'-'*46}")
        b_w, a_w = 0.0, 0.0
        for d in DIMENSIONS:
            b = sum(getattr(r.base_scores, d) for r in judge_scored) / len(
                judge_scored
            )
            a = sum(getattr(r.adapter_scores, d) for r in judge_scored) / len(
                judge_scored
            )
            print(f"  {d:<14} {b:>10.2f} {a:>10.2f} {a - b:>+10.2f}")
            b_w += b * WEIGHTS[d]
            a_w += a * WEIGHTS[d]
        print(f"  {'-'*46}")
        print(f"  {'Weighted':<14} {b_w:>10.2f} {a_w:>10.2f} {a_w - b_w:>+10.2f}")


def print_report(results: list[EvalResult]):
    """Print per-bucket breakdown + overall aggregate."""
    print(f"\n{'='*60}")
    print(f"  ADAPTER EVALUATION  ({len(results)} examples)")
    print(f"{'='*60}")

    for bucket_name in ["SHORT", "MEDIUM", "LONG"]:
        bucket_results = [r for r in results if r.bucket == bucket_name.lower()]
        _print_bucket(bucket_results, bucket_name)

    _print_bucket(results, "OVERALL")

    # Timing
    if results:
        avg_bt = sum(r.base_time for r in results) / len(results)
        avg_at = sum(r.adapter_time for r in results) / len(results)
        print(f"\n  {'Avg time (s)':<14} {avg_bt:>10.1f} {avg_at:>10.1f}")
    print()


def save_results(results: list[EvalResult], path: str = "eval_adapter_results.json"):
    """Save results to JSON."""
    data = []
    for r in results:
        entry: dict = {
            "idx": r.idx,
            "bucket": r.bucket,
            "doc_chars": r.doc_chars,
            "base_time": r.base_time,
            "adapter_time": r.adapter_time,
        }
        if r.base_rouge:
            entry["base_rouge"] = {
                "rouge1": r.base_rouge.rouge1,
                "rouge2": r.base_rouge.rouge2,
                "rougeL": r.base_rouge.rougeL,
            }
        if r.adapter_rouge:
            entry["adapter_rouge"] = {
                "rouge1": r.adapter_rouge.rouge1,
                "rouge2": r.adapter_rouge.rouge2,
                "rougeL": r.adapter_rouge.rougeL,
            }
        if r.base_scores:
            entry["base_judge"] = {d: getattr(r.base_scores, d) for d in DIMENSIONS}
        if r.adapter_scores:
            entry["adapter_judge"] = {
                d: getattr(r.adapter_scores, d) for d in DIMENSIONS
            }
        data.append(entry)

    Path(path).write_text(json.dumps(data, indent=2))
    print(f"  Results saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────


async def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found.")
        print("Run 'uv run python eval/prepare_dataset.py' first.")
        sys.exit(1)

    api_key = os.environ.get("JUDGE_API_KEY", "")
    base_url = os.environ.get("JUDGE_BASE_URL", "")
    model = os.environ.get("JUDGE_MODEL", "")

    if not all([api_key, base_url, model]):
        print("ERROR: Set JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL in eval/.env")
        sys.exit(1)

    judge = JudgeClient(api_key, base_url, model)
    all_samples = json.loads(DATASET_PATH.read_text())
    n = int(os.environ.get("EVAL_N", len(all_samples)))
    samples = all_samples[:n]
    total = len(samples)

    print(f"  Adapter Evaluation")
    print(f"  Judge:         {model}")
    print(f"  Judge API:     {base_url}")
    print(f"  Base model:    {BASE_MODEL_URL}")
    print(f"  Adapter model: {ADAPTER_MODEL_URL}")
    print(f"  Adapter path:  {ADAPTER_PATH}")
    print(f"  Dataset:       {DATASET_PATH.name} ({total} examples)")
    print()

    results: list[EvalResult] = []

    for i, sample in enumerate(samples):
        doc = sample["report"]
        ref = sample["summary"]
        bucket = sample["bucket"]
        print(
            f"[{i + 1}/{total}] idx={sample['idx']}  bucket={bucket}  {len(doc):,} chars"
        )

        # 1. Base model summary
        t0 = time.time()
        base_out = await generate_summary(BASE_MODEL_URL, doc)
        t_base = time.time() - t0

        # 2. Adapter model summary
        t0 = time.time()
        adapter_out = await generate_summary(ADAPTER_MODEL_URL, doc)
        t_adapter = time.time() - t0

        print(f"  Base:    {len(base_out):,} chars  ({t_base:.1f}s)")
        print(f"  Adapter: {len(adapter_out):,} chars  ({t_adapter:.1f}s)")

        # 3. ROUGE scores (vs reference)
        b_rouge = compute_rouge(base_out, ref)
        a_rouge = compute_rouge(adapter_out, ref)
        print(
            f"  ROUGE-1: base={b_rouge.rouge1:.4f}  adapter={a_rouge.rouge1:.4f}"
        )

        result = EvalResult(
            idx=sample["idx"],
            bucket=bucket,
            doc_chars=len(doc),
            base_summary=base_out,
            adapter_summary=adapter_out,
            base_time=t_base,
            adapter_time=t_adapter,
            base_rouge=b_rouge,
            adapter_rouge=a_rouge,
        )

        # 4. Judge scores
        scores = judge.score(
            source_document=doc,
            reference=ref,
            summary_a=base_out,
            summary_b=adapter_out,
        )
        if scores:
            result.base_scores, result.adapter_scores = scores
            print(
                f"  Judge:   base={result.base_scores.mean():.1f}  "
                f"adapter={result.adapter_scores.mean():.1f}"
            )
        else:
            print(f"  Judge:   FAILED")

        results.append(result)
        print()

    print_report(results)
    save_results(results)


if __name__ == "__main__":
    asyncio.run(main())
