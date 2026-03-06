"""LLM-as-Judge evaluation with ROUGE + multi-dimensional scoring.

Reads examples from eval/eval_dataset.json (created by prepare_dataset.py),
generates baseline + agent summaries, computes ROUGE, and sends to an external
LLM judge for multi-dimensional scoring. Reports per-bucket breakdown.

Config (via eval/.env or environment):
    JUDGE_API_KEY   — API key for the judge endpoint
    JUDGE_BASE_URL  — OpenAI-compatible base URL
    JUDGE_MODEL     — model identifier for the judge

Usage:
    cd serving && uv run python eval/eval_judge.py
    # or: make eval-judge
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from eval.common import (
    RougeScores,
    compute_rouge,
    generate_agent,
    generate_baseline,
    load_dotenv,
)

load_dotenv()

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


# ── Scoring dimensions ──────────────────────────────────────

DIMENSIONS: list[str] = [
    "coverage",
    "specificity",
    "consistency",
    "conciseness",
]

WEIGHTS: dict[str, float] = {
    "coverage":    0.30,
    "specificity": 0.30,
    "consistency": 0.25,
    "conciseness": 0.15,
}


# ── Judge prompt ────────────────────────────────────────────

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

## Summary A
{summary_a}

## Summary B
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


# ── Data classes ────────────────────────────────────────────


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
class JudgeResult:
    idx: int
    bucket: str
    doc_chars: int
    baseline_summary: str
    agent_summary: str
    baseline_time: float
    agent_time: float
    baseline_scores: Scores | None = None
    agent_scores: Scores | None = None
    baseline_rouge: RougeScores | None = None
    agent_rouge: RougeScores | None = None


# ── Judge client ────────────────────────────────────────────


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
            source_document=source_document,
            reference=reference,
            summary_a=summary_a,
            summary_b=summary_b,
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Respond with JSON only. Do not use <think> tags or chain-of-thought. Output the JSON immediately."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }

        for attempt in range(retries):
            try:
                resp = httpx.post(
                    self.url, json=payload, headers=self.headers, timeout=300,
                )
                resp.raise_for_status()
                try:
                    rjson = resp.json()
                except json.JSONDecodeError:
                    raise ValueError(f"Non-JSON response ({resp.status_code}): {resp.text[:300]}")
                msg = rjson["choices"][0]["message"]
                content = msg.get("content")
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                if content is None or content.strip() == "":
                    # Thinking model: JSON may be in reasoning field
                    if reasoning:
                        text = reasoning.strip()
                    else:
                        raise ValueError("Judge returned empty content and no reasoning")
                else:
                    text = content.strip()

                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                # Try to find JSON object in the text
                start = text.find("{")
                end = text.rfind("}") + 1
                if start == -1 or end == 0:
                    raise ValueError(f"No JSON object in response: {text[:200]}")
                data = json.loads(text[start:end])
                return (
                    Scores(**{d: data["summary_a"][d] for d in DIMENSIONS}),
                    Scores(**{d: data["summary_b"][d] for d in DIMENSIONS}),
                )
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"    Judge error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        return None


# ── Utilities ──────────────────────────────────────────────


def truncate_for_judge(doc: str, max_chars: int = 4000) -> str:
    """Truncate source document to save judge tokens.

    Keeps the first half and last half of the budget,
    since key information often appears at the start and end.
    """
    if len(doc) <= max_chars:
        return doc
    half = max_chars // 2
    return doc[:half] + "\n\n[... truncated ...]\n\n" + doc[-half:]


# ── Report printing ────────────────────────────────────────


ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]


def _print_bucket(results: list[JudgeResult], label: str):
    """Print ROUGE + Judge scores for a set of results."""
    n = len(results)
    if not n:
        return

    print(f"\n  ── {label} ({n} examples) ──")

    # ROUGE
    rouge_scored = [r for r in results if r.baseline_rouge and r.agent_rouge]
    if rouge_scored:
        print(f"\n  {'Metric':<14} {'Baseline':>10} {'Agent':>10} {'Delta':>10}")
        print(f"  {'-'*46}")
        for metric in ROUGE_METRICS:
            b = sum(getattr(r.baseline_rouge, metric) for r in rouge_scored) / len(rouge_scored)
            a = sum(getattr(r.agent_rouge, metric) for r in rouge_scored) / len(rouge_scored)
            print(f"  {metric:<14} {b:>10.4f} {a:>10.4f} {a - b:>+10.4f}")

    # Judge
    judge_scored = [r for r in results if r.baseline_scores and r.agent_scores]
    if judge_scored:
        print(f"\n  {'Dimension':<14} {'Baseline':>10} {'Agent':>10} {'Delta':>10}")
        print(f"  {'-'*46}")
        b_w, a_w = 0.0, 0.0
        for d in DIMENSIONS:
            b = sum(getattr(r.baseline_scores, d) for r in judge_scored) / len(judge_scored)
            a = sum(getattr(r.agent_scores, d) for r in judge_scored) / len(judge_scored)
            print(f"  {d:<14} {b:>10.2f} {a:>10.2f} {a - b:>+10.2f}")
            b_w += b * WEIGHTS[d]
            a_w += a * WEIGHTS[d]
        print(f"  {'-'*46}")
        print(f"  {'Weighted':<14} {b_w:>10.2f} {a_w:>10.2f} {a_w - b_w:>+10.2f}")


def print_judge_report(results: list[JudgeResult]):
    """Print per-bucket breakdown + overall aggregate."""
    print(f"\n{'='*60}")
    print(f"  LLM-AS-JUDGE EVALUATION  ({len(results)} examples)")
    print(f"{'='*60}")

    for bucket_name in ["SHORT", "MEDIUM", "LONG"]:
        bucket_results = [r for r in results if r.bucket == bucket_name.lower()]
        _print_bucket(bucket_results, bucket_name)

    _print_bucket(results, "OVERALL")

    # Timing
    if results:
        avg_bt = sum(r.baseline_time for r in results) / len(results)
        avg_at = sum(r.agent_time for r in results) / len(results)
        print(f"\n  {'Avg time (s)':<14} {avg_bt:>10.1f} {avg_at:>10.1f}")
    print()


def save_judge_results(results: list[JudgeResult], path: str = "eval_judge_results.json"):
    """Save results to JSON."""
    data = []
    for r in results:
        entry: dict = {
            "idx": r.idx,
            "bucket": r.bucket,
            "doc_chars": r.doc_chars,
            "baseline_time": r.baseline_time,
            "agent_time": r.agent_time,
        }
        if r.baseline_rouge:
            entry["baseline_rouge"] = {
                "rouge1": r.baseline_rouge.rouge1,
                "rouge2": r.baseline_rouge.rouge2,
                "rougeL": r.baseline_rouge.rougeL,
            }
        if r.agent_rouge:
            entry["agent_rouge"] = {
                "rouge1": r.agent_rouge.rouge1,
                "rouge2": r.agent_rouge.rouge2,
                "rougeL": r.agent_rouge.rougeL,
            }
        if r.baseline_scores:
            entry["baseline_judge"] = {d: getattr(r.baseline_scores, d) for d in DIMENSIONS}
        if r.agent_scores:
            entry["agent_judge"] = {d: getattr(r.agent_scores, d) for d in DIMENSIONS}
        data.append(entry)

    Path(path).write_text(json.dumps(data, indent=2))
    print(f"  Results saved to {path}")


# ── Main ───────────────────────────────────────────────────


async def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found.")
        print("Run 'make prepare-dataset' first.")
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

    print(f"  LLM-as-Judge Evaluation")
    print(f"  Judge:   {model}")
    print(f"  API:     {base_url}")
    print(f"  Dataset: {DATASET_PATH.name} ({total} examples)")
    print()

    results: list[JudgeResult] = []

    for i, sample in enumerate(samples):
        doc = sample["report"]
        ref = sample["summary"]
        bucket = sample["bucket"]
        print(f"[{i + 1}/{total}] idx={sample['idx']}  bucket={bucket}  {len(doc):,} chars")

        # 1. Baseline summary
        t0 = time.time()
        baseline_out = await generate_baseline(doc)
        t_baseline = time.time() - t0

        # 2. Agent summary
        t0 = time.time()
        agent_out = await generate_agent(doc)
        t_agent = time.time() - t0

        print(f"  Baseline: {len(baseline_out):,} chars  ({t_baseline:.1f}s)")
        print(f"  Agent:    {len(agent_out):,} chars  ({t_agent:.1f}s)")

        # 3. ROUGE scores (vs reference)
        b_rouge = compute_rouge(baseline_out, ref)
        a_rouge = compute_rouge(agent_out, ref)
        print(f"  ROUGE-1:  baseline={b_rouge.rouge1:.4f}  agent={a_rouge.rouge1:.4f}")

        result = JudgeResult(
            idx=sample["idx"],
            bucket=bucket,
            doc_chars=len(doc),
            baseline_summary=baseline_out,
            agent_summary=agent_out,
            baseline_time=t_baseline,
            agent_time=t_agent,
            baseline_rouge=b_rouge,
            agent_rouge=a_rouge,
        )

        # 4. Judge scores
        scores = judge.score(
            source_document=truncate_for_judge(doc),
            reference=ref,
            summary_a=baseline_out,
            summary_b=agent_out,
        )
        if scores:
            result.baseline_scores, result.agent_scores = scores
            print(f"  Judge:    baseline={result.baseline_scores.mean():.1f}  "
                  f"agent={result.agent_scores.mean():.1f}")
        else:
            print(f"  Judge:    FAILED")

        results.append(result)
        print()

    print_judge_report(results)
    save_judge_results(results)


if __name__ == "__main__":
    asyncio.run(main())
