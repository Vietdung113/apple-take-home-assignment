"""Evaluate 3 systems: base model, fine-tuned model, agentic pipeline.

Metrics:
- ROUGE-1, ROUGE-2, ROUGE-L (vs reference summary)
- LLM-as-judge (Qwen2.5-32B): coverage, specificity, consistency, conciseness

Config (via .env):
    BASE_MODEL_URL      — Base model server (port 8100)
    FINETUNED_MODEL_URL — Fine-tuned model server (port 8200)
    AGENT_API_URL       — Agent pipeline API (uses fine-tuned model)
    JUDGE_URL           — Judge model server (port 8001)
    JUDGE_MODEL         — Judge model name

Usage:
    cd eval
    uv run python eval_all.py
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

from common import RougeScores, compute_rouge, generate_summary, load_dotenv

load_dotenv()

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

# ── Config ───────────────────────────────────────────────────────────────

BASE_MODEL_URL = os.environ.get("BASE_MODEL_URL", "http://localhost:8100")
FINETUNED_MODEL_URL = os.environ.get("FINETUNED_MODEL_URL", "http://localhost:8200")
AGENT_API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8300")
JUDGE_URL = os.environ.get("JUDGE_URL", "http://localhost:8001")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")


# ── Judge scoring ────────────────────────────────────────────────────────

DIMENSIONS: list[str] = ["coverage", "specificity", "consistency", "conciseness"]
WEIGHTS: dict[str, float] = {
    "coverage": 0.30,
    "specificity": 0.30,
    "consistency": 0.25,
    "conciseness": 0.15,
}

JUDGE_PROMPT = """\
You are evaluating three machine-generated summaries of a government report.

## Source Document (truncated)
{source_document}

## Reference Summary (human-written)
{reference}

## Summary A (Base Model)
{summary_a}

## Summary B (Fine-tuned Model)
{summary_b}

## Summary C (Agentic Pipeline)
{summary_c}

## Task
Score each summary on 4 dimensions (1=worst, 5=best):

1. **Coverage**: Does it answer who, what, where, when, why? All key events from reference?
2. **Specificity**: Uses concrete names, numbers, dates vs vague language?
3. **Consistency**: All facts accurate vs source? Any hallucinations?
4. **Conciseness**: Gets to the point without filler or repetition?

Respond with JSON only (no markdown, no explanation):
{{"summary_a": {{"coverage": <int>, "specificity": <int>, "consistency": <int>, "conciseness": <int>}}, \
"summary_b": {{"coverage": <int>, "specificity": <int>, "consistency": <int>, "conciseness": <int>}}, \
"summary_c": {{"coverage": <int>, "specificity": <int>, "consistency": <int>, "conciseness": <int>}}}}"""


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
    finetuned_summary: str
    agent_summary: str
    base_time: float
    finetuned_time: float
    agent_time: float
    base_rouge: RougeScores
    finetuned_rouge: RougeScores
    agent_rouge: RougeScores
    base_scores: Scores | None = None
    finetuned_scores: Scores | None = None
    agent_scores: Scores | None = None


# ── Judge client ─────────────────────────────────────────────────────────


def truncate_for_judge(doc: str, max_chars: int = 4000) -> str:
    """Truncate source document to save judge tokens."""
    if len(doc) <= max_chars:
        return doc
    half = max_chars // 2
    return doc[:half] + "\n\n[... truncated ...]\n\n" + doc[-half:]


async def judge_score(
    source_document: str,
    reference: str,
    summary_a: str,
    summary_b: str,
    summary_c: str,
    retries: int = 3,
) -> tuple[Scores, Scores, Scores] | None:
    """Ask judge to score 3 summaries. Returns (scores_a, scores_b, scores_c)."""
    prompt = JUDGE_PROMPT.format(
        source_document=truncate_for_judge(source_document),
        reference=reference,
        summary_a=summary_a,
        summary_b=summary_b,
        summary_c=summary_c,
    )
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise evaluator. Respond with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(f"{JUDGE_URL}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                rjson = resp.json()
                content = rjson["choices"][0]["message"]["content"].strip()

                if not content:
                    raise ValueError("Judge returned empty content")

                # Strip markdown if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                # Extract JSON
                start = content.find("{")
                end = content.rfind("}") + 1
                if start == -1 or end == 0:
                    raise ValueError(f"No JSON in response: {content[:200]}")

                data = json.loads(content[start:end])
                return (
                    Scores(**{d: data["summary_a"][d] for d in DIMENSIONS}),
                    Scores(**{d: data["summary_b"][d] for d in DIMENSIONS}),
                    Scores(**{d: data["summary_c"][d] for d in DIMENSIONS}),
                )
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"    Judge error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)

    return None


# ── Agent client ─────────────────────────────────────────────────────────


async def generate_agent_summary(document: str) -> str:
    """Call agentic pipeline via POST /summarize."""
    async with httpx.AsyncClient(base_url=AGENT_API_URL, timeout=300.0) as client:
        resp = await client.post("/summarize", json={"document": document})
        resp.raise_for_status()
        data = resp.json()
        return data["summary"].strip()


# ── Report printing ──────────────────────────────────────────────────────

ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]


def _print_bucket(results: list[EvalResult], label: str):
    """Print ROUGE + Judge scores for a bucket."""
    n = len(results)
    if not n:
        return

    print(f"\n  ── {label} ({n} examples) ──")

    # ROUGE
    print(f"\n  {'Metric':<12} {'Base':>10} {'Finetuned':>10} {'Agent':>10}")
    print(f"  {'-'*44}")
    for metric in ROUGE_METRICS:
        b = sum(getattr(r.base_rouge, metric) for r in results) / n
        f = sum(getattr(r.finetuned_rouge, metric) for r in results) / n
        a = sum(getattr(r.agent_rouge, metric) for r in results) / n
        print(f"  {metric:<12} {b:>10.4f} {f:>10.4f} {a:>10.4f}")

    # Judge
    judge_results = [r for r in results if r.base_scores and r.finetuned_scores and r.agent_scores]
    if judge_results:
        print(f"\n  {'Dimension':<12} {'Base':>10} {'Finetuned':>10} {'Agent':>10}")
        print(f"  {'-'*44}")
        for d in DIMENSIONS:
            b = sum(getattr(r.base_scores, d) for r in judge_results) / len(judge_results)
            f = sum(getattr(r.finetuned_scores, d) for r in judge_results) / len(judge_results)
            a = sum(getattr(r.agent_scores, d) for r in judge_results) / len(judge_results)
            print(f"  {d:<12} {b:>10.2f} {f:>10.2f} {a:>10.2f}")

        print(f"  {'-'*44}")
        b_w = sum(r.base_scores.weighted() for r in judge_results) / len(judge_results)
        f_w = sum(r.finetuned_scores.weighted() for r in judge_results) / len(judge_results)
        a_w = sum(r.agent_scores.weighted() for r in judge_results) / len(judge_results)
        print(f"  {'Weighted':<12} {b_w:>10.2f} {f_w:>10.2f} {a_w:>10.2f}")


def print_report(results: list[EvalResult]):
    """Print per-bucket + overall results."""
    print(f"\n{'='*60}")
    print(f"  3-WAY EVALUATION  ({len(results)} examples)")
    print(f"{'='*60}")

    for bucket in ["SHORT", "MEDIUM", "LONG"]:
        bucket_results = [r for r in results if r.bucket == bucket.lower()]
        _print_bucket(bucket_results, bucket)

    _print_bucket(results, "OVERALL")

    # Timing
    if results:
        avg_b = sum(r.base_time for r in results) / len(results)
        avg_f = sum(r.finetuned_time for r in results) / len(results)
        avg_a = sum(r.agent_time for r in results) / len(results)
        print(f"\n  {'Avg time (s)':<12} {avg_b:>10.1f} {avg_f:>10.1f} {avg_a:>10.1f}")
    print()


def save_results(results: list[EvalResult], path: str = "eval_all_results.json"):
    """Save results to JSON."""
    data = []
    for r in results:
        entry = {
            "idx": r.idx,
            "bucket": r.bucket,
            "doc_chars": r.doc_chars,
            "base_time": r.base_time,
            "finetuned_time": r.finetuned_time,
            "agent_time": r.agent_time,
            "base_rouge": {
                "rouge1": r.base_rouge.rouge1,
                "rouge2": r.base_rouge.rouge2,
                "rougeL": r.base_rouge.rougeL,
            },
            "finetuned_rouge": {
                "rouge1": r.finetuned_rouge.rouge1,
                "rouge2": r.finetuned_rouge.rouge2,
                "rougeL": r.finetuned_rouge.rougeL,
            },
            "agent_rouge": {
                "rouge1": r.agent_rouge.rouge1,
                "rouge2": r.agent_rouge.rouge2,
                "rougeL": r.agent_rouge.rougeL,
            },
        }
        if r.base_scores:
            entry["base_judge"] = {d: getattr(r.base_scores, d) for d in DIMENSIONS}
        if r.finetuned_scores:
            entry["finetuned_judge"] = {d: getattr(r.finetuned_scores, d) for d in DIMENSIONS}
        if r.agent_scores:
            entry["agent_judge"] = {d: getattr(r.agent_scores, d) for d in DIMENSIONS}
        data.append(entry)

    Path(path).write_text(json.dumps(data, indent=2))
    print(f"  Results saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────


async def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found.")
        print("Run: uv run python prepare_dataset.py")
        sys.exit(1)

    print(f"  3-Way Evaluation")
    print(f"  Base model:      {BASE_MODEL_URL}")
    print(f"  Fine-tuned:      {FINETUNED_MODEL_URL}")
    print(f"  Agent pipeline:  {AGENT_API_URL}")
    print(f"  Judge:           {JUDGE_URL} ({JUDGE_MODEL})")
    print()

    all_samples = json.loads(DATASET_PATH.read_text())
    n = int(os.environ.get("EVAL_N", len(all_samples)))
    samples = all_samples[:n]
    total = len(samples)

    results: list[EvalResult] = []

    for i, sample in enumerate(samples):
        doc = sample["report"]
        ref = sample["summary"]
        bucket = sample["bucket"]
        print(f"[{i + 1}/{total}] idx={sample['idx']} bucket={bucket} {len(doc):,} chars")

        # 1. Base model
        t0 = time.time()
        base_out = await generate_summary(BASE_MODEL_URL, doc)
        t_base = time.time() - t0

        # 2. Fine-tuned model
        t0 = time.time()
        finetuned_out = await generate_summary(FINETUNED_MODEL_URL, doc)
        t_finetuned = time.time() - t0

        # 3. Agent pipeline
        t0 = time.time()
        agent_out = await generate_agent_summary(doc)
        t_agent = time.time() - t0

        print(f"  Base:      {len(base_out):,} chars ({t_base:.1f}s)")
        print(f"  Finetuned: {len(finetuned_out):,} chars ({t_finetuned:.1f}s)")
        print(f"  Agent:     {len(agent_out):,} chars ({t_agent:.1f}s)")

        # ROUGE
        b_rouge = compute_rouge(base_out, ref)
        f_rouge = compute_rouge(finetuned_out, ref)
        a_rouge = compute_rouge(agent_out, ref)
        print(f"  ROUGE-L: base={b_rouge.rougeL:.3f} finetuned={f_rouge.rougeL:.3f} agent={a_rouge.rougeL:.3f}")

        result = EvalResult(
            idx=sample["idx"],
            bucket=bucket,
            doc_chars=len(doc),
            base_summary=base_out,
            finetuned_summary=finetuned_out,
            agent_summary=agent_out,
            base_time=t_base,
            finetuned_time=t_finetuned,
            agent_time=t_agent,
            base_rouge=b_rouge,
            finetuned_rouge=f_rouge,
            agent_rouge=a_rouge,
        )

        # Judge
        scores = await judge_score(doc, ref, base_out, finetuned_out, agent_out)
        if scores:
            result.base_scores, result.finetuned_scores, result.agent_scores = scores
            print(f"  Judge:   base={result.base_scores.mean():.1f} "
                  f"finetuned={result.finetuned_scores.mean():.1f} "
                  f"agent={result.agent_scores.mean():.1f}")
        else:
            print(f"  Judge:   FAILED")

        results.append(result)
        print()

    print_report(results)
    save_results(results)


if __name__ == "__main__":
    asyncio.run(main())
