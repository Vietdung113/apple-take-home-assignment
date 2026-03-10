"""Generate synthetic summaries for low-quality samples using Nemotron Nano 9B.

For samples where reference summaries scored low on coverage (<4) or specificity (<4),
generate new summaries and score them. Keep only those passing quality filter.

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --max-samples 100  # For testing
    python generate_synthetic_data.py --workers 5        # Adjust concurrency
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found. Please set it in .env file")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GENERATE_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"
JUDGE_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"

DATA_DIR = Path(__file__).parent / "data"
SCORES_FILE = DATA_DIR / "scores_progress.jsonl"
OUTPUT_FILE = DATA_DIR / "synthetic_summaries.jsonl"
PROGRESS_FILE = DATA_DIR / "synthetic_progress.jsonl"

GENERATE_PROMPT = """/no_think

Summarize the following government report comprehensively.

Requirements:
1. COVERAGE: Include ALL major sections, findings, and recommendations
2. SPECIFICITY: Preserve specific numbers, dollar amounts, dates, names, statistics, and concrete details
3. STRUCTURE: Organize as: context/background → key findings → conclusions/recommendations
4. LENGTH: Match summary length to document complexity (200-600 words)

Government Report:
{document}

Summary:"""

JUDGE_PROMPT = """/no_think

Evaluate this government report summary on two dimensions:

1. COVERAGE (1-5): Does the summary comprehensively cover the key points?
2. SPECIFICITY (1-5): Does it include specific details (numbers, names, dates)?

Document:
{document}

Summary:
{summary}

Output only:
Coverage: X
Specificity: Y
"""


# ── Functions ────────────────────────────────────────────────────────────


def load_low_quality_sample_ids() -> set[int]:
    """Load sample IDs that failed quality filter (coverage<4 OR specificity<4)."""
    low_quality_ids = set()
    with open(SCORES_FILE) as f:
        for line in f:
            score = json.loads(line)
            if score.get("success") and (
                score.get("coverage", 0) < 4 or score.get("specificity", 0) < 4
            ):
                low_quality_ids.add(score["sample_id"])
    return low_quality_ids


def load_progress() -> dict[int, dict]:
    """Load already-processed synthetic samples."""
    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            for line in f:
                item = json.loads(line)
                progress[item["sample_id"]] = item
    return progress


async def generate_summary(
    client: httpx.AsyncClient, document: str, sample_id: int
) -> str | None:
    """Generate a summary using Nemotron."""
    try:
        response = await client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            json={
                "model": GENERATE_MODEL,
                "messages": [
                    {"role": "user", "content": GENERATE_PROMPT.format(document=document)}
                ],
                "temperature": 0.5,
                "max_tokens": 2048,
                "top_p": 0.9,
            },
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"].get("content", "")
        return content.strip() if content else None
    except Exception as e:
        print(f"  [GEN FAIL] Sample {sample_id}: {e}")
        return None


async def judge_summary(
    client: httpx.AsyncClient, document: str, summary: str, sample_id: int
) -> dict:
    """Score a synthetic summary on coverage + specificity."""
    try:
        response = await client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            json={
                "model": JUDGE_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(document=document, summary=summary),
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
                "top_p": 0.9,
            },
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"].get("content", "")

        coverage_match = re.search(r"Coverage[:\s]*(\d)", content, re.IGNORECASE)
        specificity_match = re.search(r"Specificity[:\s]*(\d)", content, re.IGNORECASE)

        return {
            "coverage": int(coverage_match.group(1)) if coverage_match else 0,
            "specificity": int(specificity_match.group(1)) if specificity_match else 0,
            "success": bool(coverage_match and specificity_match),
        }
    except Exception as e:
        print(f"  [JUDGE FAIL] Sample {sample_id}: {e}")
        return {"coverage": 0, "specificity": 0, "success": False}


async def process_sample(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    sample_id: int,
    document: str,
) -> dict | None:
    """Generate + judge one sample."""
    async with semaphore:
        # Generate
        summary = await generate_summary(client, document, sample_id)
        if not summary or len(summary.split()) < 30:
            return None

        # Judge
        scores = await judge_summary(client, document, summary, sample_id)
        if not scores["success"]:
            return None

        result = {
            "sample_id": sample_id,
            "document": document,
            "summary": summary,
            "coverage": scores["coverage"],
            "specificity": scores["specificity"],
            "source": "synthetic_nemotron9b",
        }

        # Save progress
        with open(PROGRESS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

        return result


async def main():
    parser = argparse.ArgumentParser(description="Generate synthetic summaries")
    parser.add_argument("--max-samples", type=int, help="Limit samples for testing")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent API requests")
    parser.add_argument(
        "--min-coverage", type=int, default=4, help="Min coverage to keep"
    )
    parser.add_argument(
        "--min-specificity", type=int, default=4, help="Min specificity to keep"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    # Step 1: Find low-quality sample IDs
    print("Loading scores...")
    low_quality_ids = load_low_quality_sample_ids()
    print(f"Low-quality samples (coverage<4 OR specificity<4): {len(low_quality_ids)}")

    # Step 2: Load progress
    progress = load_progress()
    print(f"Already processed: {len(progress)}")

    # Step 3: Load GovReport
    print("Loading GovReport dataset...")
    dataset = load_dataset("ccdv/govreport-summarization")
    all_samples = list(dataset["train"]) + list(dataset["validation"])
    print(f"Total GovReport samples: {len(all_samples)}")

    # Step 4: Filter to unprocessed low-quality samples
    to_process = []
    for sid in sorted(low_quality_ids):
        if sid not in progress and sid < len(all_samples):
            to_process.append((sid, all_samples[sid]["report"]))

    if args.max_samples:
        to_process = to_process[: args.max_samples]

    print(f"To process: {len(to_process)}")

    if not to_process:
        print("Nothing to process!")
    else:
        # Step 5: Generate + judge
        print(f"\nGenerating with {args.workers} concurrent workers...")
        client = httpx.AsyncClient()
        semaphore = asyncio.Semaphore(args.workers)

        tasks = [
            process_sample(client, semaphore, sid, doc) for sid, doc in to_process
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Generating")
        await client.aclose()

        # Merge with progress
        for r in results:
            if r:
                progress[r["sample_id"]] = r

    # Step 6: Filter high-quality synthetics
    print("\nFiltering high-quality synthetics...")
    high_quality = []
    for item in progress.values():
        if (
            item.get("coverage", 0) >= args.min_coverage
            and item.get("specificity", 0) >= args.min_specificity
        ):
            high_quality.append(item)

    print(f"High-quality synthetics: {len(high_quality)} / {len(progress)} ({len(high_quality)/max(len(progress),1)*100:.1f}%)")

    # Step 7: Save final output
    with open(OUTPUT_FILE, "w") as f:
        for item in high_quality:
            f.write(
                json.dumps({"document": item["document"], "summary": item["summary"]})
                + "\n"
            )

    print(f"\nSaved {len(high_quality)} synthetic samples to: {OUTPUT_FILE}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Synthetic Data Generation Complete")
    print(f"{'='*60}")
    print(f"Low-quality inputs:    {len(low_quality_ids)}")
    print(f"Total processed:       {len(progress)}")
    print(f"High-quality output:   {len(high_quality)}")
    print(f"Pass rate:             {len(high_quality)/max(len(progress),1)*100:.1f}%")
    print(f"Output:                {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
