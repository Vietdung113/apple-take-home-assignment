"""Prepare and filter SFT dataset using LLM-as-judge.

Step 1: Load GovReport train + validation, merge them
Step 2: Use LLM judge to score each sample on coverage + specificity
Step 3: Keep only high-quality samples (both metrics >= 4/5)
Step 4: Split 90/10 into train/validation

Usage:
    python prepare_data.py
    python prepare_data.py --max-samples 100  # For testing
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent / "config"))
from prompt_loader import get_training_prompt_instruct_model

# Load environment variables
load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found. Please set it in .env file")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
JUDGE_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"

JUDGE_PROMPT = """/no_think

Evaluate this government report summary compared to the original document on two dimensions:

1. COVERAGE (1-5): Does the summary comprehensively cover the key points from the document?
2. SPECIFICITY (1-5): Does the summary include specific details (numbers, names, dates, concrete findings)?

Document:
{document}

Summary:
{summary}

Output only:
Coverage: X
Specificity: Y
"""


# ── Functions ────────────────────────────────────────────────────────────


async def judge_sample(client: httpx.AsyncClient, document: str, summary: str, sample_id: int) -> dict:
    """Score coverage + specificity using LLM judge."""
    try:
        response = await client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": JUDGE_PROMPT.format(document=document, summary=summary)}],
                "temperature": 0.3,
                "max_tokens": 2048,  # Allow reasoning space
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
        message = result.get("choices", [{}])[0].get("message", {})

        # Try content first, then reasoning_content
        content = message.get("content") or message.get("reasoning_content", "")

        if not content:
            print(f"  [SKIP] Sample {sample_id}: Empty response")
            return {
                "sample_id": sample_id,
                "coverage": 0,
                "specificity": 0,
                "explanation": "Empty response",
                "success": False,
            }

        content = content.strip()

        # Parse scores from anywhere in the text (reasoning models put it at the end)
        coverage_match = re.search(r'Coverage[:\s]*(\d)', content, re.IGNORECASE)
        specificity_match = re.search(r'Specificity[:\s]*(\d)', content, re.IGNORECASE)

        coverage = int(coverage_match.group(1)) if coverage_match else 0
        specificity = int(specificity_match.group(1)) if specificity_match else 0

        return {
            "sample_id": sample_id,
            "coverage": coverage,
            "specificity": specificity,
            "explanation": content,
            "success": True,
        }
    except Exception as e:
        print(f"  [SKIP] Sample {sample_id}: {e}")
        return {
            "sample_id": sample_id,
            "coverage": 0,
            "specificity": 0,
            "explanation": str(e),
            "success": False,
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, help="Limit samples for testing")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent API requests")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    scores_checkpoint = output_dir / "scores_progress.jsonl"
    data_checkpoint = output_dir / "high_quality_data.jsonl"

    # Load GovReport
    print("Loading GovReport dataset...")
    dataset = load_dataset("ccdv/govreport-summarization")

    # Merge train + validation
    all_samples = list(dataset["train"]) + list(dataset["validation"])
    print(f"Total samples: {len(all_samples)}")

    if args.max_samples:
        all_samples = all_samples[:args.max_samples]
        print(f"Limited to {len(all_samples)} samples for testing")

    # Extract summaries for judging
    print("\nPreparing samples...")
    prepared = [(sample["report"], sample["summary"]) for sample in all_samples]

    # Load processed scores from checkpoint
    processed_scores = {}
    if scores_checkpoint.exists():
        print(f"Loading processed scores from checkpoint...")
        with open(scores_checkpoint) as f:
            for line in f:
                score = json.loads(line)
                processed_scores[score["sample_id"]] = score
        print(f"Loaded {len(processed_scores)} processed samples")

    # Calculate remaining samples
    remaining_count = len(prepared) - len(processed_scores)
    print(f"Total samples: {len(prepared)}, Processed: {len(processed_scores)}, Remaining: {remaining_count}")

    # Judge remaining samples
    if remaining_count > 0:
        print(f"\nJudging {remaining_count} samples with {args.workers} workers...")
        client = httpx.AsyncClient()
        semaphore = asyncio.Semaphore(args.workers)
        save_interval = 100  # Save progress every 100 samples

        async def judge_one(idx, doc_summary):
            # Skip if already processed
            if idx in processed_scores:
                return processed_scores[idx]

            async with semaphore:
                doc, summary = doc_summary
                result = await judge_sample(client, doc, summary, idx)

                # Save score to checkpoint file
                with open(scores_checkpoint, "a") as f:
                    f.write(json.dumps(result) + "\n")

                # Save high-quality data immediately if criteria met
                if result["success"] and result["coverage"] >= 4 and result["specificity"] >= 4:
                    # Use centralized prompt from config
                    messages_data = get_training_prompt_instruct_model(doc, summary)
                    with open(data_checkpoint, "a") as f:
                        f.write(json.dumps(messages_data) + "\n")

                return result

        tasks = [judge_one(i, ds) for i, ds in enumerate(prepared)]
        scores = await tqdm_asyncio.gather(*tasks, desc="Scoring")
        await client.aclose()
    else:
        print("\nAll samples already processed, using cached scores")
        scores = [processed_scores[i] for i in range(len(prepared))]

    # Filter high-quality samples (coverage >= 4 AND specificity >= 4)
    print("\nFiltering high-quality samples...")
    high_quality = []
    score_dist = {}

    for i, (score, (doc, summary)) in enumerate(zip(scores, prepared)):
        if score["success"]:
            c, s = score["coverage"], score["specificity"]
            score_dist[(c, s)] = score_dist.get((c, s), 0) + 1
            print(f"  Sample {i}: Coverage={c}, Specificity={s}")

            if c >= 4 and s >= 4:
                messages = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant for summarizing government reports."},
                        {"role": "user", "content": f"Summarize the following government report. Include specific details such as numbers, percentages, names of people and organizations, dates, and concrete findings. Ensure comprehensive coverage of all key points.\n\n{doc}"},
                        {"role": "assistant", "content": summary},
                    ]
                }
                high_quality.append(messages)
        else:
            print(f"  Sample {i}: FAILED - {score['explanation'][:100]}")

    print(f"High-quality samples (coverage ≥4 AND specificity ≥4): {len(high_quality)}")

    # Split 90/10
    random.shuffle(high_quality)
    split_idx = int(len(high_quality) * 0.9)
    train_samples = high_quality[:split_idx]
    val_samples = high_quality[split_idx:]

    # Save
    train_file = output_dir / "sft_train.jsonl"
    val_file = output_dir / "sft_validation.jsonl"

    with open(train_file, "w") as f:
        for messages in train_samples:
            f.write(json.dumps(messages) + "\n")

    with open(val_file, "w") as f:
        for messages in val_samples:
            f.write(json.dumps(messages) + "\n")

    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ High-Quality Dataset Created!")
    print(f"{'='*80}")
    print(f"Train: {len(train_samples)} samples → {train_file}")
    print(f"Validation: {len(val_samples)} samples → {val_file}")
    print(f"\nScore distribution:")
    for (c, s), count in sorted(score_dist.items(), key=lambda x: (-x[0][0], -x[0][1])):
        if count > 0:
            marker = "✓" if c >= 4 and s >= 4 else " "
            print(f"  {marker} Coverage={c}, Specificity={s}: {count:4d}")

    print(f"\n✅ Scores checkpoint: {scores_checkpoint}")
    print(f"✅ Data checkpoint: {data_checkpoint}")


if __name__ == "__main__":
    asyncio.run(main())
