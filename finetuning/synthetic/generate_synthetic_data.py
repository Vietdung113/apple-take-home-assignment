"""Generate synthetic summaries for low-quality GovReport samples using QwQ-32B.

This script:
1. Loads GovReport dataset
2. Filters out samples that already have high scores (coverage ≥4 AND specificity ≥4)
3. Uses QwQ-32B to generate new summaries for low-quality samples
4. Removes thinking tokens from QwQ output
5. Processes in parallel for speed
6. Saves results to synthetic_summaries.jsonl

Usage:
    # Generate synthetics with 10 workers
    python generate_synthetic_data.py --workers 10

    # Test with 100 samples first
    python generate_synthetic_data.py --max-samples 100 --workers 5

    # Resume from checkpoint
    python generate_synthetic_data.py --workers 10
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment
load_dotenv()

# Paths
SCORES_FILE = Path(__file__).parent.parent / "data" / "scores_progress.jsonl"
OUTPUT_FILE = Path(__file__).parent / "synthetic_summaries.jsonl"
CHECKPOINT_FILE = Path(__file__).parent / "synthetic_progress.jsonl"

# NVIDIA API
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in .env file")


def load_existing_scores():
    """Load existing scores to filter out high-quality samples."""
    high_quality_ids = set()

    if not SCORES_FILE.exists():
        print(f"Warning: Scores file not found at {SCORES_FILE}")
        return high_quality_ids

    with open(SCORES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            score = json.loads(line)
            if score.get("success") and score.get("coverage", 0) >= 4 and score.get("specificity", 0) >= 4:
                high_quality_ids.add(score["sample_id"])

    print(f"Loaded {len(high_quality_ids)} high-quality sample IDs to skip")
    return high_quality_ids


def load_processed_samples():
    """Load already processed samples from checkpoint."""
    processed_ids = set()

    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                processed_ids.add(sample["sample_id"])
        print(f"Resuming: {len(processed_ids)} samples already processed")

    return processed_ids


def remove_thinking_tokens(text: str) -> str:
    """Remove thinking tokens from QwQ output.

    QwQ often outputs thinking process in tags like <think>...</think> or similar.
    We want to extract only the final summary.
    """
    # Remove explicit thinking tags if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # QwQ typically outputs thinking followed by final answer
    # Look for patterns like "So, the summary is:" or "In summary:"
    # and take everything after that

    # Common QwQ patterns
    patterns = [
        r'(?:So,?\s*(?:the\s+)?(?:final\s+)?(?:answer|summary)\s+is[:\s]+)(.*)',
        r'(?:In\s+summary[:\s]+)(.*)',
        r'(?:Final\s+summary[:\s]+)(.*)',
        r'(?:Summary[:\s]+)(.*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            break

    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = text.strip()

    return text


def generate_summary_with_qwq(document: str, sample_id: str) -> dict | None:
    """Generate summary using QwQ-32B via NVIDIA API."""

    prompt = f"""Summarize the following government report. Requirements:

1. COVERAGE: Include ALL major sections and key points comprehensively. Cover every important topic discussed in the report.

2. SPECIFICITY: Preserve specific details such as:
   - Numbers, percentages, and statistics
   - Names of people, organizations, and places
   - Dates and time periods
   - Concrete findings and recommendations

3. CONCISENESS: Keep the summary focused and under 500 words.

Report:
{document}

Provide a comprehensive, specific summary:"""

    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )

        completion = client.chat.completions.create(
            model="qwen/qwq-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.7,
            max_tokens=4096,
            stream=True
        )

        # Collect streamed response
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content

        # Remove thinking tokens
        summary = remove_thinking_tokens(full_response)

        if not summary:
            print(f"Warning: Empty summary after removing thinking tokens for sample {sample_id}")
            return None

        return {
            "sample_id": sample_id,
            "summary": summary,
            "raw_output": full_response,  # Keep raw for debugging
            "model": "qwen/qwq-32b",
            "success": True
        }

    except Exception as e:
        print(f"Error generating summary for sample {sample_id}: {e}")
        return {
            "sample_id": sample_id,
            "success": False,
            "error": str(e)
        }


def process_sample(sample_data: tuple) -> dict | None:
    """Process a single sample (wrapper for parallel execution)."""
    idx, sample, high_quality_ids, processed_ids = sample_data

    sample_id = f"train_{idx}"

    # Skip if already high quality
    if sample_id in high_quality_ids:
        return None

    # Skip if already processed
    if sample_id in processed_ids:
        return None

    document = sample["report"]

    # Generate summary
    result = generate_summary_with_qwq(document, sample_id)

    if result and result.get("success"):
        result["document"] = document

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic summaries with QwQ-32B")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples to process (for testing)")
    args = parser.parse_args()

    print("="*80)
    print("Synthetic Summary Generation with QwQ-32B")
    print("="*80)

    # Load existing scores to filter out high-quality samples
    high_quality_ids = load_existing_scores()

    # Load checkpoint
    processed_ids = load_processed_samples()

    # Load GovReport dataset
    print("\nLoading GovReport dataset...")
    govreport = load_dataset("ccdv/govreport-test", trust_remote_code=True)
    train_data = govreport["train"]

    print(f"Total GovReport samples: {len(train_data)}")
    print(f"High-quality samples to skip: {len(high_quality_ids)}")
    print(f"Already processed: {len(processed_ids)}")

    # Prepare samples to process
    samples_to_process = []
    for idx in range(len(train_data)):
        sample_id = f"train_{idx}"
        if sample_id not in high_quality_ids and sample_id not in processed_ids:
            samples_to_process.append((idx, train_data[idx], high_quality_ids, processed_ids))

    print(f"Samples to process: {len(samples_to_process)}")

    # Limit if requested
    if args.max_samples and args.max_samples < len(samples_to_process):
        samples_to_process = samples_to_process[:args.max_samples]
        print(f"Limited to: {args.max_samples} samples")

    # Open checkpoint file for appending
    checkpoint_file = open(CHECKPOINT_FILE, "a")

    # Process in parallel
    print(f"\nGenerating summaries with {args.workers} workers...")
    print(f"Model: QwQ-32B (NVIDIA API)")
    print(f"Output: {CHECKPOINT_FILE}")
    print()

    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, sample): sample for sample in samples_to_process}

        with tqdm(total=len(samples_to_process), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()

                if result:
                    # Save to checkpoint
                    checkpoint_file.write(json.dumps(result) + "\n")
                    checkpoint_file.flush()

                    if result.get("success"):
                        success_count += 1
                    else:
                        error_count += 1

                pbar.update(1)
                pbar.set_postfix({"success": success_count, "errors": error_count})

    checkpoint_file.close()

    # Filter and save final output (only successful generations)
    print(f"\nFiltering successful generations...")
    final_summaries = []

    with open(CHECKPOINT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            result = json.loads(line)
            if result.get("success"):
                # Keep only essential fields for final output
                final_summaries.append({
                    "sample_id": result["sample_id"],
                    "document": result["document"],
                    "summary": result["summary"],
                    "model": result["model"]
                })

    # Save final output
    with open(OUTPUT_FILE, "w") as f:
        for summary in final_summaries:
            f.write(json.dumps(summary) + "\n")

    print(f"\n{'='*80}")
    print(f"Generation complete!")
    print(f"{'='*80}")
    print(f"Total processed: {success_count + error_count}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {success_count/(success_count+error_count)*100:.1f}%")
    print()
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"Checkpoint saved to: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
