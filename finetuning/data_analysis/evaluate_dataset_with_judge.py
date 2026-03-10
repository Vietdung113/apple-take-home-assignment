"""Evaluate GovReport dataset quality using LLM-as-judge.

This evaluates the REFERENCE SUMMARIES themselves (not model outputs):
1. Are references comprehensive? (Coverage vs document)
2. Are references specific? (Include numbers, dates, names)
3. Are references consistent? (No hallucinations)
4. Are references concise? (Well-structured, appropriate length)

This helps answer: "Are the reference summaries high quality ground truth?"

Usage:
    export NVIDIA_API_KEY=your_key
    python evaluate_dataset_with_judge.py --num-samples 50
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
import yaml


# ── Configuration ────────────────────────────────────────────────────────

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable not set")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
JUDGE_MODEL_NAME = "nvidia/nemotron-nano-instruct-9b"

# Load judge prompt template
config_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

JUDGE_PROMPT_TEMPLATE = config["judge_prompt"]


# ── Helper Functions ─────────────────────────────────────────────────────

async def judge_reference_quality(document: str, reference: str, semaphore: asyncio.Semaphore) -> dict:
    """Judge the quality of a reference summary.

    Note: We pass reference as both 'reference' and 'generated' to judge it
    against the document, treating it as if a model generated it.
    """
    async with semaphore:  # Rate limiting
        # Evaluate reference as if it were a generated summary
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            document=document,
            reference="",  # Empty reference
            generated=reference  # Judge the reference itself
        )

        # Simplify prompt: remove reference summary section
        prompt = prompt.replace("**Reference Summary:**\n\n\n**Generated Summary:**", "**Summary to Evaluate:**")

        async with httpx.AsyncClient(base_url=NVIDIA_BASE_URL, timeout=120.0) as client:
            payload = {
                "model": JUDGE_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.0,
            }
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            }

            try:
                resp = await client.post("/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                content = data["choices"][0]["message"]["content"].strip()

                # Parse JSON response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                result = json.loads(content)

                if not all(k in result for k in ["coverage", "specificity", "consistency", "conciseness"]):
                    return {"success": False, "error": f"Missing keys: {result}"}

                return {
                    "coverage": result["coverage"],
                    "specificity": result["specificity"],
                    "consistency": result["consistency"],
                    "conciseness": result["conciseness"],
                    "explanation": result.get("explanation", ""),
                    "success": True
                }

            except Exception as e:
                return {"success": False, "error": str(e)}


async def evaluate_samples(samples: list, max_concurrent: int = 3) -> list:
    """Evaluate multiple samples concurrently with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for sample_id, doc, ref in samples:
        task = judge_reference_quality(doc, ref, semaphore)
        tasks.append((sample_id, task))

    results = []
    for sample_id, task in tqdm_asyncio.as_completed(
        [(sid, t) for sid, t in tasks],
        total=len(tasks),
        desc="Evaluating references"
    ):
        result = await task
        results.append({
            "sample_id": sample_id,
            **result
        })
        await asyncio.sleep(0.5)  # Rate limit

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GovReport reference quality with LLM judge")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--output", default="data_analysis/reference_quality_scores.json", help="Output file")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent API calls")
    args = parser.parse_args()

    print("="*80)
    print("GovReport Reference Quality Evaluation")
    print("="*80)

    # Load dataset
    print("\nLoading GovReport dataset...")
    dataset = load_dataset("ccdv/govreport-summarization", split="train", trust_remote_code=True)

    # Sample random examples
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    print(f"Selected {len(sample_indices)} random samples")

    # Prepare samples
    samples = []
    for idx in sample_indices:
        sample = dataset[idx]
        samples.append((idx, sample["report"], sample["summary"]))

    # Evaluate with judge
    print(f"\nEvaluating references with {JUDGE_MODEL_NAME}...")
    print(f"(Rate limited to {args.max_concurrent} concurrent calls)\n")

    results = asyncio.run(evaluate_samples(samples, args.max_concurrent))

    # Aggregate statistics
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nSuccessfully evaluated: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        import numpy as np

        coverage_scores = [r["coverage"] for r in successful]
        specificity_scores = [r["specificity"] for r in successful]
        consistency_scores = [r["consistency"] for r in successful]
        conciseness_scores = [r["conciseness"] for r in successful]

        print(f"\n{'='*80}")
        print("REFERENCE SUMMARY QUALITY SCORES")
        print(f"{'='*80}")

        metrics = {
            "Coverage": coverage_scores,
            "Specificity": specificity_scores,
            "Consistency": consistency_scores,
            "Conciseness": conciseness_scores,
        }

        for metric_name, scores in metrics.items():
            mean = np.mean(scores)
            median = np.median(scores)
            std = np.std(scores)

            # Distribution
            dist = {i: scores.count(i) for i in range(1, 6)}

            print(f"\n{metric_name}:")
            print(f"  Mean: {mean:.2f} ± {std:.2f}")
            print(f"  Median: {median:.1f}")
            print(f"  Distribution: {dist}")

        # Overall quality
        overall_scores = [
            (r["coverage"] + r["specificity"] + r["consistency"] + r["conciseness"]) / 4
            for r in successful
        ]
        print(f"\nOverall Quality (average of 4 dimensions):")
        print(f"  Mean: {np.mean(overall_scores):.2f}")
        print(f"  Median: {np.median(overall_scores):.2f}")

        # Quality classification
        high_quality = sum(1 for s in overall_scores if s >= 4.0)
        medium_quality = sum(1 for s in overall_scores if 3.0 <= s < 4.0)
        low_quality = sum(1 for s in overall_scores if s < 3.0)

        print(f"\n{'='*80}")
        print("QUALITY CLASSIFICATION")
        print(f"{'='*80}")
        print(f"High quality (≥4.0): {high_quality}/{len(successful)} ({high_quality/len(successful)*100:.1f}%)")
        print(f"Medium quality (3.0-3.9): {medium_quality}/{len(successful)} ({medium_quality/len(successful)*100:.1f}%)")
        print(f"Low quality (<3.0): {low_quality}/{len(successful)} ({low_quality/len(successful)*100:.1f}%)")

        # Worst samples
        print(f"\n{'='*80}")
        print("WORST REFERENCE SUMMARIES")
        print(f"{'='*80}")

        worst = sorted(successful, key=lambda x: (x["coverage"] + x["specificity"]) / 2)[:5]
        for i, sample in enumerate(worst, 1):
            score = (sample["coverage"] + sample["specificity"]) / 2
            print(f"\n{i}. Sample {sample['sample_id']} - Avg(Coverage,Specificity): {score:.1f}")
            print(f"   Coverage: {sample['coverage']}/5")
            print(f"   Specificity: {sample['specificity']}/5")
            print(f"   Consistency: {sample['consistency']}/5")
            print(f"   Conciseness: {sample['conciseness']}/5")
            if sample.get("explanation"):
                print(f"   Explanation: {sample['explanation'][:150]}...")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "metadata": {
            "num_samples": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "judge_model": JUDGE_MODEL_NAME,
        },
        "aggregate_stats": {
            "coverage": {"mean": float(np.mean(coverage_scores)), "median": float(np.median(coverage_scores))},
            "specificity": {"mean": float(np.mean(specificity_scores)), "median": float(np.median(specificity_scores))},
            "consistency": {"mean": float(np.mean(consistency_scores)), "median": float(np.median(consistency_scores))},
            "conciseness": {"mean": float(np.mean(conciseness_scores)), "median": float(np.median(conciseness_scores))},
            "overall": {"mean": float(np.mean(overall_scores)), "median": float(np.median(overall_scores))},
        } if successful else {},
        "quality_classification": {
            "high_quality_pct": high_quality / len(successful) * 100 if successful else 0,
            "medium_quality_pct": medium_quality / len(successful) * 100 if successful else 0,
            "low_quality_pct": low_quality / len(successful) * 100 if successful else 0,
        } if successful else {},
        "detailed_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Detailed results saved to: {output_path}")

    # Final conclusions
    if successful:
        print(f"\n{'='*80}")
        print("CONCLUSIONS")
        print(f"{'='*80}")

        avg_coverage = np.mean(coverage_scores)
        avg_specificity = np.mean(specificity_scores)
        avg_consistency = np.mean(consistency_scores)

        print(f"\n1. Reference Summary Quality:")
        print(f"   - Average coverage: {avg_coverage:.2f}/5")
        print(f"   - Average specificity: {avg_specificity:.2f}/5")
        print(f"   - Average consistency: {avg_consistency:.2f}/5")

        if avg_coverage < 3.5 or avg_specificity < 3.5:
            print(f"\n2. Quality Issues Detected:")
            if avg_coverage < 3.5:
                print(f"   ⚠️  Low coverage: References may miss key points from documents")
            if avg_specificity < 3.5:
                print(f"   ⚠️  Low specificity: References lack specific details")
            print(f"   → Training on these references may teach model to be incomplete")
        else:
            print(f"\n2. Reference Quality: GOOD")
            print(f"   ✅ References are comprehensive and specific")
            print(f"   → Good ground truth for training")

        if avg_consistency < 4.0:
            print(f"\n3. Consistency Issues:")
            print(f"   ⚠️  Some references may contain hallucinations or external facts")
            print(f"   → This explains low ROUGE scores and judge score mismatches")


if __name__ == "__main__":
    main()
