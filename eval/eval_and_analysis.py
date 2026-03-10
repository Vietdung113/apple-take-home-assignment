"""Evaluation script: base model vs agent pipeline.

Usage:
    # Evaluate base model (ROUGE + Embedding)
    python eval_and_analysis.py --mode base --test-set test_set.jsonl --output results/base_results.csv

    # Evaluate agent pipeline
    python eval_and_analysis.py --mode agent --test-set test_set.jsonl --output results/agent_results.csv

    # All metrics including LLM-as-judge
    python eval_and_analysis.py --mode base --test-set test_set.jsonl --output results/base_results.csv --all
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# Add finetuning config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "finetuning" / "config"))
from prompt_loader import get_inference_prompt_instruct_model, get_judge_prompt, get_generation_params

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────

BASE_MODEL_URL = os.getenv("BASE_MODEL_URL", "http://localhost:8080") + "/v1"
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8001")

# Embedding model for semantic similarity
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
embedding_model = None  # Lazy load

# NVIDIA API for LLM-as-judge
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
JUDGE_MODEL_NAME = "nvidia/nvidia-nemotron-nano-9b-v2"


# ── Helper Functions ─────────────────────────────────────────────────────


async def call_base_model(document: str, max_tokens: int = None) -> dict:
    """Call base model via llama.cpp OpenAI-compatible chat API.

    Uses chat completions API with Qwen3's native chat format.
    """

    # Load generation params from config
    gen_params = get_generation_params()
    if max_tokens is None:
        max_tokens = gen_params["max_tokens"]

    # Load prompt as chat messages (instruct format)
    messages = get_inference_prompt_instruct_model(document)

    start = time.time()
    async with httpx.AsyncClient(base_url=BASE_MODEL_URL, timeout=120.0) as client:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": gen_params["temperature"],
            "top_p": gen_params["top_p"],
            "repeat_penalty": gen_params["repetition_penalty"],
            # Stop sequences from config to prevent rambling
            "stop": gen_params.get("stop", []),
        }
        resp = await client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Chat API returns "content" in message
        summary = data["choices"][0]["message"]["content"].strip()

        # Strip markdown formatting (model output has it, references don't)
        summary = re.sub(r'\*\*([^*]+)\*\*', r'\1', summary)
        summary = re.sub(r'^#+\s+', '', summary, flags=re.MULTILINE)
        summary = re.sub(r'\n\s*\n\s*\n+', '\n\n', summary)

        elapsed = time.time() - start

        return {
            "summary": summary,
            "time": elapsed,
            "chars": len(summary),
            "words": len(summary.split()),
        }


async def call_agent_pipeline(document: str) -> dict:
    """Call agent pipeline via FastAPI."""
    start = time.time()
    async with httpx.AsyncClient(base_url=AGENT_API_URL, timeout=300.0) as client:
        payload = {"document": document}
        resp = await client.post("/summarize", json=payload)
        resp.raise_for_status()
        data = resp.json()
        summary = data["summary"]
        elapsed = time.time() - start

        return {
            "summary": summary,
            "time": elapsed,
            "chars": len(summary),
            "words": len(summary.split()),
        }


async def call_judge(document: str, reference: str, generated: str) -> dict:
    """Call LLM judge (Llama 3.3 Nemotron 49B via NVIDIA API)."""
    # Load judge prompt from centralized config
    prompt = get_judge_prompt(document, reference, generated)

    prompt_tokens = len(prompt) // 4
    print(f"    Judge prompt: {len(prompt):,} chars (~{prompt_tokens:,} tokens)")

    max_retries = 2
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(base_url=NVIDIA_BASE_URL, timeout=600.0) as client:
                payload = {
                    "model": JUDGE_MODEL_NAME,
                    "messages": [
                        {"role": "user", "content": prompt}  # No system prompt for Nemotron Nano
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.3,  # Slightly higher for better evaluation
                    "top_p": 0.95,
                    "stream": False,
                }
                headers = {
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                }

                start = time.time()
                print(f"    Sending request to NVIDIA API... (attempt {attempt+1}/{max_retries})")

                resp = await client.post("/chat/completions", json=payload, headers=headers)

                elapsed = time.time() - start
                print(f"    Response received in {elapsed:.1f}s")

                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    raise ValueError(f"API error: {data['error']}")

                content = data["choices"][0]["message"]["content"]

                if not content:
                    raise ValueError("Empty content in response")

                content = content.strip()

                # Parse JSON response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                scores = json.loads(content)

                if not all(k in scores for k in ["coverage", "specificity", "consistency", "conciseness"]):
                    raise ValueError(f"Missing required keys in scores: {scores}")

                return scores

        except (httpx.TimeoutException, httpx.HTTPError) as e:
            print(f"    Retry {attempt+1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                print(f"    All retries failed, raising exception to skip sample")
                raise
            await asyncio.sleep(5)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"    Parse error (attempt {attempt+1}/{max_retries}): {e}")
            if 'content' in locals():
                print(f"    Raw response: {content[:300]}")
            if attempt == max_retries - 1:
                print(f"    All retries failed, raising exception to skip sample")
                raise
            await asyncio.sleep(5)


def get_embedding_model():
    """Lazy load embedding model."""
    global embedding_model
    if embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedding_model


def compute_embedding_similarity(reference: str, generated: str) -> dict:
    """Compute cosine similarity between reference and generated summaries using contextual embeddings.

    Returns:
        dict with 'embedding_similarity' (0-1, higher is better)
    """
    model = get_embedding_model()

    # Encode both summaries
    ref_embedding = model.encode(reference, convert_to_tensor=False)
    gen_embedding = model.encode(generated, convert_to_tensor=False)

    # Compute cosine similarity
    cosine_sim = np.dot(ref_embedding, gen_embedding) / (
        np.linalg.norm(ref_embedding) * np.linalg.norm(gen_embedding)
    )

    return {
        "embedding_similarity": float(cosine_sim)
    }


def compute_rouge(reference: str, generated: str) -> dict:
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "rouge1_f": scores['rouge1'].fmeasure,
        "rouge2_f": scores['rouge2'].fmeasure,
        "rougeL_f": scores['rougeL'].fmeasure,
    }


def load_test_set(test_set_path: Path) -> list[dict]:
    """Load test set from JSONL."""
    samples = []
    with open(test_set_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


async def evaluate_sample(
    sample: dict,
    sample_id: int,
    mode: str,
    use_judge: bool = False,
    use_rouge_embedding: bool = True,
) -> dict:
    """Evaluate one sample."""
    # Support both "document" (assessment format) and "report" (legacy format)
    document = sample["document"]
    reference = sample["summary"]

    print(f"\n[{sample_id}] Evaluating...")
    print(f"  Category: {sample.get('category', 'unknown')}")
    print(f"  Document: {len(document):,} chars, {len(document.split()):,} words")

    result = {
        "sample_id": sample_id,
        "category": sample.get("category", "unknown"),
        "total_tokens": sample.get("total_tokens", 0),
        "document_length": len(document.split()),
        "reference_summary": reference,
    }

    # Generate summaries based on mode
    if mode in ["base", "both"]:
        print(f"  Generating base summary...")
        base_result = await call_base_model(document)
        result["base_summary"] = base_result["summary"]
        result["base_time"] = base_result["time"]
        result["base_words"] = base_result["words"]
        print(f"    Base: {base_result['words']} words in {base_result['time']:.1f}s")

        # ROUGE + Embedding (if enabled)
        if use_rouge_embedding:
            base_rouge = compute_rouge(reference, base_result["summary"])
            result.update({f"base_{k}": v for k, v in base_rouge.items()})

            base_embedding = compute_embedding_similarity(reference, base_result["summary"])
            result.update({f"base_{k}": v for k, v in base_embedding.items()})

            print(f"    ROUGE-L: {base_rouge['rougeL_f']:.3f}, Embedding: {base_embedding['embedding_similarity']:.3f}")

    if mode in ["agent", "both"]:
        print(f"  Generating agent summary...")
        agent_result = await call_agent_pipeline(document)
        result["agent_summary"] = agent_result["summary"]
        result["agent_time"] = agent_result["time"]
        result["agent_words"] = agent_result["words"]
        print(f"    Agent: {agent_result['words']} words in {agent_result['time']:.1f}s")

        # ROUGE + Embedding (if enabled)
        if use_rouge_embedding:
            agent_rouge = compute_rouge(reference, agent_result["summary"])
            result.update({f"agent_{k}": v for k, v in agent_rouge.items()})

            agent_embedding = compute_embedding_similarity(reference, agent_result["summary"])
            result.update({f"agent_{k}": v for k, v in agent_embedding.items()})

            print(f"    ROUGE-L: {agent_rouge['rougeL_f']:.3f}, Embedding: {agent_embedding['embedding_similarity']:.3f}")

    # LLM-as-judge (if enabled)
    if use_judge:
        print(f"  Calling judge...")

        if mode in ["base", "both"]:
            base_judge = await call_judge(document, reference, result["base_summary"])
            result.update({
                "base_coverage": base_judge.get("coverage", 0),
                "base_specificity": base_judge.get("specificity", 0),
                "base_consistency": base_judge.get("consistency", 0),
                "base_conciseness": base_judge.get("conciseness", 0),
            })
            result["_base_judge_explanation"] = base_judge.get("explanation", "")

            print(f"    Judge - Coverage: {base_judge.get('coverage', 0)}/5, "
                  f"Specificity: {base_judge.get('specificity', 0)}/5, "
                  f"Consistency: {base_judge.get('consistency', 0)}/5, "
                  f"Conciseness: {base_judge.get('conciseness', 0)}/5")

        if mode in ["agent", "both"]:
            agent_judge = await call_judge(document, reference, result["agent_summary"])
            result.update({
                "agent_coverage": agent_judge.get("coverage", 0),
                "agent_specificity": agent_judge.get("specificity", 0),
                "agent_consistency": agent_judge.get("consistency", 0),
                "agent_conciseness": agent_judge.get("conciseness", 0),
            })
            result["_agent_judge_explanation"] = agent_judge.get("explanation", "")

            print(f"    Judge - Coverage: {agent_judge.get('coverage', 0)}/5, "
                  f"Specificity: {agent_judge.get('specificity', 0)}/5, "
                  f"Consistency: {agent_judge.get('consistency', 0)}/5, "
                  f"Conciseness: {agent_judge.get('conciseness', 0)}/5")

    return result


async def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization (base, agent, or both)")
    parser.add_argument(
        "--mode", choices=["base", "agent", "both"], default="both",
        help="Evaluation mode: base only, agent only, or both (default: both)"
    )
    parser.add_argument(
        "--test-set", required=True,
        help="Path to test set JSONL"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV file path"
    )

    # Metrics selection (mutually exclusive)
    metrics_group = parser.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--judge", action="store_true",
        help="Run ONLY LLM-as-judge evaluation (no ROUGE/embedding)"
    )
    metrics_group.add_argument(
        "--all", action="store_true",
        help="Run ALL metrics: ROUGE + Embedding + LLM-as-judge (slower)"
    )
    # Default (no flag): ROUGE + Embedding only

    args = parser.parse_args()

    output_file = Path(args.output)

    # Determine which metrics to run
    use_judge = args.judge or args.all
    use_rouge_embedding = not args.judge  # Skip ROUGE/embedding if --judge only

    if use_judge and not NVIDIA_API_KEY:
        print("❌ NVIDIA_API_KEY required for LLM-as-judge. Set it in .env or environment.")
        return

    # Load test set
    test_set_path = Path(args.test_set)
    print(f"Loading test set from {test_set_path}...")
    samples = load_test_set(test_set_path)
    print(f"Loaded {len(samples)} samples")

    print(f"\nMode: {args.mode}")
    if args.all:
        print(f"Metrics: ALL (ROUGE + Embedding + LLM-as-judge)")
    elif args.judge:
        print(f"Metrics: LLM-as-judge ONLY")
    else:
        print(f"Metrics: ROUGE + Embedding (default)")

    # Evaluate
    results = []
    for i, sample in enumerate(samples):
        try:
            result = await evaluate_sample(sample, i, args.mode, use_judge, use_rouge_embedding)
            results.append(result)
        except Exception as e:
            print(f"  ❌ Error on sample {i}: {e}")
            print(f"  Skipping this sample...")
            continue

    # Save full results to internal file for analysis
    df = pd.DataFrame(results)

    # Save detailed results to JSONL (for low-score analysis)
    detailed_file = output_file.parent / f"{output_file.stem}_detailed.jsonl"
    with open(detailed_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    num_success = len(results)
    num_skipped = len(samples) - num_success

    # Prepare summary statistics CSV
    summary_rows = []

    # Group by category if available
    if 'category' in df.columns:
        categories = ['8k', '16k', '32k', 'overall']
    else:
        categories = ['overall']

    for category in categories:
        if category == 'overall':
            cat_df = df
        else:
            cat_df = df[df['category'] == category]

        if len(cat_df) == 0:
            continue

        # Base model stats
        if args.mode in ["base", "both"] and 'base_summary' in cat_df.columns:
            base_row = {
                'category': category,
                'model': 'base',
                'n_samples': len(cat_df),
                'avg_time': cat_df['base_time'].mean(),
                'avg_words': cat_df['base_words'].mean(),
            }

            if use_rouge_embedding:
                base_row.update({
                    'rouge1_f': cat_df['base_rouge1_f'].mean(),
                    'rouge2_f': cat_df['base_rouge2_f'].mean(),
                    'rougeL_f': cat_df['base_rougeL_f'].mean(),
                    'embedding_similarity': cat_df['base_embedding_similarity'].mean(),
                })

            if use_judge:
                base_row.update({
                    'coverage': cat_df['base_coverage'].mean(),
                    'specificity': cat_df['base_specificity'].mean(),
                    'consistency': cat_df['base_consistency'].mean(),
                    'conciseness': cat_df['base_conciseness'].mean(),
                    'judge_avg': (cat_df['base_coverage'].mean() + cat_df['base_specificity'].mean() +
                                 cat_df['base_consistency'].mean() + cat_df['base_conciseness'].mean()) / 4,
                })

            summary_rows.append(base_row)

        # Agent model stats
        if args.mode in ["agent", "both"] and 'agent_summary' in cat_df.columns:
            agent_row = {
                'category': category,
                'model': 'agent',
                'n_samples': len(cat_df),
                'avg_time': cat_df['agent_time'].mean(),
                'avg_words': cat_df['agent_words'].mean(),
            }

            if use_rouge_embedding:
                agent_row.update({
                    'rouge1_f': cat_df['agent_rouge1_f'].mean(),
                    'rouge2_f': cat_df['agent_rouge2_f'].mean(),
                    'rougeL_f': cat_df['agent_rougeL_f'].mean(),
                    'embedding_similarity': cat_df['agent_embedding_similarity'].mean(),
                })

            if use_judge:
                agent_row.update({
                    'coverage': cat_df['agent_coverage'].mean(),
                    'specificity': cat_df['agent_specificity'].mean(),
                    'consistency': cat_df['agent_consistency'].mean(),
                    'conciseness': cat_df['agent_conciseness'].mean(),
                    'judge_avg': (cat_df['agent_coverage'].mean() + cat_df['agent_specificity'].mean() +
                                 cat_df['agent_consistency'].mean() + cat_df['agent_conciseness'].mean()) / 4,
                })

            summary_rows.append(agent_row)

    # Save summary CSV with rounded values for readability
    summary_df = pd.DataFrame(summary_rows)

    # Round numeric columns
    numeric_cols = summary_df.select_dtypes(include=['float64']).columns
    for col in numeric_cols:
        if col in ['avg_time']:
            summary_df[col] = summary_df[col].round(2)
        elif col in ['avg_words', 'n_samples']:
            summary_df[col] = summary_df[col].round(0).astype(int)
        elif col.startswith('rouge') or col == 'embedding_similarity':
            summary_df[col] = summary_df[col].round(4)
        else:  # judge scores
            summary_df[col] = summary_df[col].round(2)

    summary_df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"✅ Evaluation complete!")
    print(f"{'='*80}")
    print(f"Summary CSV: {output_file}")
    print(f"Detailed results: {detailed_file}")
    print(f"Successful: {num_success}/{len(samples)} samples")
    if num_skipped > 0:
        print(f"Skipped: {num_skipped} samples (due to errors)")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))

    # Save low-score samples if judge was used
    if use_judge and num_success > 0:
        print(f"\n{'='*80}")
        print(f"ANALYZING LOW-SCORE SAMPLES")
        print(f"{'='*80}")

        low_score_samples = []

        for mode_prefix in ["base", "agent"]:
            # Check if this mode was evaluated
            if f"{mode_prefix}_coverage" not in df.columns:
                continue

            print(f"\n{mode_prefix.upper()} mode:")

            # Calculate average judge score
            df[f'{mode_prefix}_judge_avg'] = (
                df[f'{mode_prefix}_coverage'] +
                df[f'{mode_prefix}_specificity'] +
                df[f'{mode_prefix}_consistency'] +
                df[f'{mode_prefix}_conciseness']
            ) / 4

            # Find low-score samples
            # Criteria: judge avg < 4 OR any judge dimension < 3
            low_scores = df[
                (df[f'{mode_prefix}_judge_avg'] < 4.0) |
                (df[f'{mode_prefix}_coverage'] < 3) |
                (df[f'{mode_prefix}_specificity'] < 3) |
                (df[f'{mode_prefix}_consistency'] < 3) |
                (df[f'{mode_prefix}_conciseness'] < 3)
            ]

            if len(low_scores) > 0:
                print(f"  Found {len(low_scores)} low-score samples")

                # Prepare JSON data with all relevant fields
                json_data = []
                for _, row in low_scores.iterrows():
                    sample_data = {
                        "sample_id": int(row["sample_id"]),
                        "category": row["category"],
                        "total_tokens": int(row["total_tokens"]),
                        "document_length": int(row["document_length"]),
                        "reference_summary": row["reference_summary"],
                        "generated_summary": row[f"{mode_prefix}_summary"],
                        "scores": {
                            "coverage": float(row[f"{mode_prefix}_coverage"]),
                            "specificity": float(row[f"{mode_prefix}_specificity"]),
                            "consistency": float(row[f"{mode_prefix}_consistency"]),
                            "conciseness": float(row[f"{mode_prefix}_conciseness"]),
                            "average": float(row[f"{mode_prefix}_judge_avg"])
                        },
                        "judge_explanation": row.get(f"_{mode_prefix}_judge_explanation", ""),
                        "generation_time": float(row[f"{mode_prefix}_time"]),
                        "generated_words": int(row[f"{mode_prefix}_words"])
                    }

                    # Add ROUGE scores if available
                    if f"{mode_prefix}_rouge1_f" in row:
                        sample_data["rouge_scores"] = {
                            "rouge1_f": float(row[f"{mode_prefix}_rouge1_f"]),
                            "rouge2_f": float(row[f"{mode_prefix}_rouge2_f"]),
                            "rougeL_f": float(row[f"{mode_prefix}_rougeL_f"])
                        }

                    # Add embedding similarity if available
                    if f"{mode_prefix}_embedding_similarity" in row:
                        sample_data["embedding_similarity"] = float(row[f"{mode_prefix}_embedding_similarity"])

                    json_data.append(sample_data)

                # Save to JSON file
                low_score_file = output_file.parent / f"{output_file.stem}_{mode_prefix}_low_scores.json"
                with open(low_score_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                print(f"  Saved to: {low_score_file}")

                # Print summary
                print(f"  Average scores of low-score samples:")
                print(f"    Coverage:     {low_scores[f'{mode_prefix}_coverage'].mean():.2f}")
                print(f"    Specificity:  {low_scores[f'{mode_prefix}_specificity'].mean():.2f}")
                print(f"    Consistency:  {low_scores[f'{mode_prefix}_consistency'].mean():.2f}")
                print(f"    Conciseness:  {low_scores[f'{mode_prefix}_conciseness'].mean():.2f}")
                print(f"    Overall Avg:  {low_scores[f'{mode_prefix}_judge_avg'].mean():.2f}")

                low_score_samples.append({
                    "mode": mode_prefix,
                    "file": low_score_file,
                    "count": len(low_scores)
                })
            else:
                print(f"  No low-score samples found ✅")

        if low_score_samples:
            print(f"\n💡 Low-score samples saved for analysis:")
            for item in low_score_samples:
                print(f"   {item['mode']}: {item['count']} samples → {item['file']}")


if __name__ == "__main__":
    asyncio.run(main())
