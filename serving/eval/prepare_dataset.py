"""Prepare a stratified evaluation dataset from GovReport test split.

Sorts documents by length, splits into terciles (short/medium/long),
and samples a configurable number from each bucket. Saves to JSON so
that eval_rouge.py and eval_judge.py both use the same examples.

Config (via eval/.env or environment):
    EVAL_N_SHORT   — examples from short bucket  (default: 3)
    EVAL_N_MEDIUM  — examples from medium bucket  (default: 3)
    EVAL_N_LONG    — examples from long bucket    (default: 4)
    JUDGE_SEED     — random seed for sampling     (default: 42)

Output:
    eval/eval_dataset.json — array of {idx, bucket, report, summary, doc_chars}

Usage:
    cd serving && uv run python eval/prepare_dataset.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from datasets import load_dataset

from eval.common import load_dotenv

load_dotenv()

OUTPUT_PATH = Path(__file__).parent / "eval_dataset.json"


def stratified_sample(
    ds,
    n_short: int,
    n_medium: int,
    n_long: int,
    seed: int,
) -> list[dict]:
    """Sample from each length tercile of the dataset."""
    # Build (index, doc_length) pairs and sort by length
    indexed = [(i, len(ds[i]["report"])) for i in range(len(ds))]
    indexed.sort(key=lambda x: x[1])

    n = len(indexed)
    tercile = n // 3

    buckets = {
        "short":  (indexed[:tercile], n_short),
        "medium": (indexed[tercile:2 * tercile], n_medium),
        "long":   (indexed[2 * tercile:], n_long),
    }

    rng = random.Random(seed)
    selected: list[dict] = []

    for bucket_name in ["short", "medium", "long"]:
        pool, count = buckets[bucket_name]
        count = min(count, len(pool))
        sampled = rng.sample(pool, count)
        for orig_idx, doc_len in sampled:
            selected.append({
                "idx": orig_idx,
                "bucket": bucket_name,
                "doc_chars": doc_len,
                "report": ds[orig_idx]["report"],
                "summary": ds[orig_idx]["summary"],
            })

    return selected


def main():
    n_short = int(os.environ.get("EVAL_N_SHORT", "3"))
    n_medium = int(os.environ.get("EVAL_N_MEDIUM", "3"))
    n_long = int(os.environ.get("EVAL_N_LONG", "4"))
    seed = int(os.environ.get("JUDGE_SEED", "42"))
    total = n_short + n_medium + n_long

    print(f"  Prepare Evaluation Dataset")
    print(f"  Seed:    {seed}")
    print(f"  Short:   {n_short}")
    print(f"  Medium:  {n_medium}")
    print(f"  Long:    {n_long}")
    print(f"  Total:   {total}")
    print()

    print("  Loading GovReport test split...")
    ds = load_dataset("ccdv/govreport-summarization", split="test")
    print(f"  Dataset size: {len(ds)} examples")

    samples = stratified_sample(ds, n_short=n_short, n_medium=n_medium, n_long=n_long, seed=seed)

    OUTPUT_PATH.write_text(json.dumps(samples, indent=2))
    print(f"\n  Saved {len(samples)} examples to {OUTPUT_PATH}")

    # Print summary
    for bucket in ["short", "medium", "long"]:
        bucket_samples = [s for s in samples if s["bucket"] == bucket]
        if bucket_samples:
            lengths = [s["doc_chars"] for s in bucket_samples]
            print(f"  {bucket:>6}: {len(bucket_samples)} examples, "
                  f"{min(lengths):,}–{max(lengths):,} chars")


if __name__ == "__main__":
    main()
