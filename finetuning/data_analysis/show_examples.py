"""Show diverse GovReport examples by length to understand input/output format.

Usage:
    cd finetuning && uv run python data_analysis/show_examples.py
"""

from datasets import load_dataset
import numpy as np


def show_example(ex, label, idx):
    report = ex["report"]
    summary = ex["summary"]
    r_tokens = len(report) // 4
    s_tokens = len(summary) // 4
    ratio = len(report) / len(summary) if len(summary) > 0 else 0

    print(f"\n{'='*80}")
    print(f"  EXAMPLE {idx}: {label}")
    print(f"  Report:  {len(report):,} chars (~{r_tokens:,} tokens)")
    print(f"  Summary: {len(summary):,} chars (~{s_tokens:,} tokens)")
    print(f"  Compression: {ratio:.1f}:1")
    print(f"{'='*80}")

    print(f"\n--- REPORT (first 2000 chars) ---\n")
    print(report[:2000])
    if len(report) > 2000:
        print(f"\n  [...{len(report)-2000:,} chars truncated...]")

    print(f"\n--- FULL SUMMARY ---\n")
    print(summary)
    print()


def main():
    ds = load_dataset("ccdv/govreport-summarization")
    train = ds["train"]

    lengths = [len(ex["report"]) for ex in train]
    sorted_idx = np.argsort(lengths)

    picks = [
        ("SHORT (~4K tokens)", sorted_idx[len(sorted_idx) // 10]),
        ("MEDIUM-SHORT (~6K tokens)", sorted_idx[len(sorted_idx) // 4]),
        ("MEDIAN (~10K tokens)", sorted_idx[len(sorted_idx) // 2]),
        ("LONG (~20K tokens)", sorted_idx[int(len(sorted_idx) * 0.85)]),
        ("VERY LONG (~30K tokens)", sorted_idx[int(len(sorted_idx) * 0.96)]),
    ]

    for i, (label, idx) in enumerate(picks):
        show_example(train[int(idx)], label, i + 1)


if __name__ == "__main__":
    main()
