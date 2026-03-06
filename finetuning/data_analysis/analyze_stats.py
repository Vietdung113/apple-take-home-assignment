"""Analyze GovReport dataset: format, structure, length distributions.

Usage:
    cd finetuning && uv run python data_analysis/analyze_stats.py
"""

from datasets import load_dataset
import numpy as np


def main():
    ds = load_dataset("ccdv/govreport-summarization")

    print("=" * 70)
    print("  GovReport Dataset Analysis")
    print("=" * 70)

    # ── 1. Basic structure ──────────────────────────────────
    print("\n1. BASIC STRUCTURE")
    print("-" * 50)
    print(f"Splits: {list(ds.keys())}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")
    print(f"Columns: {ds['train'].column_names}")
    print(f"Features: {ds['train'].features}")

    # ── 2. Raw examples (first 3) ──────────────────────────
    print("\n\n2. RAW EXAMPLES (first 3)")
    print("-" * 50)
    for i in range(3):
        ex = ds["train"][i]
        report = ex["report"]
        summary = ex["summary"]
        print(f"\n--- Example {i} ---")
        print(f"  Report:  {len(report):>8,} chars")
        print(f"  Summary: {len(summary):>8,} chars")
        print(f"  Report  (first 500 chars): {report[:500]}")
        print(f"  Report  (last  300 chars): ...{report[-300:]}")
        print(f"  Summary (first 500 chars): {summary[:500]}")

    # ── 3. Input (report) length distribution ──────────────
    print("\n\n3. INPUT (REPORT) LENGTH DISTRIBUTION")
    print("-" * 50)

    for split in ["train", "validation", "test"]:
        chars = np.array([len(ex["report"]) for ex in ds[split]])
        tokens = chars / 4  # rough approximation

        print(f"\n  [{split}] N={len(chars)}")
        print(f"  Characters:")
        for label, val in [
            ("Min", chars.min()),
            ("P5", np.percentile(chars, 5)),
            ("P25", np.percentile(chars, 25)),
            ("Median", np.median(chars)),
            ("Mean", chars.mean()),
            ("P75", np.percentile(chars, 75)),
            ("P95", np.percentile(chars, 95)),
            ("Max", chars.max()),
        ]:
            print(f"    {label:<8} {val:>10,.0f}")

        print(f"  Approx tokens (chars/4):")
        for label, val in [
            ("Min", tokens.min()),
            ("Median", np.median(tokens)),
            ("Mean", tokens.mean()),
            ("P95", np.percentile(tokens, 95)),
            ("Max", tokens.max()),
        ]:
            print(f"    {label:<8} {val:>10,.0f}")

        # Length bins vs Qwen2.5-3B 32K context window
        short = np.sum(tokens < 4_000)
        medium = np.sum((tokens >= 4_000) & (tokens < 16_000))
        long_ = np.sum((tokens >= 16_000) & (tokens < 32_000))
        overflow = np.sum(tokens >= 32_000)
        total = len(tokens)
        print(f"  Length bins (vs 32K context):")
        print(f"    Short    (<4K tokens):   {short:>6} ({100*short/total:>5.1f}%)")
        print(f"    Medium   (4K-16K):       {medium:>6} ({100*medium/total:>5.1f}%)")
        print(f"    Long     (16K-32K):      {long_:>6} ({100*long_/total:>5.1f}%)")
        print(f"    Overflow (>32K):         {overflow:>6} ({100*overflow/total:>5.1f}%) <- needs chunking")

    # ── 4. Output (summary) length distribution ────────────
    print("\n\n4. OUTPUT (SUMMARY) LENGTH DISTRIBUTION")
    print("-" * 50)

    for split in ["train", "validation", "test"]:
        chars = np.array([len(ex["summary"]) for ex in ds[split]])
        tokens = chars / 4
        print(f"  [{split}]  chars: min={chars.min():,}  median={np.median(chars):,.0f}"
              f"  mean={chars.mean():,.0f}  max={chars.max():,}")
        print(f"           tokens: min={tokens.min():,.0f}  median={np.median(tokens):,.0f}"
              f"  mean={tokens.mean():,.0f}  max={tokens.max():,.0f}")

    # ── 5. Compression ratio ───────────────────────────────
    print("\n\n5. COMPRESSION RATIO (input_len / output_len)")
    print("-" * 50)
    for split in ["train", "test"]:
        ratios = np.array([
            len(ex["report"]) / len(ex["summary"])
            for ex in ds[split] if len(ex["summary"]) > 0
        ])
        print(f"  [{split}]  min={ratios.min():.1f}  median={np.median(ratios):.1f}"
              f"  mean={ratios.mean():.1f}  max={ratios.max():.1f}")

    # ── 6. Text format analysis ────────────────────────────
    print("\n\n6. TEXT FORMAT ANALYSIS (sample=1000)")
    print("-" * 50)

    n = min(1000, len(ds["train"]))
    has_newlines = sum(1 for i in range(n) if "\n" in ds["train"][i]["report"])
    has_headers = sum(
        1 for i in range(n)
        if any(h in ds["train"][i]["report"]
               for h in ["Section ", "SECTION ", "Chapter ", "CHAPTER "])
    )
    has_bullets = sum(
        1 for i in range(n)
        if any(b in ds["train"][i]["report"] for b in ["- ", "* ", "• "])
    )
    avg_lines = np.mean([
        len([p for p in ds["train"][i]["report"].split("\n") if p.strip()])
        for i in range(n)
    ])

    print(f"  Reports:")
    print(f"    Has newlines:        {has_newlines:>5} ({100*has_newlines/n:.1f}%)")
    print(f"    Has section headers: {has_headers:>5} ({100*has_headers/n:.1f}%)")
    print(f"    Has bullet points:   {has_bullets:>5} ({100*has_bullets/n:.1f}%)")
    print(f"    Avg lines per report: {avg_lines:.0f}")

    has_nl_s = sum(1 for i in range(n) if "\n" in ds["train"][i]["summary"])
    has_bul_s = sum(
        1 for i in range(n)
        if any(b in ds["train"][i]["summary"] for b in ["- ", "* ", "• "])
    )
    print(f"  Summaries:")
    print(f"    Has newlines:        {has_nl_s:>5} ({100*has_nl_s/n:.1f}%)")
    print(f"    Has bullet points:   {has_bul_s:>5} ({100*has_bul_s/n:.1f}%)")

    # ── 7. Key takeaways ──────────────────────────────────
    train_tokens = np.array([len(ex["report"]) for ex in ds["train"]]) / 4
    overflow_pct = 100 * np.sum(train_tokens >= 32_000) / len(train_tokens)
    long_pct = 100 * np.sum(train_tokens >= 16_000) / len(train_tokens)
    summ_tokens = np.array([len(ex["summary"]) for ex in ds["train"]]) / 4

    print("\n\n" + "=" * 70)
    print("  KEY TAKEAWAYS")
    print("=" * 70)
    print(f"""
  FORMAT:   Reports = plain text (no newlines). Summaries = plain text paragraphs.
  INPUT:    Median ~{np.median(train_tokens):,.0f} tokens. {long_pct:.1f}% exceed 16K, {overflow_pct:.1f}% overflow 32K.
  OUTPUT:   Median ~{np.median(summ_tokens):,.0f} tokens (~800 tokens to generate).
  COMPRESS: ~{np.mean([len(e['report'])/len(e['summary']) for e in ds['train'] if len(e['summary'])>0]):.0f}:1 ratio (highly abstractive).

  PIPELINE IMPLICATIONS:
    - Segment node:  CRITICAL for {overflow_pct:.1f}% docs, helpful for {long_pct:.1f}%
    - Summarize:     Handle ~4K-16K token chunks
    - Merge:         Needed when doc splits into 2+ chunks
    - Critique/Refine: Polish abstractive output
""")


if __name__ == "__main__":
    main()
