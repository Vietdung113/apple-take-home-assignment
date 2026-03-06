"""Prepare SFT dataset for Qwen3-0.6B summarization fine-tuning.

Loads GovReport, filters samples by token budget (exact token count via
chat template), formats as chat messages, and saves train/validation/test
JSONL splits.

Usage:
    python prepare_data.py --max-tokens 8192    # → data/sft_8k_{train,validation,test}.jsonl
    python prepare_data.py --max-tokens 16384   # → data/sft_16k_{train,validation,test}.jsonl
    python prepare_data.py --max-tokens 32768   # → data/sft_32k_{train,validation,test}.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

# ── Constants ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-0.6B"

PROMPT_TEMPLATE = "Summarize the following document:\n\n{doc}\n\nSummary:"

# Quality filters
MIN_DOC_TOKENS = 200
MIN_SUM_TOKENS = 50
MIN_COMPRESSION_RATIO = 1.5


# ── Helpers ──────────────────────────────────────────────────────────────


def make_label(max_tokens: int) -> str:
    """Convert max_tokens to human-readable label: 8192 → '8k'."""
    return f"{max_tokens // 1024}k"


def build_messages(doc: str, summary: str) -> list[dict]:
    """Build chat messages for a single sample."""
    return [
        {"role": "user", "content": PROMPT_TEMPLATE.format(doc=doc)},
        {"role": "assistant", "content": summary},
    ]


def count_tokens(tokenizer, messages: list[dict]) -> int:
    """Exact token count using the model's chat template."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    return len(ids)


def process_split(
    split_data,
    tokenizer,
    max_tokens: int,
    split_name: str,
) -> list[dict]:
    """Filter and format a single dataset split."""
    rows = []
    skipped_quality = 0
    skipped_length = 0

    for example in split_data:
        doc = example["report"]
        summary = example["summary"]

        # Rough token counts for quality filtering (fast, no chat template)
        doc_tokens = len(tokenizer.encode(doc, add_special_tokens=False))
        sum_tokens = len(tokenizer.encode(summary, add_special_tokens=False))

        # Quality filters
        if doc_tokens < MIN_DOC_TOKENS or sum_tokens < MIN_SUM_TOKENS:
            skipped_quality += 1
            continue

        if doc_tokens / max(sum_tokens, 1) < MIN_COMPRESSION_RATIO:
            skipped_quality += 1
            continue

        # Build messages and check exact token count
        messages = build_messages(doc, summary)
        total_tokens = count_tokens(tokenizer, messages)

        if total_tokens > max_tokens:
            skipped_length += 1
            continue

        rows.append({"messages": messages})

    print(f"  {split_name}: {len(rows)} samples kept, "
          f"{skipped_quality} failed quality, "
          f"{skipped_length} exceeded {max_tokens} tokens")
    return rows


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT dataset filtered by token budget"
    )
    parser.add_argument(
        "--max-tokens", type=int, required=True,
        help="Maximum sequence length in tokens (e.g. 8192, 16384, 32768)",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Output directory (default: data)",
    )
    args = parser.parse_args()

    max_tokens = args.max_tokens
    label = make_label(max_tokens)
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ───────────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load dataset ─────────────────────────────────────────────────
    print("Loading GovReport dataset ...")
    raw = load_dataset("ccdv/govreport-summarization")

    # ── Process each split ───────────────────────────────────────────
    print(f"\nFiltering for max_tokens={max_tokens} ({label}):")

    splits = {
        "train": raw["train"],
        "validation": raw["validation"],
        "test": raw["test"],
    }

    for split_name, split_data in splits.items():
        rows = process_split(split_data, tokenizer, max_tokens, split_name)

        # Save JSONL
        out_path = output_dir / f"sft_{label}_{split_name}.jsonl"
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"    → {out_path} ({len(rows)} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()
