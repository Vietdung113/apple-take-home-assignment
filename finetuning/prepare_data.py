"""Prepare SFT dataset for Qwen3-0.6B summarization fine-tuning.

Loads GovReport, tokenizes all samples in parallel, then filters by
multiple token budgets at once and saves JSONL splits.

Usage:
    python prepare_data.py                                # all 3: 8k, 16k, 32k
    python prepare_data.py --max-tokens 8192              # single budget
    python prepare_data.py --max-tokens 8192 16384 32768  # explicit list
    python prepare_data.py --workers 8                    # control parallelism
"""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Constants ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_BUDGETS = [8192, 16384, 32768]

PROMPT_TEMPLATE = "Summarize the following document:\n\n{doc}\n\nSummary:"

# Quality filters
MIN_DOC_TOKENS = 200
MIN_SUM_TOKENS = 50
MIN_COMPRESSION_RATIO = 1.5


# ── Helpers ──────────────────────────────────────────────────────────────


def make_label(max_tokens: int) -> str:
    return f"{max_tokens // 1024}k"


def build_messages(doc: str, summary: str) -> list[dict]:
    return [
        {"role": "user", "content": PROMPT_TEMPLATE.format(doc=doc)},
        {"role": "assistant", "content": summary},
    ]


# ── Worker function (runs in subprocess) ─────────────────────────────────

# Global tokenizer per worker process (loaded once via initializer)
_tokenizer = None


def _init_worker(model_name: str):
    global _tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _tokenizer = AutoTokenizer.from_pretrained(model_name)


def _process_one(example: dict) -> dict | None:
    """Tokenize one sample, return messages + token count or None if bad quality."""
    doc = example["report"]
    summary = example["summary"]

    doc_tokens = len(_tokenizer.encode(doc, add_special_tokens=False))
    sum_tokens = len(_tokenizer.encode(summary, add_special_tokens=False))

    if doc_tokens < MIN_DOC_TOKENS or sum_tokens < MIN_SUM_TOKENS:
        return None
    if doc_tokens / max(sum_tokens, 1) < MIN_COMPRESSION_RATIO:
        return None

    messages = build_messages(doc, summary)
    # Format messages and tokenize
    formatted = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    total_tokens = len(_tokenizer.encode(formatted, add_special_tokens=False))

    return {"messages": messages, "total_tokens": total_tokens}


# ── Core logic ───────────────────────────────────────────────────────────


def tokenize_split(split_data, num_workers: int, split_name: str) -> list[dict]:
    """Tokenize a split in parallel. Returns list of {messages, total_tokens}."""
    # Convert HF dataset to list of dicts for multiprocessing
    records = [{"report": ex["report"], "summary": ex["summary"]} for ex in split_data]

    print(f"  {split_name}: tokenizing {len(records)} samples with {num_workers} workers ...")

    with Pool(num_workers, initializer=_init_worker, initargs=(MODEL_NAME,)) as pool:
        results = list(tqdm(
            pool.imap(_process_one, records, chunksize=64),
            total=len(records),
            desc=f"    {split_name}",
            unit="sample",
        ))

    # Filter out None (failed quality)
    valid = [r for r in results if r is not None]
    skipped = len(records) - len(valid)
    print(f"  {split_name}: {len(valid)} passed quality filters, {skipped} skipped")
    return valid


def filter_by_budget(tokenized: list[dict], max_tokens: int) -> list[dict]:
    """Filter pre-tokenized samples by token budget."""
    return [
        {"messages": r["messages"]}
        for r in tokenized
        if r["total_tokens"] <= max_tokens
    ]


def save_jsonl(rows: list[dict], path: Path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT dataset filtered by token budget"
    )
    parser.add_argument(
        "--max-tokens", type=int, nargs="*", default=None,
        help="Token budgets (default: 8192 16384 32768)",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples per split (for testing)",
    )
    args = parser.parse_args()

    budgets = args.max_tokens or DEFAULT_BUDGETS
    num_workers = args.workers or min(cpu_count(), 16)
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────
    print("Loading GovReport dataset ...")
    raw = load_dataset("ccdv/govreport-summarization")

    # ── Tokenize each split once (expensive part, parallelized) ──────
    splits = ["train", "validation", "test"]
    tokenized = {}
    for split_name in splits:
        split_data = raw[split_name]
        if args.max_samples is not None:
            split_data = split_data.select(range(min(args.max_samples, len(split_data))))
        tokenized[split_name] = tokenize_split(split_data, num_workers, split_name)

    # ── Filter by each budget and save ───────────────────────────────
    print()
    for budget in budgets:
        label = make_label(budget)
        print(f"Budget {label} (max_tokens={budget}):")
        for split_name in splits:
            rows = filter_by_budget(tokenized[split_name], budget)
            out_path = output_dir / f"sft_{label}_{split_name}.jsonl"
            save_jsonl(rows, out_path)
            print(f"  {split_name}: {len(rows)} samples → {out_path}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
