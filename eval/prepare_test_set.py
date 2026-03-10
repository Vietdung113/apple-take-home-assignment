"""Prepare stratified test set for evaluation.

Strategy:
- Load GovReport test split
- Tokenize and filter by context length (8K/16K/32K)
- Sample test set with target distribution: 50% 8K, 30% 16K, 20% 32K
- Save to JSONL for evaluation

Usage:
    python prepare_test_set.py --num-samples 100
    python prepare_test_set.py --num-samples 100 --output test_set_100.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


# ── Constants ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-0.6B"  # Updated to match project model
# Load prompts from config/prompts.yaml instead of hardcoded
import yaml
from pathlib import Path

# Load system prompt and user instruction from config
config_path = Path(__file__).parent.parent / "finetuning" / "config" / "prompts.yaml"
with open(config_path) as f:
    prompt_config = yaml.safe_load(f)

SYSTEM_PROMPT = prompt_config["system_prompt"]
USER_PROMPT_TEMPLATE = prompt_config["user_instruction"] + "\n{doc}"

# Context budgets (in tokens)
BUDGET_8K = 8192
BUDGET_16K = 16384
BUDGET_32K = 32768

# Target distribution for test set
TARGET_DISTRIBUTION = {
    "8k": 0.50,   # 50% short docs
    "16k": 0.30,  # 30% medium docs
    "32k": 0.20,  # 20% long docs
}


# ── Helper Functions ─────────────────────────────────────────────────────

def build_messages(doc: str, summary: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(doc=doc)},
        {"role": "assistant", "content": summary},
    ]


def compute_token_length(messages: list[dict], tokenizer) -> int:
    """Compute total token length after applying chat template."""
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokens = tokenizer.encode(formatted, add_special_tokens=False)
    return len(tokens)


def categorize_by_length(sample: dict, tokenizer) -> str | None:
    """Categorize sample into 8k/16k/32k based on token length.

    Returns:
        "8k", "16k", "32k", or None if too long (>32k)
    """
    doc = sample["report"]
    summary = sample["summary"]

    messages = build_messages(doc, summary)
    total_tokens = compute_token_length(messages, tokenizer)

    if total_tokens <= BUDGET_8K:
        return "8k"
    elif total_tokens <= BUDGET_16K:
        return "16k"
    elif total_tokens <= BUDGET_32K:
        return "32k"
    else:
        return None  # Too long


def stratified_sample_test_set(
    dataset,
    num_samples: int,
    target_dist: dict,
    tokenizer,
    seed: int = 42,
) -> list[dict]:
    """Sample test set with target distribution across length categories.

    Args:
        dataset: HuggingFace dataset
        num_samples: Total number of samples to select
        target_dist: Dict mapping category ("8k", "16k", "32k") to target proportion
        tokenizer: Tokenizer for length computation
        seed: Random seed

    Returns:
        List of samples with keys: report, summary, category, total_tokens
    """
    import random
    random.seed(seed)

    # Categorize all samples
    print("Categorizing samples by token length...")
    categorized = {
        "8k": [],
        "16k": [],
        "32k": [],
    }

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(dataset)} samples...")

        category = categorize_by_length(sample, tokenizer)
        if category:
            doc = sample["report"]
            summary = sample["summary"]
            messages = build_messages(doc, summary)
            total_tokens = compute_token_length(messages, tokenizer)

            categorized[category].append({
                "document": doc,  # Use "document" field per Apple assessment format
                "summary": summary,
                "category": category,
                "total_tokens": total_tokens,
            })

    print(f"\nCategory distribution in full test set:")
    for cat in ["8k", "16k", "32k"]:
        print(f"  {cat}: {len(categorized[cat])} samples")

    # Sample according to target distribution
    selected = []
    for cat, proportion in target_dist.items():
        target_count = int(num_samples * proportion)
        available = categorized[cat]

        if len(available) < target_count:
            print(f"  Warning: Only {len(available)} {cat} samples available, need {target_count}")
            target_count = len(available)

        sampled = random.sample(available, target_count)
        selected.extend(sampled)
        print(f"  Sampled {len(sampled)} from {cat} category")

    # Shuffle final list
    random.shuffle(selected)

    return selected


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare stratified test set for evaluation")
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Total number of test samples (default: 100)"
    )
    parser.add_argument(
        "--output", default="test_set.jsonl",
        help="Output JSONL file (default: test_set.jsonl)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    output_path = Path(__file__).parent / args.output

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load GovReport test set
    print("Loading GovReport test set...")
    dataset = load_dataset("ccdv/govreport-summarization", split="test")
    print(f"  Total samples: {len(dataset)}")

    # Sample stratified test set
    print(f"\nSampling {args.num_samples} test samples with distribution:")
    for cat, prop in TARGET_DISTRIBUTION.items():
        print(f"  {cat}: {prop*100:.0f}% ({int(args.num_samples * prop)} samples)")

    test_samples = stratified_sample_test_set(
        dataset,
        num_samples=args.num_samples,
        target_dist=TARGET_DISTRIBUTION,
        tokenizer=tokenizer,
        seed=args.seed,
    )

    # Save to JSONL
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {len(test_samples)}")

    from collections import Counter
    category_counts = Counter(s["category"] for s in test_samples)
    for cat in ["8k", "16k", "32k"]:
        count = category_counts[cat]
        pct = count / len(test_samples) * 100
        print(f"  {cat}: {count} samples ({pct:.1f}%)")

    # Token stats
    tokens_by_cat = {cat: [] for cat in ["8k", "16k", "32k"]}
    for sample in test_samples:
        tokens_by_cat[sample["category"]].append(sample["total_tokens"])

    print(f"\nToken length statistics:")
    for cat in ["8k", "16k", "32k"]:
        tokens = tokens_by_cat[cat]
        if tokens:
            print(f"  {cat}: min={min(tokens)}, max={max(tokens)}, mean={sum(tokens)/len(tokens):.0f}")

    print(f"\n✅ Test set saved to {output_path}")


if __name__ == "__main__":
    main()
