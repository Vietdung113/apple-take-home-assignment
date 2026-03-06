"""Quick check of generated training data."""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_data.py <jsonl_file>")
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])
    print(f"Loading tokenizer {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Checking {jsonl_path} ...")
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Total samples: {len(samples)}\n")

    token_counts = []
    for i, sample in enumerate(samples):
        messages = sample["messages"]

        # apply_chat_template returns string, then tokenize it
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        token_count = len(tokens)
        token_counts.append(token_count)

        user_len = len(messages[0]["content"])
        assistant_len = len(messages[1]["content"])

        print(f"Sample {i+1}: {token_count:,} tokens "
              f"(user: {user_len:,} chars, assistant: {assistant_len:,} chars)")

    print(f"\nToken count stats:")
    print(f"  Min: {min(token_counts):,}")
    print(f"  Max: {max(token_counts):,}")
    print(f"  Avg: {sum(token_counts) // len(token_counts):,}")


if __name__ == "__main__":
    main()
