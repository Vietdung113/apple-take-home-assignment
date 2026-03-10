"""Convert full GovReport dataset to Apple assessment format.

Apple take-home assessment requires:
- Input format: JSON with "document" field
- Output format: JSON with "summary" field

This script converts the full GovReport dataset (train + validation splits merged) to this format,
then splits 90/10 into train/val.

Usage:
    # Convert all GovReport data
    python convert_govreport_to_base_format.py

    # Custom output path
    python convert_govreport_to_base_format.py --output ../data/govreport_full/

    # Limit samples (for testing)
    python convert_govreport_to_base_format.py --max-samples 100
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Convert full GovReport dataset to Apple assessment format and split train/val 90/10"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "govreport_full",
        help="Output directory"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples (for testing)"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1 = 10%%)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Convert GovReport Dataset to Apple Assessment Format")
    print("="*80)
    print(f"Output: {args.output}")
    print(f"Val split: {args.val_split*100:.0f}%")
    print(f"Random seed: {args.seed}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print()

    # Load GovReport dataset from Hugging Face (train + validation splits)
    print("Loading GovReport dataset from Hugging Face...")
    train_ds = load_dataset("ccdv/govreport-summarization", split="train", trust_remote_code=True)
    val_ds = load_dataset("ccdv/govreport-summarization", split="validation", trust_remote_code=True)

    print(f"  Train split: {len(train_ds)} samples")
    print(f"  Validation split: {len(val_ds)} samples")

    # Merge train + validation
    dataset = concatenate_datasets([train_ds, val_ds])
    print(f"  Total merged: {len(dataset)} samples")
    print()

    # Convert to list of dicts
    print("Converting to Apple assessment format...")
    all_samples = []
    skipped = 0

    total_to_process = len(dataset) if not args.max_samples else min(len(dataset), args.max_samples)

    for idx, sample in enumerate(tqdm(dataset, desc="Converting", total=total_to_process)):
        if args.max_samples and idx >= args.max_samples:
            break

        try:
            # GovReport format: 'report' -> 'document', 'summary' -> 'summary'
            document = sample["report"].strip()
            summary = sample["summary"].strip()

            if not document or not summary:
                skipped += 1
                continue

            all_samples.append({
                "document": document,
                "summary": summary
            })

        except Exception as e:
            print(f"Warning: Failed to process sample {idx}: {e}")
            skipped += 1

    print(f"✓ Converted: {len(all_samples)} samples")
    print(f"✗ Skipped: {skipped} samples")
    print()

    # Shuffle and split 90/10
    print(f"Shuffling and splitting (seed={args.seed})...")
    random.seed(args.seed)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * (1 - args.val_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"  Train: {len(train_samples)} samples ({(1-args.val_split)*100:.0f}%)")
    print(f"  Val:   {len(val_samples)} samples ({args.val_split*100:.0f}%)")
    print()

    # Write train file
    train_file = args.output / "train.jsonl"
    print(f"Writing train data: {train_file}")
    with open(train_file, "w") as f:
        for sample in tqdm(train_samples, desc="Writing train"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Write validation file
    val_file = args.output / "val.jsonl"
    print(f"Writing validation data: {val_file}")
    with open(val_file, "w") as f:
        for sample in tqdm(val_samples, desc="Writing val"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print()
    print("="*80)
    print("Conversion Complete!")
    print("="*80)
    print(f"✓ Train: {train_file} ({train_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"✓ Val:   {val_file} ({val_file.stat().st_size / (1024*1024):.2f} MB)")
    print()


if __name__ == "__main__":
    main()
