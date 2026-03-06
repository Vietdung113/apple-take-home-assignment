"""Download GovReport dataset and verify it's cached locally.

Usage:
    cd finetuning && uv run python data_analysis/download_dataset.py
"""

from datasets import load_dataset


def main():
    print("Downloading GovReport dataset...")
    ds = load_dataset("ccdv/govreport-summarization")

    print(f"\nSplits:")
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")
    print(f"Columns: {ds['train'].column_names}")
    print(f"Features: {ds['train'].features}")
    print("\nDataset cached successfully.")


if __name__ == "__main__":
    main()
