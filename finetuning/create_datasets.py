"""Create 3 training datasets for synthetic data experiment.

Datasets:
1. Baseline: High-quality reference summaries only (2,802 samples)
2. Blend: References + equal number of synthetics (50/50 mix)
3. Full: References + ALL synthetics

Each dataset is split 90/10 into train/val.

Usage:
    python create_datasets.py
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SEED = 42

# Input files
GOVREPORT_TRAIN = DATA_DIR / "govreport_full" / "train.jsonl"
GOVREPORT_VAL = DATA_DIR / "govreport_full" / "val.jsonl"
SYNTHETIC_FILE = DATA_DIR / "synthetic_summaries.jsonl"
SCORES_FILE = DATA_DIR / "scores_progress.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: list[dict], path: Path):
    with open(path, "w") as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(samples)} samples to {path}")


def load_high_quality_references() -> list[dict]:
    """Load GovReport samples that passed quality filter (coverage>=4 AND specificity>=4)."""
    # Load scores
    scores = {}
    with open(SCORES_FILE) as f:
        for line in f:
            s = json.loads(line)
            if s.get("success") and s.get("coverage", 0) >= 4 and s.get("specificity", 0) >= 4:
                scores[s["sample_id"]] = s

    # Load full GovReport to get documents
    from datasets import load_dataset

    dataset = load_dataset("ccdv/govreport-summarization")
    all_samples = list(dataset["train"]) + list(dataset["validation"])

    references = []
    for sid in sorted(scores.keys()):
        if sid < len(all_samples):
            references.append({
                "document": all_samples[sid]["report"],
                "summary": all_samples[sid]["summary"],
                "source": "reference",
            })

    return references


def split_train_val(samples: list[dict], val_ratio: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Shuffle and split into train/val."""
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    return samples[:split_idx], samples[split_idx:]


def main():
    random.seed(SEED)

    print("=" * 60)
    print("Creating Experiment Datasets")
    print("=" * 60)

    # Load references
    print("\nLoading high-quality references...")
    references = load_high_quality_references()
    print(f"  References: {len(references)}")

    # Load synthetics
    print("Loading synthetic summaries...")
    if not SYNTHETIC_FILE.exists():
        print(f"  WARNING: {SYNTHETIC_FILE} not found!")
        print("  Run generate_synthetic_data.py first.")
        print("  Creating baseline dataset only.\n")
        synthetics = []
    else:
        synthetics = load_jsonl(SYNTHETIC_FILE)
        # Add source tag
        for s in synthetics:
            s["source"] = "synthetic"
        print(f"  Synthetics: {len(synthetics)}")

    # ── Dataset 1: Baseline (references only) ────────────────────────────
    print("\n--- Dataset 1: Baseline (references only) ---")
    baseline = [{"document": r["document"], "summary": r["summary"]} for r in references]
    train, val = split_train_val(baseline)
    save_jsonl(train, DATA_DIR / "dataset_baseline_train.jsonl")
    save_jsonl(val, DATA_DIR / "dataset_baseline_val.jsonl")

    if not synthetics:
        print("\nNo synthetics available. Only baseline dataset created.")
        return

    # ── Dataset 2: Blend (50/50 mix) ─────────────────────────────────────
    print("\n--- Dataset 2: Blend (50/50 references + synthetics) ---")
    n_synth = min(len(synthetics), len(references))
    sampled_synth = random.sample(synthetics, n_synth)
    blend = (
        [{"document": r["document"], "summary": r["summary"]} for r in references]
        + [{"document": s["document"], "summary": s["summary"]} for s in sampled_synth]
    )
    print(f"  Total: {len(references)} refs + {n_synth} synth = {len(blend)}")
    train, val = split_train_val(blend)
    save_jsonl(train, DATA_DIR / "dataset_blend_train.jsonl")
    save_jsonl(val, DATA_DIR / "dataset_blend_val.jsonl")

    # ── Dataset 3: Full (references + all synthetics) ────────────────────
    print("\n--- Dataset 3: Full (references + ALL synthetics) ---")
    full = (
        [{"document": r["document"], "summary": r["summary"]} for r in references]
        + [{"document": s["document"], "summary": s["summary"]} for s in synthetics]
    )
    print(f"  Total: {len(references)} refs + {len(synthetics)} synth = {len(full)}")
    train, val = split_train_val(full)
    save_jsonl(train, DATA_DIR / "dataset_full_train.jsonl")
    save_jsonl(val, DATA_DIR / "dataset_full_val.jsonl")

    # Summary
    print(f"\n{'='*60}")
    print("Datasets Created:")
    print(f"  1. Baseline: {len(baseline)} samples (references only)")
    print(f"  2. Blend:    {len(blend)} samples (50/50 mix)")
    print(f"  3. Full:     {len(full)} samples (all)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
