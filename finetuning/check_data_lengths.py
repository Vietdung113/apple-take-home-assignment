"""Check training data lengths to identify truncation issues."""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add config to path
sys.path.insert(0, str(Path(__file__).parent / "config"))
from prompt_loader import get_training_prompt_base_model

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-0.6B")

# Load training data
data_file = Path(__file__).parent / "data" / "govreport_full" / "train.jsonl"
print(f"Loading data from: {data_file}")

samples = []
with open(data_file) as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

# Check lengths
lengths = []
over_32k = []

print("\nChecking token lengths...")
for i, sample in enumerate(samples[:1000]):  # Check first 1000 for speed
    doc = sample["document"]
    summary = sample["summary"]

    # Format as training prompt
    prompt = get_training_prompt_base_model(doc, summary) + tokenizer.eos_token

    # Tokenize
    tokens = tokenizer(prompt, truncation=False, add_special_tokens=False)['input_ids']
    length = len(tokens)
    lengths.append(length)

    if length > 32768:
        over_32k.append({
            'index': i,
            'length': length,
            'doc_words': len(doc.split()),
            'summary_words': len(summary.split()),
        })

print("\n" + "="*80)
print("Training Data Length Analysis")
print("="*80)

import numpy as np
print(f"\nFirst 1000 samples:")
print(f"  Min:    {min(lengths):,} tokens")
print(f"  Max:    {max(lengths):,} tokens")
print(f"  Mean:   {np.mean(lengths):,.0f} tokens")
print(f"  Median: {np.median(lengths):,.0f} tokens")
print(f"  P95:    {np.percentile(lengths, 95):,.0f} tokens")
print(f"  P99:    {np.percentile(lengths, 99):,.0f} tokens")

print(f"\n⚠️  Samples >32K (will be truncated):")
print(f"  Count: {len(over_32k)}/{len(lengths)} ({len(over_32k)/len(lengths)*100:.1f}%)")

if over_32k:
    print(f"\n  Examples of samples that will be truncated:")
    for item in over_32k[:5]:
        print(f"    Sample {item['index']}: {item['length']:,} tokens ({item['doc_words']:,} words)")

print("\n" + "="*80)
print("Recommendation")
print("="*80)

if over_32k:
    print("❌ Training data contains samples >32K tokens!")
    print("   These will be silently truncated, causing:")
    print("   - Loss of information (endings cut off)")
    print("   - Model learns from incomplete documents")
    print("   - Poor performance on long documents")
    print("")
    print("Fix options:")
    print("  1. Filter out >32K samples before training")
    print("  2. Use chunking strategy for long documents")
    print("  3. Increase max_seq_length (if hardware allows)")
else:
    print("✅ All samples fit in 32K context window")
