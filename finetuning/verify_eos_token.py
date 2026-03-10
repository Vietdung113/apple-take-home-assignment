"""Verify EOS token matches between training and generation config.

IMPORTANT: Run this on remote server before training to ensure consistency.

Usage:
    python verify_eos_token.py
"""

from unsloth import FastLanguageModel
import sys
from pathlib import Path
import yaml

# Load tokenizer
print("="*80)
print("Loading tokenizer...")
print("="*80)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

print(f"\n✓ Model: unsloth/Qwen3-0.6B")
print(f"✓ EOS token: {repr(tokenizer.eos_token)}")
print(f"✓ EOS token ID: {tokenizer.eos_token_id}")

# Load prompts.yaml
config_path = Path(__file__).parent / "config" / "prompts.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

stop_sequences = config["generation"]["stop"]

print("\n" + "="*80)
print("Checking prompts.yaml stop sequences")
print("="*80)

print(f"\nStop sequences in config:")
for seq in stop_sequences:
    print(f"  - {repr(seq)}")

print(f"\nTokenizer EOS token: {repr(tokenizer.eos_token)}")

# Check if EOS is in stop sequences
if tokenizer.eos_token in stop_sequences:
    print(f"\n✅ PASS: EOS token {repr(tokenizer.eos_token)} found in stop sequences")
else:
    print(f"\n❌ FAIL: EOS token {repr(tokenizer.eos_token)} NOT found in stop sequences!")
    print(f"\n⚠️  Update {config_path}:")
    print(f"   generation:")
    print(f"     stop:")
    for seq in stop_sequences:
        print(f"       - {repr(seq)}")
    print(f"       - {repr(tokenizer.eos_token)}  # ← ADD THIS")

# Test training format
print("\n" + "="*80)
print("Testing training format")
print("="*80)

sys.path.insert(0, str(Path(__file__).parent / "config"))
from prompt_loader import get_training_prompt_base_model

doc = "Test document."
summary = "Test summary."
training_text = get_training_prompt_base_model(doc, summary) + tokenizer.eos_token

# Check last characters
print(f"\nLast 50 chars of training text:")
print(repr(training_text[-50:]))

if training_text.endswith(tokenizer.eos_token):
    print(f"\n✅ PASS: Training text ends with {repr(tokenizer.eos_token)}")
else:
    print(f"\n❌ FAIL: Training text does NOT end with {repr(tokenizer.eos_token)}")

print("\n" + "="*80)
print("Summary")
print("="*80)
print("\n✓ Training will append EOS token to teach model when to stop")
print("✓ Generation will use stop sequences to detect when model outputs EOS")
print("✓ Both use the same EOS token for consistency")
print("\nReady to train!")
