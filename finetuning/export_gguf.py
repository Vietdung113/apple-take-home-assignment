"""Export QLoRA checkpoint to GGUF format for llama.cpp inference.

Usage:
    python export_gguf.py --checkpoint output/sft_base_xxx/checkpoint-400
    python export_gguf.py --checkpoint output/sft_base_xxx/checkpoint-400 --quantization q4_k_m
"""

import argparse
import sys
from pathlib import Path

from unsloth import FastLanguageModel


def export_to_gguf(checkpoint_path: Path, quantization: str = "q4_k_m"):
    """Export checkpoint to GGUF format.

    Args:
        checkpoint_path: Path to checkpoint directory
        quantization: GGUF quantization method (q4_k_m, q5_k_m, q8_0, f16, f32)
    """

    print("=" * 80)
    print("Export QLoRA Checkpoint to GGUF")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Quantization: {quantization}")
    print()

    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    # Load checkpoint with 4-bit quantization to save memory
    print("Loading checkpoint with 4-bit quantization...")
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_path),
        max_seq_length=32768,
        load_in_4bit=True,  # Use 4-bit to reduce memory usage
        dtype=None,
    )

    print("✓ Checkpoint loaded")
    print()

    # Export to GGUF
    output_dir = checkpoint_path / "gguf"
    output_dir.mkdir(exist_ok=True)

    print(f"Exporting to GGUF ({quantization})...")
    print(f"Output: {output_dir}")
    print()

    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=quantization,
    )

    print()
    print("=" * 80)
    print("Export Complete!")
    print("=" * 80)

    # List exported files (Unsloth exports to gguf_gguf subdirectory)
    gguf_dir = checkpoint_path / "gguf_gguf"
    gguf_files = list(gguf_dir.glob("*.gguf")) if gguf_dir.exists() else []

    if gguf_files:
        print(f"\nExported files:")
        for f in gguf_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  ✓ {f.name} ({size_mb:.1f} MB)")

        print()
        print("To use with llama.cpp:")
        print(f"  llama-server --model {gguf_files[0]} --port 8100 --ctx-size 32768")
    else:
        print("\n⚠️  Warning: No .gguf files found. Check output directory.")

    print()


def main():
    parser = argparse.ArgumentParser(description="Export QLoRA checkpoint to GGUF")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16", "f32"],
        help="GGUF quantization method (default: q4_k_m)"
    )

    args = parser.parse_args()

    try:
        export_to_gguf(args.checkpoint, args.quantization)
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
