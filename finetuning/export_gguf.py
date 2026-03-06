"""Export LoRA adapter + base model to GGUF format for llama.cpp.

Usage:
    python export_gguf.py --checkpoint output/sft_8k/checkpoint-1000
    python export_gguf.py --checkpoint output/sft_8k/checkpoint-1000 --quant q4_k_m
    python export_gguf.py --checkpoint output/sft_8k/checkpoint-1000 --output my_model.gguf
"""

import argparse
import os
from pathlib import Path

from unsloth import FastLanguageModel


def export_gguf(checkpoint_path: str, output_path: str = None, quantization: str = "q4_k_m"):
    """Export merged model (base + adapter) to GGUF format."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    # Read adapter config to get base model
    import json
    config_path = checkpoint_path / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)

    base_model = adapter_config.get("base_model_name_or_path", "unsloth/Qwen3-0.6B")

    print(f"Loading base model: {base_model}")
    print(f"Loading adapter from: {checkpoint_path}")

    # Load model + adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        load_in_4bit=False,  # Load in fp16 for better GGUF quality
    )

    # Load adapter weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(checkpoint_path))

    # Merge adapter into base model
    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    # Save merged model first (needed for GGUF export)
    merged_dir = checkpoint_path.parent / f"{checkpoint_path.name}_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {merged_dir}")
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # Determine output path
    if output_path is None:
        checkpoint_name = checkpoint_path.name
        output_path = checkpoint_path.parent / f"{checkpoint_name}_{quantization}.gguf"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to GGUF: {output_path}")
    print(f"Quantization: {quantization}")

    # Export to GGUF (from merged model directory)
    model.save_pretrained_gguf(
        str(merged_dir),
        tokenizer,
        quantization_method=quantization,
    )

    # Find generated GGUF file (Unsloth creates in _gguf subdirectory)
    import shutil
    gguf_subdir = Path(str(merged_dir) + "_gguf")

    # Look in both locations
    gguf_files = list(merged_dir.glob("*.gguf")) + list(gguf_subdir.glob("*.gguf"))

    if gguf_files:
        # Find the quantized GGUF file (matches quantization method)
        quant_upper = quantization.upper().replace("_", "_")
        generated_file = None
        for f in gguf_files:
            if quant_upper in f.name.upper():
                generated_file = f
                break

        if generated_file is None:
            generated_file = gguf_files[-1]  # Fallback to most recent

        # Move to final output location
        shutil.move(str(generated_file), str(output_path))
        print(f"✅ Moved to: {output_path}")

        # Show file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Export complete!")
        print(f"   File: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"\nTest with llama.cpp:")
        print(f"   ./llama-server -m {output_path} --port 8100")

        # Cleanup merged model directories
        print(f"Cleaning up merged model directory: {merged_dir}")
        if merged_dir.exists():
            shutil.rmtree(str(merged_dir))
        if gguf_subdir.exists():
            shutil.rmtree(str(gguf_subdir))

        return str(output_path)
    else:
        raise RuntimeError("GGUF file not created!")


def main():
    parser = argparse.ArgumentParser(description="Export LoRA adapter to GGUF")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (e.g., output/sft_8k/checkpoint-1000)",
    )
    parser.add_argument(
        "--output",
        help="Output GGUF file path (default: auto-generated)",
    )
    parser.add_argument(
        "--quant",
        default="q4_k_m",
        choices=["f16", "f32", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q4_k_m", "q5_k_m", "q6_k"],
        help="Quantization method (default: q4_k_m)",
    )

    args = parser.parse_args()

    try:
        export_gguf(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            quantization=args.quant,
        )
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
