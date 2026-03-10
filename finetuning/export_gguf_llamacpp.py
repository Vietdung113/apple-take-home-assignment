"""Export QLoRA adapter to GGUF using llama.cpp convert script.

Alternative to Unsloth's export - uses llama.cpp directly.

Usage:
    python export_gguf_llamacpp.py --checkpoint output/sft_20260308_163251/checkpoint-200
"""

import argparse
import subprocess
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(checkpoint_path: Path, output_dir: Path, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Merge adapter with base model and save as HuggingFace format."""

    print("="*80)
    print("Step 1: Merge Adapter with Base Model")
    print("="*80)
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Base model:   {base_model}")
    print(f"Output:       {output_dir}")
    print()

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    print("✓ Base model loaded")

    # Load adapter
    print(f"\nLoading adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    print("✓ Adapter loaded")

    # Merge
    print("\nMerging adapter with base model...")
    model = model.merge_and_unload()
    print("✓ Merged")

    # Save merged model
    print(f"\nSaving merged model to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✓ Saved merged model")

    return output_dir


def convert_to_gguf(merged_model_dir: Path, gguf_output_dir: Path, quant: str = "Q4_K_M"):
    """Convert merged model to GGUF using llama.cpp."""

    print()
    print("="*80)
    print("Step 2: Convert to GGUF using llama.cpp")
    print("="*80)
    print(f"Input:        {merged_model_dir}")
    print(f"Output:       {gguf_output_dir}")
    print(f"Quantization: {quant}")
    print()

    gguf_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if llama.cpp is available
    llamacpp_dir = Path.home() / ".unsloth" / "llama.cpp"

    if not llamacpp_dir.exists():
        print("Installing llama.cpp...")
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llamacpp_dir)
        ], check=True)

        # Build llama.cpp
        print("Building llama.cpp...")
        subprocess.run(["make", "-C", str(llamacpp_dir)], check=True)

    # Convert to FP16 GGUF first
    convert_script = llamacpp_dir / "convert_hf_to_gguf.py"
    fp16_output = gguf_output_dir / "model-fp16.gguf"

    print(f"\nConverting to FP16 GGUF...")
    subprocess.run([
        "python", str(convert_script),
        str(merged_model_dir),
        "--outfile", str(fp16_output),
        "--outtype", "f16"
    ], check=True)

    print(f"✓ FP16 GGUF created: {fp16_output}")

    # Quantize to target format
    quantize_bin = llamacpp_dir / "llama-quantize"
    final_output = gguf_output_dir / f"model-{quant.lower()}.gguf"

    print(f"\nQuantizing to {quant}...")
    subprocess.run([
        str(quantize_bin),
        str(fp16_output),
        str(final_output),
        quant
    ], check=True)

    print(f"✓ Quantized GGUF created: {final_output}")

    # Show file sizes
    print()
    print("="*80)
    print("Export Complete!")
    print("="*80)
    print(f"\nFP16 GGUF:       {fp16_output.name} ({fp16_output.stat().st_size / 1e9:.2f} GB)")
    print(f"Quantized GGUF:  {final_output.name} ({final_output.stat().st_size / 1e9:.2f} GB)")
    print(f"\nOutput directory: {gguf_output_dir}")
    print()

    return final_output


def main():
    parser = argparse.ArgumentParser(description="Export adapter to GGUF using llama.cpp")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to adapter checkpoint"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model (default: Qwen/Qwen2.5-3B-Instruct)"
    )

    parser.add_argument(
        "--quant",
        type=str,
        default="Q4_K_M",
        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"],
        help="Quantization method (default: Q4_K_M)"
    )

    args = parser.parse_args()

    # Paths
    checkpoint_path = args.checkpoint
    merged_dir = checkpoint_path / "merged"
    gguf_dir = checkpoint_path / "gguf"

    # Step 1: Merge adapter
    merge_adapter(checkpoint_path, merged_dir, args.base_model)

    # Step 2: Convert to GGUF
    convert_to_gguf(merged_dir, gguf_dir, args.quant)


if __name__ == "__main__":
    main()
