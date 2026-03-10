import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset, concatenate_datasets
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent / "config"))
from prompt_loader import get_training_prompt_base_model

# Load environment variables
load_dotenv()

# Paths
DEFAULT_CONFIG = Path(__file__).parent / "config" / "training.yaml"
OUTPUT_DIR = Path(__file__).parent / "output" / f"sft_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(data_file: Path, val_file: Path | None, config: dict, max_samples: int | None = None) -> tuple[Dataset, Dataset]:
    """Load training data and validation data (split or separate file).

    Data format: {"document": "...", "summary": "..."}
    """

    # Load training data
    print(f"Loading training data from: {data_file}")
    samples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"  Loaded: {len(samples)} samples")

    # Limit samples if requested
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]
        print(f"  Limited to: {len(samples)} samples")

    # Load or split validation data
    if val_file and val_file.exists():
        # Use separate validation file
        print(f"Loading validation data from: {val_file}")
        val_samples = []
        with open(val_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    val_samples.append(json.loads(line))

        train_samples = samples
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val:   {len(val_samples)} samples (from file)")
    else:
        # Split from training data
        print(f"Splitting validation from training data")
        seed = config["hardware"]["seed"]
        val_split = config["data"]["val_split"]

        random.seed(seed)
        random.shuffle(samples)
        split_idx = int(len(samples) * (1 - val_split))

        val_samples = samples[split_idx:]
        train_samples = samples[:split_idx]

        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val:   {len(val_samples)} samples (split {val_split*100:.0f}%)")

    return Dataset.from_list(train_samples), Dataset.from_list(val_samples)


def train(
    train_ds: Dataset,
    val_ds: Dataset,
    output_dir: Path,
    config: dict,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    resume_from: str | None = None,
):
    """Train QLoRA adapter with Unsloth + TRL on BASE model."""

    # Load model name from config
    model_name = config["model_name"]
    max_seq_length = config["max_seq_length"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    log_cfg = config["logging"]
    hw_cfg = config["hardware"]

    # Override config with CLI arguments
    num_epochs = num_epochs if num_epochs else train_cfg["num_epochs"]
    batch_size = batch_size if batch_size else train_cfg["batch_size"]
    grad_accum = grad_accum if grad_accum else train_cfg["grad_accum_steps"]

    # Load base model with 4-bit quantization
    print(f"\nLoading model: {model_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  4-bit quantization: enabled")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,  # Auto-detect
    )

    # Disable Qwen3 thinking budget for summarization task
    if hasattr(model.config, 'think_budget'):
        model.config.think_budget = 0
        print("  Disabled Qwen3 thinking budget (not needed for summarization)")

    # Apply QLoRA adapters
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume_from)
    else:
        adapter_type = "DoRA" if lora_cfg.get("use_dora", False) else "QLoRA"
        print(f"\nApplying {adapter_type} adapters:")
        print(f"  Rank: {lora_cfg['rank']}")
        print(f"  Alpha: {lora_cfg['alpha']}")
        print(f"  Dropout: {lora_cfg['dropout']}")
        if lora_cfg.get("use_dora", False):
            print(f"  DoRA: enabled (magnitude + direction learning)")
        print(f"  Target modules: {', '.join(lora_cfg['target_modules'])}")

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=hw_cfg["seed"],
            use_dora=lora_cfg.get("use_dora", False),
        )

    # Preprocessing: format samples
    def format_sample(example):
        """Format sample for training. Uses centralized prompt from config/prompts.yaml"""
        document = example["document"]
        summary = example["summary"]
        # Add EOS token so model learns when to stop
        text = get_training_prompt_base_model(document, summary) + tokenizer.eos_token

        # Add token length for filtering
        tokens = tokenizer(text, truncation=False, add_special_tokens=False)['input_ids']
        example["text"] = text
        example["token_length"] = len(tokens)
        return example

    print("\nPreprocessing data...")
    num_workers = min(8, len(train_ds) // 100) if len(train_ds) > 100 else None
    train_ds = train_ds.map(format_sample, num_proc=num_workers)
    val_ds = val_ds.map(format_sample, num_proc=None)

    # Filter samples >32K tokens
    print(f"\nFiltering samples >32K tokens...")
    train_before = len(train_ds)
    val_before = len(val_ds)

    train_ds = train_ds.filter(lambda x: x["token_length"] <= max_seq_length)
    val_ds = val_ds.filter(lambda x: x["token_length"] <= max_seq_length)

    train_filtered = train_before - len(train_ds)
    val_filtered = val_before - len(val_ds)

    print(f"  Train: Removed {train_filtered}/{train_before} samples ({train_filtered/train_before*100:.1f}%)")
    print(f"  Val:   Removed {val_filtered}/{val_before} samples ({val_filtered/val_before*100:.1f}%)")
    print(f"  Final: {len(train_ds)} train, {len(val_ds)} val")

    # Sort by token length to stabilize GPU memory usage
    # Similar-length samples are batched together → less padding waste, no memory spikes
    train_ds = train_ds.sort("token_length")
    print(f"  Sorted train by token_length (min={train_ds[0]['token_length']}, max={train_ds[-1]['token_length']})")

    # Show example
    print("\nExample (first 500 chars):")
    print("-" * 80)
    print(train_ds[0]["text"][:500] + "...")
    print("-" * 80)

    # Training configuration
    effective_batch = batch_size * grad_accum
    print(f"\nTraining configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Warmup ratio: {train_cfg['warmup_ratio']}")
    print(f"  Weight decay: {train_cfg['weight_decay']}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # WandB configuration
    wandb_project = "qwen3-govreport-summarization"
    wandb_run_name = f"{model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if os.getenv("WANDB_DISABLED", "false").lower() != "true":
        import wandb

        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_RUN_NAME"] = wandb_run_name

        # Initialize with tags and config
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=[
                model_name.split('/')[-1],  # e.g., "Qwen3-0.6B"
                "govreport",
                "summarization",
                "qlora",
                f"r{lora_cfg['rank']}",
                f"bs{batch_size}x{grad_accum}",
            ],
            config={
                "model": model_name,
                "dataset": "govreport",
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "max_seq_length": max_seq_length,
                "lora_rank": lora_cfg["rank"],
                "lora_alpha": lora_cfg["alpha"],
                "learning_rate": train_cfg["learning_rate"],
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": batch_size * grad_accum,
            }
        )

        print(f"\nWandB logging:")
        print(f"  Project: {wandb_project}")
        print(f"  Run: {wandb_run_name}")
        print(f"  URL: {wandb.run.url}")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 2),
        gradient_accumulation_steps=grad_accum,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optimizer"],
        max_seq_length=max_seq_length,
        packing=False,
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=log_cfg["steps"],
        save_steps=log_cfg["save_steps"],
        eval_steps=log_cfg["eval_steps"],
        eval_strategy="steps",
        save_total_limit=log_cfg["save_total_limit"],
        bf16=hw_cfg["use_bf16"],
        seed=hw_cfg["seed"],
        dataloader_num_workers=hw_cfg["dataloader_num_workers"],
        report_to="wandb" if os.getenv("WANDB_DISABLED", "false").lower() != "true" else "none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
    )

    # Train
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")

    trainer.train(resume_from_checkpoint=resume_from if resume_from else None)

    # Save adapter
    adapter_dir = output_dir / "adapter"
    print(f"\n{'='*80}")
    print(f"Saving adapter to: {adapter_dir}")
    print(f"{'='*80}")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"\n✅ Training complete!")
    print(f"   Adapter saved: {adapter_dir}")

    return model, tokenizer


def export_gguf(model, tokenizer, output_dir: Path, quant: str = "q4_k_m"):
    """Export adapter to GGUF format for llama.cpp."""
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Exporting to GGUF format: {quant}")
    print(f"Output: {gguf_dir}")
    print(f"{'='*80}\n")

    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=quant,
    )

    print(f"\n✅ GGUF export complete!")
    print(f"   Model: {gguf_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune base model for summarization")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML config file")
    parser.add_argument("--data", type=Path, help="Path to training data JSONL file (default from config)")
    parser.add_argument("--val-data", type=Path, help="Path to validation data JSONL file (default from config)")
    parser.add_argument("--max-samples", type=int, help="Limit training samples (for testing)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Per-device batch size (overrides config)")
    parser.add_argument("--grad-accum", type=int, help="Gradient accumulation steps (overrides config)")
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint directory")
    parser.add_argument("--export-gguf", action="store_true", help="Export to GGUF after training")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Use data paths from config if not provided via CLI
    if not args.data:
        args.data = Path(__file__).parent / config["data"]["train_file"]
    if not args.val_data:
        args.val_data = Path(__file__).parent / config["data"]["val_file"]

    # Load data
    train_ds, val_ds = load_data(args.data, args.val_data, config, max_samples=args.max_samples)

    # Train
    model, tokenizer = train(
        train_ds,
        val_ds,
        output_dir=args.output_dir,
        config=config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        resume_from=args.resume_from,
    )

    # Export GGUF if requested
    if args.export_gguf:
        export_gguf(model, tokenizer, args.output_dir)

    print("\n" + "="*80)
    print("All done! 🎉")
    print("="*80)


if __name__ == "__main__":
    main()
