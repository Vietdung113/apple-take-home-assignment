"""SFT fine-tuning Qwen3-0.6B for summarization with QLoRA + TRL SFTTrainer.

Trains on pre-filtered JSONL data (from prepare_data.py) with automatic
chat template application and loss masking on user tokens.

Run on a vast.ai GPU instance via setup.sh.

Usage:
    python train_sft.py --config configs/sft_8k.yaml                          # full run
    python train_sft.py --config configs/sft_8k.yaml --max-samples 10         # smoke test
    python train_sft.py --config configs/sft_8k.yaml --num-epochs 2           # 2 epochs
    python train_sft.py --config configs/sft_8k.yaml --max-samples 10 --num-epochs 2  # smoke test 2 epochs
    python train_sft.py --config configs/sft_8k.yaml --export-gguf            # train + GGUF
"""

import argparse
import json
import pathlib
import sys

import torch
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def detect_device():
    """Detect available device: CUDA GPU, MPS (Apple Silicon), or CPU."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Using CUDA GPU: {gpu_name}")
        return device, True
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"🍎 Using Apple Silicon MPS (Metal)")
        return device, False  # MPS doesn't support Unsloth
    else:
        device = "cpu"
        print(f"💻 Using CPU (slow, not recommended for training)")
        return device, False


# Detect device
DEVICE, USE_UNSLOTH = detect_device()

# Import Unsloth only if CUDA GPU available
if USE_UNSLOTH:
    from unsloth import FastLanguageModel
else:
    # Fallback to standard transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data loading ─────────────────────────────────────────────────────────


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_sft_data(config: dict, max_samples: int | None = None) -> tuple[Dataset, Dataset]:
    """Load pre-filtered JSONL files produced by prepare_data.py."""
    base_dir = pathlib.Path(__file__).parent
    data_dir = base_dir / config.get("data_dir", "data")
    label = config["data_label"]

    train_path = data_dir / f"sft_{label}_train.jsonl"
    val_path = data_dir / f"sft_{label}_validation.jsonl"

    print(f"Loading data: {train_path}")
    train_rows = load_jsonl(str(train_path))
    val_rows = load_jsonl(str(val_path))

    if max_samples is not None and max_samples > 0:
        train_rows = train_rows[:max_samples]
        val_rows = val_rows[:max(2, max_samples // 5)]

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    return train_ds, val_ds


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning with Unsloth")
    parser.add_argument("--config", default="configs/sft_8k.yaml")
    parser.add_argument("--max-samples", type=int, help="Override training sample count")
    parser.add_argument("--num-epochs", type=int, help="Override num_train_epochs")
    parser.add_argument("--export-gguf", action="store_true", help="Export GGUF after training")
    args = parser.parse_args()

    config_path = pathlib.Path(__file__).parent / args.config
    config = load_config(config_path)

    # Override config with CLI args
    if args.num_epochs is not None:
        config["num_train_epochs"] = args.num_epochs

    max_seq = config.get("max_seq_length", 8192)
    lora_rank = config.get("lora_rank", 32)

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading model: {config['model']}")

    if USE_UNSLOTH:
        # CUDA GPU: Use Unsloth (fast)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model"],
            max_seq_length=max_seq,
            load_in_4bit=True,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.6,
        )

        # Apply QLoRA with Unsloth
        print("Applying QLoRA adapters (Unsloth) ...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            lora_alpha=config.get("lora_alpha", lora_rank),
            lora_dropout=config.get("lora_dropout", 0),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.get("seed", 42),
        )
    else:
        # CPU/MPS: Use standard transformers + PEFT
        print(f"⚠️  Warning: Training on {DEVICE} will be slower than CUDA GPU")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with 4-bit quantization if possible (CPU doesn't support it)
        if DEVICE == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config["model"],
                quantization_config=bnb_config,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # CPU/MPS: Load in fp32 or bf16
            dtype = torch.bfloat16 if DEVICE == "mps" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                config["model"],
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            if DEVICE == "mps":
                model = model.to("mps")

        # Apply LoRA
        print("Applying QLoRA adapters (PEFT) ...")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=config.get("lora_alpha", lora_rank),
            target_modules=config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            lora_dropout=config.get("lora_dropout", 0),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # ── Data ─────────────────────────────────────────────────────────
    train_ds, val_ds = load_sft_data(config, max_samples=args.max_samples)

    # Preprocess: add "text" field with formatted chat template
    def add_text_field(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return example

    train_ds = train_ds.map(add_text_field)
    val_ds = val_ds.map(add_text_field)

    # ── SFT Trainer ──────────────────────────────────────────────────
    output_dir = config.get("output_dir", "output/sft")

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=config.get("learning_rate", 2e-4),
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=config.get("weight_decay", 0.1),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        optim="paged_adamw_8bit",
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        num_train_epochs=config.get("num_train_epochs", 1),
        max_seq_length=max_seq,
        packing=False,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 200),
        eval_steps=config.get("eval_steps", 200),
        eval_strategy="steps",
        save_total_limit=3,
        max_grad_norm=1.0,
        bf16=config.get("bf16", True),
        seed=config.get("seed", 42),
        report_to="wandb" if config.get("use_wandb", True) else "none",
    )

    # ── Init wandb ───────────────────────────────────────────────────
    if config.get("use_wandb", True):
        import wandb
        wandb.init(
            project=config.get("wandb_project", "qwen3-govreport-sft"),
            name=config.get("wandb_run_name", f"sft-{config['data_label']}"),
            config=config,
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",  # Use preprocessed text field
    )

    print(f"\n{'='*60}")
    print(f"Starting SFT training ({config['data_label']}) ...")
    print(f"  max_seq_length: {max_seq}")
    print(f"  batch_size: {config.get('per_device_train_batch_size', 4)}")
    print(f"  grad_accum: {config.get('gradient_accumulation_steps', 4)}")
    print(f"  effective_batch: {config.get('per_device_train_batch_size', 4) * config.get('gradient_accumulation_steps', 4)}")
    print(f"  train samples: {len(train_ds)}")
    print(f"{'='*60}\n")

    trainer.train()

    # ── Save adapter ─────────────────────────────────────────────────
    adapter_dir = config.get("adapter_dir", f"{output_dir}/adapter")
    print(f"\nSaving LoRA adapter to {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # ── Export GGUF ──────────────────────────────────────────────────
    if args.export_gguf:
        gguf_dir = config.get("gguf_dir", f"{output_dir}/gguf")
        quant = config.get("gguf_quant", "q4_k_m")
        print(f"\nExporting GGUF ({quant}) to {gguf_dir}")
        model.save_pretrained_gguf(
            gguf_dir, tokenizer, quantization_method=quant,
        )
        print(f"GGUF model saved to: {gguf_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
