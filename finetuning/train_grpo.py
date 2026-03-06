"""QLoRA fine-tuning Qwen3-0.6B for summarization with GRPO (RL).

Reward functions:
  - ROUGE-L F1 between generated summary and reference summary
  - Length penalty: penalise summaries that are too short or too long
  - Format reward: encourage clean, no-tag output

Run on a vast.ai GPU instance via setup.sh.

Usage:
    python train.py                         # configs/train.yaml, smoke test
    python train.py --max-samples 0         # full dataset
    python train.py --export-gguf           # train + export GGUF
"""

import argparse
import re
import pathlib
import yaml
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ── Prompt template (matches serving pipeline exactly) ───────────────────

SUMMARIZE_CHUNK_PROMPT = """\
Read the following text carefully.

Step 1: Identify the main topic and key entities.
Step 2: List the 3-5 most important facts or events.
Step 3: Write a concise summary covering these key points.

Text:
{chunk}

Summary: /no_think"""


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Reward functions ─────────────────────────────────────────────────────
# Signature: (prompts, completions, **kwargs) -> list[float]
# `completions` is list of [{"role": "assistant", "content": "..."}]
# Extra dataset columns are passed via **kwargs.

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_reward_func(completions, answer, **kwargs) -> list[float]:
    """Reward = ROUGE-L F1 between generated summary and reference (0-2 scale)."""
    rewards = []
    for completion, ref in zip(completions, answer):
        gen = completion[0]["content"].strip()
        gen = re.sub(r"<think>.*?</think>\s*", "", gen, flags=re.DOTALL).strip()
        score = _scorer.score(ref, gen)
        rewards.append(score["rougeL"].fmeasure * 2.0)  # scale to 0-2
    return rewards


def length_reward_func(completions, answer, **kwargs) -> list[float]:
    """Penalise summaries too short (<30% of ref) or too long (>200% of ref)."""
    rewards = []
    for completion, ref in zip(completions, answer):
        gen = completion[0]["content"].strip()
        gen = re.sub(r"<think>.*?</think>\s*", "", gen, flags=re.DOTALL).strip()
        ref_len = max(len(ref.split()), 1)
        gen_len = len(gen.split())
        ratio = gen_len / ref_len
        if 0.3 <= ratio <= 2.0:
            rewards.append(0.5)
        elif ratio < 0.1 or ratio > 3.0:
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards


def no_tag_reward_func(completions, **kwargs) -> list[float]:
    """Penalise leftover thinking tags or markdown artifacts."""
    rewards = []
    for completion in completions:
        gen = completion[0]["content"]
        has_tags = bool(re.search(r"<think>|</think>|```", gen))
        rewards.append(-0.5 if has_tags else 0.3)
    return rewards


# ── Data preparation ─────────────────────────────────────────────────────

def prepare_dataset(config: dict):
    """Load GovReport → format as GRPO prompts with 'answer' column."""
    ds_name = config["dataset"]
    max_train = config.get("max_train_samples", 0)
    max_val = config.get("max_val_samples", 0)
    max_seq = config.get("max_seq_length", 4096)
    max_chunk_chars = (max_seq - 600) * 4

    print(f"Loading dataset: {ds_name}")
    raw = load_dataset(ds_name)

    train_ds = raw["train"]
    val_ds = raw["validation"]

    if max_train > 0:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_val > 0:
        val_ds = val_ds.select(range(min(max_val, len(val_ds))))

    def format_row(example):
        doc = example["report"][:max_chunk_chars]
        example["prompt"] = [
            {"role": "user", "content": SUMMARIZE_CHUNK_PROMPT.format(chunk=doc)},
        ]
        example["answer"] = example["summary"][:2000]
        return example

    train_ds = train_ds.map(format_row, desc="Formatting train")
    val_ds = val_ds.map(format_row, desc="Formatting val")

    # Keep only needed columns
    keep = ["prompt", "answer"]
    train_ds = train_ds.select_columns(keep)
    val_ds = val_ds.select_columns(keep)

    print(f"  Train: {len(train_ds)} examples")
    print(f"  Val:   {len(val_ds)} examples")
    return train_ds, val_ds


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with Unsloth")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--max-samples", type=int, help="Override max_train_samples")
    parser.add_argument("--export-gguf", action="store_true", help="Export GGUF after training")
    args = parser.parse_args()

    config_path = pathlib.Path(__file__).parent / args.config
    config = load_config(config_path)

    if args.max_samples is not None:
        config["max_train_samples"] = args.max_samples
        config["max_val_samples"] = max(2, args.max_samples // 5)

    max_seq = config.get("max_seq_length", 4096)
    lora_rank = config.get("lora_rank", 16)

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading model: {config['model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"],
        max_seq_length=max_seq,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
    )

    # ── Apply QLoRA ──────────────────────────────────────────────────
    print("Applying QLoRA adapters ...")
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

    # ── Data ─────────────────────────────────────────────────────────
    train_ds, val_ds = prepare_dataset(config)

    # ── GRPO Trainer ─────────────────────────────────────────────────
    output_dir = config.get("output_dir", "output/grpo")
    max_prompt_length = config.get("max_prompt_length", 3072)
    max_completion_length = max_seq - max_prompt_length

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=config.get("learning_rate", 5e-6),
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=config.get("weight_decay", 0.1),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        optim="paged_adamw_8bit",
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        num_generations=config.get("num_generations", 4),
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=config.get("max_steps", 250),
        save_steps=config.get("save_steps", 50),
        logging_steps=config.get("logging_steps", 1),
        max_grad_norm=0.1,
        bf16=config.get("bf16", True),
        seed=config.get("seed", 42),
        report_to="wandb" if config.get("use_wandb", True) else "none",
    )

    # ── Init wandb ───────────────────────────────────────────────────
    if config.get("use_wandb", True):
        import wandb
        wandb.init(
            project=config.get("wandb_project", "qwen3-govreport-grpo"),
            name=config.get("wandb_run_name"),
            config=config,
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            rouge_reward_func,
            length_reward_func,
            no_tag_reward_func,
        ],
        args=training_args,
        train_dataset=train_ds,
    )

    print(f"\n{'='*60}")
    print("Starting GRPO training ...")
    print(f"{'='*60}\n")

    trainer.train()

    # ── Save adapter ─────────────────────────────────────────────────
    adapter_dir = config.get("adapter_dir", f"{output_dir}/adapter")
    print(f"\nSaving LoRA adapter to {adapter_dir}")
    model.save_lora(adapter_dir)

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
