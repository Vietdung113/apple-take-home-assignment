"""GRPO training with LLM-as-reward (Nemotron 9B).

GRPO (Group Relative Policy Optimization) trains the model to maximize
LLM judge scores instead of just imitating reference summaries.

Usage:
    # Train from SFT checkpoint
    python train_grpo.py --sft-checkpoint output/sft_base_xxx/checkpoint-2080

    # Test reward function first
    python train_grpo.py --test-reward --num-samples 3
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import torch
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent / "config"))
from prompt_loader import get_inference_prompt_base_model

# Load environment variables
load_dotenv()

# Paths
DEFAULT_CONFIG = Path(__file__).parent / "config" / "grpo.yaml"
OUTPUT_DIR = Path(__file__).parent / "output" / f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Nemotron API
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
JUDGE_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"


def load_config(config_path: Path) -> dict:
    """Load GRPO configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Reward Functions ─────────────────────────────────────────────────────────


async def judge_reward_func(completions, document, **kwargs) -> list[float]:
    """Async reward function: call Nemotron 9B judge to score summaries.

    GRPOTrainer calls this with completions (list of strings) and dataset columns.
    Returns list of floats (rewards).
    """
    config = load_config(DEFAULT_CONFIG)
    weights = config["grpo"]["reward_weights"]
    judge_cfg = config["grpo"]["judge"]
    semaphore = asyncio.Semaphore(judge_cfg["max_concurrent"])

    async def score_one(doc: str, summary: str) -> float:
        async with semaphore:
            prompt = f"""Rate this summary on 4 dimensions (1-5 each). Respond in JSON only.

Document:
{doc}

Summary:
{summary}

Rate:
1. coverage: Does it capture all key points? (1-5)
2. specificity: Does it include specific details like numbers, dates, names? (1-5)
3. consistency: Is it factually consistent with the document? (1-5)
4. conciseness: Is it concise and well-structured? (1-5)

JSON response:
{{"coverage": <1-5>, "specificity": <1-5>, "consistency": <1-5>, "conciseness": <1-5>}}"""

            async with httpx.AsyncClient(base_url=NVIDIA_BASE_URL, timeout=judge_cfg["timeout"]) as client:
                headers = {
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": judge_cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.0,
                }

                for attempt in range(judge_cfg.get("retry_attempts", 2)):
                    try:
                        resp = await client.post("/chat/completions", json=payload, headers=headers)
                        resp.raise_for_status()
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"].strip()

                        # Parse JSON
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0].strip()

                        scores = json.loads(content)

                        # Weighted reward normalized to [0, 1]
                        reward = (
                            scores.get("coverage", 3) * weights["coverage"]
                            + scores.get("specificity", 3) * weights["specificity"]
                            + scores.get("consistency", 3) * weights["consistency"]
                            + scores.get("conciseness", 3) * weights["conciseness"]
                        ) / 5.0
                        return reward

                    except Exception as e:
                        if attempt == judge_cfg.get("retry_attempts", 2) - 1:
                            print(f"  Warning: Judge failed after retries: {e}")
                            return 0.6  # Neutral reward on failure
                        await asyncio.sleep(1)

            return 0.6

    # Score all completions
    tasks = [score_one(doc, comp) for doc, comp in zip(document, completions)]
    rewards = await asyncio.gather(*tasks)
    return list(rewards)


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_data(config: dict, max_samples: int | None = None) -> Dataset:
    """Load training data formatted for GRPOTrainer.

    GRPOTrainer expects dataset with 'prompt' column.
    Additional columns (like 'document') are passed to reward functions.
    """
    data_file = Path(__file__).parent / config["data"]["train_file"]
    print(f"Loading data from: {data_file}")

    samples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    # Format for GRPOTrainer: needs 'prompt' column
    formatted = []
    for sample in samples:
        prompt = get_inference_prompt_base_model(sample["document"])
        formatted.append({
            "prompt": prompt,
            "document": sample["document"],  # Passed to reward function
        })

    ds = Dataset.from_list(formatted)
    print(f"  Loaded {len(ds)} samples")
    return ds


# ── Test Reward ──────────────────────────────────────────────────────────────


async def test_reward_function(config: dict, num_samples: int = 3):
    """Test reward function on a few samples."""
    if not NVIDIA_API_KEY:
        print("Error: NVIDIA_API_KEY not found in .env")
        return

    print("=" * 80)
    print("Testing Reward Function")
    print("=" * 80)

    # Load model
    model_name = config["model"]["name"]
    print(f"\nLoading model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded\n")

    # Load test data
    data_file = Path(__file__).parent / config["data"]["train_file"]
    samples = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))

    print(f"Testing {len(samples)} samples\n")

    gen_cfg = config["grpo"]["generation"]

    for i, sample in enumerate(samples, 1):
        print(f"--- Sample {i}/{len(samples)} ---")
        document = sample["document"]
        print(f"Document: {len(document)} chars")

        # Generate summary
        prompt = get_inference_prompt_base_model(document)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"Summary: {len(summary.split())} words")

        # Score with judge
        rewards = await judge_reward_func(
            completions=[summary],
            document=[document],
        )
        print(f"Reward: {rewards[0]:.3f}")
        print()

    print("=" * 80)
    print("Reward function working!")
    print("=" * 80)


# ── GRPO Training ────────────────────────────────────────────────────────────


def train_grpo(config: dict, sft_checkpoint: str | None = None, output_dir: Path = OUTPUT_DIR):
    """Train with GRPO using TRL GRPOTrainer."""
    if not NVIDIA_API_KEY:
        print("Error: NVIDIA_API_KEY not found in .env")
        return

    print("=" * 80)
    print("GRPO Training")
    print("=" * 80)

    model_name = config["model"]["name"]
    grpo_cfg = config["grpo"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    log_cfg = config["logging"]
    hw_cfg = config["hardware"]

    # Load model
    model_source = sft_checkpoint if sft_checkpoint else model_name
    print(f"\nLoading model: {model_source}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )

    # Disable thinking budget
    if hasattr(model.config, "think_budget"):
        model.config.think_budget = 0
        print("  Disabled Qwen3 thinking budget")

    # Apply LoRA if not loading from SFT checkpoint
    if not sft_checkpoint:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=hw_cfg["seed"],
        )

    # Load data
    max_samples = config["data"].get("max_train_samples")
    train_ds = load_data(config, max_samples=max_samples)

    # GRPOConfig
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optimizer"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=hw_cfg["use_bf16"],
        seed=hw_cfg["seed"],
        logging_steps=log_cfg["steps"],
        save_steps=log_cfg["save_steps"],
        save_total_limit=log_cfg["save_total_limit"],
        # GRPO-specific
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=grpo_cfg["generation"]["max_tokens"],
        beta=grpo_cfg.get("beta", 0.0),
        report_to="wandb" if os.getenv("WANDB_DISABLED", "false").lower() != "true" else "none",
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=judge_reward_func,
        train_dataset=train_ds,
    )

    # Train
    print(f"\nStarting GRPO training...")
    print(f"  Output: {output_dir}")
    print(f"  Generations per sample: {grpo_cfg['num_generations']}")
    print(f"  Reward weights: {grpo_cfg['reward_weights']}")
    print(f"  Beta (KL penalty): {grpo_cfg.get('beta', 0.0)}")
    print()

    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    print(f"\nModel saved to: {output_dir / 'final'}")
    print("GRPO training complete!")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="GRPO training with LLM-as-reward")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="GRPO config file")
    parser.add_argument("--sft-checkpoint", type=str, help="SFT checkpoint to start from")
    parser.add_argument("--test-reward", action="store_true", help="Test reward function only")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples for testing")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.test_reward:
        asyncio.run(test_reward_function(config, args.num_samples))
    else:
        train_grpo(config, args.sft_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
