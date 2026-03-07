"""Progressive training: 8K → 16K → 32K with curriculum learning.

Strategy:
1. Train on 8K data (3 epochs) - learn short document summarization
2. Continue from 8K checkpoint on 16K data (2 epochs) - adapt to medium docs
3. Continue from 16K checkpoint on 32K data (1 epoch) - adapt to long docs

This curriculum learning approach helps the model progressively learn to handle
longer contexts while building on previously learned patterns.

Base model: Qwen3-0.6B-Instruct (instruction-tuned, better at following prompts)

Usage:
    python train_progressive.py --output-base output/progressive
    python train_progressive.py --output-base output/progressive --smoke-test
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime


def run_stage(
    stage_name: str,
    config_path: str,
    num_epochs: int,
    output_dir: str,
    adapter_from: str = None,
    smoke_test: bool = False,
):
    """Run one training stage."""
    print(f"\n{'='*80}")
    print(f"STAGE: {stage_name}")
    print(f"{'='*80}")
    print(f"  Config: {config_path}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Output: {output_dir}")
    if adapter_from:
        print(f"  Resume from adapter: {adapter_from}")
    print()

    cmd = [
        "uv", "run", "python", "train_sft.py",
        "--config", config_path,
        "--num-epochs", str(num_epochs),
    ]

    if smoke_test:
        cmd.extend(["--max-samples", "50"])

    if adapter_from:
        cmd.extend(["--resume-from", adapter_from])

    try:
        result = subprocess.run(cmd, check=True, cwd=pathlib.Path(__file__).parent)
        print(f"\n✅ {stage_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {stage_name} failed with error: {e}")
        return False


def find_last_checkpoint(output_dir: str) -> str | None:
    """Find the last checkpoint directory."""
    output_path = pathlib.Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )

    if checkpoints:
        return str(checkpoints[-1])
    return None


def copy_checkpoint_as_base(checkpoint_dir: str, target_config: str):
    """Copy checkpoint to be used as base model for next stage.

    This modifies the config to point to the checkpoint adapter.
    """
    import yaml

    config_path = pathlib.Path(target_config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update config to use checkpoint as base
    # Note: For Unsloth, we need to load base model then apply adapter separately
    # This is handled in the training script
    print(f"  Using checkpoint {checkpoint_dir} for next stage")
    print(f"  Note: Will manually load adapter in next stage")


def main():
    parser = argparse.ArgumentParser(description="Progressive training 8K→16K→32K")
    parser.add_argument(
        "--output-base",
        default="output/progressive",
        help="Base output directory"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with max-samples=50 for testing"
    )
    parser.add_argument(
        "--start-from",
        choices=["8k", "16k", "32k"],
        default="8k",
        help="Start from this stage (skip previous stages)"
    )
    args = parser.parse_args()

    base_dir = pathlib.Path(__file__).parent
    output_base = base_dir / args.output_base
    output_base.mkdir(parents=True, exist_ok=True)

    # Log file
    log_file = output_base / f"progressive_training_{datetime.now():%Y%m%d_%H%M%S}.log"
    print(f"Logging to: {log_file}")

    stages = [
        {
            "name": "Stage 1: 8K (3 epochs)",
            "config": "configs/sft_8k.yaml",
            "epochs": 3,
            "output": output_base / "stage1_8k",
            "checkpoint_from": None,
        },
        {
            "name": "Stage 2: 16K (2 epochs)",
            "config": "configs/sft_16k.yaml",
            "epochs": 2,
            "output": output_base / "stage2_16k",
            "checkpoint_from": output_base / "stage1_8k",
        },
        {
            "name": "Stage 3: 32K (1 epoch)",
            "config": "configs/sft_32k.yaml",
            "epochs": 1,
            "output": output_base / "stage3_32k",
            "checkpoint_from": output_base / "stage2_16k",
        },
    ]

    # Filter stages based on start_from
    stage_names = {"8k": 0, "16k": 1, "32k": 2}
    start_idx = stage_names[args.start_from]
    stages = stages[start_idx:]

    print(f"\n{'='*80}")
    print("PROGRESSIVE TRAINING PLAN")
    print(f"{'='*80}")
    for i, stage in enumerate(stages, start=start_idx+1):
        print(f"{i}. {stage['name']}")
        print(f"   Config: {stage['config']}")
        print(f"   Epochs: {stage['epochs']}")
        print(f"   Output: {stage['output']}")
        if stage['checkpoint_from']:
            print(f"   Resume from: {stage['checkpoint_from']}")
    print()

    if args.smoke_test:
        print("⚠️  SMOKE TEST MODE: Using max-samples=50")
        print()

    input("Press ENTER to start training, or Ctrl+C to cancel...")

    # Run stages sequentially
    for stage in stages:
        # Update output_dir in config dynamically
        import yaml
        import tempfile

        config_path = base_dir / stage["config"]
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Create temporary config with updated output_dir
        config["output_dir"] = str(stage["output"])
        config["num_train_epochs"] = stage["epochs"]

        # Find adapter from previous stage if needed
        adapter_from = None
        if stage["checkpoint_from"]:
            # Use the adapter directory from previous stage
            adapter_dir = pathlib.Path(stage["checkpoint_from"]) / "adapter"
            if adapter_dir.exists():
                adapter_from = str(adapter_dir)
                print(f"  Found adapter from previous stage: {adapter_from}")
            else:
                print(f"  Warning: Adapter not found at {adapter_dir}")
                print(f"  Will train from scratch")

        # Save temporary config
        temp_config = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, dir=base_dir
        )
        yaml.dump(config, temp_config)
        temp_config.close()

        success = run_stage(
            stage_name=stage["name"],
            config_path=temp_config.name,
            num_epochs=stage["epochs"],
            output_dir=str(stage["output"]),
            adapter_from=adapter_from,
            smoke_test=args.smoke_test,
        )

        # Cleanup temp config
        pathlib.Path(temp_config.name).unlink()

        if not success:
            print(f"\n❌ Training failed at {stage['name']}")
            print("Stopping progressive training.")
            sys.exit(1)

    print(f"\n{'='*80}")
    print("✅ PROGRESSIVE TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Final model: {stages[-1]['output']}")
    print()


if __name__ == "__main__":
    main()
