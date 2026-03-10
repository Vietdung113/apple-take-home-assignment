# Synthetic Data Generation Experiment

## Overview

This experiment tests whether synthetic summaries can improve model performance on specificity while maintaining coverage.

**Problem:** 84.7% of GovReport reference summaries lack specificity (score <4), causing the model to learn an "abstract" style instead of a detailed "executive summary" style.

**Solution:** Generate synthetic summaries with high specificity using Nemotron Nano 9B, then train on mixed datasets.

## Two-Tier Approach

### Tier 1: Enrichment (13,671 samples)
**Target:** Summaries with good coverage (≥4) but low specificity (<4)

**Strategy:** Enrich existing summaries by adding:
- Exact numbers, percentages, dollar amounts
- Specific dates and timeframes
- Full agency/program names
- Concrete examples from the document

**Prompt:** Uses reference summary as baseline, asks model to add more details

### Tier 2: Generation (1,963 samples)
**Target:** Summaries with both low coverage (<4) and low specificity (<4)

**Strategy:** Generate completely new summaries following expert analyst principles

**Prompt:** Full requirements for comprehensive + specific summaries

## Quick Start

### Step 1: Generate Synthetic Summaries

```bash
cd finetuning

# Set your NVIDIA API key
export NVIDIA_API_KEY=your_key_here

# Generate both tiers (recommended)
uv run python generate_synthetic_data_v2.py --tier both

# Or generate individually
uv run python generate_synthetic_data_v2.py --tier 1  # Enrichment only (~6-8 hours)
uv run python generate_synthetic_data_v2.py --tier 2  # Generation only (~1-2 hours)
```

**Expected output:**
- `data/synthetic_tier1_summaries.jsonl` - Enriched summaries (Tier 1)
- `data/synthetic_tier2_summaries.jsonl` - Generated summaries (Tier 2)
- Progress files for resuming if interrupted

**Pass rate expectation:**
- Tier 1: ~20-30% (enrichment is easier)
- Tier 2: ~10-20% (full generation is harder)

### Step 2: Create Training Datasets

```bash
# Combine synthetics with references to create 3 datasets
uv run python create_training_datasets.py
```

**Output:**
1. `data/dataset_baseline.jsonl` - High-quality references only (~2.8K samples)
2. `data/dataset_blend.jsonl` - 50/50 mix of references + synthetics (~5-6K samples)
3. `data/dataset_full.jsonl` - All references + all synthetics (~6-8K samples)

### Step 3: Train Three Adapters

```bash
# Baseline adapter (references only)
uv run python train_sft_base.py \
  --config config/training.yaml \
  --data data/dataset_baseline.jsonl \
  --output-dir output/adapter_baseline \
  --export-gguf

# Blend adapter (50/50 mix)
uv run python train_sft_base.py \
  --config config/training.yaml \
  --data data/dataset_blend.jsonl \
  --output-dir output/adapter_blend \
  --export-gguf

# Full synthetic adapter (all data)
uv run python train_sft_base.py \
  --config config/training.yaml \
  --data data/dataset_full.jsonl \
  --output-dir output/adapter_full \
  --export-gguf
```

**Training time:** ~2-3 hours per adapter on A100 40GB

### Step 4: Evaluate and Compare

```bash
cd ../eval

# Start inference servers for each adapter
llama-server --model ../models/Qwen3-0.6B-Q4_K_M.gguf --port 8100 &  # Base model
llama-server --model ../finetuning/output/adapter_baseline/final/gguf/model-Q4_K_M.gguf --port 8101 &
llama-server --model ../finetuning/output/adapter_blend/final/gguf/model-Q4_K_M.gguf --port 8102 &
llama-server --model ../finetuning/output/adapter_full/final/gguf/model-Q4_K_M.gguf --port 8103 &

# Run comparative evaluation
uv run python compare_adapters.py \
  --models base,baseline,blend,full \
  --ports 8100,8101,8102,8103 \
  --test-set test_set.jsonl \
  --output results/synthetic_comparison.csv
```

## Expected Results

### Hypothesis

| Adapter | ROUGE-L | Judge (Coverage) | Judge (Specificity) | Outcome |
|---------|---------|------------------|---------------------|---------|
| Base (no adapter) | 0.40 | 3.5 | 3.2 | Baseline |
| Baseline (refs only) | 0.45 | 3.8 | 3.6 | Current best |
| **Blend (50/50)** | **0.47** ↑ | **4.0** ↑ | **4.2** ↑ | **Expected winner** |
| Full Synthetic | 0.44 | 3.9 | 4.0 | Good but diluted |

**Why Blend should win:**
- Maintains high-quality reference patterns (ROUGE anchor)
- Learns high-specificity style from synthetics
- Balanced training signal (not overwhelmed by either source)

**Why Full Synthetic might underperform:**
- Too many synthetics dilute reference quality
- Model confusion between reference vs. synthetic styles
- Potential circular validation artifacts

## Quality Control

Synthetic summaries are filtered using Nemotron Nano 9B judge:
- **Keep only:** Coverage ≥4 AND Specificity ≥4
- **Discard:** Lower quality synthetics that don't meet threshold

This ensures we only add high-quality data to training.

## Resume from Interruption

The generation script supports resuming:

```bash
# If interrupted, simply re-run the same command
uv run python generate_synthetic_data_v2.py --tier both

# It will skip already processed samples and continue
```

Progress is saved to:
- `data/synthetic_tier1_progress.jsonl`
- `data/synthetic_tier2_progress.jsonl`

## Cost and Time

**Total time estimate:** ~11-15 hours
- Tier 1 generation: 6-8 hours (13,671 samples)
- Tier 2 generation: 1-2 hours (1,963 samples)
- Dataset creation: 5 minutes
- Training 3 adapters: 6-9 hours (3 × 2-3 hours)
- Evaluation: 2 hours

**Cost:** FREE (Nemotron Nano 9B via NVIDIA API)

## Troubleshooting

### API Rate Limiting
If you hit rate limits, the script includes automatic retry with backoff. You can also:
- Reduce concurrency by increasing `time.sleep()` values
- Run in smaller batches using `--tier 1` and `--tier 2` separately

### Generation Quality Issues
If pass rates are too low (<10%), try:
- Checking NVIDIA API key is valid
- Reviewing `data/synthetic_tier*_progress.jsonl` for error patterns
- Adjusting quality threshold in `generate_synthetic_data_v2.py`

### OOM During Training
If you run out of memory:
- Reduce `batch_size` in `config/training.yaml`
- Increase `grad_accum_steps` to maintain effective batch size
- Use gradient checkpointing (already enabled)

## Next Steps

1. **If Blend wins:** Use blend dataset for final model submission
2. **If Baseline wins:** Stick with high-quality references, document negative result
3. **If Full wins:** Use full synthetic dataset, but verify no circular validation

Either way, document findings in final report for Apple assessment.
