# Complete Workflow: Training → Evaluation

End-to-end flow from fine-tuning to evaluation on vast.ai.

---

## Overview

```
1. Prepare data (8K tokens)
2. Train adapter (QLoRA SFT)
3. Export GGUF (merge + quantize)
4. Deploy 3 servers (base, fine-tuned, judge)
5. Run 3-way evaluation
6. Generate report
```

---

## Step 1: Prepare Training Data

**Local or vast.ai:**

```bash
cd finetuning
uv sync
uv run python prepare_data.py --max-tokens 8192
# Output: data/sft_8k_{train,validation,test}.jsonl
```

Upload to vast.ai if prepared locally:

```bash
scp -P <port> data/sft_8k_*.jsonl root@<host>:/workspace/repo/finetuning/data/
```

---

## Step 2: Train QLoRA Adapter

**On vast.ai GPU:**

```bash
ssh -p <port> root@<host>
cd /workspace/repo/finetuning

# Clone repo if needed
git pull

# Train (full dataset)
bash setup.sh sft 8k

# Or smoke test (10 samples, 2 epochs)
bash setup.sh sft 8k --max-samples 10 --num-epochs 2
```

**Output:**
- `output/sft_8k/adapter/` — LoRA weights
- Training logs → wandb (if configured)

---

## Step 3: Export GGUF

**On vast.ai (after training):**

```bash
cd /workspace/repo/finetuning

# Export merged GGUF (Q4_K_M quantization)
bash setup.sh sft 8k --export-gguf

# Copy to models directory
mkdir -p /workspace/models
cp output/sft_8k/gguf/*.gguf /workspace/models/Qwen3-0.6B-sft-8k-Q4_K_M.gguf
```

**Output:**
- `Qwen3-0.6B-sft-8k-Q4_K_M.gguf` (~400MB)

---

## Step 4: Deploy 3 llama.cpp Servers

**On vast.ai:**

### 4.1 Download Base + Judge Models

```bash
cd /workspace/repo/inference-server-eval
bash download_models.sh
```

Downloads:
- `Qwen3-0.6B-Q4_K_M.gguf` (base, ~400MB)
- `Qwen2.5-32B-Instruct-Q4_K_M.gguf` (judge, ~15GB)

### 4.2 Start 3 Servers

```bash
bash setup_servers.sh
```

Servers:
- **Port 8100:** Base model
- **Port 8200:** Fine-tuned model
- **Port 8001:** Judge model

Verify:

```bash
curl http://localhost:8100/health
curl http://localhost:8200/health
curl http://localhost:8001/health
```

Logs:

```bash
tail -f /tmp/llama_base.log
tail -f /tmp/llama_finetuned.log
tail -f /tmp/llama_judge.log
```

---

## Step 5: Start Agent API

**Terminal 4 on vast.ai:**

```bash
cd /workspace/repo/serving
uv sync

# Point agent to fine-tuned model
export INFERENCE_BASE_URL=http://localhost:8200/v1

# Start FastAPI
uv run uvicorn api_service.main:app --host 0.0.0.0 --port 8300
```

**Port 8300:** Agent pipeline (uses fine-tuned model at :8200)

---

## Step 6: Prepare Eval Dataset

**On vast.ai:**

```bash
cd /workspace/repo/eval
uv sync
uv run python prepare_dataset.py
```

**Output:**
- `eval_dataset.json` (10 samples: 3 short, 3 medium, 4 long)

---

## Step 7: Configure Eval Environment

```bash
cd eval
cp .env.example .env
cat > .env <<EOF
BASE_MODEL_URL=http://localhost:8100
FINETUNED_MODEL_URL=http://localhost:8200
AGENT_API_URL=http://localhost:8300
JUDGE_URL=http://localhost:8001
JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
EOF
```

---

## Step 8: Run Evaluation

```bash
cd eval
uv run python eval_all.py
```

**What happens:**

For each test sample:
1. Generate 3 summaries (base / fine-tuned / agent)
2. Compute ROUGE scores (vs reference)
3. LLM-as-judge scoring (4 dimensions)
4. Aggregate results by bucket (short/medium/long)

**Output:**

```
  3-WAY EVALUATION  (10 examples)
  ════════════════════════════════════════════════════════

  ── OVERALL (10 examples) ──

  Metric       Base      Finetuned      Agent
  --------------------------------------------
  rouge1     0.3421       0.4123     0.4456
  rouge2     0.1821       0.2341     0.2678
  rougeL     0.3012       0.3823     0.4123

  Dimension  Base      Finetuned      Agent
  --------------------------------------------
  coverage     3.2         4.1         4.5
  specificity  3.0         4.3         4.6
  consistency  3.5         4.0         4.2
  conciseness  3.8         3.9         3.7
  --------------------------------------------
  Weighted     3.28        4.13        4.42

  Avg time (s) 1.2         1.3         4.8

  Results saved to eval_all_results.json
```

---

## Step 9: Download Results

```bash
# From local machine
scp -P <port> root@<host>:/workspace/repo/eval/eval_all_results.json .
```

---

## Architecture Summary

```
vast.ai instance (RTX 4090 24GB):

  ┌─────────────────────────────────────────────┐
  │  Training                                   │
  │  ├─ finetuning/train_sft.py                │
  │  └─ output: adapter + GGUF                  │
  └─────────────────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────────┐
  │  Inference Servers (llama.cpp)              │
  │  ├─ :8100  Base GGUF                        │
  │  ├─ :8200  Fine-tuned GGUF                  │
  │  └─ :8001  Judge GGUF (32B)                 │
  └─────────────────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────────┐
  │  Agent API                                  │
  │  └─ :8300  FastAPI (uses :8200)             │
  └─────────────────────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────────┐
  │  Evaluation                                 │
  │  ├─ Generate 3 summaries                    │
  │  ├─ Compute ROUGE                           │
  │  ├─ LLM-as-judge                            │
  │  └─ output: eval_all_results.json           │
  └─────────────────────────────────────────────┘
```

---

## Resource Requirements

**GPU Memory (RTX 4090 24GB):**
- Base model: ~2GB
- Fine-tuned model: ~2GB
- Judge model (32B Q4): ~16GB
- Agent API: ~500MB
- **Total: ~20.5GB** ✓ fits

**Disk:**
- Models: ~20GB
- Training output: ~2GB
- Total: ~25GB

**Time Estimates:**
- Training (8K, full dataset ~7.7K samples): ~2-4 hours
- GGUF export: ~5 minutes
- Evaluation (10 samples): ~10-15 minutes

---

## Troubleshooting

### Server won't start

```bash
# Check logs
tail -100 /tmp/llama_base.log
tail -100 /tmp/llama_finetuned.log
tail -100 /tmp/llama_judge.log

# Check GPU memory
nvidia-smi

# Stop all servers
cd inference-server-eval
bash setup_servers.sh --stop
```

### Eval fails

```bash
# Test each server individually
curl http://localhost:8100/v1/models
curl http://localhost:8200/v1/models
curl http://localhost:8001/v1/models
curl http://localhost:8300/health

# Check agent logs
# (FastAPI stdout)
```

### Out of memory

Reduce judge model size:
```bash
# Use smaller judge (7B instead of 32B)
export JUDGE_GGUF=/workspace/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
bash setup_servers.sh
```

---

## Next Steps

After evaluation:
1. **Compare results** across 8K/16K/32K adapters
2. **Analyze trade-offs** (accuracy vs context length vs latency)
3. **Iterate** on training hyperparameters
4. **Deploy** best model to production (Mac via inference-server/)
