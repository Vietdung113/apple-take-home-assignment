# Evaluation

3-way comparison: **base model vs fine-tuned model vs agentic pipeline**.

## Metrics

- **ROUGE-1, ROUGE-2, ROUGE-L** (n-gram overlap with reference)
- **LLM-as-judge** (Qwen2.5-32B): coverage, specificity, consistency, conciseness

## Setup

### 1. Prepare eval dataset

```bash
cd serving
uv run python eval/prepare_dataset.py
# → eval/eval_dataset.json (10 samples: 3 short, 3 medium, 4 long)
```

### 2. Start inference servers on vast.ai

See `inference-server-eval/README.md`:

```bash
ssh -p <port> root@<vast.ai-host>
cd /workspace/repo/inference-server-eval

# Download models
bash download_models.sh

# Export fine-tuned GGUF
cd ../finetuning
bash setup.sh sft 8k --export-gguf
cp output/sft_8k/gguf/*.gguf /workspace/models/Qwen3-0.6B-sft-8k-Q4_K_M.gguf

# Start 3 llama.cpp servers
cd ../inference-server-eval
bash setup_servers.sh
```

### 3. Start agent API (uses fine-tuned model)

```bash
# Terminal 4 on vast.ai
cd /workspace/repo/serving
export INFERENCE_BASE_URL=http://localhost:8200/v1  # use fine-tuned model
uv run uvicorn api_service.main:app --host 0.0.0.0 --port 8300
```

### 4. Configure eval/.env

```bash
cd serving
cp eval/.env.example eval/.env
# URLs should point to vast.ai servers (or use SSH tunnel)
```

### 5. Run evaluation

```bash
uv run python eval/eval_all.py
```

## Output

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

## Servers Architecture

```
vast.ai instance (RTX 4090 24GB):

  Port 8100: llama.cpp → Base GGUF (Qwen3-0.6B)
  Port 8200: llama.cpp → Fine-tuned GGUF (Qwen3-0.6B + adapter)
  Port 8001: llama.cpp → Judge GGUF (Qwen2.5-32B Q4_K_M)
  Port 8300: FastAPI   → Agent pipeline (calls :8200)
```

## Notes

- Agent is slower due to multi-step pipeline (chunking → extract → merge → summarize)
- Judge evaluates all 3 summaries simultaneously for fair comparison
- ROUGE measures n-gram overlap (good baseline, but misses paraphrasing)
- LLM-as-judge captures semantic quality better than ROUGE
