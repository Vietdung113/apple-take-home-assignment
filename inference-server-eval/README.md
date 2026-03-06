## Evaluation Inference Servers

3 llama.cpp servers for evaluating base model, fine-tuned model, and judge on vast.ai.

## Setup

### 1. Download models

```bash
cd inference-server-eval
bash download_models.sh
# Downloads:
#   - Qwen3-0.6B-Q4_K_M.gguf (base, ~400MB)
#   - Qwen2.5-32B-Instruct-Q4_K_M.gguf (judge, ~15GB)
```

### 2. Export fine-tuned GGUF

```bash
cd ../finetuning
bash setup.sh sft 8k --export-gguf
# → output/sft_8k/gguf/*.gguf

# Copy to models dir
cp output/sft_8k/gguf/*.gguf /workspace/models/Qwen3-0.6B-sft-8k-Q4_K_M.gguf
```

### 3. Start servers

```bash
cd inference-server-eval
bash setup_servers.sh
```

Servers:
- **Port 8100:** Base model (Qwen3-0.6B)
- **Port 8200:** Fine-tuned model (Qwen3-0.6B + adapter)
- **Port 8001:** Judge model (Qwen2.5-32B)

### 4. Test

```bash
curl http://localhost:8100/health
curl http://localhost:8200/health
curl http://localhost:8001/health
```

### 5. Stop servers

```bash
bash setup_servers.sh --stop
```

## Resource Requirements

- **GPU:** RTX 4090 24GB or A100 40GB
- **RAM:** 32GB+
- **Disk:** ~20GB for models

## Logs

```bash
tail -f /tmp/llama_base.log
tail -f /tmp/llama_finetuned.log
tail -f /tmp/llama_judge.log
```

## Custom model paths

```bash
export BASE_GGUF=/path/to/base.gguf
export FINETUNED_GGUF=/path/to/finetuned.gguf
export JUDGE_GGUF=/path/to/judge.gguf
bash setup_servers.sh
```
