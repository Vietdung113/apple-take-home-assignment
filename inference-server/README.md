# Inference Server

OpenAI-compatible inference server for Qwen3-0.6B with continuous batching.

## Quick Start

### Mac (Apple Silicon) — native llama.cpp with Metal GPU

```bash
# Install llama.cpp
brew install llama.cpp

# Start server (auto-downloads model from HuggingFace)
cd inference-server && ./start.sh
```

Server runs at `http://localhost:8100` with 4 parallel slots and Metal GPU acceleration.

### NVIDIA GPU — vLLM via Docker

```bash
cd inference-server && docker compose --profile gpu up -d
```

### CPU-only — llama.cpp via Docker

```bash
cd inference-server && docker compose --profile cpu up -d
```

## Test

```bash
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

## Configuration

Edit `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8100 | Server port |
| `MODEL_REPO` | unsloth/Qwen3-0.6B-GGUF | HuggingFace repo (for start.sh) |
| `MODEL_FILE` | Qwen3-0.6B-Q4_K_M.gguf | GGUF filename |
| `CTX_SIZE` | 4096 | Max context length |
| `PARALLEL` | 4 | Concurrent request slots |
| `GPU_LAYERS` | -1 | GPU layers (-1 = all) |

## Using finetuned model

```bash
MODEL_PATH=/path/to/finetuned.gguf ./start.sh
```
