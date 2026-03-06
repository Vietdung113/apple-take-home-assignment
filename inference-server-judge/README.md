# Judge Inference Server

Qwen2.5-32B-Instruct LLM-as-judge via vLLM OpenAI-compatible API.

## Setup on vast.ai

```bash
# SSH into vast.ai instance
ssh -p <port> root@<host>

# Clone repo
cd /workspace
git clone https://github.com/Vietdung113/apple-take-home-assignment.git repo

# Start judge server
cd repo/inference-server-judge
bash setup_judge.sh
# → starts vLLM on port 8001

# Monitor startup (takes 2-3 min to load 32B model)
tail -f /tmp/vllm_judge.log

# Test
curl http://localhost:8001/v1/models
```

## Stop server

```bash
bash setup_judge.sh --stop
```

## Custom port

```bash
PORT=8002 bash setup_judge.sh
```

## Requirements

- GPU: A100 40GB+ or 2x RTX 4090
- RAM: 64GB+ recommended
- vLLM auto-installed by script
