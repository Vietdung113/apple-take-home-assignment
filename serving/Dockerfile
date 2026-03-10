FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy prompt config (shared with training/eval)
COPY finetuning/config/prompts.yaml finetuning/config/prompts.yaml
COPY finetuning/config/prompt_loader.py finetuning/config/prompt_loader.py

# Copy serving project
COPY serving/pyproject.toml serving/pyproject.toml
COPY serving/api_service/ serving/api_service/

# Install dependencies
RUN cd serving && uv pip install --system -e .

WORKDIR /app/serving

EXPOSE 8001

CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
