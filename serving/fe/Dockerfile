FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
COPY app.py .

RUN uv pip install --system -e .

EXPOSE 7860

CMD ["python", "app.py"]
