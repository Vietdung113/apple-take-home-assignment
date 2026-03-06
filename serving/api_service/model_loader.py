"""Async inference client for OpenAI-compatible server (llama.cpp / vLLM)."""

import os
import re

import httpx

INFERENCE_BASE_URL = os.environ.get("INFERENCE_BASE_URL", "http://localhost:8100/v1")

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=INFERENCE_BASE_URL, timeout=120.0)
    return _client


def _strip_think_tags(text: str) -> str:
    """Strip Qwen3 thinking tags (closed or unclosed)."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


async def generate(prompt: str, max_new_tokens: int = 1024) -> str:
    """Send chat completion request to inference server."""
    client = _get_client()
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
        "top_p": 0.9,
    }
    resp = await client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    # If content is empty but reasoning has text, use reasoning
    text = content if content.strip() else reasoning

    return _strip_think_tags(text)
