"""Async inference client for OpenAI-compatible server (llama.cpp / vLLM)."""

import os
import re
import sys
from pathlib import Path

import httpx

# Load system prompt from centralized config (same source as training/eval)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "finetuning" / "config"))
from prompt_loader import get_system_prompt, get_generation_params  # noqa: E402

INFERENCE_BASE_URL = os.environ.get("INFERENCE_BASE_URL", "http://localhost:8080/v1")
SYSTEM_PROMPT = get_system_prompt()
_GEN_PARAMS = get_generation_params()

def _get_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=INFERENCE_BASE_URL, timeout=600.0)


def _strip_think_tags(text: str) -> str:
    """Strip Qwen3 thinking tags (closed or unclosed)."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


async def generate(
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float | None = None,
    repetition_penalty: float | None = None,
    system_prompt: str | None = None,
) -> str:
    """Send chat completion request to inference server.

    Uses the same system prompt and generation params as training/eval.

    Args:
        prompt: User message content
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (default from prompts.yaml)
        repetition_penalty: Penalty for repeating tokens (default from prompts.yaml)
        system_prompt: Override system prompt (default: training system prompt)
    """
    sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    temp = temperature if temperature is not None else _GEN_PARAMS["temperature"]
    rep_penalty = repetition_penalty if repetition_penalty is not None else _GEN_PARAMS["repetition_penalty"]

    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temp,
        "top_p": _GEN_PARAMS["top_p"],
        "repetition_penalty": rep_penalty,
        "stop": _GEN_PARAMS.get("stop", []),
    }

    async with _get_client() as client:
        resp = await client.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            print(f"Error from inference server: {resp.status_code}")
            print(f"Response: {resp.text[:500]}")
        resp.raise_for_status()
        data = resp.json()

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    # If content is empty but reasoning has text, use reasoning
    text = content if content.strip() else reasoning

    return _strip_think_tags(text)
