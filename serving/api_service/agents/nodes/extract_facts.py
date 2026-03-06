"""Node: extract facts from all chunks in parallel."""

import asyncio

from api_service.agents.state import SummaryState
from api_service.agents.prompts import EXTRACT_CHUNK_FACTS
from api_service.model_loader import generate


async def _extract_one_chunk(chunk: str, chunk_idx: int) -> str:
    """Extract facts from a single chunk."""
    prompt = EXTRACT_CHUNK_FACTS.format(chunk=chunk)
    chunk_words = len(chunk.split())
    max_tokens = min(600, max(150, chunk_words // 6))
    result = await generate(prompt, max_new_tokens=max_tokens)
    n = result.count("\n- ") + (1 if result.startswith("- ") else 0)
    print(f"    chunk[{chunk_idx}]: {n} facts ({len(result):,} chars)")
    return result


async def extract_facts_node(state: SummaryState) -> dict:
    """Extract facts from all chunks in parallel via async."""
    chunks = state["chunks"]
    print(f"  Extract: {len(chunks)} chunks in parallel ...")

    tasks = [_extract_one_chunk(c, i) for i, c in enumerate(chunks)]
    chunk_facts = await asyncio.gather(*tasks)

    return {"chunk_facts": list(chunk_facts)}
