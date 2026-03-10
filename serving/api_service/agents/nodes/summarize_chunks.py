"""Node: Hierarchical summarization for long docs (≥25K tokens).

Level 1: Extract key information from each chunk (parallel)
Level 2: Merge section summaries into final summary (context → findings → implications)
"""

import asyncio

from api_service.agents.state import SummaryState
from api_service.agents.prompts import get_summarize_chunk_prompt, get_merge_summaries_prompt
from api_service.model_loader import generate


async def _summarize_one_chunk(chunk: str, chunk_idx: int, total_chunks: int) -> str:
    """Level 1: Extract key information from a single chunk."""
    prompt = get_summarize_chunk_prompt(chunk)

    # Adaptive token limit based on chunk size
    chunk_tokens = len(chunk) // 4  # Rough estimate
    max_tokens = min(1024, max(512, chunk_tokens // 10))  # 10:1 compression

    result = await generate(prompt, max_new_tokens=max_tokens)

    word_count = len(result.split())
    print(f"    Section {chunk_idx+1}/{total_chunks}: {word_count} words (~{len(result):,} chars)")
    return result


async def summarize_chunks_node(state: SummaryState) -> dict:
    """Hierarchical summarization: chunk summaries → merged final summary."""
    chunks = state["chunks"]
    print(f"  Hierarchical: {len(chunks)} sections...")

    # Level 1: Summarize each chunk sequentially (parallel requires --parallel N on server)
    print(f"  Level 1: Summarizing {len(chunks)} sections...")
    chunk_summaries = []
    for i, c in enumerate(chunks):
        result = await _summarize_one_chunk(c, i, len(chunks))
        chunk_summaries.append(result)

    # Calculate total intermediate summary size
    total_summary_chars = sum(len(s) for s in chunk_summaries)
    print(f"  Total section summaries: {total_summary_chars:,} chars (~{total_summary_chars//4:,} tokens)")

    # Level 2: Merge all summaries
    print(f"  Level 2: Synthesizing final summary...")
    combined_summaries = "\n\n".join([
        f"Section {i+1}:\n{s}"
        for i, s in enumerate(chunk_summaries)
    ])

    prompt = get_merge_summaries_prompt(combined_summaries)
    final_summary = await generate(prompt, max_new_tokens=1024)

    final_word_count = len(final_summary.split())
    print(f"  Final: {final_word_count} words (~{len(final_summary):,} chars)")

    return {"draft_summary": final_summary}
