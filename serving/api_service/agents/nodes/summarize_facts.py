"""Node: Pass 2 — write coherent prose summary from extracted facts only."""

from api_service.agents.state import SummaryState
from api_service.agents.prompts import SUMMARIZE_FACTS
from api_service.model_loader import generate


async def summarize_facts_node(state: SummaryState) -> dict:
    """Summarize using only the extracted facts (no full document)."""
    target_words = state["target_words"]

    prompt = SUMMARIZE_FACTS.format(
        extracted_facts=state["extracted_facts"],
        target_words=target_words,
    )
    max_tokens = int(target_words * 1.5) + 100
    summary = await generate(prompt, max_new_tokens=max_tokens)

    print(f"  Summarize: target ~{target_words} words, "
          f"got ~{len(summary.split())} words ({len(summary):,} chars)")

    return {"draft_summary": summary.strip()}
