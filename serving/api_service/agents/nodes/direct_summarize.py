"""Node: Direct summarization for short/medium documents (<15K tokens)."""

from api_service.agents.state import SummaryState
from api_service.model_loader import generate

DIRECT_SUMMARIZE_PROMPT = """\
Summarize the following document:

{document}

Summary: /no_think"""


async def direct_summarize_node(state: SummaryState) -> dict:
    """Directly summarize document without chunking."""
    doc = state["document"]
    target_words = state["target_words"]

    print(f"  Direct summarize: {len(doc):,} chars → target ~{target_words} words")

    prompt = DIRECT_SUMMARIZE_PROMPT.format(document=doc)
    max_tokens = int(target_words * 1.5) + 100
    summary = await generate(prompt, max_new_tokens=max_tokens)

    print(f"  Generated: {len(summary):,} chars (~{len(summary.split())} words)")

    return {"draft_summary": summary.strip()}
