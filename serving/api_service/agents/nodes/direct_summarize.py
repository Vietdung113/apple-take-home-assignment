"""Node: Direct summarization for short/medium documents (<25K tokens).

Uses the same prompt structure as training:
  System: "You are an expert summarizing government reports..." (from model_loader)
  User: user_instruction + document + summary_instruction (from prompts.yaml)
"""

from api_service.agents.state import SummaryState
from api_service.agents.prompts import get_direct_summarize_prompt
from api_service.model_loader import generate


async def direct_summarize_node(state: SummaryState) -> dict:
    """Directly summarize document using training-aligned prompt."""
    doc = state["document"]

    print(f"  Direct summarize: {len(doc):,} chars")

    prompt = get_direct_summarize_prompt(doc)

    summary = await generate(prompt, max_new_tokens=1024)

    print(f"  Generated: {len(summary):,} chars (~{len(summary.split())} words)")

    return {"draft_summary": summary.strip()}
