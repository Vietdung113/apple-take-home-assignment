"""Node: merge and deduplicate facts from all chunks."""

from difflib import SequenceMatcher

from api_service.agents.state import SummaryState

SIMILARITY_THRESHOLD = 0.75
MAX_FACTS = 30


def _parse_facts(text: str) -> list[str]:
    """Parse bullet-point facts from text."""
    facts = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            facts.append(line[2:].strip())
    return facts


def _is_duplicate(fact: str, existing: list[str]) -> bool:
    """Check if fact is too similar to any existing fact."""
    for e in existing:
        ratio = SequenceMatcher(None, fact.lower(), e.lower()).ratio()
        if ratio > SIMILARITY_THRESHOLD:
            return True
    return False


def merge_facts_node(state: SummaryState) -> dict:
    """Merge facts from all chunks, deduplicate, ensure even coverage."""
    chunk_facts = state["chunk_facts"]
    n_chunks = len(chunk_facts)

    # Parse all facts per chunk
    facts_per_chunk = [_parse_facts(cf) for cf in chunk_facts]
    total_raw = sum(len(f) for f in facts_per_chunk)

    # Ensure even coverage: round-robin selection from chunks
    # This guarantees beginning, middle, end of doc are all represented
    merged: list[str] = []
    max_rounds = max(len(f) for f in facts_per_chunk) if facts_per_chunk else 0

    for round_idx in range(max_rounds):
        if len(merged) >= MAX_FACTS:
            break
        for chunk_idx in range(n_chunks):
            if round_idx < len(facts_per_chunk[chunk_idx]):
                fact = facts_per_chunk[chunk_idx][round_idx]
                if not _is_duplicate(fact, merged):
                    merged.append(fact)
                    if len(merged) >= MAX_FACTS:
                        break

    # Format as bullet points
    extracted = "\n".join(f"- {f}" for f in merged)

    print(f"  Merge: {total_raw} raw facts → {len(merged)} after dedup "
          f"({len(extracted):,} chars)")

    return {"extracted_facts": extracted}
