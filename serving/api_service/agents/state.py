"""Pipeline state schema."""

from typing import TypedDict


class SummaryState(TypedDict, total=False):
    document: str
    chunks: list[str]            # Document split into chunks (only for long docs)
    chunk_facts: list[str]       # Facts extracted per chunk (only for long docs)
    extracted_facts: str         # Merged + deduped facts (only for long docs)
    target_words: int
    draft_summary: str
    final_summary: str
    is_long_document: bool       # Routing: True if doc ≥15K tokens
