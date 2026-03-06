"""Pipeline state schema."""

from typing import TypedDict


class SummaryState(TypedDict, total=False):
    document: str
    chunks: list[str]            # Document split into chunks
    chunk_facts: list[str]       # Facts extracted per chunk
    extracted_facts: str         # Merged + deduped facts
    target_words: int
    draft_summary: str
    final_summary: str
