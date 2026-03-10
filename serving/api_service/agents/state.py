"""Pipeline state schema."""

from typing import TypedDict


class SummaryState(TypedDict, total=False):
    document: str
    chunks: list[str]            # Document split into chunks (only for long docs)
    draft_summary: str
    final_summary: str
    is_long_document: bool       # Routing: True if doc ≥25K tokens
