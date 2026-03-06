"""Prompt templates for the summarization pipeline."""

# ── Direct path: simple summarization (short/medium docs) ──

DIRECT_SUMMARIZE = """\
/no_think
Summarize this government report:

{document}"""

# ── Extract path (long docs): Pass 1 — EXTRACT key facts from a chunk ──

EXTRACT_CHUNK_FACTS = """\
/no_think
Extract key facts from this government report section (as bullet points):

{chunk}"""

# ── Extract path (long docs): Pass 2 — SUMMARIZE from extracted facts ──

SUMMARIZE_FACTS = """\
/no_think
Summarize this government report using these facts:

{extracted_facts}"""
