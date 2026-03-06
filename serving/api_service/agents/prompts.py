"""Prompt templates for the summarization pipeline."""

# ── Direct path: simple summarization (short/medium docs) ──

DIRECT_SUMMARIZE = """\
/no_think
Summarize this government report:

{document}"""

# ── Extract path (long docs): Pass 1 — EXTRACT key facts from a chunk ──

EXTRACT_CHUNK_FACTS = """\
/no_think
Extract all important facts from this government report section.
Copy exact names, numbers, dates, and amounts.
Write each fact as one short bullet point starting with "- ".
Do not explain or interpret. Only copy facts.

Text:
{chunk}"""

# ── Extract path (long docs): Pass 2 — SUMMARIZE from extracted facts ──

SUMMARIZE_FACTS = """\
/no_think
Write a {target_words}-word summary of this government report using only these facts.
Combine related facts into sentences. Use all facts. Do not add new information.

Facts:
{extracted_facts}"""
