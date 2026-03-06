"""Prompt templates for the extractive-first summarization pipeline."""

# ── Pass 1: EXTRACT key facts from a chunk ──

EXTRACT_CHUNK_FACTS = """\
Extract all important facts from this text.
Copy exact names, numbers, dates, and amounts.
Write each fact as one short bullet point starting with "- ".
Do not explain or interpret. Only copy facts.

Text:
{chunk}

Facts: /no_think"""

# ── Pass 2: SUMMARIZE from extracted facts only ──

SUMMARIZE_FACTS = """\
Write a {target_words}-word summary using only these facts.
Combine related facts into sentences. Use all facts. Do not add new information.

Facts:
{extracted_facts}

Summary: /no_think"""
