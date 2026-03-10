"""End-to-end tests for the summarization pipeline.

Runs real GovReport documents through the API and validates:
- API returns successful response
- Output is non-empty and reasonable length
- Summary contains substantive content (not just boilerplate)

Requires running services (docker compose up).

Usage:
    cd serving && uv run pytest tests/test_e2e.py -v
    cd serving && uv run pytest tests/test_e2e.py -v -k 8k      # run only 8k test
    cd serving && uv run pytest tests/test_e2e.py -v -k gt32k    # run only >32k test
"""

import json
import os
import time
from pathlib import Path

import httpx
import pytest

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")
DATA_DIR = Path(__file__).parent / "data"


def load_sample(name: str) -> dict:
    path = DATA_DIR / f"{name}.json"
    return json.loads(path.read_text())


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_8k():
    return load_sample("sample_8k")


@pytest.fixture(scope="module")
def sample_16k():
    return load_sample("sample_16k")


@pytest.fixture(scope="module")
def sample_32k():
    return load_sample("sample_32k")


@pytest.fixture(scope="module")
def sample_gt32k():
    return load_sample("sample_gt32k")


# ── Helpers ──────────────────────────────────────────────────────────────

async def run_pipeline(document: str) -> dict:
    """Call the /summarize API and return result with timing."""
    start = time.time()
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=600.0) as client:
        resp = await client.post("/summarize", json={"document": document})
        resp.raise_for_status()
    elapsed = time.time() - start
    data = resp.json()
    data["_elapsed"] = elapsed
    return data


def assert_valid_summary(result: dict, doc_chars: int):
    """Common assertions for any pipeline result."""
    summary = result["summary"]

    # Summary exists and is non-empty
    assert summary, "Summary should not be empty"
    assert len(summary.strip()) > 50, f"Summary too short: {len(summary)} chars"

    # Summary is shorter than document
    assert len(summary) < doc_chars, "Summary should be shorter than document"

    # Summary has multiple words (not garbage)
    word_count = len(summary.split())
    assert word_count >= 20, f"Summary has too few words: {word_count}"


# ── Tests ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_8k_direct_path(sample_8k):
    """8K doc (~15K chars) should produce valid summary."""
    doc = sample_8k["document"]
    result = await run_pipeline(doc)

    assert_valid_summary(result, len(doc))

    summary = result["summary"]
    print(f"\n[8K] {len(doc):,} chars -> {len(summary.split())} words in {result['_elapsed']:.1f}s")
    print(f"  Summary preview: {summary[:200]}...")


@pytest.mark.asyncio
async def test_16k_direct_path(sample_16k):
    """16K doc (~65K chars) should produce valid summary."""
    doc = sample_16k["document"]
    result = await run_pipeline(doc)

    assert_valid_summary(result, len(doc))

    summary = result["summary"]
    print(f"\n[16K] {len(doc):,} chars -> {len(summary.split())} words in {result['_elapsed']:.1f}s")
    print(f"  Summary preview: {summary[:200]}...")


@pytest.mark.asyncio
async def test_32k_direct_path(sample_32k):
    """32K doc (~89K chars) should produce valid summary."""
    doc = sample_32k["document"]
    result = await run_pipeline(doc)

    assert_valid_summary(result, len(doc))

    summary = result["summary"]
    print(f"\n[32K] {len(doc):,} chars -> {len(summary.split())} words in {result['_elapsed']:.1f}s")
    print(f"  Summary preview: {summary[:200]}...")


@pytest.mark.asyncio
async def test_gt32k_hierarchical_path(sample_gt32k):
    """GT32K doc (~189K chars) should produce valid summary."""
    doc = sample_gt32k["document"]
    result = await run_pipeline(doc)

    assert_valid_summary(result, len(doc))

    summary = result["summary"]
    print(f"\n[GT32K] {len(doc):,} chars -> {len(summary.split())} words in {result['_elapsed']:.1f}s")
    print(f"  Summary preview: {summary[:200]}...")
