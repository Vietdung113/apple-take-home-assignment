"""Test adaptive routing: short docs use direct path, long docs use extract path."""

import asyncio

from api_service.agents.graph import pipeline


async def test_short_doc():
    """Test direct summarization path for short document."""
    print("\n" + "="*60)
    print("TEST 1: Short Document (Direct Path)")
    print("="*60)

    # Short document (~500 words, ~2K chars)
    short_doc = """
The Federal Reserve announced today a quarter-point interest rate increase,
bringing the federal funds rate to 5.25%. This marks the tenth consecutive
rate hike as the central bank continues its fight against inflation.

Fed Chair Jerome Powell stated that while inflation has shown signs of
moderating, it remains above the target level of 2%. The Consumer Price Index
rose 4.2% year-over-year in the latest report, down from a peak of 9.1%
last summer but still elevated.

The rate decision was unanimous among Federal Open Market Committee members.
Powell indicated that future rate decisions will be data-dependent, with the
Fed closely monitoring economic indicators including employment, consumer
spending, and inflation trends.

Financial markets reacted positively to the announcement, with major stock
indices rising as investors had anticipated the move. The S&P 500 gained 1.2%
while the Nasdaq Composite increased 1.5%.

Economists predict the Fed may pause rate increases later this year if
inflation continues to decline. However, the central bank has emphasized
its commitment to bringing inflation back to target levels, even if it
means accepting slower economic growth in the near term.
""".strip()

    print(f"Document: {len(short_doc):,} chars (~{len(short_doc.split())} words)")
    print(f"Expected: DIRECT path\n")

    result = await pipeline.ainvoke({"document": short_doc})

    is_long = result.get("is_long_document", False)
    summary = result["final_summary"]

    print(f"\nActual path: {'EXTRACT' if is_long else 'DIRECT'}")
    print(f"Chunks created: {len(result.get('chunks', []))}")
    print(f"Facts extracted: {'Yes' if result.get('extracted_facts') else 'No'}")
    print(f"Summary length: {len(summary):,} chars (~{len(summary.split())} words)")
    print(f"\nSummary:\n{summary}")

    assert not is_long, "Short doc should use DIRECT path"
    assert len(result.get("chunks", [])) == 0, "Short doc should not be chunked"
    assert not result.get("extracted_facts"), "Short doc should not extract facts"
    print("\n✅ PASSED: Short doc used direct path")


async def test_long_doc():
    """Test extract-merge-summarize path for long document."""
    print("\n" + "="*60)
    print("TEST 2: Long Document (Extract Path)")
    print("="*60)

    # Long document (~20K+ chars, 60+ sentences to trigger chunking)
    long_doc = " ".join([
        f"This is sentence number {i} containing important information about "
        f"the topic of summarization and natural language processing systems. "
        f"It discusses various aspects of machine learning, neural networks, "
        f"and transformer architectures that enable modern AI capabilities. "
        f"The research in this area continues to advance rapidly with new "
        f"techniques and methodologies being developed constantly."
        for i in range(400)  # ~20K chars
    ])

    print(f"Document: {len(long_doc):,} chars (~{len(long_doc.split())} words)")
    print(f"Expected: EXTRACT path\n")

    result = await pipeline.ainvoke({"document": long_doc})

    is_long = result.get("is_long_document", False)
    summary = result["final_summary"]
    chunks = result.get("chunks", [])
    facts = result.get("extracted_facts", "")

    print(f"\nActual path: {'EXTRACT' if is_long else 'DIRECT'}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Facts extracted: {'Yes' if facts else 'No'}")
    print(f"Summary length: {len(summary):,} chars (~{len(summary.split())} words)")
    print(f"\nSummary:\n{summary[:500]}...")

    assert is_long, "Long doc should use EXTRACT path"
    assert len(chunks) > 0, "Long doc should be chunked"
    assert facts, "Long doc should extract facts"
    print("\n✅ PASSED: Long doc used extract path")


async def main():
    print("\n" + "="*60)
    print("ADAPTIVE ROUTING TESTS")
    print("="*60)

    await test_short_doc()
    await test_long_doc()

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
