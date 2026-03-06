"""End-to-end test of the chunked extractive pipeline with per-step timing.

Usage:
    cd serving && uv run python tests/test_e2e.py          # first short doc
    cd serving && uv run python tests/test_e2e.py medium    # first medium doc
    cd serving && uv run python tests/test_e2e.py long      # first long doc
    cd serving && uv run python tests/test_e2e.py longest   # longest doc in dataset
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from api_service.agents.graph import pipeline


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "short"

    dataset_path = Path(__file__).parent.parent.parent / "eval" / "eval_dataset.json"
    samples = json.loads(dataset_path.read_text())

    if mode == "longest":
        s = max(samples, key=lambda x: len(x["report"]))
    elif mode == "long":
        bucket = [x for x in samples if x["bucket"] == "long"]
        s = bucket[0] if bucket else samples[-1]
    elif mode == "medium":
        bucket = [x for x in samples if x["bucket"] == "medium"]
        s = bucket[0] if bucket else samples[-1]
    else:
        bucket = [x for x in samples if x["bucket"] == "short"]
        s = bucket[0] if bucket else samples[0]

    doc = s["report"]
    ref = s["summary"]
    print(f"Using eval dataset idx={s['idx']} bucket={s['bucket']}")
    print(f"Document: {len(doc):,} chars ({len(doc.split()):,} words)")
    print(f"Reference: {len(ref):,} chars")
    print(f"\n{'='*60}")
    print("Running pipeline ...")
    print(f"{'='*60}\n")

    t_total = time.time()
    result = await pipeline.ainvoke({"document": doc})
    t_total = time.time() - t_total

    summary = result["final_summary"]
    is_long = result.get("is_long_document", False)
    facts = result.get("extracted_facts", "")
    n_facts = facts.count("\n- ") + (1 if facts.startswith("- ") else 0) if facts else 0

    print(f"\n{'='*60}")
    print(f"  RESULT  (total: {t_total:.1f}s)")
    print(f"{'='*60}")
    print(f"  Path: {'EXTRACT' if is_long else 'DIRECT'}")
    print(f"  Chunks: {len(result.get('chunks', []))}")
    if is_long:
        print(f"  Facts extracted: {n_facts}")
    print(f"  Summary: {len(summary):,} chars (~{len(summary.split())} words)")

    if is_long and facts:
        print(f"\n--- EXTRACTED FACTS ---\n")
        print(facts)

    print(f"\n--- AGENT SUMMARY ---\n")
    print(summary)
    print(f"\n--- REFERENCE (first 1000 chars) ---\n")
    print(ref[:1000])


if __name__ == "__main__":
    asyncio.run(main())
