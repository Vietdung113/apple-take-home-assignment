"""ROUGE-only evaluation: baseline vs agent pipeline.

Reads examples from eval/eval_dataset.json (created by prepare_dataset.py)
and reports ROUGE-1/2/L F1 scores for each bucket and overall.

Usage:
    cd serving && uv run python eval/eval_rouge.py
    # or: make eval-rouge
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from eval.common import (
    RougeScores,
    compute_rouge,
    generate_agent,
    generate_baseline,
    load_dotenv,
)

load_dotenv()

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]


# ── Data classes ───────────────────────────────────────────


@dataclass
class RougeResult:
    idx: int
    bucket: str
    doc_chars: int
    baseline_rouge: RougeScores
    agent_rouge: RougeScores
    baseline_time: float
    agent_time: float


@dataclass
class RougeEvalReport:
    results: list[RougeResult] = field(default_factory=list)

    def _print_bucket(self, results: list[RougeResult], label: str):
        n = len(results)
        if not n:
            return

        print(f"\n  ── {label} ({n} examples) ──")
        print(f"  {'Metric':<14} {'Baseline':>10} {'Agent':>10} {'Delta':>10}")
        print(f"  {'-'*46}")

        for metric in ROUGE_METRICS:
            b = sum(getattr(r.baseline_rouge, metric) for r in results) / n
            a = sum(getattr(r.agent_rouge, metric) for r in results) / n
            print(f"  {metric:<14} {b:>10.4f} {a:>10.4f} {a - b:>+10.4f}")

    def print_report(self):
        n = len(self.results)
        if not n:
            print("No results.")
            return

        print(f"\n{'='*55}")
        print(f"  ROUGE Scores (F1) — {n} examples")
        print(f"{'='*55}")

        # Per-bucket breakdown
        for bucket_name in ["SHORT", "MEDIUM", "LONG"]:
            bucket_results = [r for r in self.results if r.bucket == bucket_name.lower()]
            self._print_bucket(bucket_results, bucket_name)

        # Overall
        self._print_bucket(self.results, "OVERALL")

        # Timing
        avg_bt = sum(r.baseline_time for r in self.results) / n
        avg_at = sum(r.agent_time for r in self.results) / n
        print(f"\n  {'Avg time (s)':<14} {avg_bt:>10.1f} {avg_at:>10.1f}")
        print()

    def save(self, path: str = "eval_rouge_results.json"):
        data = []
        for r in self.results:
            data.append({
                "idx": r.idx,
                "bucket": r.bucket,
                "doc_chars": r.doc_chars,
                "baseline_time": r.baseline_time,
                "agent_time": r.agent_time,
                "baseline_rouge": {
                    "rouge1": r.baseline_rouge.rouge1,
                    "rouge2": r.baseline_rouge.rouge2,
                    "rougeL": r.baseline_rouge.rougeL,
                },
                "agent_rouge": {
                    "rouge1": r.agent_rouge.rouge1,
                    "rouge2": r.agent_rouge.rouge2,
                    "rougeL": r.agent_rouge.rougeL,
                },
            })

        Path(path).write_text(json.dumps(data, indent=2))
        print(f"  Results saved to {path}")


# ── Main ───────────────────────────────────────────────────


async def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found.")
        print("Run 'make prepare-dataset' first.")
        sys.exit(1)

    all_samples = json.loads(DATASET_PATH.read_text())
    n = int(os.environ.get("EVAL_N", len(all_samples)))
    samples = all_samples[:n]
    total = len(samples)

    print(f"  ROUGE Evaluation")
    print(f"  Dataset: {DATASET_PATH.name} ({total} examples)")
    print()

    report = RougeEvalReport()

    for i, sample in enumerate(samples):
        doc = sample["report"]
        ref = sample["summary"]
        bucket = sample["bucket"]
        print(f"[{i + 1}/{total}] idx={sample['idx']}  bucket={bucket}  {len(doc):,} chars")

        # 1. Baseline summary
        t0 = time.time()
        baseline_out = await generate_baseline(doc)
        t_baseline = time.time() - t0

        # 2. Agent summary
        t0 = time.time()
        agent_out = await generate_agent(doc)
        t_agent = time.time() - t0

        print(f"  Baseline: {len(baseline_out):,} chars  ({t_baseline:.1f}s)")
        print(f"  Agent:    {len(agent_out):,} chars  ({t_agent:.1f}s)")

        # 3. ROUGE scores (vs reference)
        b_rouge = compute_rouge(baseline_out, ref)
        a_rouge = compute_rouge(agent_out, ref)
        print(f"  ROUGE-1:  baseline={b_rouge.rouge1:.4f}  agent={a_rouge.rouge1:.4f}")
        print(f"  ROUGE-2:  baseline={b_rouge.rouge2:.4f}  agent={a_rouge.rouge2:.4f}")
        print(f"  ROUGE-L:  baseline={b_rouge.rougeL:.4f}  agent={a_rouge.rougeL:.4f}")

        report.results.append(RougeResult(
            idx=sample["idx"],
            bucket=bucket,
            doc_chars=len(doc),
            baseline_rouge=b_rouge,
            agent_rouge=a_rouge,
            baseline_time=t_baseline,
            agent_time=t_agent,
        ))
        print()

    report.print_report()
    report.save()


if __name__ == "__main__":
    asyncio.run(main())
