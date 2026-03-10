"""Compare base model vs fine-tuned model on same samples.

This helps determine if fine-tuning is actually helping or hurting.

Usage:
    # Start both servers first:
    # Port 8100: Fine-tuned model
    # Port 8200: Base model (no fine-tuning)

    python compare_base_vs_finetuned.py --num-samples 10
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm_asyncio

# Add finetuning config to path
sys.path.insert(0, str(Path(__file__).parent.parent / "finetuning" / "config"))
from prompt_loader import get_inference_prompt_base_model, get_generation_params


FINETUNED_URL = "http://localhost:8100/v1"  # Fine-tuned
BASE_URL = "http://localhost:8200/v1"       # Base (no fine-tuning)


async def generate_summary(url: str, prompt: str, gen_params: dict) -> dict:
    """Generate summary from model."""
    try:
        async with httpx.AsyncClient(base_url=url, timeout=120.0) as client:
            payload = {
                "prompt": prompt,
                "max_tokens": gen_params["max_tokens"],
                "temperature": gen_params["temperature"],
                "top_p": gen_params["top_p"],
                "repeat_penalty": gen_params["repetition_penalty"],
                "stop": ["\n\n\nGovernment Report:", "\n\nExpert Summary:", "\n\n\n", "<|endoftext|>"],
            }

            resp = await client.post("/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["text"].strip()

            return {
                "summary": text,
                "words": len(text.split()),
                "chars": len(text),
                "success": True
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def compare_sample(sample_id: int, document: str, reference: str, gen_params: dict) -> dict:
    """Compare base vs fine-tuned on one sample."""
    prompt = get_inference_prompt_base_model(document)

    # Generate from both models in parallel
    finetuned_task = generate_summary(FINETUNED_URL, prompt, gen_params)
    base_task = generate_summary(BASE_URL, prompt, gen_params)

    finetuned, base = await asyncio.gather(finetuned_task, base_task)

    # Simple quality metrics
    result = {
        "sample_id": sample_id,
        "reference_words": len(reference.split()),
        "finetuned": finetuned,
        "base": base,
    }

    # Word overlap with reference (simple metric)
    if finetuned.get("success") and base.get("success"):
        ref_words = set(reference.lower().split())

        ft_words = set(finetuned["summary"].lower().split())
        base_words = set(base["summary"].lower().split())

        result["finetuned_overlap"] = len(ref_words & ft_words) / len(ref_words) if ref_words else 0
        result["base_overlap"] = len(ref_words & base_words) / len(ref_words) if ref_words else 0

    return result


async def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of test samples")
    parser.add_argument("--test-set", default="test_set.jsonl", help="Test set file")
    parser.add_argument("--output", default="comparison_results.json", help="Output file")
    args = parser.parse_args()

    print("="*80)
    print("Base vs Fine-tuned Comparison")
    print("="*80)

    # Check both servers are running
    print("\nChecking servers...")
    for name, url in [("Fine-tuned", FINETUNED_URL), ("Base", BASE_URL)]:
        try:
            async with httpx.AsyncClient(base_url=url, timeout=5.0) as client:
                await client.get("/health")
            print(f"  ✅ {name} server running on {url}")
        except:
            print(f"  ❌ {name} server NOT running on {url}")
            print(f"\nPlease start both servers:")
            print(f"  Fine-tuned: ./start_model.sh <finetuned.gguf> --port 8100")
            print(f"  Base: llama-server --model ../models/Qwen3-0.6B-Q4_K_M.gguf --port 8200")
            return

    # Load test samples
    print(f"\nLoading {args.num_samples} test samples...")
    test_path = Path(__file__).parent / args.test_set
    samples = []
    with open(test_path) as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            sample = json.loads(line)
            doc = sample.get("document") or sample.get("report")
            samples.append((i, doc, sample["summary"]))

    print(f"Loaded {len(samples)} samples")

    # Get generation params
    gen_params = get_generation_params()
    print(f"\nGeneration params:")
    print(f"  max_tokens: {gen_params['max_tokens']}")
    print(f"  temperature: {gen_params['temperature']}")

    # Compare
    print(f"\nGenerating summaries from both models...\n")

    tasks = [compare_sample(sid, doc, ref, gen_params) for sid, doc, ref in samples]
    results = []

    for task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Comparing"):
        result = await task
        results.append(result)

    # Show actual comparisons
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)

    successful = [r for r in results if r["finetuned"].get("success") and r["base"].get("success")]

    if not successful:
        print("❌ No successful comparisons")
        return

    # Load references for display
    references = {}
    with open(test_path) as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            sample = json.loads(line)
            references[i] = sample["summary"]

    for r in successful[:3]:  # Show first 3 samples
        sid = r["sample_id"]
        ref = references.get(sid, "N/A")

        print(f"\n{'='*80}")
        print(f"Sample {sid}")
        print(f"{'='*80}")

        print(f"\n[REFERENCE] ({len(ref.split())} words)")
        print(ref[:600])
        if len(ref) > 600:
            print("...")

        print(f"\n[FINE-TUNED - Checkpoint 300] ({r['finetuned']['words']} words)")
        print(r['finetuned']['summary'][:600])
        if len(r['finetuned']['summary']) > 600:
            print("...")

        print(f"\n[BASE MODEL - No Fine-tuning] ({r['base']['words']} words)")
        print(r['base']['summary'][:600])
        if len(r['base']['summary']) > 600:
            print("...")

        print(f"\n→ Which is better? (visually compare content quality)")
        print(f"  - Coverage: Does it capture key points?")
        print(f"  - Specificity: Does it include numbers/dates/names?")
        print(f"  - Structure: Is it well-organized?")
        print(f"  - Fluency: Is it coherent and readable?")

    # Save full outputs for detailed analysis
    output_path = Path(args.output)

    # Enrich with references
    for r in results:
        sid = r["sample_id"]
        if sid in references:
            r["reference"] = references[sid]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Full outputs saved to: {output_path}")
    print(f"   Open this file to read all summaries in detail")

    # Manual conclusion
    print(f"\n" + "="*80)
    print("YOUR ASSESSMENT")
    print("="*80)
    print(f"\nAfter reading the above comparisons, ask yourself:")
    print(f"  1. Is fine-tuned output MORE comprehensive than base?")
    print(f"  2. Does fine-tuned include MORE specific details (numbers, dates)?")
    print(f"  3. Is fine-tuned MORE coherent and well-structured?")
    print(f"  4. Does fine-tuned BETTER match reference style?")
    print(f"\nIf YES to most → Fine-tuning is working (even at step 300)")
    print(f"If NO to most → Need to train longer or investigate issues")


if __name__ == "__main__":
    asyncio.run(main())
