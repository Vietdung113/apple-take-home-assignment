"""Convert high quality training data to base model format (plain text).

Apple take-home assessment requires:
- Input format: Plain text / JSON with "document" field
- Output format: Plain text / JSON with "summary" field

This script converts chat template format to simple document/summary pairs.

Usage:
    python convert_to_base_format.py

    # Custom input/output paths
    python convert_to_base_format.py \
        --input ../data/high_quality_data.jsonl \
        --output ../data/base_model_format/
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def extract_document_summary(messages: list) -> tuple[str, str] | None:
    """Extract document and summary from chat template format.

    Args:
        messages: List of message dicts with role and content

    Returns:
        Tuple of (document, summary) or None if parsing fails
    """
    # Find user message containing document
    document = None
    summary = None

    for msg in messages:
        if msg["role"] == "user":
            # Extract document from user message
            # Format: "Summarize the following government report. ...\n\n[DOCUMENT]"
            content = msg["content"]

            # Find document after the instruction
            if "\n\n" in content:
                parts = content.split("\n\n", 1)
                if len(parts) == 2:
                    document = parts[1].strip()

        elif msg["role"] == "assistant":
            # Summary is assistant's response
            summary = msg["content"].strip()

    if document and summary:
        return document, summary

    return None


def convert_to_json_format(input_file: Path, output_file: Path):
    """Convert to JSON format (one object per line).

    Output format:
    {"document": "...", "summary": "..."}
    """

    print(f"Converting {input_file} to JSON format...")
    print(f"Output: {output_file}")

    converted = 0
    skipped = 0

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in tqdm(f_in, desc="Converting"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                messages = data.get("messages", [])

                result = extract_document_summary(messages)
                if result:
                    document, summary = result

                    # Write in simple format
                    json_obj = {
                        "document": document,
                        "summary": summary
                    }
                    f_out.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                    converted += 1
                else:
                    skipped += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                skipped += 1

    print(f"✓ Converted: {converted} samples")
    print(f"✗ Skipped: {skipped} samples")
    print(f"✓ Output: {output_file}")


def convert_to_plain_text(input_file: Path, output_dir: Path):
    """Convert to separate plain text files for documents and summaries.

    Creates:
    - documents.txt: One document per line
    - summaries.txt: One summary per line (aligned with documents)
    """

    print(f"Converting {input_file} to plain text format...")
    print(f"Output directory: {output_dir}")

    doc_file = output_dir / "documents.txt"
    sum_file = output_dir / "summaries.txt"

    converted = 0
    skipped = 0

    with open(input_file) as f_in, \
         open(doc_file, "w") as f_doc, \
         open(sum_file, "w") as f_sum:

        for line in tqdm(f_in, desc="Converting"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                messages = data.get("messages", [])

                result = extract_document_summary(messages)
                if result:
                    document, summary = result

                    # Write one per line
                    f_doc.write(document.replace("\n", " ") + "\n")
                    f_sum.write(summary.replace("\n", " ") + "\n")
                    converted += 1
                else:
                    skipped += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                skipped += 1

    print(f"✓ Converted: {converted} samples")
    print(f"✗ Skipped: {skipped} samples")
    print(f"✓ Documents: {doc_file}")
    print(f"✓ Summaries: {sum_file}")


def create_readme(output_dir: Path, stats: dict):
    """Create README explaining the data format."""

    readme = output_dir / "README.md"

    content = f"""# Base Model Training Data Format

Data converted from chat template format to plain text format for base model training.

## Format

### JSON Format (`base_model_data.jsonl`)

Each line is a JSON object:

```json
{{
  "document": "Long government report text...",
  "summary": "Concise summary with specific details..."
}}
```

### Plain Text Format

- `documents.txt`: One document per line (newlines replaced with spaces)
- `summaries.txt`: One summary per line (aligned with documents)

## Statistics

- Total samples: {stats['total']}
- Source: GovReport dataset with LLM-as-judge filtering
- Quality criteria: Coverage ≥4 AND Specificity ≥4

## Usage

### For Base Model Training (Text Continuation)

```python
# Load data
with open('documents.txt') as f_doc, open('summaries.txt') as f_sum:
    for doc, summary in zip(f_doc, f_sum):
        # Train format: "Document: {{doc}}\\nSummary: {{summary}}"
        training_text = f"Document: {{doc.strip()}}\\nSummary: {{summary.strip()}}"
```

### For Evaluation (JSON Format)

```python
import json

with open('base_model_data.jsonl') as f:
    for line in f:
        data = json.loads(line)
        document = data['document']
        summary = data['summary']
        # Use for inference testing
```

## Comparison with Chat Template Format

**Chat Template (for instruction-tuned models):**
```json
{{
  "messages": [
    {{"role": "system", "content": "You are a helpful assistant..."}},
    {{"role": "user", "content": "Summarize...\\n\\n[DOCUMENT]"}},
    {{"role": "assistant", "content": "[SUMMARY]"}}
  ]
}}
```

**Base Format (for base models):**
```json
{{
  "document": "[DOCUMENT]",
  "summary": "[SUMMARY]"
}}
```

## Notes

- This format aligns with Apple take-home assessment requirements
- Suitable for base model training (text continuation)
- Can be easily converted back to chat template if needed
"""

    with open(readme, "w") as f:
        f.write(content)

    print(f"✓ README: {readme}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert high quality data to base model format"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "high_quality_data.jsonl",
        help="Input file (chat template format)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "base_model_format",
        help="Output directory"
    )

    parser.add_argument(
        "--format",
        choices=["json", "text", "both"],
        default="both",
        help="Output format: json, text, or both (default: both)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Convert High Quality Data to Base Model Format")
    print("="*80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print()

    # Count total samples
    total_samples = 0
    with open(args.input) as f:
        for line in f:
            if line.strip():
                total_samples += 1

    # Convert
    if args.format in ["json", "both"]:
        json_output = args.output / "base_model_data.jsonl"
        convert_to_json_format(args.input, json_output)
        print()

    if args.format in ["text", "both"]:
        convert_to_plain_text(args.input, args.output)
        print()

    # Create README
    stats = {"total": total_samples}
    create_readme(args.output, stats)

    print()
    print("="*80)
    print("Conversion Complete!")
    print("="*80)
    print(f"\nOutput directory: {args.output}")
    print("\nFiles created:")
    for f in sorted(args.output.glob("*")):
        size = f.stat().st_size / (1024*1024) if f.is_file() else 0
        print(f"  - {f.name:<30} ({size:.2f} MB)" if f.is_file() else f"  - {f.name}/")
    print()


if __name__ == "__main__":
    main()
