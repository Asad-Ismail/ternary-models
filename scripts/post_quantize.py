#!/usr/bin/env python3
"""
Post-quantization: update model card with real metrics and push to HuggingFace.

Usage:
    python scripts/post_quantize.py <model-dir> <hub-repo-id> [model-card-template]

Example:
    python scripts/post_quantize.py \
        ./gemma-4-E4B-it-ternary \
        AsadIsmail/gemma-4-E4B-it-ternary \
        model_cards/gemma-4-E4B-it-ternary.md
"""

import json
import re
import sys
from pathlib import Path


def read_report(model_dir: Path) -> dict:
    report_path = model_dir / "broad_generative_report.json"
    if not report_path.exists():
        print(f"Warning: {report_path} not found")
        return {}
    with open(report_path) as f:
        return json.load(f)


def format_size(bytes_val: int) -> str:
    gb = bytes_val / (1024**3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = bytes_val / (1024**2)
    return f"{mb:.0f} MB"


def update_model_card(template_path: Path, report: dict, output_path: Path):
    """Replace TBD placeholders in model card with real values."""
    summary = report.get("summary", {})

    text = template_path.read_text()

    # Replace TBD values with actuals
    replacements = {
        "full_model_effective_bits": summary.get("full_model_effective_bits"),
        "compression_ratio": summary.get("compression_ratio"),
        "avg_relative_error": summary.get("avg_relative_error", summary.get("avg_reconstruction_error")),
        "quantized_modules": summary.get("quantized_modules"),
    }

    if replacements["full_model_effective_bits"]:
        text = re.sub(
            r"\*TBD after quantization\*",
            f"{replacements['full_model_effective_bits']:.1f}",
            text,
        )
        text = re.sub(
            r"\*TBD\*",
            lambda m: "TBD",  # Leave remaining TBDs for now
            text,
            count=0,
        )

    # Replace specific TBD fields in the table
    if replacements["full_model_effective_bits"]:
        text = text.replace(
            "| **Full-model effective bits** | *TBD*",
            f"| **Full-model effective bits** | {replacements['full_model_effective_bits']:.1f}",
        )
        text = text.replace(
            "| **Full-model effective bits** | *TBD after quantization*",
            f"| **Full-model effective bits** | {replacements['full_model_effective_bits']:.1f}",
        )

    if replacements["compression_ratio"]:
        text = text.replace(
            "| **Compression ratio** | *TBD*",
            f"| **Compression ratio** | {replacements['compression_ratio']:.2f}x",
        )

    if replacements["avg_relative_error"]:
        text = text.replace(
            "| **Avg reconstruction error** | *TBD*",
            f"| **Avg reconstruction error** | {replacements['avg_relative_error']:.4f}",
        )

    output_path.write_text(text)
    print(f"Updated model card: {output_path}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    hub_repo = sys.argv[2]
    template = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    if not model_dir.exists():
        print(f"Error: {model_dir} does not exist")
        sys.exit(1)

    report = read_report(model_dir)
    if not report:
        print("No report found. Skipping model card update.")
    else:
        summary = report.get("summary", {})
        print(f"Quantization results:")
        print(f"  Method: {report.get('method', '?')}")
        print(f"  Modules quantized: {summary.get('quantized_modules', '?')}")
        print(f"  Effective bits: {summary.get('full_model_effective_bits', '?')}")
        print(f"  Compression ratio: {summary.get('compression_ratio', '?')}")
        print(f"  Avg relative error: {summary.get('avg_relative_error', '?')}")

        # Update model card if template provided
        if template and template.exists():
            readme_path = model_dir / "README.md"
            update_model_card(template, report, readme_path)

    print(f"\nTo push to HuggingFace:")
    print(f"  hf upload {hub_repo} {model_dir} .")


if __name__ == "__main__":
    main()
