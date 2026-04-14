#!/usr/bin/env python3
"""Update README.md model table from quantization reports."""

import json
import sys
from pathlib import Path


def read_report(output_dir: str) -> dict:
    """Read broad_generative_report.json from a quantized model directory."""
    report_path = Path(output_dir) / "broad_generative_report.json"
    if not report_path.exists():
        return {}
    with open(report_path) as f:
        return json.load(f)


def format_size(bytes_val: int) -> str:
    """Format bytes to human-readable size."""
    gb = bytes_val / (1024**3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = bytes_val / (1024**2)
    return f"{mb:.0f} MB"


def main():
    models = [
        {
            "name": "Gemma 4-E4B-it",
            "dir": "gemma-4-E4B-it-ternary",
            "type": "VLM (any-to-any)",
            "hf_link": "AsadIsmail/gemma-4-E4B-it-ternary",
        },
        {
            "name": "Qwen2.5-VL-7B",
            "dir": "Qwen2.5-VL-7B-Instruct-ternary",
            "type": "VLM",
            "hf_link": "AsadIsmail/Qwen2.5-VL-7B-Instruct-ternary",
        },
        {
            "name": "SmolVLM2-2.2B",
            "dir": "SmolVLM2-2.2B-Instruct-ternary",
            "type": "VLM",
            "hf_link": "AsadIsmail/SmolVLM2-2.2B-Instruct-ternary",
        },
        {
            "name": "Whisper-large-v3",
            "dir": "whisper-large-v3-ternary",
            "type": "Speech-to-Text",
            "hf_link": "AsadIsmail/whisper-large-v3-ternary",
        },
    ]

    base_dir = Path(__file__).parent.parent

    rows = []
    for m in models:
        model_dir = base_dir / m["dir"]
        report = read_report(model_dir)

        summary = report.get("summary", {})
        model_info = report.get("model_info", {})

        total_params = sum(
            c.get("parameter_count", 0)
            for c in model_info.get("components", [])
        )
        params_str = f"{total_params / 1e9:.1f}B" if total_params else "TBD"

        fp16_size = total_params * 2 if total_params else 0
        fp16_str = format_size(fp16_size) if fp16_size else "TBD"

        disk = summary.get("disk_bytes")
        disk_str = format_size(disk) if disk else "TBD"

        ratio = summary.get("compression_ratio")
        ratio_str = f"{ratio:.2f}x" if ratio else "TBD"

        bits = summary.get("full_model_effective_bits")
        quality_str = f"{bits:.1f} bits/param" if bits else "TBD"

        row = (
            f"| [{m['name']}](https://huggingface.co/{m['hf_link']}) "
            f"| {m['type']} | {params_str} | {fp16_str} | {disk_str} "
            f"| {ratio_str} | {quality_str} "
            f"| [link](https://huggingface.co/{m['hf_link']}) |"
        )
        rows.append(row)

    header = "| Model | Type | Params | FP16 Size | Ternary Size | Compression | Quality | HuggingFace |"
    separator = "|-------|------|--------|-----------|-------------|-------------|---------|-------------|"

    print(header)
    print(separator)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
