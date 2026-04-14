#!/usr/bin/env python3
"""
Validate all quantized models in the collection.

For each model directory, loads it, runs a test generation, and reports results.
"""

import json
import sys
import time
from pathlib import Path


VENV_PYTHON = "/Users/asad.ismail/personal/ternary-quant/.venv/bin/python"

MODELS = {
    "gemma-4-E4B-it-ternary": {
        "type": "vlm",
        "prompt": "Describe this image",
        "description": "Gemma 4-E4B (multimodal VLM)",
    },
    "Qwen2.5-VL-7B-Instruct-ternary": {
        "type": "vlm",
        "prompt": "What do you see in this image?",
        "description": "Qwen2.5-VL-7B (VLM)",
    },
    "SmolVLM2-2.2B-Instruct-ternary": {
        "type": "vlm",
        "prompt": "Describe this image briefly",
        "description": "SmolVLM2-2.2B (compact VLM)",
    },
    "whisper-large-v3-ternary": {
        "type": "audio",
        "description": "Whisper-large-v3 (speech-to-text)",
    },
}


def validate_text_model(model_dir: str, prompt: str = "Hello, how are you?"):
    """Validate a text/VLM model by running generation."""
    from ternary_quant.inference import load_ternary_model, generate_text

    t0 = time.perf_counter()
    model, tokenizer = load_ternary_model(model_dir, runtime_mode="cached")
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    output = generate_text(model, tokenizer, prompt, max_new_tokens=50)
    gen_time = time.perf_counter() - t0

    tokens = len(tokenizer.encode(output))
    tok_s = tokens / gen_time if gen_time > 0 else 0

    return {
        "load_time_s": round(load_time, 1),
        "gen_time_s": round(gen_time, 2),
        "tokens": tokens,
        "tok_per_s": round(tok_s, 1),
        "output_preview": output[:200],
        "success": len(output.strip()) > 0,
    }


def validate_from_report(model_dir: str):
    """Read and validate the quantization report."""
    report_path = Path(model_dir) / "broad_generative_report.json"
    if not report_path.exists():
        return {"error": "No report found"}

    with open(report_path) as f:
        report = json.load(f)

    summary = report.get("summary", {})
    return {
        "method": report.get("method", "?"),
        "modules_quantized": summary.get("quantized_modules", "?"),
        "effective_bits": summary.get("full_model_effective_bits", "?"),
        "compression_ratio": summary.get("compression_ratio", "?"),
        "avg_relative_error": summary.get("avg_relative_error", "?"),
    }


def main():
    base_dir = Path("/Users/asad.ismail/personal/ternary-models")

    print("=" * 60)
    print("Ternary Models Collection — Validation Report")
    print("=" * 60)

    for model_name, info in MODELS.items():
        model_dir = base_dir / model_name
        print(f"\n--- {info['description']} ---")
        print(f"Directory: {model_dir}")

        if not model_dir.exists():
            print("  Status: NOT YET QUANTIZED")
            continue

        # Check report
        report = validate_from_report(str(model_dir))
        if "error" in report:
            print(f"  Report: {report['error']}")
        else:
            print(f"  Method: {report['method']}")
            print(f"  Modules: {report['modules_quantized']}")
            print(f"  Effective bits: {report['effective_bits']}")
            print(f"  Compression: {report['compression_ratio']}x")
            print(f"  Avg error: {report['avg_relative_error']}")

        # Run generation test (only for text/vlm models)
        if info["type"] in ("text", "vlm"):
            try:
                result = validate_text_model(str(model_dir), info.get("prompt", "Hello"))
                print(f"  Load time: {result['load_time_s']}s")
                print(f"  Speed: {result['tok_per_s']} tok/s")
                print(f"  Output: {result['output_preview'][:100]}...")
                print(f"  Passed: {'YES' if result['success'] else 'NO'}")
            except Exception as e:
                print(f"  Generation test FAILED: {e}")

    print("\n" + "=" * 60)
    print("Validation complete.")


if __name__ == "__main__":
    main()
