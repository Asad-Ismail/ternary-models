#!/usr/bin/env python3
"""
Compare ternary-quantized models against FP16 baselines.

Runs generation on each quantized model and reports quality + performance.
"""

import argparse
import time
from pathlib import Path


def benchmark_model(model_dir: str, runtime_mode: str = "cached", prompt: str = "Explain quantum computing in one paragraph."):
    """Load and benchmark a ternary-quantized model."""
    from ternary_quant.inference import load_ternary_model, generate_text

    print(f"\nBenchmarking: {model_dir}")
    print(f"  Runtime mode: {runtime_mode}")

    # Load
    t0 = time.perf_counter()
    model, tokenizer = load_ternary_model(model_dir, runtime_mode=runtime_mode)
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time:.1f}s")

    # Generate
    t0 = time.perf_counter()
    output = generate_text(model, tokenizer, prompt, max_new_tokens=100)
    gen_time = time.perf_counter() - t0

    # Count tokens
    tokens = len(tokenizer.encode(output))
    tok_per_sec = tokens / gen_time if gen_time > 0 else 0

    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Tokens generated: {tokens}")
    print(f"  Speed: {tok_per_sec:.1f} tok/s")
    print(f"  Output: {output[:200]}...")

    return {
        "model": model_dir,
        "load_time": load_time,
        "gen_time": gen_time,
        "tokens": tokens,
        "tok_per_sec": tok_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ternary models")
    parser.add_argument("model_dirs", nargs="+", help="Quantized model directories")
    parser.add_argument("--runtime", default="cached", help="Runtime mode")
    parser.add_argument("--prompt", default="Explain quantum computing in one paragraph.")
    args = parser.parse_args()

    results = []
    for model_dir in args.model_dirs:
        if Path(model_dir).exists():
            try:
                r = benchmark_model(model_dir, args.runtime, args.prompt)
                results.append(r)
            except Exception as e:
                print(f"  Error: {e}")

    if results:
        print("\n=== Summary ===")
        print(f"{'Model':<40} {'Load (s)':<10} {'tok/s':<10}")
        print("-" * 60)
        for r in results:
            print(f"{r['model']:<40} {r['load_time']:<10.1f} {r['tok_per_sec']:<10.1f}")


if __name__ == "__main__":
    main()
