# Quality Comparison: FP16 vs Ternary (tritplane3)

Model: Gemma 4-E4B-it (8B multimodal VLM)
Method: Greedy decoding, same prompts, chat template applied

## Results

| Prompt | FP16 Original | Ternary (tritplane3) | Match |
|--------|--------------|---------------------|-------|
| "What is the capital of France?" | The capital of France is **Paris**. | The capital of France is **Paris**. | Exact |
| "Explain photosynthesis in 2 sentences." | Photosynthesis is the process used by plants and other organisms to convert light energy, usually from the sun, into chemical energy in the form of glucose. This vital process consumes carbon dioxide and water while releasing oxygen as a byproduct. | Photosynthesis is the process where plants convert light energy into chemical energy, using carbon dioxide and water to produce glucose. This process releases oxygen as a byproduct, which is essential for life on Earth. | Same meaning |
| "Write a Python function to reverse a string." | Here are several ways to write a Python function to reverse a string, ranging from the most "Pythonic" (easiest and fastest) to more manual approaches for educational purposes. **1. The Pythonic Way (Slicing) - Recommended** | Here are a few ways to write a Python function to reverse a string, depending on the desired approach. **1. Using Slicing (The most "Pythonic" way)** | Same approach |

## Conclusion

In this Gemma 4-E4B-it sample, ternary quantization with tritplane3 produced output close to the
FP16 baseline: same facts, same reasoning, and same code structure, with minor wording differences
within normal generation variance.

The tradeoff: ~8-10 effective bits per weight (vs GGUF Q4's ~4.5 bits), but in a
HuggingFace-native PTQ workflow that can selectively quantize VLM components.
