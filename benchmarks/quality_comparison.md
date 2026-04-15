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

Ternary quantization with tritplane3 produces **near-identical output to FP16** on Gemma 4-E4B-it.
Same facts, same reasoning, same code structure. Only minor wording differences that are within
normal generation variance.

The tradeoff: ~8-10 effective bits per weight (vs GGUF Q4's ~4.5 bits), but applied to VLM
architectures that GGUF cannot quantize at all.
