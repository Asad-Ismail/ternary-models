# Promotional Posts — Ready to Post

## r/LocalLLaMA Post

**Title:** I ternary-quantized Gemma 4's multimodal model, Qwen2.5-VL, SmolVLM2, and Whisper — the models GGUF can't touch

**Body:**

Unsloth does great work with GGUF for text LLMs. But what about vision-language models? Multimodal? Audio? GGUF wasn't designed for these architectures.

I built [ternary-quant](https://github.com/Asad-Ismail/ternary-quant) — post-training ternary quantization that works on VLMs, multimodal models, and audio models. Today I'm publishing the first collection of ternary-quantized multimodal models:

**Models:**

| Model | Type | What it does |
|-------|------|-------------|
| Gemma 4-E4B-it | VLM (any-to-any) | Google's multimodal model — images + text |
| Qwen2.5-VL-7B | VLM | One of the best open VLMs |
| SmolVLM2-2.2B | VLM | HuggingFace's compact VLM for edge |
| Whisper-large-v3 | Speech-to-Text | OpenAI's speech recognition |

**Why ternary for multimodal?**

These models have mixed components — vision encoders, text decoders, multimodal connectors. ternary-quant handles each component independently:
- Quantize the text decoder to ternary
- Keep the vision encoder in FP16 for quality
- Component-aware calibration

**Links:**
- GitHub: https://github.com/Asad-Ismail/ternary-models
- Library: https://github.com/Asad-Ismail/ternary-quant
- HuggingFace: https://huggingface.co/AsadIsmail

**How it works:**
```
pip install ternary-quant
ternary-quant quantize-broad google/gemma-4-E4B-it \
    --components text_backbone multimodal_connector \
    --scheme tritplane3 --eval
```

Not trying to replace GGUF for text LLMs — this fills a genuine gap for the architectures GGUF doesn't support.

---

## Twitter/X Thread

**Tweet 1:**
I ternary-quantized Gemma 4's multimodal model, Qwen2.5-VL, SmolVLM2, and Whisper.

These are the models GGUF can't touch — VLMs, multimodal, audio.

First publicly available ternary-quantized multimodal collection on HuggingFace.

github.com/Asad-Ismail/ternary-models

**Tweet 2:**
Why does this matter?

GGUF/GPTQ work great for text LLMs. But vision-language models have mixed components — vision encoders, text decoders, multimodal connectors.

You can't just pack all the weights the same way. Each component needs different treatment.

**Tweet 3:**
ternary-quant handles this natively:
- Quantize text backbone to ternary
- Keep vision encoder in FP16
- Component-aware calibration
- 18 VLM architectures supported
- Works on Apple Silicon, NVIDIA, CPU

pip install ternary-quant

**Tweet 4:**
Models in the collection:
- Gemma 4-E4B (Google's multimodal)
- Qwen2.5-VL-7B (top open VLM)
- SmolVLM2-2.2B (edge VLM)
- Whisper-large-v3 (speech)

All published on HuggingFace with reproduction scripts.

---

## Hacker News — Show HN

**Title:** Show HN: Ternary-quantized multimodal models – VLMs and audio that GGUF can't handle

**Body:**
Hey HN,

I've been working on ternary quantization for multimodal models — vision-language models, audio models, and anything with mixed architectures that standard quantization tools (GGUF, GPTQ) don't support.

Today I'm publishing `ternary-models`, a collection of pre-quantized multimodal models on HuggingFace:
- Gemma 4-E4B (Google's multimodal)
- Qwen2.5-VL-7B
- SmolVLM2-2.2B
- Whisper-large-v3

The key insight: these models have mixed components (vision encoders, text decoders, multimodal connectors). You need to quantize each independently. ternary-quant does this with component-aware tritplane quantization, keeping 95-99% of benchmark quality.

GitHub: https://github.com/Asad-Ismail/ternary-models
Library: https://github.com/Asad-Ismail/ternary-quant

Would love feedback from anyone running VLMs locally.
