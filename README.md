# ternary-models

Pre-quantized ternary models for VLMs, multimodal, and audio — the models GGUF can't touch.

[Unsloth](https://github.com/unslothai/unsloth) owns GGUF for text LLMs. We do everything else: **vision-language models, any-to-any multimodal, speech-to-text** — quantized to ternary weights with [ternary-quant](https://github.com/Asad-Ismail/ternary-quant) and published on [HuggingFace](https://huggingface.co/AsadIsmail).

## Why ternary quantization for multimodal?

GGUF and GPTQ work great for text-only LLMs. But try quantizing a vision-language model, an any-to-any multimodal model, or Whisper with them — you can't. These architectures have mixed components (vision encoders, text decoders, multimodal connectors, audio encoders) that need **component-aware quantization**.

ternary-quant handles this natively:
- Quantize the text backbone while keeping the vision encoder in FP16
- Quantize both vision + text with different strategies
- Quantize audio encoder/decoder selectively
- 95-99% quality retention on benchmarks

## Models

| Model | Type | Params | FP16 Size | Ternary Size | Compression | Modules | HuggingFace |
|-------|------|--------|-----------|-------------|-------------|---------|-------------|
| [Gemma 4-E4B-it](https://huggingface.co/AsadIsmail/gemma-4-E4B-it-ternary) | VLM (multimodal) | 8B | ~16 GB | 4.2 GB | 3.8x | 342 layers | [link](https://huggingface.co/AsadIsmail/gemma-4-E4B-it-ternary) |
| [Qwen2.5-VL-7B](https://huggingface.co/AsadIsmail/Qwen2.5-VL-7B-Instruct-ternary) | VLM | 7.6B | ~12.7 GB | 7.2 GB | 1.8x | 196 layers | [link](https://huggingface.co/AsadIsmail/Qwen2.5-VL-7B-Instruct-ternary) |
| [SmolVLM2-2.2B](https://huggingface.co/AsadIsmail/SmolVLM2-2.2B-Instruct-ternary) | VLM | 2.2B | ~4.4 GB | — | 1.47x | 169 layers | [link](https://huggingface.co/AsadIsmail/SmolVLM2-2.2B-Instruct-ternary) |
| [Whisper-large-v3](https://huggingface.co/AsadIsmail/whisper-large-v3-ternary) | Speech-to-Text | 1.5B | ~3 GB | 944 MB | 1.8x | 320 layers | [link](https://huggingface.co/AsadIsmail/whisper-large-v3-ternary) |

## Quick start

```bash
pip install ternary-quant

# Load and run any model from the collection
from ternary_quant.inference import load_ternary_model, generate_text

# Vision-Language Model
model, processor = load_ternary_model(
    "AsadIsmail/gemma-4-E4B-it-ternary",
    runtime_mode="metal"  # Apple Silicon — use "cached" for NVIDIA/CPU
)

# Text generation
model, tokenizer = load_ternary_model(
    "AsadIsmail/Qwen2.5-VL-7B-Instruct-ternary",
    runtime_mode="cached"
)
output = generate_text(model, tokenizer, "Describe this image", max_new_tokens=100)
```

## Reproduce any model

Every model in this collection can be reproduced with a single command:

```bash
# Clone and set up
git clone git@github.com:Asad-Ismail/ternary-models.git
cd ternary-models

# Quantize any model (example: Gemma 4-E4B)
pip install ternary-quant
ternary-quant quantize-broad google/gemma-4-E4B-it \
    --output ./gemma-4-E4B-it-ternary \
    --components text_backbone multimodal_connector \
    --scheme tritplane3 \
    --dtype float16 \
    --eval \
    --push-to-hub AsadIsmail/gemma-4-E4B-it-ternary
```

See [scripts/](scripts/) for the exact command used for each model.

## Supported hardware

| Runtime | Hardware | How |
|---------|----------|-----|
| `cached` | Any (NVIDIA, CPU, Apple Silicon) | Dequantize once at load. Fastest. |
| `metal` | Apple Silicon (M1-M4) | Native Metal kernels, 2.3x memory savings |
| `triton_memory` | NVIDIA GPU | Packed ternary in VRAM, fits oversized models |
| `gemlite` | NVIDIA GPU | Best batch throughput |

## What makes this different from Unsloth/GGUF?

| Feature | GGUF (Unsloth) | ternary-quant |
|---------|----------------|---------------|
| Text LLMs | Excellent | Good |
| Vision-Language Models | Not supported | Full support (18 VLM architectures) |
| Any-to-any Multimodal | Not supported | Full support |
| Audio/Speech models | Not supported | Full support (Whisper, etc.) |
| Component-aware quantization | No | Yes (quantize vision/text/audio independently) |
| Apple Silicon Metal kernels | Via llama.cpp | Native Metal shaders |

## Contributing a model

Want to see a specific model quantized? Open an issue or submit a PR:

```bash
# Check if a model is compatible
ternary-quant check <model-id>

# Quantize it
ternary-quant quantize-broad <model-id> \
    --output ./<model-name>-ternary \
    --components <components> \
    --scheme tritplane3 \
    --eval
```

## License

Apache 2.0. Individual model licenses follow their base model's license.
