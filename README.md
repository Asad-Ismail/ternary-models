# ternary-models

Pre-quantized models produced with
[ternary-quant](https://github.com/Asad-Ismail/ternary-quant) and published on
[Hugging Face](https://huggingface.co/AsadIsmail).

This repo is the model-release companion to `ternary-quant`. The library handles
post-training ternary quantization and inference; this repo tracks release
metadata, model cards, reproduction scripts, validation notes, and the Gradio
playground.

## Positioning

GGUF/llama.cpp and Unsloth are strong defaults for local text-LLM deployment,
and llama.cpp supports selected multimodal models through `libmtmd`.

`ternary-quant` targets a different workflow:

- Hugging Face/Python-native post-training ternary quantization.
- Component-aware quantization for mixed architectures such as VLMs, audio
  models, seq2seq models, and diffusion/video transformers.
- Explicit control over which components are quantized and which stay in
  higher precision, for example keeping a vision encoder in FP16 while
  quantizing a text backbone or decoder.
- Packed ternary runtime modes for CUDA/Triton and Apple Silicon Metal, plus a
  fast `cached` mode that dequantizes once at load.

The claim is not that this beats GGUF/Unsloth for standard decoder-only LLMs.
For plain local text generation, GGUF/Unsloth will usually be the more mature
deployment path. The value here is component-aware ternary PTQ for model
families that are awkward to handle with text-LLM-first quantization tooling.

## Hub Collection

As of 2026-04-27, the Hugging Face collection contains these public
ternary/ternary-derived releases:

| Model | Task | Base model | Notes |
|---|---|---|---|
| [Qwen3-1.7B-ternary](https://huggingface.co/AsadIsmail/Qwen3-1.7B-ternary) | Text generation | `Qwen/Qwen3-1.7B` | Early `ternary-quant` text model. |
| [Qwen2-VL-2B-ternary](https://huggingface.co/AsadIsmail/Qwen2-VL-2B-ternary) | Image-text-to-text | `Qwen/Qwen2-VL-2B-Instruct` | VLM checkpoint. |
| [Gemma4-E2B-ternary](https://huggingface.co/AsadIsmail/Gemma4-E2B-ternary) | Image-text-to-text | `google/gemma-4-E2B-it` | Early Gemma 4 multimodal checkpoint. |
| [Qwen2.5-VL-7B-Instruct-ternary](https://huggingface.co/AsadIsmail/Qwen2.5-VL-7B-Instruct-ternary) | Image-text-to-text | `Qwen/Qwen2.5-VL-7B-Instruct` | VLM checkpoint. |
| [gemma-4-E4B-it-ternary](https://huggingface.co/AsadIsmail/gemma-4-E4B-it-ternary) | Image-text-to-text | `google/gemma-4-E4B-it` | Gemma 4 multimodal checkpoint. |
| [SmolVLM2-2.2B-Instruct-ternary](https://huggingface.co/AsadIsmail/SmolVLM2-2.2B-Instruct-ternary) | Image-text-to-text | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Compact VLM checkpoint. |
| [whisper-large-v3-ternary](https://huggingface.co/AsadIsmail/whisper-large-v3-ternary) | Automatic speech recognition | `openai/whisper-large-v3` | Decoder-focused Whisper quantization. |
| [gemma-4-31B-it-ternary](https://huggingface.co/AsadIsmail/gemma-4-31B-it-ternary) | Image-text-to-text | `google/gemma-4-31B-it` | Large Gemma 4 multimodal checkpoint. |
| [CogVideoX-2b-ternary](https://huggingface.co/AsadIsmail/CogVideoX-2b-ternary) | Text-to-video | `zai-org/CogVideoX-2b` | Diffusers-compatible video model artifact. |
| [CogVideoX-5b-ternary](https://huggingface.co/AsadIsmail/CogVideoX-5b-ternary) | Text-to-video | `zai-org/CogVideoX-5b` | Diffusers-compatible video model artifact. |
| [SmolVLM2-500M-Video-Instruct-ternary](https://huggingface.co/AsadIsmail/SmolVLM2-500M-Video-Instruct-ternary) | Image-text-to-text | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | Small video-capable VLM. |
| [SmolVLM2-256M-Video-Instruct-ternary](https://huggingface.co/AsadIsmail/SmolVLM2-256M-Video-Instruct-ternary) | Image-text-to-text | `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | Edge-oriented video-capable VLM. |
| [whisper-medium-ternary](https://huggingface.co/AsadIsmail/whisper-medium-ternary) | Automatic speech recognition | `openai/whisper-medium` | Whisper ASR checkpoint. |
| [whisper-small-ternary](https://huggingface.co/AsadIsmail/whisper-small-ternary) | Automatic speech recognition | `openai/whisper-small` | Smaller Whisper ASR checkpoint. |
| [Wan2.1-T2V-1.3B-ternary](https://huggingface.co/AsadIsmail/Wan2.1-T2V-1.3B-ternary) | Text-to-video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Diffusers-compatible video model artifact. |
| [Wan2.2-TI2V-5B-ternary](https://huggingface.co/AsadIsmail/Wan2.2-TI2V-5B-ternary) | Text/image-to-video | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Diffusers-compatible video model artifact. |

The local repo currently contains reproduction assets for a subset of the Hub
collection: Qwen2.5-VL-7B, Gemma 4 E4B, Gemma 4 31B, SmolVLM2-2.2B, and
Whisper-large-v3. Some newer Hub releases may not yet have local model-card
folders or scripts in this repository.

## What Is Actually Quantized?

`ternary-quant` is component-first. Depending on the architecture and script,
it may quantize only selected components rather than every parameter in the
model.

Examples:

- VLMs: quantize the language backbone and/or multimodal connector while
  preserving sensitive vision components when needed.
- Whisper/audio: quantize selected encoder or decoder components.
- Video diffusion transformers: publish Diffusers-compatible artifacts; check
  the individual model card for whether packed ternary storage or dequantized
  compatibility storage is used.

Do not assume a table-level compression number applies uniformly to every
parameter. For exact size, quantized components, validation notes, and runtime
requirements, use the individual Hugging Face model card.

## Quick Start

```bash
pip install ternary-quant
```

```python
from ternary_quant.inference import load_ternary_model, generate_text

model, tokenizer = load_ternary_model(
    "AsadIsmail/Qwen3-1.7B-ternary",
    device="auto",
    runtime_mode="cached",
)

output = generate_text(
    model,
    tokenizer,
    "Describe what component-aware quantization means.",
    max_new_tokens=100,
)
print(output)
```

For image, audio, and video models, follow the usage block on the specific
model card. Processor/tokenizer behavior differs by architecture.

## Reproducing Releases

Scripts in [`scripts/`](scripts/) capture the commands used for the local
release subset. Example:

```bash
pip install ternary-quant

ternary-quant quantize-broad google/gemma-4-E4B-it \
    --output ./gemma-4-E4B-it-ternary \
    --components text_backbone multimodal_connector \
    --scheme tritplane3 \
    --dtype float16 \
    --eval \
    --push-to-hub AsadIsmail/gemma-4-E4B-it-ternary
```

Reproduction is model-dependent. Larger VLMs and video models need more RAM,
may require model-license access, and may need architecture-specific component
selection.

## Runtime Modes

| Runtime | Hardware | Behavior |
|---|---|---|
| `cached` | CPU, NVIDIA, Apple Silicon | Dequantizes selected weights at load. Usually fastest and simplest. |
| `metal` | Apple Silicon | Uses packed ternary Metal kernels where supported. |
| `triton_memory` | NVIDIA GPU | Uses packed ternary CUDA/Triton path to reduce peak VRAM where supported. |
| `gemlite` | NVIDIA GPU | Uses packed low-bit GEMM/GEMV path where supported by the installed stack. |

Runtime support depends on model family, installed `ternary-quant` version,
hardware, and optional dependencies.

## Comparison With GGUF / Unsloth

| Area | GGUF / Unsloth | `ternary-quant` / this repo |
|---|---|---|
| Standard text LLM deployment | Mature, fast, broad local ecosystem. | Supported, but not the main advantage. |
| Selected VLM deployment | Supported in llama.cpp through `libmtmd` for supported models. | HF-native component-aware PTQ path for selected VLMs. |
| Audio / ASR | llama.cpp has experimental multimodal audio support; coverage depends on model support. | Whisper checkpoints are published; component choices are model-specific. |
| Quantization type | Mixed GGUF low-bit formats, including dynamic layer/model-specific schemes. | Ternary PTQ: grouped asymmetric ternary and TritPlane-style multi-plane ternary. |
| Component control | Deployment-oriented; component control depends on converter/runtime support. | Explicit component selection such as text backbone, decoder, encoder, connector, or vision backbone. |
| Best use case | Local deployment of supported models. | Research and release workflow for component-aware ternary quantization in the HF/Python stack. |

## Validation Scope

`ternary-quant` reports 95.4-99.2% FP16 quality retention on its measured
decoder-LLM lm-eval suite. That number should not be read as a universal claim
for every VLM, ASR, or video model in this repository.

For multimodal and audio releases, validation varies by model card and may
include reconstruction error, smoke-test prompts, coherent-output checks, or
task-specific examples. Formal benchmark coverage is still model-dependent.

## License

Repository code and metadata are Apache 2.0. Each model artifact follows the
license and terms of its base model.
