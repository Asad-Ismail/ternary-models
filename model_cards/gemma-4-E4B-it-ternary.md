---
language:
- en
library_name: transformers
tags:
- ternary-quant
- quantization
- ternary
- vlm
- multimodal
- gemma
- google
base_model: google/gemma-4-E4B-it
pipeline_tag: image-text-to-text
license: gemma
---

# Gemma 4-E4B-it — Ternary Quantized

Ternary-quantized version of [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it),
produced with [ternary-quant](https://github.com/Asad-Ismail/ternary-quant).

**This is the first publicly available ternary-quantized multimodal model on HuggingFace.**

Gemma 4-E4B is Google's multimodal model that processes both images and text. This ternary-quantized
version retains the multimodal capabilities while significantly reducing model size — something GGUF
and GPTQ cannot do for vision-language architectures.

## Quantization details

| Metric | Value |
|--------|-------|
| **Scheme** | tritplane3 (3-plane progressive ternary) |
| **Components quantized** | text_backbone, multimodal_connector |
| **Vision encoder** | Kept in FP16 (preserving image understanding quality) |
| **Group size** | 32 |
| **Calibration iterations** | 10 |
| **Full-model effective bits** | *TBD after quantization* |
| **Compression ratio** | *TBD* |
| **Avg reconstruction error** | *TBD* |

## Usage

```python
from ternary_quant.inference import load_ternary_model

# Load model (auto-detects best device: CUDA > MPS > CPU)
model, processor = load_ternary_model(
    "AsadIsmail/gemma-4-E4B-it-ternary",
    runtime_mode="metal"  # Use "cached" for NVIDIA GPU or CPU
)

# Image understanding
from PIL import Image
image = Image.open("photo.jpg")
inputs = processor(text="Describe this image", images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Hardware requirements

| Runtime | Min RAM | Speed | Notes |
|---------|---------|-------|-------|
| `metal` (Apple Silicon) | ~8 GB | Good | Native Metal kernels |
| `cached` (any device) | ~12 GB | Fastest | Dequantizes once at load |
| `triton_memory` (NVIDIA) | ~6 GB | Moderate | Packed ternary in VRAM |

## Reproduce

```bash
pip install ternary-quant
ternary-quant quantize-broad google/gemma-4-E4B-it \
    --output ./gemma-4-E4B-it-ternary \
    --components text_backbone multimodal_connector \
    --scheme tritplane3 \
    --dtype float16 \
    --eval
```

## Part of the ternary-models collection

See [github.com/Asad-Ismail/ternary-models](https://github.com/Asad-Ismail/ternary-models) for the
full collection of ternary-quantized multimodal models.

## Citation

```bibtex
@software{ternary_quant,
  author = {Ismail, Asad},
  title = {ternary-quant: Post-training ternary quantization for HuggingFace generative models},
  url = {https://github.com/Asad-Ismail/ternary-quant},
  year = {2026}
}
```
