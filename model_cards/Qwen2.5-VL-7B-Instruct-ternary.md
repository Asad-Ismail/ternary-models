---
language:
- en
- zh
library_name: transformers
tags:
- ternary-quant
- quantization
- ternary
- vlm
- multimodal
- qwen
base_model: Qwen/Qwen2.5-VL-7B-Instruct
pipeline_tag: image-text-to-text
license: apache-2.0
---

# Qwen2.5-VL-7B-Instruct — Ternary Quantized

Ternary-quantized version of [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
produced with [ternary-quant](https://github.com/Asad-Ismail/ternary-quant).

Qwen2.5-VL is one of the most capable open-weight vision-language models. This ternary-quantized
version makes it accessible on consumer hardware while preserving multimodal understanding — a
capability that GGUF quantization cannot provide for VLM architectures.

## Quantization details

| Metric | Value |
|--------|-------|
| **Scheme** | tritplane3 (3-plane progressive ternary) |
| **Components quantized** | text_backbone, multimodal_connector (196 linear layers) |
| **Vision encoder** | Kept in FP16 |
| **Stored size** | 7341 MB (~7.2 GB) |
| **FP16 size** | 13051 MB (~12.7 GB) |
| **Compression ratio** | 1.8x |

## Usage

```python
from ternary_quant.inference import load_ternary_model

model, processor = load_ternary_model(
    "AsadIsmail/Qwen2.5-VL-7B-Instruct-ternary",
    runtime_mode="metal"  # "cached" for NVIDIA/CPU
)

from PIL import Image
image = Image.open("photo.jpg")
inputs = processor(text="What is shown in this image?", images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Reproduce

```bash
pip install ternary-quant
ternary-quant quantize-broad Qwen/Qwen2.5-VL-7B-Instruct \
    --output ./Qwen2.5-VL-7B-Instruct-ternary \
    --components text_backbone multimodal_connector \
    --scheme tritplane3 --dtype float16 --eval
```

## Part of the ternary-models collection

[github.com/Asad-Ismail/ternary-models](https://github.com/Asad-Ismail/ternary-models)
