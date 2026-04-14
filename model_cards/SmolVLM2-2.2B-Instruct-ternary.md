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
- smolvlm
- edge
base_model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
pipeline_tag: image-text-to-text
license: apache-2.0
---

# SmolVLM2-2.2B-Instruct — Ternary Quantized

Ternary-quantized version of [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct),
produced with [ternary-quant](https://github.com/Asad-Ismail/ternary-quant).

SmolVLM2 is HuggingFace's compact vision-language model designed for edge deployment. The
ternary-quantized version pushes it even further — making it feasible for mobile and IoT devices.

## Quantization details

| Metric | Value |
|--------|-------|
| **Scheme** | tritplane3 (3-plane progressive ternary) |
| **Components quantized** | text_backbone, multimodal_connector (169 linear layers) |
| **Vision encoder** | Kept in FP16 |
| **Full-model effective bits** | 10.92 |
| **Compression ratio** | 1.47x |
| **Avg reconstruction error** | 0.1236 |
| **Validation** | Passed (correctly describes demo image) |

## Usage

```python
from ternary_quant.inference import load_ternary_model

model, processor = load_ternary_model(
    "AsadIsmail/SmolVLM2-2.2B-Instruct-ternary",
    runtime_mode="cached"
)

from PIL import Image
image = Image.open("photo.jpg")
inputs = processor(text="Describe this image", images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Reproduce

```bash
pip install ternary-quant
ternary-quant quantize-broad HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --output ./SmolVLM2-2.2B-Instruct-ternary \
    --components text_backbone multimodal_connector \
    --scheme tritplane3 --dtype float16 --eval
```

## Part of the ternary-models collection

[github.com/Asad-Ismail/ternary-models](https://github.com/Asad-Ismail/ternary-models)
