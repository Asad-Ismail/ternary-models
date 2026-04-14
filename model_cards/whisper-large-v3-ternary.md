---
language:
- en
- zh
- de
- es
- ru
- ko
- fr
- ja
- pt
- tr
- pl
- ca
- nl
- ar
- sv
- it
- id
- hi
- fi
- vi
- he
- uk
- el
- ms
- cs
- ro
- da
- hu
- ta
- "no"
- th
- ur
- hr
- bg
- lt
- la
- mi
- ml
- cy
- sk
- te
- fa
- lv
- bn
- sr
- az
- sl
- kn
- et
- mk
- br
- eu
- is
- hy
- ne
- mn
- bs
- kk
- sq
- sw
- gl
- mr
- pa
- si
- km
- sn
- yo
- so
- af
- oc
- ka
- be
- tg
- sd
- gu
- am
- yi
- lo
- uz
- fo
- ht
- ps
- tk
- nn
- mt
- sa
- lb
- my
- bo
- tl
- mg
- as
- tt
- haw
- ln
- ha
- ba
- jw
- su
- yue
library_name: transformers
tags:
- ternary-quant
- quantization
- ternary
- audio
- speech-to-text
- whisper
base_model: openai/whisper-large-v3
pipeline_tag: automatic-speech-recognition
license: apache-2.0
---

# Whisper-large-v3 — Ternary Quantized

Ternary-quantized version of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3),
produced with [ternary-quant](https://github.com/Asad-Ismail/ternary-quant).

This demonstrates ternary-quant's unique capability to quantize **audio/speech models** — something
GGUF and GPTQ were not designed for. The decoder is ternary-quantized while the audio encoder is
preserved in FP16 for maximum transcription quality.

## Quantization details

| Metric | Value |
|--------|-------|
| **Scheme** | tritplane3 (3-plane progressive ternary) |
| **Components quantized** | decoder (320 linear layers) |
| **Audio encoder** | Kept in FP16 (preserving audio understanding quality) |
| **Stored size** | 943.7 MB |
| **FP16 size** | 1677.7 MB |
| **Compression ratio** | 1.8x |

## Usage

```python
from ternary_quant.inference import load_ternary_model
import torch

model, processor = load_ternary_model(
    "AsadIsmail/whisper-large-v3-ternary",
    runtime_mode="cached"
)

# Transcribe audio
import librosa
audio, sr = librosa.load("audio.mp3", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    predicted_ids = model.generate(**inputs)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])
```

## Reproduce

```bash
pip install ternary-quant
ternary-quant quantize-broad openai/whisper-large-v3 \
    --output ./whisper-large-v3-ternary \
    --components decoder \
    --scheme tritplane3 --dtype float16 --eval
```

## Part of the ternary-models collection

[github.com/Asad-Ismail/ternary-models](https://github.com/Asad-Ismail/ternary-models)
