#!/bin/bash
# Quantize Whisper-large-v3 (speech-to-text, ~1.5B params)
# Requires: pip install ternary-quant
# Hardware: 8GB+ RAM (lightweight)

set -euo pipefail

MODEL="openai/whisper-large-v3"
OUTPUT="./whisper-large-v3-ternary"
HUB_REPO="AsadIsmail/whisper-large-v3-ternary"

echo "=== Quantizing ${MODEL} ==="
echo "Components: decoder"
echo "Scheme: tritplane3 (3-plane progressive ternary)"

ternary-quant quantize-broad "${MODEL}" \
    --output "${OUTPUT}" \
    --components decoder \
    --scheme tritplane3 \
    --dtype float16 \
    --n-iter 10 \
    --group-size 32 \
    --seq-len 160 \
    --calibration-batch-size 2 \
    --eval \
    --runtime-mode cached \
    --push-to-hub "${HUB_REPO}"

echo "=== Done. Model at: ${OUTPUT} ==="
echo "=== HuggingFace: https://huggingface.co/${HUB_REPO} ==="
