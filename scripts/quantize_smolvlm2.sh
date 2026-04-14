#!/bin/bash
# Quantize SmolVLM2-2.2B-Instruct (compact VLM, ~2.2B params)
# Requires: pip install ternary-quant
# Hardware: 16GB+ RAM (runs on any modern machine)

set -euo pipefail

MODEL="HuggingFaceTB/SmolVLM2-2.2B-Instruct"
OUTPUT="./SmolVLM2-2.2B-Instruct-ternary"
HUB_REPO="AsadIsmail/SmolVLM2-2.2B-Instruct-ternary"

echo "=== Quantizing ${MODEL} ==="
echo "Components: text_backbone, multimodal_connector"
echo "Scheme: tritplane3 (3-plane progressive ternary)"

ternary-quant quantize-broad "${MODEL}" \
    --output "${OUTPUT}" \
    --components text_backbone multimodal_connector \
    --scheme tritplane3 \
    --dtype float16 \
    --n-iter 10 \
    --group-size 32 \
    --seq-len 160 \
    --calibration-batch-size 2 \
    --eval \
    --runtime-mode metal \
    --push-to-hub "${HUB_REPO}"

echo "=== Done. Model at: ${OUTPUT} ==="
echo "=== HuggingFace: https://huggingface.co/${HUB_REPO} ==="
