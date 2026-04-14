#!/bin/bash
# Quantize Gemma 4-E4B-it (any-to-any multimodal VLM, ~8B params)
# Requires: pip install ternary-quant
# Hardware: 48GB+ RAM recommended (Apple Silicon or NVIDIA GPU)

set -euo pipefail

MODEL="google/gemma-4-E4B-it"
OUTPUT="./gemma-4-E4B-it-ternary"
HUB_REPO="AsadIsmail/gemma-4-E4B-it-ternary"

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
