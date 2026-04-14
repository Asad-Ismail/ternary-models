#!/bin/bash
# Quantize Qwen2.5-VL-7B-Instruct (vision-language model, ~7.6B params)
# Requires: pip install ternary-quant
# Hardware: 32GB+ RAM recommended

set -euo pipefail

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT="./Qwen2.5-VL-7B-Instruct-ternary"
HUB_REPO="AsadIsmail/Qwen2.5-VL-7B-Instruct-ternary"

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
