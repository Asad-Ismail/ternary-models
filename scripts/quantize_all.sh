#!/bin/bash
# Quantize all models in the ternary-models collection.
# Run from the repo root: ./scripts/quantize_all.sh
#
# Requirements:
#   pip install ternary-quant
#   hf auth login (for push-to-hub)
#
# Hardware: 48GB+ RAM recommended for the full suite.
# Smaller models (SmolVLM2, Whisper) need only 8-16GB.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Ternary Models Collection ==="
echo "Quantizing all models sequentially..."
echo ""

# 1. SmolVLM2-2.2B (smallest, fastest — good smoke test)
echo "[1/4] SmolVLM2-2.2B-Instruct"
bash "${SCRIPT_DIR}/quantize_smolvlm2.sh"
echo ""

# 2. Whisper-large-v3 (audio, small)
echo "[2/4] Whisper-large-v3"
bash "${SCRIPT_DIR}/quantize_whisper_large.sh"
echo ""

# 3. Gemma 4-E4B (multimodal VLM)
echo "[3/4] Gemma 4-E4B-it"
bash "${SCRIPT_DIR}/quantize_gemma4_e4b.sh"
echo ""

# 4. Qwen2.5-VL-7B (VLM)
echo "[4/4] Qwen2.5-VL-7B-Instruct"
bash "${SCRIPT_DIR}/quantize_qwen25_vl_7b.sh"
echo ""

echo "=== All models quantized and published ==="
