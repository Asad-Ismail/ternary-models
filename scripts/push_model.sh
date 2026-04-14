#!/bin/bash
# Push a quantized model directory to HuggingFace Hub
# Usage: ./scripts/push_model.sh <model-dir> <hub-repo-id>
# Example: ./scripts/push_model.sh ./gemma-4-E4B-it-ternary AsadIsmail/gemma-4-E4B-it-ternary

set -euo pipefail

MODEL_DIR="${1:?Usage: push_model.sh <model-dir> <hub-repo-id>}"
HUB_REPO="${2:?Usage: push_model.sh <model-dir> <hub-repo-id>}"

if [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: ${MODEL_DIR} does not exist"
    exit 1
fi

echo "=== Pushing ${MODEL_DIR} to ${HUB_REPO} ==="

# Create repo if it doesn't exist
hf repos create "${HUB_REPO}" --type model 2>/dev/null || echo "Repo already exists"

# Upload the model
hf upload "${HUB_REPO}" "${MODEL_DIR}" .

echo "=== Published: https://huggingface.co/${HUB_REPO} ==="
