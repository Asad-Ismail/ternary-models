#!/bin/bash
# Deploy the Ternary Model Playground to HuggingFace Spaces
# Requires: hf auth login

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_DIR="${REPO_DIR}/space"
SPACE_REPO="AsadIsmail/ternary-model-playground"

echo "=== Deploying Ternary Model Playground to HF Spaces ==="

# Create the Space repo if it doesn't exist
hf repos create "${SPACE_REPO}" --type space --space-sdk gradio 2>/dev/null || echo "Space already exists"

# Upload the space files
hf upload "${SPACE_REPO}" "${SPACE_DIR}" . --repo-type space

echo "=== Deployed to: https://huggingface.co/spaces/${SPACE_REPO} ==="
