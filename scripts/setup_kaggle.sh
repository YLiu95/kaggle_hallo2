#!/usr/bin/env bash
#
# Deterministic environment bootstrap for Kaggle T4 notebooks.
# Pins the CUDA-enabled PyTorch stack before installing the rest of the Python deps.

set -euo pipefail

PYTORCH_VERSION="2.2.2"
TORCHVISION_VERSION="0.17.2"
TORCHAUDIO_VERSION="2.2.2"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

echo "[1/2] Installing CUDA-enabled PyTorch stack..."
pip install --upgrade pip
pip install \
  torch=="${PYTORCH_VERSION}" \
  torchvision=="${TORCHVISION_VERSION}" \
  torchaudio=="${TORCHAUDIO_VERSION}" \
  --extra-index-url "${PYTORCH_INDEX_URL}"

echo "[2/2] Installing Hallo2 dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Environment ready."
