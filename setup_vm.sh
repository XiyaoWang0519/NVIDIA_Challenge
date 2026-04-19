#!/bin/bash
# One-shot setup script for A100 80GB GPU instances (RunPod / Lambda Cloud / GCP)
# Run as: bash setup_vm.sh
set -e

# Use sudo only if available (not in Docker containers where we're already root)
run_as_root() { if command -v sudo &>/dev/null; then sudo "$@"; else "$@"; fi; }

echo "=== 1/7 System packages ==="
run_as_root apt-get update -qq 2>/dev/null || true
run_as_root apt-get install -y -qq git curl wget python3 python3-venv python3-pip build-essential 2>/dev/null || true

echo "=== 2/7 Install uv ==="
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=== 3/7 Verify GPU ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}'); assert torch.cuda.is_available(), 'No GPU found!'; mem_gb=torch.cuda.get_device_properties(0).total_memory/1024**3; print(f'VRAM: {mem_gb:.0f}GB'); assert mem_gb >= 70, f'Need 80GB, got {mem_gb:.0f}GB'"

echo "=== 4/7 Install Python dependencies ==="
uv sync

echo "=== 5/7 Download competition data ==="
uv run python prepare.py || true

echo "=== 6/7 Download model weights (this takes ~30 min) ==="
# Note: prepare.py has the old model name. The correct HuggingFace ID is NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
# We download by the correct name, and prepare.py's train.py will find it via HuggingFace cache.
MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
uv run python -c "
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
token = os.environ.get('HF_TOKEN')
print(f'Downloading $MODEL_ID...')
AutoTokenizer.from_pretrained('$MODEL_ID', trust_remote_code=True, token=token)
AutoModelForCausalLM.from_pretrained('$MODEL_ID', torch_dtype=torch.bfloat16, trust_remote_code=True, token=token)
print('Done!')
"

echo ""
echo "========================================="
echo "  Setup complete! Run training with:"
echo "  uv run train.py 2>&1 | tee run.log"
echo "========================================="
