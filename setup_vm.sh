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
# Use huggingface-cli to download without loading (avoids mamba_ssm import issue)
MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
uv run huggingface-cli download "$MODEL_ID" --token "${HF_TOKEN:-}" --local-dir-use-symlinks True 2>&1 | tail -5
echo "Model download complete."

echo ""
echo "========================================="
echo "  Setup complete! Run training with:"
echo "  uv run train.py 2>&1 | tee run.log"
echo "========================================="
