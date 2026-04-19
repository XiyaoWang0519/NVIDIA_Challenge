#!/bin/bash
# One-shot setup script for A100 80GB GPU instances (RunPod / Lambda Cloud / GCP)
# Run as: bash setup_vm.sh
set -e

echo "=== 1/7 System packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq git curl wget python3 python3-venv python3-pip build-essential > /dev/null

echo "=== 2/7 Install uv ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "=== 3/7 Verify GPU ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}'); assert torch.cuda.is_available(), 'No GPU found!'; assert torch.cuda.get_device_properties(0).total_mem >= 70*1024**3, f'GPU has {torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB, need 80GB'"

echo "=== 4/7 Clone repo ==="
if [ ! -d "NVIDIA_Challenge" ]; then
    git clone https://github.com/XiyaoWang0519/NVIDIA_Challenge.git
fi
cd NVIDIA_Challenge/autorae

echo "=== 5/7 Install Python dependencies ==="
uv sync

echo "=== 6/7 Download competition data ==="
uv run python prepare.py

echo "=== 7/7 Download model weights (this takes ~30 min) ==="
uv run python -c "
from prepare import MODEL_NAME
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print(f'Downloading {MODEL_NAME}...')
AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True)
print('Done!')
"

echo ""
echo "========================================="
echo "  Setup complete! Run training with:"
echo "  cd NVIDIA_Challenge/autorae"
echo "  uv run train.py 2>&1 | tee run.log"
echo "========================================="
