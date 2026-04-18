#!/bin/bash
#===============================================================================
# Google Cloud VM Setup Script for autoresearch-nemotron
# 
# Run this script on a fresh Google Cloud VM with GPU to set up
# the training environment for NVIDIA Nemotron Model Reasoning Challenge.
#
# Usage: ./setup_gcp.sh
#===============================================================================

set -e

echo "==============================================================================="
echo "NVIDIA Nemotron Model Reasoning Challenge - GCP Setup"
echo "==============================================================================="

#-------------------------------------------------------------------------------
# 1. System Updates
#-------------------------------------------------------------------------------
echo ""
echo "[1/8] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y build-essential git curl wget htop tmux vim

#-------------------------------------------------------------------------------
# 2. Install NVIDIA Driver & CUDA
#-------------------------------------------------------------------------------
echo ""
echo "[2/8] Installing NVIDIA drivers and CUDA..."

# Check if NVIDIA driver is already installed
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver already installed:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    # Install NVIDIA driver
    sudo apt-get install -y nvidia-driver-535
    
    # Reboot may be needed - for now, continue
    echo "Driver installed. Reboot may be required for full GPU access."
fi

#-------------------------------------------------------------------------------
# 3. Install CUDA Toolkit
#-------------------------------------------------------------------------------
echo ""
echo "[3/8] Installing CUDA Toolkit..."

# Download and install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Verify CUDA
nvcc --version

#-------------------------------------------------------------------------------
# 4. Install Python and uv
#-------------------------------------------------------------------------------
echo ""
echo "[4/8] Installing Python and uv..."

# Install Python 3.11
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

#-------------------------------------------------------------------------------
# 5. Clone Repository
#-------------------------------------------------------------------------------
echo ""
echo "[5/8] Setting up autoresearch-nemotron..."

# Create working directory
mkdir -p ~/projects
cd ~/projects

# Clone or update repo
if [ -d "autoresearch-nemotron" ]; then
    echo "Repository already exists, pulling latest..."
    cd autoresearch-nemotron
    git pull
else
    echo "Cloning autoresearch-nemotron..."
    git clone https://github.com/karpathy/autoresearch.git autoresearch-nemotron
    # Note: You'll need to add your fork or copy files
fi

cd autoresearch-nemotron

#-------------------------------------------------------------------------------
# 6. Install Python Dependencies
#-------------------------------------------------------------------------------
echo ""
echo "[6/8] Installing Python dependencies..."

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    accelerate>=0.27.0 \
    datasets>=2.18.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    tqdm>=4.66.0 \
    pyyaml>=6.0.0 \
    regex>=2023.0.0 \
    huggingface-hub>=0.20.0

#-------------------------------------------------------------------------------
# 7. Install Flash Attention (Optional but Recommended)
#-------------------------------------------------------------------------------
echo ""
echo "[7/8] Installing Flash Attention..."

pip install flash-attn --no-build-isolation --no-cache-dir

#-------------------------------------------------------------------------------
# 8. Verify Installation
#-------------------------------------------------------------------------------
echo ""
echo "[8/8] Verifying installation..."

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Verify GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# Verify transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo ""
echo "==============================================================================="
echo "Setup Complete!"
echo "==============================================================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Prepare data: python prepare.py"
echo "3. Start training: python train.py"
echo ""
echo "For screen session: screen -S training"
echo "For results sync: gcloud compute scp ..."
echo ""
