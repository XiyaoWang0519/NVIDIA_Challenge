# GCP Setup for Autorae

Guide for running `autorae` experiments on Google Cloud Platform.

## Recommended GCP Configuration

### Instance type

Nemotron-3-Nano-30B is a **MoE model** (30B total params, ~3.6B active per token). In bf16, the weights alone are ~60GB. **You need a single A100 80GB** for LoRA fine-tuning.

| Component | Requirement |
|---|---|
| GPU | **A100 80GB** (minimum — bf16 weights + LoRA training uses ~60GB VRAM) |
| CPU | 16+ vCPU |
| RAM | 128GB+ |
| Disk | 300GB+ SSD (model weights ~60GB + data + checkpoints) |

> **Warning**: A100 40GB is NOT sufficient. bf16 model weights alone are ~60GB.
> If you only have 40GB GPUs, you'd need QLoRA (4-bit quantization + LoRA) or 2x A100-40GB.

### Estimated cost

- A100 80GB on-demand: ~$3.67/hr
- A100 80GB preemptible/spot: ~$1.11/hr (recommended — experiments are restartable)
- At ~12 experiments/hr, that's ~$0.09 per experiment on preemptible

## Setup steps

### 1. Create the instance

```bash
# Via gcloud CLI
gcloud compute instances create autorae-runner \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=300GB \
  --boot-disk-type=pd-ssd \
  --scopes=storage-full
```

Or via Cloud Console: Compute Engine → VM Instances → Create Instance → Choose A100 GPU.

### 2. Install NVIDIA drivers

```bash
# Connect to instance
gcloud compute ssh autorae-runner --zone=us-central1-a

# Install drivers
sudo apt update
sudo apt install build-essential
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-drivers-535
```

### 3. Install CUDA and cuDNN

```bash
# Install CUDA 12.x (required for flash attention + bf16)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Install Python and uv

```bash
# Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 5. Clone the repo and install dependencies

```bash
# Clone your repo (adjust URL)
git clone https://github.com/YOUR_USERNAME/NVIDIA_Challenge.git
cd NVIDIA_Challenge/autorae

# Install dependencies
uv sync

# Verify GPU is visible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### 6. Download model weights

```bash
# Login to HuggingFace (required for Nemotron)
huggingface-cli login

# Or use a token
export HF_TOKEN="your_hf_token"

# Download model (this takes ~30 min for 30B bf16)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Nemotron-3-Nano-30B...')
AutoTokenizer.from_pretrained('nvidia/Nemotron-3-Nano-30B-A3B-bf16', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('nvidia/Nemotron-3-Nano-30B-A3B-bf16', torch_dtype=torch.bfloat16, trust_remote_code=True)
print('Done!')
"
```

### 7. Download competition data

```bash
# Option A: Via Kaggle API
pip install kaggle
mkdir -p ~/.kaggle
# Add your kaggle.json credentials
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge

# Option B: Via prepare.py (if it supports direct download)
uv run python prepare.py
```

### 8. Run experiments

```bash
# Single experiment
uv run train.py > run.log 2>&1

# For long-running autonomous experiments, use screen/tmux
screen -S autorae
uv run train.py > run.log 2>&1
# Detach with Ctrl+A, D
```

## Using preemptible instances (recommended for cost savings)

Autoresearch experiments are restartable, so preemptible instances work well:

```bash
# Create preemptible instance
gcloud compute instances create autorae-runner \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=300GB \
  --boot-disk-type=pd-ssd \
  --preemptible \
  --scopes=storage-full

# Set up auto-restart script (optional)
# Add to crontab:
# @reboot cd /path/to/autorae && uv run train.py > run.log 2>&1
```

## Storage setup

For persistent storage of model weights and data:

```bash
# Create a persistent disk
gcloud compute disks create autorae-data \
  --size=500GB \
  --zone=us-central1-a \
  --type=pd-ssd

# Attach to instance
gcloud compute instances attach-disk autorae-runner \
  --disk=aundae-data \
  --zone=us-central1-a

# Mount
sudo mkfs.ext4 -F /dev/sdb
sudo mount /dev/sdb /mnt/data
```

## SSH key setup for passwordless git

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# Add to GitHub/GitLab
```

## Alternative: Google Cloud Run / Vertex AI

For fully managed experiments, consider:

```bash
# Vertex AI Custom Training
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=autorae-experiment \
  --executor-image-uri=your-container-image \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=1,NVIDIA_TESLA_A100
```

## Cost optimization tips

1. **Use preemptible instances**: 70% cheaper, suitable for restartable experiments
2. **Detach disks**: Save $0.10/GB/month by keeping only model weights on persistent disk
3. **Use Deep Learning VM images**: Pre-configured with CUDA, cuDNN, PyTorch
4. **Spot instances on Vertex AI**: Even cheaper for batch experiments
5. **Check committed use discounts**: 1-year commitments can reduce costs by 60%

## Monitoring

```bash
# GPU utilization
nvidia-smi -l 1

# Training progress
tail -f run.log

# Resource monitoring
htop
```
