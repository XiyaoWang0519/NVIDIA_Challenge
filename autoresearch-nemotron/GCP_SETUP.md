# Google Cloud Setup for NVIDIA Nemotron Model Reasoning Challenge

## Overview

This guide sets up the `autoresearch-nemotron` project on Google Cloud with GPU support for training Nemotron-3-Nano-30B with LoRA.

## Prerequisites

- Google Cloud account with $240 credit
- Google Cloud SDK installed (`gcloud` CLI)
- Project created in Google Cloud Console

## Step 1: Initial Setup

### Install Google Cloud SDK

```bash
# Download and install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize
gcloud init

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

## Step 2: Create GPU VM Instance

### Recommended Configuration

| Setting | Value |
|---------|-------|
| **Machine Type** | `g2-standard-8` (1 x NVIDIA L4) or `g2-standard-12` (1 x NVIDIA RTX 6000) |
| **Boot Disk** | Ubuntu 22.04 LTS, 100GB SSD |
| **GPU** | NVIDIA L4 (24GB) or RTX 6000 (48GB) |
| **External IP** | Ephemeral |

### Create Instance Command

```bash
# For L4 GPU (24GB VRAM - good for LoRA)
gcloud compute instances create autoresearch-nemotron \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform
```

Or for RTX 6000 (48GB VRAM - better for larger batches):

```bash
gcloud compute instances create autoresearch-nemotron-rtx \
    --zone=us-central1-a \
    --machine-type=g2-standard-12 \
    --accelerator=type=nvidia-rtx-6000,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform
```

## Step 3: Connect to VM

```bash
# Get external IP
gcloud compute instances describe autoresearch-nemotron \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].networkIP)'

# SSH into VM
gcloud compute ssh autoresearch-nemotron \
    --zone=us-central1-a
```

## Step 4: Install Dependencies

Run the setup script on the VM:

```bash
# Clone your repo (or copy files)
git clone https://github.com/YOUR_USERNAME/autoresearch-nemotron.git
cd autoresearch-nemotron

# Run setup
chmod +x setup_gcp.sh
./setup_gcp.sh
```

## Step 5: Start Training

```bash
cd autoresearch-nemotron

# Screen session for persistent training
screen -S training

# Prepare data
python prepare.py

# Start training (5 min experiment)
python train.py

# Detach screen: Ctrl+A, D
```

## Cost Estimation

| Resource | Usage | Estimated Cost |
|----------|-------|----------------|
| G2-standard-8 (L4) | 10 hours/week | ~$2.10/hour |
| Storage | 100GB | ~$10/month |
| Egress | Minimal | ~$0 |

With $240 credit:
- ~100 hours of L4 GPU training
- Or ~50 hours of RTX 6000 training

## Monitoring

### Check GPU Usage
```bash
nvidia-smi
```

### Monitor Training
```bash
# View training log
tail -f train.log

# Check results
cat results.tsv
```

### Sync Results Locally
```bash
# Download results
gcloud compute scp autoresearch-nemotron:~/autoresearch-nemotron/results.tsv ./results.tsv \
    --zone=us-central1-a
```

## Instance Management

```bash
# Stop instance (save money)
gcloud compute instances stop autoresearch-nemotron --zone=us-central1-a

# Start instance
gcloud compute instances start autoresearch-nemotron --zone=us-central1-a

# Delete instance (when done)
gcloud compute instances delete autoresearch-nemotron --zone=us-central1-a
```

## Alternative: Use Cloud Shell

For quick experiments, use Cloud Shell (free):

1. Go to [shell.cloud.google.com](https://shell.cloud.google.com)
2. Clone repo and run setup
3. Note: Cloud Shell has no GPU, but good for:
   - Data preparation
   - Code development
   - Small-scale testing

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
nvcc --version

# Reload nvidia module
sudo modprobe nvidia
nvidia-smi
```

### Out of Memory
```bash
# Reduce batch size in config.py
DEVICE_BATCH_SIZE = 1  # Reduce from default
GRADIENT_ACCUMULATION_STEPS = 16  # Increase
```

### Training Too Slow
```bash
# Use mixed precision
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch train.py
```
