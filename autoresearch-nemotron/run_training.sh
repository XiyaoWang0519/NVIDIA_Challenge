#!/bin/bash
#===============================================================================
# Training Launcher Script for Google Cloud VM
#
# Runs the autoresearch training loop on GCP with proper resource monitoring.
# Designed to run in a screen session for persistence.
#
# Usage: ./run_training.sh [options]
#   options:
#     --time TIME      Training time budget in seconds (default: 300 = 5 min)
#     --tag TAG        Experiment tag for results
#     --grpo           Use GRPO training (default: SFT)
#     --sft            Use SFT training
#===============================================================================

set -e

# Configuration
TIME_BUDGET=${TIME_BUDGET:-300}
EXP_TAG=${EXP_TAG:-$(date +%Y%m%d_%H%M%S)}
TRAINING_MODE=${TRAINING_MODE:-sft}  # sft or grpo

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --time)
            TIME_BUDGET="$2"
            shift 2
            ;;
        --tag)
            EXP_TAG="$2"
            shift 2
            ;;
        --grpo)
            TRAINING_MODE="grpo"
            shift
            ;;
        --sft)
            TRAINING_MODE="sft"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}NVIDIA Nemotron Model Reasoning Challenge - Training Launcher${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${YELLOW}Experiment Tag:${NC} $EXP_TAG"
echo -e "${YELLOW}Time Budget:${NC} ${TIME_BUDGET}s"
echo -e "${YELLOW}Training Mode:${NC} $TRAINING_MODE"
echo ""

#-------------------------------------------------------------------------------
# Environment Setup
#-------------------------------------------------------------------------------
cd ~/projects/autoresearch-nemotron

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Run setup_gcp.sh first"
    exit 1
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_DISABLE_PROGRESS_BARS=1

#-------------------------------------------------------------------------------
# Pre-flight Checks
#-------------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Pre-flight checks...${NC}"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo -e "${RED}Warning: nvidia-smi not found${NC}"
fi

# Check disk space
DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')
echo -e "${GREEN}Disk available:${NC} $DISK_AVAIL"

#-------------------------------------------------------------------------------
# Data Preparation
#-------------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[2/5] Preparing data...${NC}"

if [ ! -f "data/train.jsonl" ]; then
    echo "Running data preparation..."
    python prepare.py
else
    echo "Data already prepared, skipping..."
fi

#-------------------------------------------------------------------------------
# Configure Training
#-------------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[3/5] Configuring training...${NC}"

# Update config for GCP
cat >> config.py << EOF

# GCP Training Config (auto-generated)
TIME_BUDGET = $TIME_BUDGET
EXP_TAG = "$EXP_TAG"
EOF

#-------------------------------------------------------------------------------
# Training Run
#-------------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[4/5] Starting training...${NC}"
echo -e "${GREEN}Log file: train_${EXP_TAG}.log${NC}"
echo ""

# Create results file if not exists
touch results.tsv

# Run training with full logging
python train.py 2>&1 | tee train_${EXP_TAG}.log

#-------------------------------------------------------------------------------
# Results Summary
#-------------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[5/5] Training complete!${NC}"

# Extract key metrics
if [ -f "train_${EXP_TAG}.log" ]; then
    echo ""
    echo -e "${GREEN}Results:${NC}"
    grep "^val_accuracy:" train_${EXP_TAG}.log 2>/dev/null || echo "Accuracy not found in log"
    grep "^training_seconds:" train_${EXP_TAG}.log 2>/dev/null || echo "Time not found in log"
    grep "^peak_vram_mb:" train_${EXP_TAG}.log 2>/dev/null || echo "Memory not found in log"
fi

#-------------------------------------------------------------------------------
# Auto-commit Results
#-------------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}Committing results...${NC}"

# Update results.tsv with latest experiment
RESULTS_FILE="results.tsv"
if [ -f "$RESULTS_FILE" ]; then
    echo -e "${GREEN}Current results:${NC}"
    tail -5 "$RESULTS_FILE"
fi

echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Training session complete!${NC}"
echo -e "${BLUE}===============================================================================${NC}"
