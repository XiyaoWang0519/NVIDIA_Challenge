#!/bin/bash
#===============================================================================
# Monitor Training Script
#
# Monitors training progress on Google Cloud VM from local machine.
# Run this locally, not on the VM.
#
# Usage: ./monitor.sh [instance_name] [zone] [interval]
#   defaults: instance_name=autoresearch-nemotron, zone=us-central1-a, interval=30
#===============================================================================

INSTANCE_NAME=${1:-autoresearch-nemotron}
ZONE=${2:-us-central1-a}
INTERVAL=${3:-30}

echo "Monitoring $INSTANCE_NAME (refreshing every ${INTERVAL}s)"
echo "Press Ctrl+C to stop"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to get and display status
show_status() {
    # Get GPU stats
    GPU_STATS=$(gcloud compute ssh "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --command="nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>/dev/null || echo "N/A,N/A,N/A,N/A")
    
    # Get training status
    TRAINING_STATUS=$(gcloud compute ssh "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --command="ps aux | grep 'python train.py' | grep -v grep || echo 'Not running'" 2>/dev/null)
    
    # Get latest log line
    LAST_LOG=$(gcloud compute ssh "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --command="tail -1 ~/projects/autoresearch-nemotron/train.log 2>/dev/null || echo 'No log'" 2>/dev/null)
    
    # Get latest results
    LAST_RESULT=$(gcloud compute ssh "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --command="tail -1 ~/projects/autoresearch-nemotron/results.tsv 2>/dev/null || echo 'No results'" 2>/dev/null)
    
    # Clear line and print
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC}"
    echo -e "GPU: $GPU_STATS"
    echo -e "Status: $TRAINING_STATUS"
    echo -e "Last log: $LAST_LOG"
    echo -e "Last result: $LAST_RESULT"
    echo "---"
}

# Initial status
show_status

# Continuous monitoring
while true; do
    sleep "$INTERVAL"
    show_status
done
