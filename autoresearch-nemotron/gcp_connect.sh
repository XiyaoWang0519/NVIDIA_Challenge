#!/bin/bash
#===============================================================================
# GCP Connect Script
#
# Starts the Google Cloud VM and opens an SSH session.
# Run this locally to quickly connect to your training VM.
#
# Usage: ./gcp_connect.sh [instance_name] [zone]
#===============================================================================

INSTANCE_NAME=${1:-autoresearch-nemotron}
ZONE=${2:-us-central1-a}

echo "Checking instance status..."

# Check if instance exists
if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &>/dev/null; then
    echo "Instance '$INSTANCE_NAME' not found in zone '$ZONE'"
    echo "Creating new instance..."
    
    # Create new instance with L4 GPU
    gcloud compute instances create "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --machine-type=g2-standard-8 \
        --accelerator=type=nvidia-l4,count=1 \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-ssd \
        --maintenance-policy=TERMINATE \
        --scopes=cloud-platform
    
    echo "Instance created. Waiting for it to start..."
    sleep 30
fi

# Check if instance is running
STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(status)')

if [ "$STATUS" != "RUNNING" ]; then
    echo "Instance is $STATUS. Starting..."
    gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE"
    echo "Waiting for instance to start..."
    sleep 30
else
    echo "Instance is already running."
fi

# Get external IP
IP=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo ""
echo "Instance: $INSTANCE_NAME"
echo "External IP: $IP"
echo ""

# SSH into instance
echo "Connecting to VM..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE"
