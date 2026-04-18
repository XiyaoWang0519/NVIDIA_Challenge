#!/bin/bash
#===============================================================================
# Sync Results Script
#
# Syncs training results from Google Cloud VM to local machine.
# Run this locally, not on the VM.
#
# Usage: ./sync_results.sh [instance_name] [zone]
#   defaults: instance_name=autoresearch-nemotron, zone=us-central1-a
#===============================================================================

INSTANCE_NAME=${1:-autoresearch-nemotron}
ZONE=${2:-us-central1-a}
LOCAL_DIR="/Users/xiyaowang/Documents/Projects/NVIDIA_Challenge/autoresearch-nemotron"

echo "Syncing results from $INSTANCE_NAME ($ZONE) to local..."
echo ""

# Sync results.tsv
echo "Syncing results.tsv..."
gcloud compute scp "$INSTANCE_NAME:~/projects/autoresearch-nemotron/results.tsv" \
    "$LOCAL_DIR/results_remote.tsv" \
    --zone="$ZONE" 2>/dev/null || echo "No results.tsv found"

# Sync training logs
echo "Syncing training logs..."
gcloud compute scp "$INSTANCE_NAME:~/projects/autoresearch-nemotron/train_*.log" \
    "$LOCAL_DIR/" \
    --zone="$ZONE" 2>/dev/null || echo "No logs found"

# Sync checkpoints (if any)
echo "Syncing checkpoints..."
gcloud compute scp --recurse "$INSTANCE_NAME:~/projects/autoresearch-nemotron/checkpoints/" \
    "$LOCAL_DIR/checkpoints_remote/" \
    --zone="$ZONE" 2>/dev/null || echo "No checkpoints found"

echo ""
echo "Sync complete!"
echo "Files saved to: $LOCAL_DIR/"
