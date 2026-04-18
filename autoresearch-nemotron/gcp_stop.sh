#!/bin/bash
#===============================================================================
# GCP Stop Script
#
# Stops the Google Cloud VM to save costs when not training.
# Run this locally when done for the day.
#
# Usage: ./gcp_stop.sh [instance_name] [zone]
#===============================================================================

INSTANCE_NAME=${1:-autoresearch-nemotron}
ZONE=${2:-us-central1-a}

echo "Stopping instance $INSTANCE_NAME..."

# Check if instance is running
STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(status)' 2>/dev/null)

if [ "$STATUS" == "RUNNING" ]; then
    gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE"
    echo "Instance stopped."
else
    echo "Instance is not running (current status: $STATUS)"
fi

# Show cost estimate
echo ""
echo "VM stopped. You won't be charged for compute while stopped."
echo "Data on the boot disk will persist (~$5/month for 100GB)."
echo ""
echo "To restart: ./gcp_connect.sh $INSTANCE_NAME $ZONE"
