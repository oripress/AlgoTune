#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" 2>/dev/null || true

AWS_REGION="${AWS_REGION:-us-east-1}"

echo "========================================="
echo "Shutting Down ALL AWS Batch Resources"
echo "========================================="
echo ""

# Get all compute environments
CE_NAMES=$(aws batch describe-compute-environments \
  --region "$AWS_REGION" \
  --query 'computeEnvironments[*].computeEnvironmentName' \
  --output text)

if [ -z "$CE_NAMES" ]; then
  echo "No compute environments found"
  exit 0
fi

echo "→ Found compute environments: $CE_NAMES"
echo ""

# Scale down and disable each compute environment
for CE_NAME in $CE_NAMES; do
  echo "→ Processing: $CE_NAME"

  # Get current state
  CURRENT_STATE=$(aws batch describe-compute-environments \
    --compute-environments "$CE_NAME" \
    --region "$AWS_REGION" \
    --query 'computeEnvironments[0].state' \
    --output text 2>/dev/null || echo "UNKNOWN")

  echo "  Current state: $CURRENT_STATE"

  # Enable if disabled (needed to modify)
  if [ "$CURRENT_STATE" = "DISABLED" ]; then
    echo "  → Enabling to allow modifications..."
    aws batch update-compute-environment \
      --compute-environment "$CE_NAME" \
      --state ENABLED \
      --region "$AWS_REGION" 2>&1 || true
    sleep 2
  fi

  # Scale to 0
  echo "  → Scaling to 0 vCPUs..."
  aws batch update-compute-environment \
    --compute-environment "$CE_NAME" \
    --compute-resources '{"minvCpus":0,"desiredvCpus":0}' \
    --region "$AWS_REGION" 2>&1 || true

  sleep 1

  # Disable
  echo "  → Disabling..."
  aws batch update-compute-environment \
    --compute-environment "$CE_NAME" \
    --state DISABLED \
    --region "$AWS_REGION" 2>&1 || true

  echo "  ✓ Done"
  echo ""
done

echo "→ Waiting 5 seconds for changes to propagate..."
sleep 5

# Get current instance ID
CURRENT_INSTANCE=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
if [ -n "$CURRENT_INSTANCE" ]; then
  echo "→ Protecting current instance: $CURRENT_INSTANCE"
else
  echo "→ Not running on EC2"
fi

# Find and terminate all Batch instances
echo ""
echo "→ Finding all Batch EC2 instances..."
ALL_INSTANCE_IDS=$(aws ec2 describe-instances \
  --filters "Name=tag-key,Values=aws:batch:compute-environment" \
            "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[*].Instances[*].InstanceId' \
  --output text \
  --region "$AWS_REGION" 2>/dev/null || echo "")

if [ -n "$ALL_INSTANCE_IDS" ]; then
  echo "  Found instances: $ALL_INSTANCE_IDS"

  # Filter out current instance
  INSTANCE_IDS=""
  for instance in $ALL_INSTANCE_IDS; do
    if [ "$instance" != "$CURRENT_INSTANCE" ]; then
      INSTANCE_IDS="$INSTANCE_IDS $instance"
    else
      echo "  Skipping current instance: $instance"
    fi
  done
  INSTANCE_IDS=$(echo "$INSTANCE_IDS" | xargs)

  if [ -n "$INSTANCE_IDS" ]; then
    echo "  → Terminating: $INSTANCE_IDS"
    aws ec2 terminate-instances \
      --instance-ids $INSTANCE_IDS \
      --region "$AWS_REGION" 2>&1
    echo "  ✓ Instances terminated"
  else
    echo "  ✓ No instances to terminate (only current instance)"
  fi
else
  echo "  ✓ No Batch instances found"
fi

echo ""
echo "========================================="
echo "✓ All AWS Batch resources shut down"
echo "========================================="
