#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

AWS_REGION="${AWS_REGION:-us-east-1}"
CE_NAME="${BATCH_COMPUTE_ENV_NAME:-AlgoTuneCE}"
QUEUE_NAME="${BATCH_JOB_QUEUE_NAME:-AlgoTuneQueue-v2}"

echo "========================================="
echo "Updating disk size to ${BATCH_ROOT_VOLUME_GB}GB"
echo "========================================="
echo ""

# Step 1: Disable the job queue
echo "→ Step 1: Disabling job queue..."
aws batch update-job-queue \
  --job-queue "$QUEUE_NAME" \
  --state DISABLED \
  --region "$AWS_REGION" 2>/dev/null || true

echo "  Waiting for queue to be disabled..."
for i in {1..20}; do
  STATE=$(aws batch describe-job-queues \
    --job-queues "$QUEUE_NAME" \
    --region "$AWS_REGION" \
    --query 'jobQueues[0].state' \
    --output text 2>/dev/null || echo "UNKNOWN")
  
  STATUS=$(aws batch describe-job-queues \
    --job-queues "$QUEUE_NAME" \
    --region "$AWS_REGION" \
    --query 'jobQueues[0].status' \
    --output text 2>/dev/null || echo "UNKNOWN")

  echo "  State: $STATE, Status: $STATUS"
  
  if [ "$STATE" = "DISABLED" ] && [ "$STATUS" = "VALID" ]; then
    echo "  ✓ Job queue disabled"
    break
  fi
  
  sleep 5
done

# Step 2: Disable the CE
echo ""
echo "→ Step 2: Disabling compute environment..."
aws batch update-compute-environment \
  --compute-environment "$CE_NAME" \
  --state DISABLED \
  --region "$AWS_REGION" 2>/dev/null || true

echo "  Waiting for CE to be disabled..."
for i in {1..30}; do
  STATE=$(aws batch describe-compute-environments \
    --compute-environments "$CE_NAME" \
    --region "$AWS_REGION" \
    --query 'computeEnvironments[0].state' \
    --output text 2>/dev/null || echo "UNKNOWN")
  
  STATUS=$(aws batch describe-compute-environments \
    --compute-environments "$CE_NAME" \
    --region "$AWS_REGION" \
    --query 'computeEnvironments[0].status' \
    --output text 2>/dev/null || echo "UNKNOWN")

  echo "  State: $STATE, Status: $STATUS"
  
  if [ "$STATE" = "DISABLED" ] && [ "$STATUS" = "VALID" ]; then
    echo "  ✓ Compute environment disabled"
    break
  fi
  
  sleep 10
done

# Step 3: Delete the job queue
echo ""
echo "→ Step 3: Deleting job queue..."
aws batch delete-job-queue \
  --job-queue "$QUEUE_NAME" \
  --region "$AWS_REGION" 2>/dev/null || true

echo "  Waiting for queue deletion..."
for i in {1..20}; do
  STATUS=$(aws batch describe-job-queues \
    --job-queues "$QUEUE_NAME" \
    --region "$AWS_REGION" \
    --query 'jobQueues[0].status' \
    --output text 2>/dev/null || echo "DELETED")

  if [ "$STATUS" = "DELETED" ] || [ "$STATUS" = "None" ]; then
    echo "  ✓ Job queue deleted"
    break
  fi

  echo "  Status: $STATUS (waiting...)"
  sleep 5
done

# Step 4: Delete the compute environment
echo ""
echo "→ Step 4: Deleting compute environment..."
aws batch delete-compute-environment \
  --compute-environment "$CE_NAME" \
  --region "$AWS_REGION"

echo "  Waiting for CE deletion..."
for i in {1..24}; do
  STATUS=$(aws batch describe-compute-environments \
    --compute-environments "$CE_NAME" \
    --region "$AWS_REGION" \
    --query 'computeEnvironments[0].status' \
    --output text 2>/dev/null || echo "DELETED")

  if [ "$STATUS" = "DELETED" ] || [ "$STATUS" = "None" ]; then
    echo "  ✓ Compute environment deleted"
    break
  fi

  echo "  Status: $STATUS (waiting...)"
  sleep 5
done

# Step 5: Recreate everything with new disk size
echo ""
echo "→ Step 5: Creating new CE and queue with ${BATCH_ROOT_VOLUME_GB}GB disk..."
python3 "$SCRIPT_DIR/create_batch_resources.py"

echo ""
echo "========================================="
echo "✓ Disk size updated to ${BATCH_ROOT_VOLUME_GB}GB"
echo "========================================="
