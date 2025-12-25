#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

echo "=================================="
echo "AWS Batch Cleanup & Fix"
echo "=================================="
echo ""

# 1. Get all job queues and detach compute environments
echo "→ Step 1: Detaching compute environments from job queues..."
QUEUES=$(aws batch describe-job-queues --region $AWS_REGION --query 'jobQueues[*].jobQueueName' --output text)
for queue in $QUEUES; do
  echo "  → Disabling and detaching: $queue"
  aws batch update-job-queue \
    --job-queue "$queue" \
    --state DISABLED \
    --compute-environment-order '[]' \
    --region $AWS_REGION 2>&1 || true
done

echo "  → Waiting 10s for queue updates to propagate..."
sleep 10

# 2. Disable and delete all compute environments
echo ""
echo "→ Step 2: Disabling all compute environments..."
CES=$(aws batch describe-compute-environments --region $AWS_REGION --query 'computeEnvironments[*].computeEnvironmentName' --output text)
for ce in $CES; do
  echo "  → Disabling: $ce"
  
  # Enable first if disabled (needed to modify)
  STATE=$(aws batch describe-compute-environments --compute-environments "$ce" --region $AWS_REGION --query 'computeEnvironments[0].state' --output text)
  if [ "$STATE" = "DISABLED" ]; then
    aws batch update-compute-environment --compute-environment "$ce" --state ENABLED --region $AWS_REGION 2>&1 || true
    sleep 2
  fi
  
  # Scale to 0
  aws batch update-compute-environment \
    --compute-environment "$ce" \
    --compute-resources '{"minvCpus":0,"desiredvCpus":0}' \
    --region $AWS_REGION 2>&1 || true
  
  # Disable
  aws batch update-compute-environment \
    --compute-environment "$ce" \
    --state DISABLED \
    --region $AWS_REGION 2>&1 || true
done

echo "  → Waiting 15s for compute environments to disable..."
sleep 15

# 3. Delete all job queues
echo ""
echo "→ Step 3: Deleting all job queues..."
for queue in $QUEUES; do
  echo "  → Deleting: $queue"
  aws batch delete-job-queue --job-queue "$queue" --region $AWS_REGION 2>&1 || true
done

echo "  → Waiting 10s for queue deletion..."
sleep 10

# 4. Delete all compute environments
echo ""
echo "→ Step 4: Deleting all compute environments..."
for ce in $CES; do
  echo "  → Deleting: $ce"
  aws batch delete-compute-environment --compute-environment "$ce" --region $AWS_REGION 2>&1 || true
done

echo "  → Waiting 15s for deletion to complete..."
sleep 15

# 5. Terminate any orphaned EC2 instances
echo ""
echo "→ Step 5: Terminating orphaned EC2 instances..."
CURRENT_INSTANCE=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
BATCH_INSTANCES=$(aws ec2 describe-instances \
  --filters "Name=tag-key,Values=aws:batch:compute-environment" \
            "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[*].Instances[*].InstanceId' \
  --output text \
  --region $AWS_REGION 2>/dev/null || echo "")

if [ -n "$BATCH_INSTANCES" ]; then
  INSTANCES_TO_TERMINATE=""
  for instance in $BATCH_INSTANCES; do
    if [ "$instance" != "$CURRENT_INSTANCE" ]; then
      INSTANCES_TO_TERMINATE="$INSTANCES_TO_TERMINATE $instance"
    fi
  done
  
  if [ -n "$INSTANCES_TO_TERMINATE" ]; then
    echo "  → Terminating: $INSTANCES_TO_TERMINATE"
    aws ec2 terminate-instances --instance-ids $INSTANCES_TO_TERMINATE --region $AWS_REGION || true
  else
    echo "  → No instances to terminate (only current instance)"
  fi
else
  echo "  → No Batch instances found"
fi

echo ""
echo "=================================="
echo "✓ Cleanup complete!"
echo "=================================="
echo ""
echo "Now run: python3 aws/create_batch_resources.py"
echo "This will create fresh resources from your .env configuration"
