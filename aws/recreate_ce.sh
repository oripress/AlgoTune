#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

AWS_REGION="${AWS_REGION:-us-east-1}"
CE_NAME="${BATCH_COMPUTE_ENV_NAME:-AlgoTuneCE}"

echo "========================================="
echo "Recreating Compute Environment: $CE_NAME"
echo "========================================="
echo ""

# Step 1: Delete the old CE (only if already disabled)
STATE=$(aws batch describe-compute-environments \
  --compute-environments "$CE_NAME" \
  --region "$AWS_REGION" \
  --query 'computeEnvironments[0].state' \
  --output text 2>/dev/null || echo "UNKNOWN")
if [ "$STATE" != "DISABLED" ]; then
  echo "→ Cannot delete $CE_NAME because state is $STATE."
  echo "  Disable it manually if you want to recreate it."
  exit 1
fi

echo "→ Deleting $CE_NAME..."
aws batch delete-compute-environment \
  --compute-environment "$CE_NAME" \
  --region "$AWS_REGION" || true

echo "→ Waiting for deletion to complete (this may take 1-2 minutes)..."
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

echo ""
echo "→ Step 2: Creating new $CE_NAME with minvCpus=0..."
python3 "$SCRIPT_DIR/create_batch_resources.py"

echo ""
echo "========================================="
echo "✓ Compute Environment Recreated"
echo "========================================="
echo ""
echo "New settings:"
aws batch describe-compute-environments \
  --compute-environments "$CE_NAME" \
  --region "$AWS_REGION" \
  --query 'computeEnvironments[0].{Name:computeEnvironmentName,State:state,MinvCpus:computeResources.minvCpus,MaxvCpus:computeResources.maxvCpus}' \
  --output table
