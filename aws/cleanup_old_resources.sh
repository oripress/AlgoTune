#!/usr/bin/env bash
# Run this later to clean up old orphaned resources once they finish deleting
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

echo "Cleaning up old orphaned resources..."

# Delete old queue if it's still around
if aws batch describe-job-queues --job-queues AlgoTuneQueue --region $AWS_REGION 2>/dev/null | grep -q AlgoTuneQueue; then
  echo "→ Deleting old queue AlgoTuneQueue..."
  aws batch update-job-queue --job-queue AlgoTuneQueue --state DISABLED --compute-environment-order '[]' --region $AWS_REGION 2>&1 || true
  sleep 5
  aws batch delete-job-queue --job-queue AlgoTuneQueue --region $AWS_REGION 2>&1 || true
fi

# Delete old compute environments
for ce in AlgoTuneCE-v2 AlgoTuneCE-r6a AlgoTuneCE-r6a-multi; do
  if aws batch describe-compute-environments --compute-environments $ce --region $AWS_REGION 2>/dev/null | grep -q $ce; then
    STATE=$(aws batch describe-compute-environments --compute-environments $ce --region $AWS_REGION --query 'computeEnvironments[0].state' --output text 2>/dev/null || echo "UNKNOWN")
    if [ "$STATE" != "DISABLED" ]; then
      echo "→ Skipping delete (state=$STATE): $ce"
      continue
    fi
    echo "→ Deleting old CE: $ce..."
    aws batch delete-compute-environment --compute-environment $ce --region $AWS_REGION 2>&1 || true
  fi
done

echo "✓ Cleanup complete!"
