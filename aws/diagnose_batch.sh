#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" 2>/dev/null || true

AWS_REGION="${AWS_REGION:-us-east-1}"

echo "========================================="
echo "AWS Batch Diagnostics"
echo "========================================="
echo ""

# Check compute environments
echo "→ Compute Environments:"
aws batch describe-compute-environments \
  --region "$AWS_REGION" \
  --query 'computeEnvironments[*].[computeEnvironmentName,state,status]' \
  --output table

echo ""
echo "→ Detailed Compute Environment Info:"
aws batch describe-compute-environments \
  --region "$AWS_REGION" \
  --query 'computeEnvironments[*].{Name:computeEnvironmentName,State:state,Status:status,MinvCPUs:computeResources.minvCpus,DesiredvCPUs:computeResources.desiredvCpus,MaxvCPUs:computeResources.maxvCpus}' \
  --output table

echo ""
echo "→ Job Queues:"
aws batch describe-job-queues \
  --region "$AWS_REGION" \
  --query 'jobQueues[*].[jobQueueName,state,status]' \
  --output table

echo ""
echo "→ Jobs by Status:"
for status in SUBMITTED PENDING RUNNABLE STARTING RUNNING SUCCEEDED FAILED; do
  count=$(aws batch list-jobs \
    --job-queue "${BATCH_JOB_QUEUE_NAME:-AlgoTuneJobQueue}" \
    --job-status "$status" \
    --region "$AWS_REGION" \
    --query 'length(jobSummaryList)' \
    --output text 2>/dev/null || echo "0")
  echo "  $status: $count"
done

echo ""
echo "→ Active Jobs (SUBMITTED/PENDING/RUNNABLE/STARTING/RUNNING):"
for status in SUBMITTED PENDING RUNNABLE STARTING RUNNING; do
  aws batch list-jobs \
    --job-queue "${BATCH_JOB_QUEUE_NAME:-AlgoTuneJobQueue}" \
    --job-status "$status" \
    --region "$AWS_REGION" \
    --query 'jobSummaryList[*].[jobId,jobName,status]' \
    --output table 2>/dev/null || true
done

echo ""
echo "→ EC2 Instances with Batch tags:"
aws ec2 describe-instances \
  --filters "Name=tag-key,Values=aws:batch:compute-environment" \
            "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,Tags[?Key==`aws:batch:compute-environment`].Value|[0],LaunchTime]' \
  --output table \
  --region "$AWS_REGION" 2>/dev/null || echo "No instances found"

echo ""
echo "→ ALL EC2 Instances (running/pending):"
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType,LaunchTime]' \
  --output table \
  --region "$AWS_REGION" 2>/dev/null || echo "No instances found"

echo ""
echo "========================================="
echo "Diagnostics Complete"
echo "========================================="
