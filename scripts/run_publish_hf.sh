#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/aws/.env" ]; then
  set -a
  source "$ROOT_DIR/aws/.env"
  set +a
fi

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

export S3_RESULTS_BUCKET="${S3_RESULTS_BUCKET:-algotune-results-926341122891}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export BATCH_JOB_QUEUE_NAME="${BATCH_JOB_QUEUE_NAME:-AlgoTuneQueue-v2}"
export BATCH_JOB_DEF_NAME="${BATCH_JOB_DEF_NAME:-AlgoTuneJobDef}"
export ALGOTUNE_HF_DATASET="${ALGOTUNE_HF_DATASET:-oripress/AlgoTune}"
export ALGOTUNE_HF_REVISION="${ALGOTUNE_HF_REVISION:-main}"
export DATASET_RETENTION_DAYS="${DATASET_RETENTION_DAYS:-7}"
export BATCH_COMPUTE_ENV_NAME="${BATCH_COMPUTE_ENV_NAME:-AlgoTuneCE}"

echo "Using bucket: $S3_RESULTS_BUCKET"
echo "Using queue: $BATCH_JOB_QUEUE_NAME"
echo "Using job def: $BATCH_JOB_DEF_NAME"
echo "Using HF repo: $ALGOTUNE_HF_DATASET"
echo "Using compute env: $BATCH_COMPUTE_ENV_NAME"

QUEUE_STATUS=$(aws batch describe-job-queues \
  --job-queues "$BATCH_JOB_QUEUE_NAME" \
  --region "$AWS_REGION" \
  --query 'jobQueues[0].status' \
  --output text || true)
QUEUE_STATE=$(aws batch describe-job-queues \
  --job-queues "$BATCH_JOB_QUEUE_NAME" \
  --region "$AWS_REGION" \
  --query 'jobQueues[0].state' \
  --output text || true)

if [ "$QUEUE_STATE" != "ENABLED" ]; then
  echo "Enabling job queue $BATCH_JOB_QUEUE_NAME (state: $QUEUE_STATE)"
  aws batch update-job-queue \
    --job-queue "$BATCH_JOB_QUEUE_NAME" \
    --state ENABLED \
    --region "$AWS_REGION"
fi

for _ in $(seq 1 30); do
  QUEUE_STATUS=$(aws batch describe-job-queues \
    --job-queues "$BATCH_JOB_QUEUE_NAME" \
    --region "$AWS_REGION" \
    --query 'jobQueues[0].status' \
    --output text || true)
  QUEUE_STATE=$(aws batch describe-job-queues \
    --job-queues "$BATCH_JOB_QUEUE_NAME" \
    --region "$AWS_REGION" \
    --query 'jobQueues[0].state' \
    --output text || true)
  if [ "$QUEUE_STATUS" = "VALID" ] && [ "$QUEUE_STATE" = "ENABLED" ]; then
    break
  fi
  echo "Waiting for queue to become ENABLED/VALID (state: $QUEUE_STATE, status: $QUEUE_STATUS)"
  sleep 5
done

if [ "$QUEUE_STATUS" != "VALID" ] || [ "$QUEUE_STATE" != "ENABLED" ]; then
  echo "Queue not ready (state: $QUEUE_STATE, status: $QUEUE_STATUS). Aborting."
  exit 1
fi

CE_STATE=$(aws batch describe-compute-environments \
  --compute-environments "$BATCH_COMPUTE_ENV_NAME" \
  --region "$AWS_REGION" \
  --query 'computeEnvironments[0].state' \
  --output text || true)

if [ "$CE_STATE" != "ENABLED" ]; then
  echo "Enabling compute environment $BATCH_COMPUTE_ENV_NAME (state: $CE_STATE)"
  aws batch update-compute-environment \
    --compute-environment "$BATCH_COMPUTE_ENV_NAME" \
    --state ENABLED \
    --region "$AWS_REGION"
fi

for _ in $(seq 1 30); do
  CE_STATE=$(aws batch describe-compute-environments \
    --compute-environments "$BATCH_COMPUTE_ENV_NAME" \
    --region "$AWS_REGION" \
    --query 'computeEnvironments[0].state' \
    --output text || true)
  CE_STATUS=$(aws batch describe-compute-environments \
    --compute-environments "$BATCH_COMPUTE_ENV_NAME" \
    --region "$AWS_REGION" \
    --query 'computeEnvironments[0].status' \
    --output text || true)
  if [ "$CE_STATE" = "ENABLED" ] && [ "$CE_STATUS" = "VALID" ]; then
    break
  fi
  echo "Waiting for compute env to become ENABLED/VALID (state: $CE_STATE, status: $CE_STATUS)"
  sleep 5
done

if [ "$CE_STATE" != "ENABLED" ] || [ "$CE_STATUS" != "VALID" ]; then
  echo "Compute env not ready (state: $CE_STATE, status: $CE_STATUS). Aborting."
  exit 1
fi

python3 "$ROOT_DIR/scripts/publish_datasets_hf_parallel.py" \
  --all-tasks \
  --s3-bucket "$S3_RESULTS_BUCKET" \
  --shutdown
