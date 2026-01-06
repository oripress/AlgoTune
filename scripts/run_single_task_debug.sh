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

# Backward compatibility: map old token names to HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
  if [ -n "${HUGGING_FACE_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGING_FACE_TOKEN"
  elif [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
  fi
fi

export S3_RESULTS_BUCKET="${S3_RESULTS_BUCKET:-algotune-results-926341122891}"
export ALGOTUNE_HF_BATCH_FILES="${ALGOTUNE_HF_BATCH_FILES:-25}"

TASK_NAME="${1:-max_common_subgraph}"

python3 "$ROOT_DIR/scripts/publish_datasets_hf.py" \
  --tasks "$TASK_NAME" \
  --s3-bucket "$S3_RESULTS_BUCKET"
