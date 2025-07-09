#!/usr/bin/env bash

#SBATCH --time=00:30:00     # Adjust as needed
#SBATCH --mem=10G           # Reduced to prevent massive datasets
#SBATCH --cpus-per-task=1
#SBATCH --output=tests/logs/generate_%x.out
#SBATCH --error=tests/logs/generate_%x.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Usage check
# -----------------------------------------------------------------------------
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <task_name> <target_time_ms>" >&2
  exit 1
fi

TASK_NAME="$1"
TARGET_TIME_MS="$2"

echo "$(date): Starting dataset generation for Task='${TASK_NAME}', target_time_ms=${TARGET_TIME_MS}" >&2
echo "$(date): Slurm Job ID: $SLURM_JOB_ID" >&2

# --- Load environment from unified config ---
PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR")   # assume submission from repo root
# If we're in scripts/, go up one level to get to the actual project root
if [[ "$PROJECT_ROOT" == */scripts ]]; then
    PROJECT_ROOT=$(dirname "$PROJECT_ROOT")
fi

# Try config.env first, then fall back to slurm/run_config.env
if [ -f "$PROJECT_ROOT/config.env" ]; then
    echo "$(date): Loading environment from config.env..." >&2
    source "$PROJECT_ROOT/config.env"
elif [ -f "$PROJECT_ROOT/slurm/run_config.env" ]; then
    echo "$(date): Loading environment from slurm/run_config.env..." >&2
    source "$PROJECT_ROOT/slurm/run_config.env"
else
    echo "Error: No configuration file found (config.env or slurm/run_config.env)" >&2
    exit 1
fi

if [ -z "${TEMP_DIR_STORAGE:-}" ] || [ -z "${SINGULARITY_IMAGE:-}" ] || [ -z "${DATA_DIR:-}" ]; then
    echo "Error: Missing TEMP_DIR_STORAGE or SINGULARITY_IMAGE or DATA_DIR in configuration" >&2
    exit 1
fi

# --- Prepare dataset directory ---
SHARED_DATASET_DIR_BASE="${DATA_DIR}"
TASK_DATASET_DIR="$SHARED_DATASET_DIR_BASE/$TASK_NAME"
echo "$(date): Target dataset directory: $TASK_DATASET_DIR" >&2

# Clean / recreate
echo "$(date): Cleaning old dataset directory and recreating…" >&2
rm -rf "$TASK_DATASET_DIR"
mkdir -p "$TASK_DATASET_DIR"
echo "$(date): Sleeping 1s to allow FS consistency…" >&2
sleep 1

# --- Run generation inside Singularity ---
GENERATION_START_TIME=$(date +%s%3N)
echo "$(date): Launching singularity for dataset generation…" >&2

set +e   # allow capturing exit code manually
singularity exec \
    --pwd /app \
    --bind "$PROJECT_ROOT:/app" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${TEMP_DIR_STORAGE}:${TEMP_DIR_STORAGE}" \
    --env PYTHONPATH="/app:${PYTHONPATH:-}" \
    --env CODE_DIR="/app" \
    --env TEMP_DIR_STORAGE="${TEMP_DIR_STORAGE}" \
    --env TARGET_TIME_MS="${TARGET_TIME_MS}" \
    "${SINGULARITY_IMAGE}" \
    bash -c "\
      set -e; \
      if [ -n \"${OVERRIDE_K:-}\" ]; then \
        python3 AlgoTuner/scripts/generate_and_annotate.py \"$TASK_NAME\" --data-dir \"$TASK_DATASET_DIR\" --k \"${OVERRIDE_K:-}\"; \
      else \
        python3 AlgoTuner/scripts/generate_and_annotate.py \"$TASK_NAME\" --data-dir \"$TASK_DATASET_DIR\"; \
      fi; \
    "
PY_EXIT=$?
set -e

GENERATION_END_TIME=$(date +%s%3N)
GENERATION_DURATION_MS=$((GENERATION_END_TIME - GENERATION_START_TIME))

if [ $PY_EXIT -ne 0 ]; then
  echo "Error: In‐container dataset generation FAILED for '$TASK_NAME' (exit code $PY_EXIT)." >&2
  exit 1
fi

echo "$(date): Dataset generation succeeded for '$TASK_NAME' in ${GENERATION_DURATION_MS} ms." >&2

# --- In‐container file‐presence check (to confirm a *.jsonl exists) ---
echo "$(date): Checking for at least one '${TASK_NAME}_*_train.jsonl' in $TASK_DATASET_DIR …" >&2
sleep 2   # give filesystem a moment
if find "$TASK_DATASET_DIR" -maxdepth 1 -name "${TASK_NAME}_*_train.jsonl" -print -quit | grep -q .; then
    echo "$(date): SUCCESS – found at least one training file." >&2
    exit 0
else
    echo "$(date): FAILURE – no '${TASK_NAME}_*_train.jsonl' found!" >&2
    echo "$(date): Listing contents of $TASK_DATASET_DIR for debugging:" >&2
    ls -la "$TASK_DATASET_DIR" >&2
    exit 1
fi