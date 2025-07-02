#!/usr/bin/env bash

#SBATCH --time=00:05:00 # Short job
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=tests/logs/%x_%A_%a.out # Use %a for consistency, though it's not an array job
#SBATCH --error=tests/logs/%x_%A_%a.err

set -euo pipefail
set -x

# --- Environment Setup ---
PROJECT_ROOT=$(realpath "${SLURM_SUBMIT_DIR:-.}") 

echo "$(date): Starting summary update script for $SLURM_JOB_NAME (Task: $1, Dataset Dir: $2)" >&2
echo "$(date): Job ID: $SLURM_JOB_ID, Submitted from: $SLURM_SUBMIT_DIR" >&2

if [ "$#" -ne 2 ]; then 
  echo "Usage: $0 <task_name> <path_to_pregenerated_dataset>" >&2
  exit 1
fi

TASK_NAME="$1"
PREGENERATED_DATA_DIR="$2" 

# Find the actual train file instead of assuming train.jsonl
# Use ls + head to get one matching file; assumes only one train file per dir
actual_train_file=$(ls -1 "$PREGENERATED_DATA_DIR"/*_train.jsonl 2>/dev/null | head -n 1)

# --- Sanity Checks ---
if [ ! -d "$PREGENERATED_DATA_DIR" ]; then
    echo "$(date): ERROR: Dataset directory '$PREGENERATED_DATA_DIR' not found!" >&2
    # Exit with error because the dependency should ensure it exists.
    # If it doesn't, something went wrong upstream.
    exit 1 
fi

echo "$(date): Checking contents of $PREGENERATED_DATA_DIR ..." >&2
ls -la "$PREGENERATED_DATA_DIR" >&2 || echo "$(date): Failed to list directory $PREGENERATED_DATA_DIR (rc=$?)" >&2

if [ -z "$actual_train_file" ] || [ ! -f "$actual_train_file" ]; then
    echo "$(date): ERROR: Could not find a unique training file (*_train.jsonl) in '$PREGENERATED_DATA_DIR'!" >&2
    # Exit with error, as the generation job this depends on should have created it.
    exit 1
fi

echo "$(date): Found training file: $actual_train_file" >&2

# --- Get Dataset Parameter 'n' (k value) from Filename --- 
# Use grep/sed to extract the number after _n and before _size
dataset_n=$(basename "$actual_train_file" | grep -oP '_n\K\d+(?=_size)')

echo "$(date): DEBUG - Parsed dataset_n = [$dataset_n] from $actual_train_file" >&2

if ! [[ "$dataset_n" =~ ^[0-9]+$ ]]; then
    echo "$(date): ERROR: Could not parse 'n' (k value) from filename '$actual_train_file'!" >&2
    exit 1
fi
echo "$(date): Parsed dataset parameter n (k value) = $dataset_n for task $TASK_NAME" >&2

# --- Update Summary Report using flock and jq --- 
echo "$(date): Updating summary report with n=$dataset_n..." >&2
REPORTS_DIR="$PROJECT_ROOT/tests/reports"
SUMMARY_FILE="$REPORTS_DIR/summary.json"
LOCK_FILE="$REPORTS_DIR/summary.lock" # Use the same lock file as agent.sh

# Ensure reports directory exists (though it should by now)
mkdir -p "$REPORTS_DIR" 

# Use flock to safely update the JSON summary
(
  flock -x 200 # Acquire exclusive lock
  # Atomically update summary.json with dataset size 'n' for the task
  jq --arg task "$TASK_NAME" --argjson n "$dataset_n" '.[$task] = (.[$task] // {}) | .[$task]["n"] = $n' "$SUMMARY_FILE" > "${SUMMARY_FILE}.tmp" && mv "${SUMMARY_FILE}.tmp" "$SUMMARY_FILE"
  echo "$(date): Summary file update complete for task $TASK_NAME (set n=$dataset_n)." >&2
) 200>"$LOCK_FILE" # Associate file descriptor 200 with the lock file

echo "$(date): Summary update script finished successfully." >&2
exit 0 