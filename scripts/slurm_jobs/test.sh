#!/usr/bin/env bash

# Job name is now set via sbatch --job-name in submit_test.sh
#SBATCH --output=slurm/outputs/%x_%J.txt # Use job name (%x) and job ID (%J) for output
#SBATCH --error=slurm/errors/%x_%J.err  # Use job name (%x) and job ID (%J) for error
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
# Exit on error and catch errors in pipelines.
set -e
set -o pipefail

################################################################################
# 1) Establish project root, source environment, set up directories
################################################################################

# Use the SLURM submit directory as the project root.
PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR")

# Source shared configuration FIRST (script will fail if not found).
if [ -f "$PROJECT_ROOT/config.env" ]; then
    source "$PROJECT_ROOT/config.env"
elif [ -f "$PROJECT_ROOT/slurm/run_config.env" ]; then
    source "$PROJECT_ROOT/slurm/run_config.env"
else
    echo "Error: No configuration file found (config.env or slurm/run_config.env)"
    exit 1
fi

# Output/Error/Log directories should be created by the submission script (submit_test.sh)

# Compute the tasks directory based on the project root (if needed).
# TASKS_DIR=$(realpath "$PROJECT_ROOT/tasks") # This wasn't used, removing

# Load any environment variables from .env
set -o allexport
source "$PROJECT_ROOT/.env"
set +o allexport

MODEL="dummy"

################################################################################
# 2) Get task details from environment variables (passed from submit_test.sh)
################################################################################

# Check if required environment variables are set
if [ -z "$TASK_NAME" ]; then
    echo "Error: TASK_NAME environment variable is not set. This script expects TASK_NAME to be provided via sbatch --export."
    exit 1
fi
if [ -z "$TASK_N" ]; then
    echo "Error: TASK_N environment variable is not set."
    exit 1
fi
if [ -z "$TASK_DATASET_SIZE" ]; then
    echo "Error: TASK_DATASET_SIZE environment variable is not set."
    exit 1
fi
if [ -z "$TASK_TARGET_TIME_MS" ]; then
    echo "Error: TASK_TARGET_TIME_MS environment variable is not set."
    exit 1
fi
if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH environment variable is not set."
    exit 1
fi
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Provided DATASET_PATH directory does not exist: $DATASET_PATH"
    exit 1
fi

echo "Running Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Processing Task: $TASK_NAME"
echo "Parameters: N=$TASK_N, DatasetSize=$TASK_DATASET_SIZE, TargetTime=$TASK_TARGET_TIME_MS"
echo "Dataset Path: $DATASET_PATH"

################################################################################
# 3) Execute the task inside Singularity
################################################################################

echo "Running test simulation for task ${TASK_NAME}..."

# Debugging before singularity exec
echo "[DEBUG] Host TEMP_DIR_STORAGE: ${TEMP_DIR_STORAGE}"
echo "[DEBUG] Host DATA_DIR: ${DATA_DIR}"
echo "[DEBUG] Host SINGULARITY_IMAGE: ${SINGULARITY_IMAGE}"

# Verify host paths before binding
echo "[DEBUG] Checking host paths before singularity exec:"
ls -ld "${PROJECT_ROOT}"
ls -ld "${DATA_DIR}" # Main data dir
ls -ld "${DATASET_PATH}" # Specific task data dir (should exist)
ls -ld "${TEMP_DIR_STORAGE}" || echo "[WARN] Host TEMP_DIR_STORAGE does not exist or is not accessible."

# Ensure TEMP_DIR_STORAGE exists (important for the bind mount source)
echo "[DEBUG] Ensuring TEMP_DIR_STORAGE exists: ${TEMP_DIR_STORAGE}"
mkdir -p "${TEMP_DIR_STORAGE}"

# Create a unique temporary directory for this task run on the host
# Use Slurm Job ID and Task Name for uniqueness
if [ -n "$SLURM_JOB_ID" ]; then
    # Sanitize TASK_NAME for directory usage (replace potential slashes, etc.)
    SANITIZED_TASK_NAME=$(echo "$TASK_NAME" | tr '/' '_') 
    HOST_TEMP_CODE_DIR=$(mktemp -d "${TEMP_DIR_STORAGE}/test_${SLURM_JOB_ID}_${SANITIZED_TASK_NAME}_XXXXXX")
else
    SANITIZED_TASK_NAME=$(echo "$TASK_NAME" | tr '/' '_')
    HOST_TEMP_CODE_DIR=$(mktemp -d "${TEMP_DIR_STORAGE}/test_pid_${$}_${SANITIZED_TASK_NAME}_XXXXXX")
fi

# Define a cleanup function to remove the temporary directory on exit
cleanup() {
    echo "Cleaning up temporary directory: $HOST_TEMP_CODE_DIR"
    rm -rf "$HOST_TEMP_CODE_DIR"
}
trap cleanup EXIT # Register the cleanup function to run on script exit (normal or error)

echo "[DEBUG] Created host temporary directory: $HOST_TEMP_CODE_DIR"
# Since TEMP_DIR_STORAGE is bind-mounted to itself, the path inside container is the same
CONTAINER_CODE_DIR="$HOST_TEMP_CODE_DIR"

echo "[DEBUG] Value of AGENT_MODE before singularity exec: >>>${AGENT_MODE}<<<"
echo "[DEBUG] Attempting singularity exec with direct python command..."

echo "[DEBUG] Checking SINGULARITY_IMAGE variable: >>>${SINGULARITY_IMAGE}<<<"
ls -l "${SINGULARITY_IMAGE}"

# Execute the python test script directly inside singularity (single line)
# Prepend /app to PYTHONPATH to prioritize bind-mounted code
# Pass all relevant task parameters via environment variables
singularity exec \
    --pwd /app \
    --bind "${PROJECT_ROOT}:/app" \
    --bind "${TEMP_DIR_STORAGE}:${TEMP_DIR_STORAGE}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${PROJECT_ROOT}/logs:/app/logs" \
    --env PYTHONPATH="/app:${PYTHONPATH}" \
    --env DATA_DIR="${DATA_DIR}" \
    --env TEMP_DIR_STORAGE="${TEMP_DIR_STORAGE}" \
    --env TASK_NAME="${TASK_NAME}" \
    --env MODEL="${MODEL}" \
    --env TASK_N="${TASK_N}" \
    --env TASK_DATASET_SIZE="${TASK_DATASET_SIZE}" \
    --env TASK_TARGET_TIME_MS="${TASK_TARGET_TIME_MS}" \
    --env DATASET_PATH="${DATASET_PATH}" \
    --env WORKSPACE="/" \
    --env CODE_DIR="${CONTAINER_CODE_DIR}" \
    --env AGENT_MODE="${AGENT_MODE}" \
    "${SINGULARITY_IMAGE}" \
    python3 /app/AlgoTuner/tests/run_tests.py --model "${MODEL}" --task "${TASK_NAME}"

# Update final message
echo "Task ${TASK_NAME} (Job ID: ${SLURM_JOB_ID}) completed."

# Removed the loop that created scripts
# Removed the sbatch submission within the loop
# Removed the waiting loop (Slurm handles waiting for array tasks)
# Removed the cleanup loop for run_task_*.sh scripts