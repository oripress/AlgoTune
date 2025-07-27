#!/usr/bin/env bash
#SBATCH --output=slurm/outputs/eval-%A_%a.out # Relative path for output
#SBATCH --error=slurm/errors/eval-%A_%a.err  # Relative path for error
#SBATCH --time=47:59:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

# Source shared configuration (script will fail if not found)
if [ -f config.env ]; then
    source config.env
elif [ -f slurm/run_config.env ]; then
    source slurm/run_config.env
else
    echo "Error: No configuration file found (config.env or slurm/run_config.env)"
    exit 1
fi

# Define PROJECT_ROOT based on the submission directory
PROJECT_ROOT=$(realpath "${SLURM_SUBMIT_DIR:-.}") # Use CWD as fallback if not in SLURM

# Use EVAL_DATA_DIR if provided, otherwise fall back to the default
if [ ! -z "$EVAL_DATA_DIR" ]; then
    export DATA_DIR="$EVAL_DATA_DIR"
    echo "[DEBUG] Using EVAL_DATA_DIR as DATA_DIR: $DATA_DIR"
else
    export DATA_DIR="${PROJECT_ROOT}/../data"
    echo "[DEBUG] Using default DATA_DIR: $DATA_DIR"
fi

# Ensure the logs directory exists on the host for bind mounting
mkdir -p "${PROJECT_ROOT}/logs"

# Ensure the base temp storage directory exists
mkdir -p "${TEMP_DIR_STORAGE}"

# Create a unique temporary directory for this task run on the host
# Use Slurm Job ID and Task ID for uniqueness if available, otherwise use process ID
if [ -n "$SLURM_JOB_ID" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    HOST_TEMP_CODE_DIR=$(mktemp -d "${TEMP_DIR_STORAGE}/eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXXXX")
else
    HOST_TEMP_CODE_DIR=$(mktemp -d "${TEMP_DIR_STORAGE}/eval_pid_${$}_XXXXXX")
fi

# Define a cleanup function to remove the temporary directory on exit
cleanup() {
    echo "Cleaning up temporary directory: $HOST_TEMP_CODE_DIR"
    rm -rf "$HOST_TEMP_CODE_DIR"
}
trap cleanup EXIT # Register the cleanup function to run on script exit (normal or error)

echo "[DEBUG] Created host temporary directory: $HOST_TEMP_CODE_DIR"

# Determine the path of the temporary directory inside the container
# Since TEMP_DIR_STORAGE is bind-mounted to itself, the path is the same
CONTAINER_CODE_DIR="$HOST_TEMP_CODE_DIR"

# Source the main .env file from the project root to load API keys etc.
set -o allexport
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "[DEBUG] Sourcing environment variables from ${PROJECT_ROOT}/.env"
    source "${PROJECT_ROOT}/.env"
else
    echo "[WARN] ${PROJECT_ROOT}/.env file not found."
fi
set +o allexport

# Read task and model from array task file
if [ -z "$EVAL_TASK_FILE" ]; then
    echo "Error: EVAL_TASK_FILE environment variable is not set."
    exit 1
fi

if [ ! -f "$EVAL_TASK_FILE" ]; then
    echo "Error: Task file not found: $EVAL_TASK_FILE"
    exit 1
fi

# Get the task and model for this array index
TASK_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EVAL_TASK_FILE")
if [ -z "$TASK_LINE" ]; then
    echo "Error: No task found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse model and task from tab-separated line
EVAL_MODEL_NAME=$(echo "$TASK_LINE" | cut -f1)
EVAL_TASK_NAME=$(echo "$TASK_LINE" | cut -f2)

if [ -z "$EVAL_TASK_NAME" ] || [ -z "$EVAL_MODEL_NAME" ]; then
    echo "Error: Could not parse task line: $TASK_LINE"
    exit 1
fi

# Construct a job identifier string (using Slurm Job ID if available)
if [ -n "$SLURM_JOB_ID" ]; then
    JOB_IDENTIFIER="Job $SLURM_JOB_ID Array $SLURM_ARRAY_TASK_ID"
else
    JOB_IDENTIFIER="Process $$"
fi

echo "Evaluating task $EVAL_TASK_NAME for model $EVAL_MODEL_NAME ($JOB_IDENTIFIER)"

# Debugging host environment variables
echo "[DEBUG] Host TEMP_DIR_STORAGE: ${TEMP_DIR_STORAGE}"
echo "[DEBUG] Host DATA_DIR: ${DATA_DIR}"
echo "[DEBUG] Host SINGULARITY_IMAGE: ${SINGULARITY_IMAGE}"
echo "[DEBUG] Host PROJECT_ROOT: ${PROJECT_ROOT}"
echo "[DEBUG] Host TEMP_CODE_DIR (for cleanup): $HOST_TEMP_CODE_DIR"
echo "[DEBUG] Container CODE_DIR: $CONTAINER_CODE_DIR"

# Pass necessary environment variables to the singularity container
env_vars=()
# Prefer per-task DATASET_PATH if set, otherwise fall back to DATA_DIR
if [ ! -z "$DATASET_PATH" ]; then
    env_vars+=("--env" "DATA_DIR=${DATASET_PATH}")
elif [ ! -z "$DATA_DIR" ]; then
    env_vars+=("--env" "DATA_DIR=${DATA_DIR}")
fi
[ ! -z "$TEMP_DIR_STORAGE" ] && env_vars+=("--env" "TEMP_DIR_STORAGE=${TEMP_DIR_STORAGE}")
# Pass CODE_DIR pointing to the *temporary* directory path inside the container
env_vars+=("--env" "CODE_DIR=${CONTAINER_CODE_DIR}")
# Pass evaluation-specific variables
[ ! -z "$EVAL_TASK_NAME" ] && env_vars+=("--env" "EVAL_TASK_NAME=${EVAL_TASK_NAME}")
[ ! -z "$EVAL_MODEL_NAME" ] && env_vars+=("--env" "EVAL_MODEL_NAME=${EVAL_MODEL_NAME}")
[ ! -z "$EVAL_RESULTS_DIR" ] && env_vars+=("--env" "EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR}")
[ ! -z "$EVAL_GENERATION_FILE" ] && env_vars+=("--env" "EVAL_GENERATION_FILE=${EVAL_GENERATION_FILE}")
[ ! -z "$EVAL_OUTPUT_FILE" ] && env_vars+=("--env" "EVAL_OUTPUT_FILE=${EVAL_OUTPUT_FILE}")
[ ! -z "$EVAL_NUM_RUNS" ] && env_vars+=("--env" "EVAL_NUM_RUNS=${EVAL_NUM_RUNS}")
# Pass config file path for the container
env_vars+=("--env" "ALGOTUNE_CONFIG_PATH=/app/AlgoTuner/config/config.yaml")

# Define bind mounts
bind_mounts=()
bind_mounts+=("--bind" "${PROJECT_ROOT}:/app") # Mount project root to /app
bind_mounts+=("--bind" "${PROJECT_ROOT}/logs:/app/logs")
# Mount the base temporary storage directory. The specific task's temp dir is inside this.
[ ! -z "$TEMP_DIR_STORAGE" ] && bind_mounts+=("--bind" "${TEMP_DIR_STORAGE}:${TEMP_DIR_STORAGE}")
# Mount the task-specific data directory if available, otherwise the base DATA_DIR
if [ ! -z "$DATASET_PATH" ]; then
    bind_mounts+=("--bind" "${DATASET_PATH}:${DATASET_PATH}")
elif [ ! -z "$DATA_DIR" ]; then
    bind_mounts+=("--bind" "${DATA_DIR}:${DATA_DIR}")
fi

echo "Executing singularity for evaluation task $EVAL_TASK_NAME..."

# Execute the evaluation script inside the container for the specific task
singularity exec \
    --pwd /app \
    --env PYTHONPATH="/app:${PYTHONPATH}" \
    "${env_vars[@]}" \
    "${bind_mounts[@]}" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -lc $'echo "[INSIDE] Kernel: $(uname -a)"; \
    echo "[INSIDE] lscpu:"; lscpu | sed "s/^/  /"; \
    echo "[INSIDE] CPU affinity for PID $$: $(taskset -pc $$)"; \
    echo "[INSIDE] Nice level: $(nice)"; \
    echo "[INSIDE] CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"; \
    echo "[INSIDE] DATA_DIR=${DATA_DIR}"; \
    echo "[INSIDE] ALGOTUNE_CONFIG_PATH=${ALGOTUNE_CONFIG_PATH}"; \
    echo "[INSIDE] pwd=$(pwd)"; \
    export PYTHONPATH="/app:${PYTHONPATH}"; \
    python3 /app/scripts/evaluate_results.py \
        --models "$EVAL_MODEL_NAME" \
        --tasks "$EVAL_TASK_NAME" \
        --results-dir "$EVAL_RESULTS_DIR" \
        --generation-file "$EVAL_GENERATION_FILE" \
        --data-dir "$DATA_DIR" \
        --num-runs "$EVAL_NUM_RUNS" \
        --output "$EVAL_OUTPUT_FILE" \
        --max-workers 1'

echo "Evaluation of task $EVAL_TASK_NAME for model $EVAL_MODEL_NAME finished."
# The 'trap cleanup EXIT' will automatically handle removing HOST_TEMP_CODE_DIR