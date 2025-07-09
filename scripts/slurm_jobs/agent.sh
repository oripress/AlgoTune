#!/usr/bin/env bash
#SBATCH --output=slurm/outputs/slurm-%A_%a.out # Relative path for output
#SBATCH --error=slurm/errors/slurm-%A_%a.err  # Relative path for error
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

# Override DATA_DIR to point to the local data directory under the repo
export DATA_DIR="${PROJECT_ROOT}/../data"
echo "[DEBUG] Overriding DATA_DIR in agent.sh: $DATA_DIR"

# Ensure the logs directory exists on the host for bind mounting
mkdir -p "${PROJECT_ROOT}/logs"

# Ensure the base temp storage directory exists
mkdir -p "${TEMP_DIR_STORAGE}"

# Create a unique temporary directory for this task run on the host
# Use Slurm Job ID and Task ID for uniqueness if available, otherwise use process ID
if [ -n "$SLURM_JOB_ID" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    HOST_TEMP_CODE_DIR=$(mktemp -d "${TEMP_DIR_STORAGE}/task_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXXXX")
else
    HOST_TEMP_CODE_DIR=$(mktemp -d "${TEMP_DIR_STORAGE}/task_pid_${$}_XXXXXX")
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

# Task Identification using exported TASK_NAME variable
if [ -z "$TASK_NAME" ]; then
    echo "Error: TASK_NAME environment variable is not set. Ensure submit_agent.sh exports it."
    exit 1
fi

# MODEL is also expected to be exported by submit_agent.sh
if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable is not set. Ensure submit_agent.sh exports it."
    exit 1
fi

# Construct a job identifier string (using Slurm Job ID if available)
if [ -n "$SLURM_JOB_ID" ]; then
    JOB_IDENTIFIER="Job $SLURM_JOB_ID"
else
    JOB_IDENTIFIER="Process $$"
fi

echo "Running task $TASK_NAME ($JOB_IDENTIFIER) with Model $MODEL"

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
# Pass MODEL (already in script env from sbatch --export)
[ ! -z "$MODEL" ] && env_vars+=("--env" "MODEL=${MODEL}")
# Pass AGENT_MODE so it's set inside the container
[ ! -z "$AGENT_MODE" ] && env_vars+=("--env" "AGENT_MODE=${AGENT_MODE}")
# Pass API Keys if found in the sourced .env file
[ ! -z "$DEEPSEEK_API_KEY" ] && env_vars+=("--env" "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}")
[ ! -z "$OPENAI_API_KEY" ] && env_vars+=("--env" "OPENAI_API_KEY=${OPENAI_API_KEY}")
[ ! -z "$GEMINI_API_KEY" ] && env_vars+=("--env" "GEMINI_API_KEY=${GEMINI_API_KEY}")
[ ! -z "$CLAUDE_API_KEY" ] && env_vars+=("--env" "CLAUDE_API_KEY=${CLAUDE_API_KEY}")
[ ! -z "$TOGETHER_API_KEY" ] && env_vars+=("--env" "TOGETHER_API_KEY=${TOGETHER_API_KEY}")
# Pass task-specific parameters from submit_agent.sh
[ ! -z "$TASK_NAME" ] && env_vars+=("--env" "TASK_NAME=${TASK_NAME}")
[ ! -z "$TASK_N" ] && env_vars+=("--env" "TASK_N=${TASK_N}")
[ ! -z "$TASK_DATASET_SIZE" ] && env_vars+=("--env" "TASK_DATASET_SIZE=${TASK_DATASET_SIZE}")
[ ! -z "$TASK_TARGET_TIME_MS" ] && env_vars+=("--env" "TASK_TARGET_TIME_MS=${TASK_TARGET_TIME_MS}")
# Pass summary file and reports directory paths (adjusted for container)
if [ ! -z "$SUMMARY_FILE" ]; then
    if [[ "$SUMMARY_FILE" == "$PROJECT_ROOT"* ]]; then
        CONTAINER_SUMMARY_FILE="/app${SUMMARY_FILE#$PROJECT_ROOT}"
    else
        CONTAINER_SUMMARY_FILE="$SUMMARY_FILE"
    fi
    env_vars+=("--env" "SUMMARY_FILE=${CONTAINER_SUMMARY_FILE}")
fi
if [ ! -z "$REPORTS_DIR" ]; then
    if [[ "$REPORTS_DIR" == "$PROJECT_ROOT"* ]]; then
        CONTAINER_REPORTS_DIR="/app${REPORTS_DIR#$PROJECT_ROOT}"
    else
        CONTAINER_REPORTS_DIR="$REPORTS_DIR"
    fi
    env_vars+=("--env" "REPORTS_DIR=${CONTAINER_REPORTS_DIR}")
fi
# Pass config file path for the container
env_vars+=("--env" "ALGOTUNE_CONFIG_PATH=/app/AlgoTuner/config/config.yaml")
# Pass any other necessary environment variables from the host or .env file
# Example: env_vars+=("--env" "SOME_API_KEY=${SOME_API_KEY}")

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
# Add bind mount for Google credentials using the path from run_config.env
[ ! -z "$GOOGLE_CREDS_HOST_PATH" ] && bind_mounts+=("--bind" "${GOOGLE_CREDS_HOST_PATH}:/credentials/google_creds.json")

echo "Executing singularity for task $TASK_NAME..."
echo "[DEBUG] Host SUMMARY_FILE: ${SUMMARY_FILE}"
echo "[DEBUG] Host REPORTS_DIR: ${REPORTS_DIR}"
echo "[DEBUG] Container SUMMARY_FILE: ${CONTAINER_SUMMARY_FILE:-Not Set}"
echo "[DEBUG] Container REPORTS_DIR: ${CONTAINER_REPORTS_DIR:-Not Set}"

# Execute the agent script inside the container for the specific task, with environment logging
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
    echo "[INSIDE] SUMMARY_FILE=${SUMMARY_FILE}"; \
    echo "[INSIDE] REPORTS_DIR=${REPORTS_DIR}"; \
    echo "[INSIDE] DATA_DIR=${DATA_DIR}"; \
    echo "[INSIDE] ALGOTUNE_CONFIG_PATH=${ALGOTUNE_CONFIG_PATH}"; \
    echo "[INSIDE] pwd=$(pwd)"; \
    echo "[INSIDE] ls -la /app/AlgoTuner/config/ | head -5:"; ls -la /app/AlgoTuner/config/ | head -5; \
    export PYTHONPATH="/app:${PYTHONPATH}"; \
    python3 /app/AlgoTuner/main.py --task "$TASK_NAME" --model "$MODEL"'

echo "Task $TASK_NAME finished."
# The 'trap cleanup EXIT' will automatically handle removing HOST_TEMP_CODE_DIR