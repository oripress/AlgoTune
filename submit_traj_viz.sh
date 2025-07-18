#!/usr/bin/env bash

# Source the configuration to get partition info
if [ -f "config.env" ]; then
    source config.env
else
    echo "Error: config.env not found!"
    exit 1
fi

# Check if the partition variable is set
if [ -z "$SLURM_PARTITION_VIZ" ]; then
    echo "Error: SLURM_PARTITION_VIZ is not set in config.env"
    exit 1
fi

echo "Submitting trajectories_to_html.py to partition(s): $SLURM_PARTITION_VIZ using Conda env: ${CONDA_ENV_VIZ}"

# Define the paths relative to the submission directory
PROJECT_ROOT=$(pwd) # Or use SLURM_SUBMIT_DIR if always submitting from project root
PYTHON_SCRIPT="${PROJECT_ROOT}/trajectories_to_html/trajectories_to_html.py"
OUTPUT_LOG="${PROJECT_ROOT}/slurm/outputs/traj_viz_%j.log"
ERROR_LOG="${PROJECT_ROOT}/slurm/errors/traj_viz_%j.err"

# Ensure output/error directories exist
mkdir -p "${PROJECT_ROOT}/slurm/outputs" "${PROJECT_ROOT}/slurm/errors"

# Submit the Python script directly via sbatch using conda run
sbatch --partition="$SLURM_PARTITION_VIZ" \
       --job-name="traj_viz" \
       --output="${OUTPUT_LOG}" \
       --error="${ERROR_LOG}" \
       --mem=16gb \
       --time=00:25:00 \
       --wrap="singularity exec ${SINGULARITY_IMAGE} python3 '${PYTHON_SCRIPT}'"

sbatch_exit_code=$?
if [ $sbatch_exit_code -ne 0 ]; then
    echo "Error: sbatch command failed with exit code $sbatch_exit_code"
    exit 1
else
    echo "sbatch command submitted successfully (check sacct for job status)."
fi

# Note: The Conda path needs to be updated above. 