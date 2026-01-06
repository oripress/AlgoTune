#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v singularity >/dev/null 2>&1; then
    echo "Error: singularity not found in PATH."
    exit 1
fi

if [ -f "$PROJECT_ROOT/config.env" ]; then
    source "$PROJECT_ROOT/config.env"
elif [ -f "$PROJECT_ROOT/slurm/run_config.env" ]; then
    source "$PROJECT_ROOT/slurm/run_config.env"
else
    echo "Error: No configuration file found (config.env or slurm/run_config.env)"
    exit 1
fi

if [ -z "${TEMP_DIR_STORAGE:-}" ]; then
    export TEMP_DIR_STORAGE="/tmp"
fi

source "$PROJECT_ROOT/scripts/slurm_jobs/resolve_singularity_image.sh"
resolve_singularity_image

echo "Resolved Singularity image: $SINGULARITY_IMAGE"
echo "Running quick container check..."
singularity exec "$SINGULARITY_IMAGE" python3 -c "import sys; print('ok', sys.version)"
