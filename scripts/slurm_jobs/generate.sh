#!/usr/bin/env bash
#SBATCH --time=00:30:00
#SBATCH --mem=14G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/outputs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

# -----------------------------------------------------------------------------
# generate.sh – run exactly one eval sub‐job (one of a 3‐job array),
# collect JSON, and append to summary.json under .[task].baseline_runs.[run_id]
#
# Fix: ensure that the temporary helper is invoked by its full, bound‐path
# inside the container, rather than assuming "/tmp/<basename>".
#
# Usage (automatically invoked by sbatch from the driver):
#   sbatch --array=0-2 --dependency=afterok:<gen_job_id> \
#           generate.sh <task_name> <dataset_dir>
# -----------------------------------------------------------------------------

set -euo pipefail
set -x    # For debugging

# ----------------------------------------------------------------------
# 0.  Project-root discovery and config loading
# ----------------------------------------------------------------------
PROJECT_ROOT=$(realpath "${SLURM_SUBMIT_DIR:-.}")
# If we're in scripts/, go up one level to get to the actual project root
if [[ "$PROJECT_ROOT" == */scripts ]]; then
    PROJECT_ROOT=$(dirname "$PROJECT_ROOT")
fi

# Load environment configuration
if [ -f "$PROJECT_ROOT/config.env" ]; then
    source "$PROJECT_ROOT/config.env"
    echo "$(date): Loaded config from $PROJECT_ROOT/config.env" >&2
elif [ -f "$PROJECT_ROOT/slurm/run_config.env" ]; then
    source "$PROJECT_ROOT/slurm/run_config.env"
    echo "$(date): Loaded config from $PROJECT_ROOT/slurm/run_config.env" >&2
else
    echo "$(date): WARNING: No config file found (config.env or slurm/run_config.env)" >&2
fi

# Define the summary file path
SUMMARY_FILE="$PROJECT_ROOT/reports/generation.json"

# Ensure the reports directory exists
mkdir -p "$(dirname "$SUMMARY_FILE")"

# ----------------------------------------------------------------------
# 1.  Early temp-script naming (host vs. container)
# ----------------------------------------------------------------------
TASK_NAME="${1:-unknown_task}"
PREGENERATED_DATA_DIR="$2"    # e.g. "/…/DATA_DIR/$TASK_NAME"
RUN_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Create a unique name for the helper script
TEMP_SCRIPT_BASENAME="temp_timing_helper_${TASK_NAME}_${RUN_ID}.py"
TEMP_PYTHON_SCRIPT_HOST="${TEMP_DIR_STORAGE:-/tmp}/${TEMP_SCRIPT_BASENAME}"
# Now point the "container" path to the same absolute location,
# because we bind "${TEMP_DIR_STORAGE}:${TEMP_DIR_STORAGE}" below.
TEMP_PYTHON_SCRIPT_CONTAINER="${TEMP_PYTHON_SCRIPT_HOST}"

echo "$(date): Starting eval job – task='${TASK_NAME}', run_id=${RUN_ID}, dataset='${PREGENERATED_DATA_DIR}'" >&2

# ----------------------------------------------------------------------
# 2.  Ensure tests/logs & tests/reports exist (for summary.json and logs)
# ----------------------------------------------------------------------
mkdir -p "$PROJECT_ROOT/tests/logs" "$PROJECT_ROOT/tests/reports"

# ----------------------------------------------------------------------
# 3.  Get target_time_ms from environment variable (passed by SLURM job)
# ----------------------------------------------------------------------
# Use the TARGET_TIME_MS environment variable passed from the submit script
if [ -n "${TARGET_TIME_MS:-}" ]; then
    # Use the environment variable from SLURM job
    TARGET_TIME_MS="${TARGET_TIME_MS}"
else
    echo "Error: TARGET_TIME_MS environment variable not set" >&2
    echo "This should be passed from the SLURM job submission" >&2
    exit 1
fi

echo "$(date): Using TARGET_TIME_MS=${TARGET_TIME_MS}" >&2

cat > "$TEMP_PYTHON_SCRIPT_HOST" <<'PYHELPER'
#!/usr/bin/env python3
import os
import sys
import json
import logging
import traceback
import numpy as np
from pathlib import Path
import glob

logging.basicConfig(level=logging.INFO, format="%(levelname)s:python_eval:%(message)s")

# Ensure project root (/app) and AlgoTune (if needed) are on PYTHONPATH
sys.path.insert(0, "/app")
if "/app/AlgoTune" not in sys.path:
    sys.path.insert(0, "/app/AlgoTune")

import argparse
from AlgoTuner.utils.streaming_json import stream_jsonl
from AlgoTuner.utils.evaluator.loader import load_task
from AlgoTuner.utils.evaluator.main import evaluate_baseline_dataset
from AlgoTuner.config.loader import load_config
from AlgoTuner.utils.evaluator.runner import DATASET_RUNS, DATASET_WARMUPS

parser = argparse.ArgumentParser()
parser.add_argument("task_name")
parser.add_argument("data_dir")
parser.add_argument("run_id", type=int)
parser.add_argument("target_time_ms", type=int)
args = parser.parse_args()

task_name = args.task_name
data_dir = args.data_dir
run_id = args.run_id
target_time_ms = args.target_time_ms

results = {
    "success": False,
    "error": None,
    "avg_min_ms": None,
    "std_min_ms": None,
    "target_time_ms": target_time_ms
}

# Apply pysat fixes for multiprocessing compatibility
try:
    from AlgoTuner.utils.pysat_fix import apply_pysat_fixes
    apply_pysat_fixes()
    logging.info("Applied pysat fixes for multiprocessing compatibility")
except Exception as exc:
    logging.warning(f"Failed to apply pysat fixes: {exc}")

try:
    logging.info(f"Loading task '{task_name}', setting target_time_ms={target_time_ms}")
    task_instance = load_task(task_name=task_name, data_dir=data_dir)
    if task_instance is None:
        raise ValueError(f"load_task returned None for '{task_name}'")

    if hasattr(task_instance, "_target_time_ms"):
        task_instance._target_time_ms = target_time_ms
    elif hasattr(task_instance, "set_target_time"):
        task_instance.set_target_time(target_time_ms)

    baseline_times_filename = f"baseline_times_{task_name}_{run_id}_{target_time_ms}_{os.getpid()}.json"
    baseline_times_filepath = os.path.join(os.environ.get("TEMP_DIR_STORAGE", "/tmp"), baseline_times_filename)

    # Find the train JSONL file for the task
    train_files = glob.glob(os.path.join(data_dir, f"{task_name}_T{target_time_ms}ms_n*_size*_train.jsonl"))
    if not train_files:
        raise FileNotFoundError(f"No train JSONL file found in {data_dir} for task {task_name} with target time {target_time_ms}ms")
    train_jsonl = train_files[0]
    dataset_iterable = stream_jsonl(train_jsonl)

    logging.info(f"Evaluating baseline with target_time={target_time_ms} → writing to {baseline_times_filepath}")
    returned_fp = evaluate_baseline_dataset(
        task_obj=task_instance,
        dataset_iterable=dataset_iterable,
        num_runs=DATASET_RUNS,
        warmup_runs=DATASET_WARMUPS,
        output_file=baseline_times_filepath,
        jsonl_path=train_jsonl  # Pass the path so it can stream directly
    )
    logging.info(f"Evaluation done; reading results from {returned_fp}")

    with open(returned_fp, "r") as f:
        problem_times_dict = json.load(f)

    min_times_ms = [
        float(v) for v in problem_times_dict.values()
        if v is not None and isinstance(v, (int, float)) and float(v) > 0
    ]
    if not min_times_ms:
        raise ValueError("No valid positive baseline times found → treating as failure.")

    results["avg_min_ms"] = float(np.mean(min_times_ms))
    results["std_min_ms"] = float(np.std(min_times_ms))
    results["success"] = True
    logging.info(f"SUCCESS: avg={results['avg_min_ms']:.2f} ms, std={results['std_min_ms']:.2f} ms")

    try:
        os.remove(returned_fp)
    except OSError:
        pass

except Exception as e:
    err_msg = f"target_time={target_time_ms}ms failed: {e}"
    logging.error(err_msg)
    logging.debug(traceback.format_exc())
    results["error"] = err_msg

print(json.dumps({
    "task_name": task_name,
    "run_id": run_id,
    "success": results["success"],
    "error": results["error"],
    "avg_min_ms": results["avg_min_ms"],
    "std_min_ms": results["std_min_ms"],
    "target_time_ms": results["target_time_ms"]
}))
PYHELPER

chmod +x "$TEMP_PYTHON_SCRIPT_HOST"

# ----------------------------------------------------------------------
# 4.  Run the Python helper inside Singularity
# ----------------------------------------------------------------------
EVAL_START=$(date +%s%3N)

PY_OUTPUT=$(
  set -o pipefail; \
  singularity exec ${SINGULARITY_FLAGS:-} \
    --bind "$PROJECT_ROOT:/app" \
    --bind "${PREGENERATED_DATA_DIR}:${PREGENERATED_DATA_DIR}" \
    --bind "${TEMP_DIR_STORAGE}:${TEMP_DIR_STORAGE}" \
    --env PYTHONPATH="/app:/app/AlgoTune:${PYTHONPATH:-}" \
    --env TEMP_DIR_STORAGE="${TEMP_DIR_STORAGE}" \
    --env CODE_DIR="/app" \
    "${SINGULARITY_IMAGE}" \
    python3 "$TEMP_PYTHON_SCRIPT_CONTAINER" \
       "$TASK_NAME" "$PREGENERATED_DATA_DIR" "$RUN_ID" "$TARGET_TIME_MS" 2>&1
)
SING_EXIT=$?

# Check for OOM kill in SLURM logs
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Check SLURM logs for OOM kill messages
    if dmesg 2>/dev/null | grep -q "Killed process.*${SLURM_JOB_ID}" || \
       journalctl -q --since="5 minutes ago" 2>/dev/null | grep -q "oom_kill.*${SLURM_JOB_ID}" || \
       echo "$PY_OUTPUT" | grep -q "MemoryError\|OutOfMemoryError\|killed.*memory"; then
        echo "$(date): OOM KILL DETECTED for job ${SLURM_JOB_ID}" >&2
        PY_OUTPUT="${PY_OUTPUT}\nERROR: Out of memory kill detected during evaluation"
        SING_EXIT=137  # Standard OOM exit code
    fi
fi

EVAL_END=$(date +%s%3N)
EVAL_MS=$((EVAL_END - EVAL_START))

LOG_FILE="$PROJECT_ROOT/tests/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${RUN_ID}.log"
{
  echo "---- Python helper stdout+stderr ----"
  echo "$PY_OUTPUT"
  echo "---- end Python helper output ----"
} >> "$LOG_FILE"

# ----------------------------------------------------------------------
# 5. Convert helper output → RESULTS_JSON, then atomically update summary.json
# ----------------------------------------------------------------------
if (( SING_EXIT != 0 )); then
    ERR=$(echo "$PY_OUTPUT" | grep -E '(Error|Exception|Traceback)' | head -n1)
    ERR=${ERR:-"singularity exit $SING_EXIT"}
    RESULTS_JSON=$(jq -n \
      --arg tn "$TASK_NAME" \
      --argjson run "$RUN_ID" \
      --arg msg "$ERR" \
      --argjson d "$EVAL_MS" \
      '{ task_name: $tn,
         run_id: $run,
         success: false,
         error: $msg,
         avg_min_ms: null,
         std_min_ms: null,
         target_time_ms: null,
         eval_duration_ms: $d }')
else
    JLINE=$(echo "$PY_OUTPUT" | sed '/^$/d' | tail -n1)
    if echo "$JLINE" | jq -e . >/dev/null 2>&1; then
        RESULTS_JSON=$(echo "$JLINE" | \
          jq --argjson d "$EVAL_MS" --arg tn "$TASK_NAME" --argjson run "$RUN_ID" \
             '. + { eval_duration_ms: $d, task_name: $tn, run_id: $run }')
    else
        RESULTS_JSON=$(jq -n \
          --arg tn "$TASK_NAME" \
          --argjson run "$RUN_ID" \
          --arg msg "no json from helper" \
          --argjson d "$EVAL_MS" \
          '{ task_name: $tn,
             run_id: $run,
             success: false,
             error: $msg,
             avg_min_ms: null,
             std_min_ms: null,
             target_time_ms: null,
             eval_duration_ms: $d }')
    fi
fi

echo "$(date): RESULTS_JSON → $RESULTS_JSON" >&2

# ----------------------------------------------------------------------
# 6.  Atomic update of summary.json: append/update .baseline_runs[run_id] for this task
# ----------------------------------------------------------------------
LOCK_FILE="$PROJECT_ROOT/reports/generation.lock"
TMP_SUM2=$(mktemp)

# Clean up stale lock file if it exists and is older than 5 minutes
if [[ -f "$LOCK_FILE" ]]; then
    if find "$LOCK_FILE" -mmin +5 2>/dev/null | grep -q .; then
        echo "$(date): Removing stale lock file $LOCK_FILE" >&2
        rm -f "$LOCK_FILE"
    fi
fi

success_val=$(echo "$RESULTS_JSON" | jq '.success')
avg_min_val=$(echo "$RESULTS_JSON" | jq '.avg_min_ms')
std_min_val=$(echo "$RESULTS_JSON" | jq '.std_min_ms')
eval_dur_val=$(echo "$RESULTS_JSON" | jq '.eval_duration_ms')
tgt_time_val=$(echo "$RESULTS_JSON" | jq '.target_time_ms')

# Find the train file for this task and target_time
TRAIN_FILE=$(ls "$PREGENERATED_DATA_DIR"/${TASK_NAME}_T${TARGET_TIME_MS}ms_n*_size*_train.jsonl | head -n1)
if [[ -z "$TRAIN_FILE" ]]; then
  echo "Could not find train file for $TASK_NAME at T=${TARGET_TIME_MS}ms" >&2
  exit 1
fi

# Extract n and dataset_size from the filename
N_VAL=$(basename "$TRAIN_FILE" | sed -E 's/.*_n([0-9]+)_size[0-9]+_train\.jsonl/\1/')
DATASET_SIZE=$(basename "$TRAIN_FILE" | sed -E 's/.*_n[0-9]+_size([0-9]+)_train\.jsonl/\1/')

(
  flock -w 60 200 || { echo "Could not lock $LOCK_FILE" >&2; exit 1; }

  # Initialize generation.json if it doesn't exist or is empty (inside lock to prevent races)
  if [[ ! -f "$SUMMARY_FILE" ]] || [[ ! -s "$SUMMARY_FILE" ]]; then
    echo '{}' > "$SUMMARY_FILE"
    echo "$(date): Initialized $SUMMARY_FILE (locked)" >&2
  fi

  jq --arg task "$TASK_NAME" \
     --argjson run_id "$RUN_ID" \
     --argjson success "$success_val" \
     --argjson avg_min "$avg_min_val" \
     --argjson std_min "$std_min_val" \
     --argjson dur "$eval_dur_val" \
     --argjson tgt "$tgt_time_val" \
     --arg n "$N_VAL" \
     --arg dataset_size "$DATASET_SIZE" \
     '
     (.[$task] //= {}) |
     (.[$task].target_time_ms) = $tgt |
     (.[$task].n) = ($n | tonumber) |
     (.[$task].dataset_size) = ($dataset_size | tonumber) |
     (.[$task].baseline_runs //= {}) |
     (.[$task].baseline_runs[$run_id|tostring] = {
         success: $success,
         avg_min_ms: $avg_min,
         std_min_ms: $std_min,
         eval_duration_ms: $dur
       })
     ' "$SUMMARY_FILE" > "$TMP_SUM2" && mv "$TMP_SUM2" "$SUMMARY_FILE"

  echo " → summary.json updated for '$TASK_NAME' run $RUN_ID" >&2
) 200>"$LOCK_FILE"

rm -f "$TMP_SUM2"

# ----------------------------------------------------------------------
# 7.  Exit 0 if success==true, else exit 1 so that array dependency fails
# ----------------------------------------------------------------------
if [[ $(echo "$RESULTS_JSON" | jq -r '.success') == "true" ]]; then
    echo "$(date): Eval job for '$TASK_NAME' run $RUN_ID finished OK." >&2
    exit 0
else
    echo "$(date): Eval job for '$TASK_NAME' run $RUN_ID finished with FAILURE." >&2
    exit 1
fi