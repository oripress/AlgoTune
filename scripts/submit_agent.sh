#!/usr/bin/env bash

# Exit on error and catch errors in pipelines.
set -e
set -o pipefail

export SKIP_DATASET_GEN=1  # Skip dataset generation, reuse existing data
export AGENT_MODE=1 

# Maximum number of tasks to submit (default 999)
MAX_TASKS=250

# Parse flags
RETRY_NA=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --retry-na)
            RETRY_NA=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Check for minimum number of arguments (model name)
if [ $# -lt 1 ]; then
    echo "Usage: $0 [--retry-na] <model_name> [task_index_1 task_index_2 ...]"
    echo "  --retry-na: Retry tasks with N/A results and entries without matching log files"
    echo "Example: $0 my-cool-model 30 51"
    echo "Example: $0 --retry-na my-other-model  (runs all tasks, retries N/A)"
    exit 1
fi

# First argument is the model name
MODEL_NAME="$1"
shift # Remove the model name from the argument list, leaving only task indices

echo "Using model: $MODEL_NAME"
if [[ "$RETRY_NA" == "true" ]]; then
    echo "Retry N/A mode enabled: will clean up and retry tasks with N/A results and entries without matching log files"
fi

# Helper function to check if a result is N/A
is_result_na() {
    local task_name="$1"
    local model_name="$2"
    local summary_file="$3"
    
    # Check if result exists and is "N/A"
    local result=$(jq -r --arg task "$task_name" --arg model "$model_name" \
        '.[$task][$model].final_speedup // "null"' "$summary_file" 2>/dev/null)
    
    if [[ "$result" == "N/A" ]]; then
        return 0  # Is N/A
    else
        return 1  # Not N/A or doesn't exist
    fi
}

# Helper function to map log model name back to full model name in config
map_log_model_to_full_name() {
    local log_model_name="$1"
    
    # Known mappings based on how model names are shortened in the script
    case "$log_model_name" in
        "DeepSeek-R1")
            echo "deepseek-ai/DeepSeek-R1"
            ;;
        "gemini-2.5-pro-preview-06-05")
            echo "gemini/gemini-2.5-pro-preview-06-05"
            ;;
        "gemini-2.5-pro-preview-05-06")
            echo "gemini/gemini-2.5-pro-preview-05-06"
            ;;
        "gemini-2.5-pro")
            echo "vertex_ai/gemini-2.5-pro"
            ;;
        "o4-mini"|"claude-3-7-sonnet-20250219"|"claude-opus-4-20250514"|"deepseek-coder"|"deepseek-reasoner")
            echo "$log_model_name"  # These don't have prefixes
            ;;
        "dummy")
            echo ""  # Dummy logs have no corresponding model
            ;;
        *)
            echo "$log_model_name"  # Default fallback
            ;;
    esac
}

# Helper function to check if a log file has corresponding entry in summary
log_has_summary_entry() {
    local task_name="$1"
    local full_model_name="$2"
    local summary_file="$3"
    
    if [[ -z "$full_model_name" ]]; then
        return 1  # No valid model name (e.g., dummy logs)
    fi
    
    # Check if entry exists in summary (regardless of N/A status)
    if jq --exit-status --arg task "$task_name" --arg model "$full_model_name" \
        '.[$task][$model] != null' "$summary_file" > /dev/null 2>&1; then
        return 0  # Entry exists
    else
        return 1  # No entry
    fi
}

# Helper function to check if a summary entry has matching log files
summary_entry_has_logs() {
    local task_name="$1"
    local model_name="$2"
    
    # Get the shortened model name as it appears in log files
    local model_name_for_log=${model_name##*/}
    local log_prefix="${task_name}_${model_name_for_log}_"
    
    # Check if any log files exist for this task+model combination
    if find "${PROJECT_ROOT}/logs/" -maxdepth 1 -name "${log_prefix}*.log" -print -quit 2>/dev/null | grep -q .; then
        return 0  # Log files exist
    else
        return 1  # No log files found
    fi
}

# Helper function to clean up orphaned log files for the specified model
cleanup_orphaned_logs() {
    local target_model_name="$1"
    local summary_file="$2"
    
    echo " ---> Checking for orphaned log files for model: $target_model_name"
    
    # Get the shortened model name as it appears in log files
    local target_model_short=${target_model_name##*/}
    
    # Find all log files for this model
    local orphaned_count=0
    while IFS= read -r -d '' log_file; do
        local filename=$(basename "$log_file")
        
        # Extract task and model from filename: {task}_{model}_{timestamp}.log
        # Use the target model short name to split correctly since we know what we're looking for
        if [[ "$filename" =~ ^(.+)_${target_model_short}_[0-9]{8}_[0-9]{6}\.log$ ]]; then
            local log_task_name="${BASH_REMATCH[1]}"
            local log_model_short="$target_model_short"
            
            # Only process files for our target model
            if [[ "$log_model_short" == "$target_model_short" ]]; then
                # Map back to full model name
                local full_model_name=$(map_log_model_to_full_name "$log_model_short")
                
                # Check if this log has a corresponding summary entry
                if ! log_has_summary_entry "$log_task_name" "$full_model_name" "$summary_file"; then
                    echo " ---> Orphaned log found: $filename (task: $log_task_name, model: $full_model_name)"
                    rm -f "$log_file"
                    orphaned_count=$((orphaned_count + 1))
                fi
            fi
        fi
    done < <(find "${PROJECT_ROOT}/logs/" -maxdepth 1 -name "*_${target_model_short}_*.log" -print0 2>/dev/null)
    
    if [[ $orphaned_count -gt 0 ]]; then
        echo " ---> Removed $orphaned_count orphaned log file(s) for model $target_model_name"
    else
        echo " ---> No orphaned log files found for model $target_model_name"
    fi
}

# Helper function to check if log exists for given task/model combination
log_exists_for_task_model() {
    local task_name="$1"
    local model_name="$2"
    
    # Extract model name after last slash for log file pattern
    local model_name_for_log=${model_name##*/}
    local log_prefix="${task_name}_${model_name_for_log}_"
    
    # Check if any log files exist with this prefix
    if find "${PROJECT_ROOT}/logs/" -maxdepth 1 -name "${log_prefix}*.log" -print -quit 2>/dev/null | grep -q .; then
        return 0  # Log exists
    else
        return 1  # No log exists
    fi
}

# Helper function to clean up orphaned summary entry
cleanup_orphaned_summary_entry() {
    local task_name="$1"
    local model_name="$2"
    local summary_file="$3"
    
    echo " ---> Removing orphaned summary entry for ${task_name}/${model_name}"
    
    # Remove entry from summary JSON using jq
    local temp_file=$(mktemp)
    if jq --arg task "$task_name" --arg model "$model_name" \
        'if .[$task] then .[$task] = (.[$task] | del(.[$model])) else . end | 
         if .[$task] == {} then del(.[$task]) else . end' \
        "$summary_file" > "$temp_file" 2>/dev/null; then
        mv "$temp_file" "$summary_file"
        echo " ---> Removed orphaned entry from summary file"
    else
        rm -f "$temp_file"
        echo " ---> Warning: Could not update summary file"
    fi
}

# Helper function to clean up orphaned log files
cleanup_orphaned_logs() {
    local task_name="$1"
    local model_name="$2"
    
    # Extract model name after last slash for log file pattern
    local model_name_for_log=${model_name##*/}
    local log_prefix="${task_name}_${model_name_for_log}_"
    
    echo " ---> Removing orphaned logs matching pattern: ${log_prefix}*.log"
    find "${PROJECT_ROOT}/logs/" -maxdepth 1 -name "${log_prefix}*.log" -print -delete 2>/dev/null || echo " ---> No matching log files found to delete."
}

# Helper function to clean up N/A entries and corresponding logs
cleanup_na_entry() {
    local task_name="$1"
    local model_name="$2"
    local summary_file="$3"
    
    echo " ---> Cleaning up N/A entry for ${task_name}/${model_name}"
    
    # Remove log files for this specific task+model combination
    local model_name_for_log=${model_name##*/}  # Extract part after last /
    local log_prefix="${task_name}_${model_name_for_log}_"
    echo " ---> Removing logs matching pattern: ${log_prefix}*.log"
    find "${PROJECT_ROOT}/logs/" -maxdepth 1 -name "${log_prefix}*.log" -print -delete 2>/dev/null || echo " ---> No matching log files found to delete."
    
    # Remove entry from summary JSON using jq
    local temp_file=$(mktemp)
    if jq --arg task "$task_name" --arg model "$model_name" \
        'if .[$task] then .[$task] = (.[$task] | del(.[$model])) else . end | 
         if .[$task] == {} then del(.[$task]) else . end' \
        "$summary_file" > "$temp_file" 2>/dev/null; then
        mv "$temp_file" "$summary_file"
        echo " ---> Removed N/A entry from summary file"
    else
        rm -f "$temp_file"
        echo " ---> Warning: Could not update summary file"
    fi
}

# Determine the project root assuming this script is run from the root.
PROJECT_ROOT=$(pwd)
echo "PROJECT_ROOT: $PROJECT_ROOT"

# Define report directory and summary file path
REPORTS_DIR="$PROJECT_ROOT/reports"
SUMMARY_FILE="$REPORTS_DIR/agent_summary.json"
echo "Reports directory: $REPORTS_DIR"
echo "Summary file: $SUMMARY_FILE"

# Source the configuration
if [ -f "${PROJECT_ROOT}/config.env" ]; then
    set -o allexport
    source "${PROJECT_ROOT}/config.env"
    set +o allexport
elif [ -f "${PROJECT_ROOT}/slurm/run_config.env" ]; then
    set -o allexport
    source "${PROJECT_ROOT}/slurm/run_config.env"
    set +o allexport
else
    echo "Error: No configuration file found (config.env or slurm/run_config.env)"
    exit 1
fi

# Override DATA_DIR to point to the local data directory under the repo
export DATA_DIR="${PROJECT_ROOT}/../data"
echo "Overriding DATA_DIR: $DATA_DIR"

# Ensure necessary env vars are set from config
if [ -z "$SLURM_PARTITIONS_DEFAULT" ]; then
    echo "Error: SLURM_PARTITIONS_DEFAULT is not set in configuration"
    exit 1
fi
if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR is not set in configuration"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: DATA_DIR directory does not exist: $DATA_DIR"
    exit 1
fi
echo "Using DATA_DIR: $DATA_DIR"

# Minimal directory setup
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${REPORTS_DIR}" # Ensure reports directory exists
mkdir -p "${PROJECT_ROOT}/slurm/outputs"
mkdir -p "${PROJECT_ROOT}/slurm/errors"

# Initialize summary file if it doesn't exist
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "{}" > "$SUMMARY_FILE"
    echo "Initialized empty summary file: $SUMMARY_FILE"
fi

# Clean up orphaned log files and entries without logs when using --retry-na
if [[ "$RETRY_NA" == "true" ]]; then
    cleanup_orphaned_logs "$MODEL_NAME" "$SUMMARY_FILE"
    
    # Also clean up summary entries that don't have corresponding log files
    echo " ---> Checking for summary entries without matching log files for model: $MODEL_NAME"
    
    # Get all tasks for this model from the summary
    entries_without_logs=0
    while IFS= read -r task_name; do
        if [[ -n "$task_name" ]] && ! summary_entry_has_logs "$task_name" "$MODEL_NAME"; then
            echo " ---> Summary entry without log files found: ${task_name}/${MODEL_NAME}"
            cleanup_na_entry "$task_name" "$MODEL_NAME" "$SUMMARY_FILE"
            entries_without_logs=$((entries_without_logs + 1))
        fi
    done < <(jq -r --arg model "$MODEL_NAME" 'to_entries[] | select(.value[$model] != null) | .key' "$SUMMARY_FILE" 2>/dev/null)
    
    if [[ $entries_without_logs -gt 0 ]]; then
        echo " ---> Removed $entries_without_logs summary entries without matching log files for model $MODEL_NAME"
    else
        echo " ---> All summary entries for model $MODEL_NAME have matching log files"
    fi
fi

# --- Task selection: choose tasks with sufficient baseline runs and no nulls ---
TIMINGS_SUMMARY_FILE_PATH="$PROJECT_ROOT/reports/generation.json"
NUM_BASELINE_RUNS_REQUIRED=3

echo "Selecting tasks from $TIMINGS_SUMMARY_FILE_PATH based on baseline run count >= $NUM_BASELINE_RUNS_REQUIRED and non-null statistics..."
# Populate TASK_DATA array: task_name:n:dataset_size:target_time_ms
TASK_DATA=($(jq -r --argjson req_runs "$NUM_BASELINE_RUNS_REQUIRED" '
  to_entries[]
  | select(.value.baseline_runs and (.value.baseline_runs | keys | length >= $req_runs))
  | select((.value.baseline_runs | any(.[]; (.avg_min_ms==null or .std_min_ms==null))) | not)
  | "\(.key):\(.value.n):\(.value.dataset_size):\(.value.target_time_ms)"
' "$TIMINGS_SUMMARY_FILE_PATH"))

# Number of tasks to process
NUM_TASKS=${#TASK_DATA[@]}
if [ "$NUM_TASKS" -eq 0 ]; then
    echo "Error: No tasks meet baseline timing criteria in $TIMINGS_SUMMARY_FILE_PATH"
    exit 1
fi

echo "Found $NUM_TASKS tasks to submit: ${TASK_DATA[*]}"

# Limit tasks to MAX_TASKS (set above)
if [ "$NUM_TASKS" -gt "$MAX_TASKS" ]; then
    echo "Limiting to first $MAX_TASKS tasks (of $NUM_TASKS total)"
    TASK_DATA=( "${TASK_DATA[@]:0:$MAX_TASKS}" )
    NUM_TASKS=${#TASK_DATA[@]}
fi

# Determine the list of task indices to run
TASK_INDICES_TO_RUN=()
if [ $# -gt 0 ]; then
    # Task names/indices provided (remaining arguments after shift)
    echo "Specific tasks provided: $@"
    for task_spec in "$@"; do # Use remaining arguments
        if [[ "$task_spec" =~ ^[0-9]+$ ]]; then
            # It's a numeric index - validate range
            MAX_INDEX=$((NUM_TASKS - 1))
            if [ "$task_spec" -gt "$MAX_INDEX" ]; then
                echo "Error: Invalid task index '$task_spec'. Must be a number between 0 and $MAX_INDEX."
                exit 1
            fi
            TASK_INDICES_TO_RUN+=("$task_spec")
        else
            # It's a task name - find its index in the filtered TASK_DATA array
            found_index=""
            for i in "${!TASK_DATA[@]}"; do
                TASK_INFO="${TASK_DATA[$i]}"
                IFS=':' read -r TASK_NAME_CHECK _ _ _ <<< "$TASK_INFO"
                if [ "$TASK_NAME_CHECK" = "$task_spec" ]; then
                    found_index="$i"
                    break
                fi
            done
            if [ -z "$found_index" ]; then
                echo "Error: Task '$task_spec' not found in available tasks with sufficient baseline runs."
                echo "Available tasks:"
                for TASK_INFO in "${TASK_DATA[@]}"; do
                    IFS=':' read -r TASK_NAME_SHOW _ _ _ <<< "$TASK_INFO"
                    echo "  $TASK_NAME_SHOW"
                done | head -20
                echo "... (and more)"
                exit 1
            fi
            TASK_INDICES_TO_RUN+=("$found_index")
        fi
    done
    echo "Will submit specific tasks for indices: ${TASK_INDICES_TO_RUN[*]}"
elif [ "$NUM_TASKS" -gt 0 ]; then
    # No tasks provided: run all tasks
    TASK_INDICES_TO_RUN=($(seq 0 $((NUM_TASKS - 1))))
    echo "No specific tasks provided. Found $NUM_TASKS tasks. Submitting all tasks."
else
    # No tasks and no tasks found (already handled, but for clarity)
    echo "Error: No tasks discovered and no specific tasks requested."
    exit 1
fi

echo "Submitting agent jobs..."
JOB_IDS=()
SUBMITTED_COUNT=0
SKIPPED_COUNT=0

# Iterate through specified task indices for single MODEL_NAME
for i in "${TASK_INDICES_TO_RUN[@]}"; do
    TASK_INFO="${TASK_DATA[$i]}"
    # Parse TASK_INFO (task_name:n:dataset_size:target_time_ms)
    IFS=':' read -r TASK_NAME TASK_N TASK_DATASET_SIZE TASK_TARGET_TIME_MS <<< "$TASK_INFO"

    # Construct DATASET_PATH and resolve it to an absolute, canonical path
    UNRESOLVED_DATASET_PATH="${DATA_DIR}/${TASK_NAME}"
    RESOLVED_DATASET_PATH=$(realpath "$UNRESOLVED_DATASET_PATH")
    echo " ---> Resolved DATASET_PATH for task $TASK_NAME: $RESOLVED_DATASET_PATH" # For debugging

    # --- Check if result already exists in summary file AND log exists ---
    echo "Checking summary and logs for Task: $TASK_NAME, Model: $MODEL_NAME..."
    
    # Check if summary entry exists
    has_summary_entry=false
    if jq --exit-status --arg task "$TASK_NAME" --arg model "$MODEL_NAME" \
        '.[$task][$model] != null' "$SUMMARY_FILE" > /dev/null 2>&1; then
        has_summary_entry=true
    fi
    
    # Check if log exists
    has_log=false
    if log_exists_for_task_model "$TASK_NAME" "$MODEL_NAME"; then
        has_log=true
    fi
    
    echo " ---> Summary entry exists: $has_summary_entry, Log exists: $has_log"
    
    # Decision logic based on the new default behavior
    if [[ "$RETRY_NA" == "true" ]]; then
        # With --retry-na: only skip if result exists and is NOT N/A
        if jq --exit-status --arg task "$TASK_NAME" --arg model "$MODEL_NAME" \
            '.[$task][$model] != null and .[$task][$model].final_speedup != "N/A"' \
            "$SUMMARY_FILE" > /dev/null 2>&1; then
            echo " ---> Valid (non-N/A) result found in $SUMMARY_FILE. Skipping job submission."
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
        elif is_result_na "$TASK_NAME" "$MODEL_NAME" "$SUMMARY_FILE"; then
            echo " ---> N/A result found. Cleaning up and retrying..."
            cleanup_na_entry "$TASK_NAME" "$MODEL_NAME" "$SUMMARY_FILE"
            echo " ---> Proceeding with job submission after cleanup."
        else
            echo " ---> No result found. Proceeding with submission."
        fi
    else
        # New default behavior: skip only if BOTH summary entry AND log exist
        if [[ "$has_summary_entry" == "true" ]] && [[ "$has_log" == "true" ]]; then
            echo " ---> Both summary entry and log found. Skipping job submission."
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
        elif [[ "$has_summary_entry" == "true" ]] && [[ "$has_log" == "false" ]]; then
            echo " ---> Summary entry found but no log. Cleaning up orphaned summary entry..."
            cleanup_orphaned_summary_entry "$TASK_NAME" "$MODEL_NAME" "$SUMMARY_FILE"
            echo " ---> Proceeding with job submission after cleanup."
        elif [[ "$has_summary_entry" == "false" ]] && [[ "$has_log" == "true" ]]; then
            echo " ---> Log found but no summary entry. Cleaning up orphaned logs..."
            cleanup_orphaned_logs "$TASK_NAME" "$MODEL_NAME"
            echo " ---> Proceeding with job submission after cleanup."
        else
            echo " ---> No result found. Proceeding with submission."
        fi
    fi
    # --- End Check ---

    # --- Delete existing logs for this combo before submitting ---
    # Construct log file prefix pattern
    MODEL_NAME_FOR_LOG=${MODEL_NAME##*/}
    LOG_PREFIX="${TASK_NAME}_${MODEL_NAME_FOR_LOG}_"
    echo " ---> Cleaning up previous logs for ${TASK_NAME}_${MODEL_NAME_FOR_LOG}"
    # Clean up both .out and .err files from SLURM directories
    find "${PROJECT_ROOT}/slurm/outputs/" -maxdepth 1 -name "${LOG_PREFIX}*.out" -print -delete 2>/dev/null || echo " ---> No previous .out files found to delete."
    find "${PROJECT_ROOT}/slurm/errors/" -maxdepth 1 -name "${LOG_PREFIX}*.err" -print -delete 2>/dev/null || echo " ---> No previous .err files found to delete."
    # --- End Log Cleanup ---

    # Construct job name
    TASK_NAME_SHORT=${TASK_NAME:0:5}
    MODEL_NAME_SHORT=${MODEL_NAME%%/*} # Get part before / if present
    MODEL_NAME_SHORT=${MODEL_NAME_SHORT:0:5}
    JOB_NAME="${TASK_NAME_SHORT}_${MODEL_NAME_SHORT}"

    echo "Submitting job for Task: $TASK_NAME, Model: $MODEL_NAME (Job name: $JOB_NAME, Index: $i)"
    echo "Params: n=${TASK_N}, dataset_size=${TASK_DATASET_SIZE}, target_time_ms=${TASK_TARGET_TIME_MS}, Data: ${RESOLVED_DATASET_PATH}"

    # Submit individual job, exporting necessary variables
    # Ensure RUNSCRIPT_PATH points to the correct agent runner script
    RUNSCRIPT_PATH="${PROJECT_ROOT}/scripts/slurm_jobs/agent.sh"
    if [ ! -f "$RUNSCRIPT_PATH" ]; then
         echo "Error: Agent runscript not found at $RUNSCRIPT_PATH"
         exit 1
    fi

    JOB_ID=$(sbatch --parsable \
                    --job-name="${JOB_NAME}" \
                    --partition="${SLURM_PARTITIONS_DEFAULT}" \
                    --time=48:00:00 \
                    --output="${PROJECT_ROOT}/slurm/outputs/${LOG_PREFIX}%j.out" \
                    --error="${PROJECT_ROOT}/slurm/errors/${LOG_PREFIX}%j.err" \
                    --export=ALL,TASK_NAME="${TASK_NAME}",MODEL="${MODEL_NAME}",TASK_N="${TASK_N}",TASK_DATASET_SIZE="${TASK_DATASET_SIZE}",TASK_TARGET_TIME_MS="${TASK_TARGET_TIME_MS}",DATASET_PATH="${RESOLVED_DATASET_PATH}",REPORTS_DIR="${REPORTS_DIR}",SUMMARY_FILE="${SUMMARY_FILE}" \
                    "$RUNSCRIPT_PATH")
    JOB_IDS+=("$JOB_ID")
    SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
    echo "Submitted job with ID: $JOB_ID"
done

echo "-----------------------------------"
echo "Total tasks considered: $NUM_TASKS"
echo "Jobs Submitted: $SUBMITTED_COUNT"
echo "Jobs Skipped (result found in agent summary): $SKIPPED_COUNT" # Clarified skip reason
echo "Job IDs Submitted: ${JOB_IDS[*]}"
echo "-----------------------------------"