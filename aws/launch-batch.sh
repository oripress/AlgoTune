#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

#######################################################
# AlgoTune AWS Batch Launcher
# One script to launch optimization jobs on AWS Batch
#######################################################

# Avoid interactive AWS CLI pager hangs during cleanup.
export AWS_PAGER=""

echo "╔════════════════════════════════════════╗"
echo "║  AlgoTune AWS Batch Launcher           ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Resolve paths relative to this script so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

#######################################################
# Interactive Configuration
#######################################################

echo "→ Configuring batch job..."
echo ""

# Prompt for model
echo "Which model do you want to use?"
echo ""
echo "Available models (from config.yaml):"
CONFIG_PATH="$ROOT_DIR/AlgoTuner/config/config.yaml"
# Extract model names from config.yaml and display numbered
MODEL_SELECTION=$(python3 -c "
import yaml
import sys
try:
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    models = list(config.get('models', {}).keys())
    # Display numbered from bottom to top (reverse order)
    for i, model in enumerate(reversed(models), 1):
        # Strip openrouter/provider/ prefix for display
        display_name = model.split('/')[-1] if '/' in model else model
        model_cfg = config.get('models', {}).get(model, {})
        effort = (model_cfg.get('reasoning') or {}).get('effort')
        if effort == 'medium':
            display_name = f\"{display_name} (medium)\"
        print(f'{i}|{model}|{display_name}')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" "$CONFIG_PATH")

if [ $? -ne 0 ]; then
  echo "❌ ERROR: Could not load models from config.yaml"
  exit 1
fi

# Display models
echo "$MODEL_SELECTION" | while IFS='|' read -r num model display_name; do
  echo "  $num) $display_name"
done

echo ""
read -p "Model (number or full name): " MODEL_INPUT

if [ -z "$MODEL_INPUT" ]; then
  echo "❌ ERROR: Model is required"
  exit 1
fi

# Check if input is a number
if [[ "$MODEL_INPUT" =~ ^[0-9]+$ ]]; then
  # User entered a number - get the model name
  MODEL=$(echo "$MODEL_SELECTION" | awk -F'|' -v num="$MODEL_INPUT" '$1 == num {print $2}')
  if [ -z "$MODEL" ]; then
    echo "❌ ERROR: Invalid model number: $MODEL_INPUT"
    exit 1
  fi
  echo "  → Selected: $MODEL"
else
  # User entered the full name
  MODEL="$MODEL_INPUT"
fi
MODEL_DISPLAY="${MODEL##*/}"

# Prompt for task selection
echo ""
echo "Which tasks do you want to run?"
echo "  1) All tasks (154 tasks)"
echo "  2) Specific tasks (comma-separated)"
echo ""
read -p "Choice [1/2]: " TASK_CHOICE

while true; do
  case "$TASK_CHOICE" in
    1)
      TASK_ARGS="--all-tasks"
      echo "  → Will run all 154 tasks"
      break
      ;;
    2)
      while true; do
        echo "Enter tasks separated by commas (e.g., svm,pca,kmeans):"
        read -p "Tasks: " TASKS_INPUT
        if [ -z "$TASKS_INPUT" ]; then
          echo "❌ ERROR: Task list cannot be empty"
          continue
        fi

        # Validate that all tasks exist
        INVALID_TASKS=""
        IFS=',' read -ra TASK_ARRAY <<< "$TASKS_INPUT"
        for task in "${TASK_ARRAY[@]}"; do
          task=$(echo "$task" | xargs)  # trim whitespace
          if [ ! -d "$ROOT_DIR/AlgoTuneTasks/$task" ]; then
            INVALID_TASKS="$INVALID_TASKS $task"
          fi
        done

        if [ -n "$INVALID_TASKS" ]; then
          echo "❌ ERROR: The following tasks do not exist:$INVALID_TASKS"

          # For each invalid task, show closest matches using edit distance
          for invalid in $INVALID_TASKS; do
            echo ""
            echo "   Did you mean one of these? (closest matches for '$invalid'):"
            python3 - <<PYTHON "$invalid" "$ROOT_DIR"
import sys
from pathlib import Path

def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

invalid_task = sys.argv[1]
root_dir = sys.argv[2]
tasks_dir = Path(root_dir) / "AlgoTuneTasks"

# Get all task directories
tasks = [d.name for d in tasks_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]

# Calculate edit distance for each task
distances = [(task, edit_distance(invalid_task.lower(), task.lower())) for task in tasks]
distances.sort(key=lambda x: x[1])

# Show top 5 closest matches
for task, dist in distances[:5]:
    print(f"   - {task}")
PYTHON
          done
          echo ""
          continue
        fi

        TASK_ARGS="--tasks $TASKS_INPUT"
        echo "  → Will run specified tasks"
        break
      done
      break
      ;;
    *)
      echo "❌ ERROR: Invalid choice"
      read -p "Choice [1/2]: " TASK_CHOICE
      ;;
  esac
done

echo ""

# Check for already-completed tasks
echo "→ Checking for already-completed tasks..."
# First, get the task list to check
case "$TASK_CHOICE" in
  1)
    # Get all tasks from AlgoTuneTasks directory
    ALL_TASKS_LIST=$(find "$ROOT_DIR/AlgoTuneTasks" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | grep -v __pycache__ | sort)
    ;;
  2)
    # For specific tasks, use the provided list
    ALL_TASKS_LIST=$(echo "$TASKS_INPUT" | tr ',' '\n')
    ;;
esac

# Check for completed tasks
SKIP_COMPLETED="no"
COMPLETED_COUNT=$(echo "$ALL_TASKS_LIST" | python3 "$SCRIPT_DIR/check_completed.py" --model "$MODEL" --quiet | wc -l)
TOTAL_COUNT=$(echo "$ALL_TASKS_LIST" | wc -l)
ALREADY_DONE=$((TOTAL_COUNT - COMPLETED_COUNT))

if [ "$ALREADY_DONE" -gt 0 ]; then
  echo "  Found $ALREADY_DONE completed task(s) for model: $MODEL"
  echo ""
  read -p "Skip already-completed tasks? [Y/n]: " SKIP_CHOICE
  if [[ ! "$SKIP_CHOICE" =~ ^[Nn]$ ]]; then
    SKIP_COMPLETED="yes"
    echo "  ✓ Will skip $ALREADY_DONE completed task(s)"

    # Filter the task list with interactive prompt for N/A tasks
    FILTERED_TASKS=$(echo "$ALL_TASKS_LIST" | python3 "$SCRIPT_DIR/check_completed.py" --model "$MODEL" --interactive)
    TASKS_INPUT=$(echo "$FILTERED_TASKS" | tr '\n' ',' | sed 's/,$//')
    TASK_ARGS="--tasks $TASKS_INPUT"
    TASK_LIST="$FILTERED_TASKS"

    if [ -z "$TASKS_INPUT" ]; then
      echo ""
      echo "  ✓ All tasks already completed! Nothing to do."
      exit 0
    fi
  else
    echo "  → Will run all tasks (including completed ones)"
  fi
else
  echo "  ✓ No completed tasks found, will run all selected tasks"
  NO_COMPLETED_TASKS="yes"
fi

if [ -z "${TASK_LIST:-}" ]; then
  TASK_LIST="$ALL_TASKS_LIST"
fi

echo ""

if [ "${NO_COMPLETED_TASKS:-}" != "yes" ]; then
  read -p "Retry N/A runs? [y/N]: " RETRY_NA
  if [[ "$RETRY_NA" =~ ^[Yy]$ ]]; then
    echo "→ Preparing retry for N/A tasks..."
    RETRY_TASKS_FILE="$(mktemp)"
    python3 "$SCRIPT_DIR/retry_na.py" \
      --model "$MODEL" \
      --model-display "$MODEL_DISPLAY" \
      --logs-dir "$ROOT_DIR/logs" \
      --job-map "$ROOT_DIR/reports/job_map.json" \
      --summary-file "$ROOT_DIR/reports/agent_summary.json" \
      --out "$RETRY_TASKS_FILE" || true

    if [ -s "$RETRY_TASKS_FILE" ]; then
      TASKS_INPUT=$(tr '\n' ',' < "$RETRY_TASKS_FILE" | sed 's/,$//')
      TASK_ARGS="--tasks $TASKS_INPUT"
      TASK_LIST=$(cat "$RETRY_TASKS_FILE")
      echo "  ✓ Will retry $(wc -l < "$RETRY_TASKS_FILE") task(s)"
    else
      echo "  ✓ No N/A tasks to retry"
    fi
    rm -f "$RETRY_TASKS_FILE"
  fi
fi

# Check aws/.env exists (AWS-specific config)
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  echo "❌ ERROR: aws/.env file not found"
  echo "   Run ./aws/setup-aws.sh to configure AWS Batch"
  exit 1
fi

# Load environment variables
set -a

# Load root .env for API keys (if exists)
if [ -f "$ROOT_DIR/.env" ]; then
  source "$ROOT_DIR/.env"
fi

# Load aws/.env for AWS configuration
source "$SCRIPT_DIR/.env"

set +a

# Verify AWS credentials work
echo "→ Verifying AWS credentials..."
if ! aws sts get-caller-identity &>/dev/null; then
  echo "❌ ERROR: AWS credentials invalid"
  echo "   Check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in aws/.env"
  exit 1
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "  ✓ Authenticated as account: $ACCOUNT_ID"
echo ""

# Job queues
SPOT_QUEUE_NAME="${BATCH_JOB_QUEUE_NAME_SPOT:-$BATCH_JOB_QUEUE_NAME}"
ONDEMAND_QUEUE_NAME="${BATCH_JOB_QUEUE_NAME_ONDEMAND:-$BATCH_JOB_QUEUE_NAME}"

# Instance types for pricing display
ONDEMAND_INSTANCE_TYPES="${BATCH_INSTANCE_TYPES_ONDEMAND:-${BATCH_INSTANCE_TYPES:-}}"
SPOT_INSTANCE_TYPES="${BATCH_INSTANCE_TYPES_SPOT:-${BATCH_INSTANCE_TYPES:-}}"
ONDEMAND_INSTANCE_TYPE="$(echo "$ONDEMAND_INSTANCE_TYPES" | cut -d',' -f1 | xargs)"
SPOT_INSTANCE_TYPE="$(echo "$SPOT_INSTANCE_TYPES" | cut -d',' -f1 | xargs)"

fetch_price() {
  local instance_type="$1"
  local capacity_type="$2"
  if [ -z "$instance_type" ]; then
    echo ""
    return
  fi
  python3 "$SCRIPT_DIR/get_instance_pricing.py" \
    --instance-type "$instance_type" \
    --region "$AWS_REGION" \
    --capacity-type "$capacity_type" 2>/dev/null || true
}

format_price_line() {
  local instance_type="$1"
  local price="$2"
  if [ -n "$instance_type" ] && [ -n "$price" ]; then
    printf "%s @ $%.4f/hr" "$instance_type" "$price"
  elif [ -n "$instance_type" ]; then
    printf "%s @ N/A" "$instance_type"
  else
    printf "N/A"
  fi
}

ONDEMAND_PRICE_PER_HOUR="$(fetch_price "$ONDEMAND_INSTANCE_TYPE" "on-demand")"
SPOT_PRICE_PER_HOUR="$(fetch_price "$SPOT_INSTANCE_TYPE" "spot")"

# Choose capacity type
echo "Which capacity type do you want to use?"
echo "  1) On-Demand (default) - $(format_price_line "$ONDEMAND_INSTANCE_TYPE" "$ONDEMAND_PRICE_PER_HOUR")"
if [ -n "${BATCH_JOB_QUEUE_NAME_SPOT:-}" ]; then
  echo "  2) Spot (cheaper, may interrupt; auto-retry on On-Demand) - $(format_price_line "$SPOT_INSTANCE_TYPE" "$SPOT_PRICE_PER_HOUR")"
else
  echo "  2) Spot (cheaper, may interrupt; auto-retry on On-Demand) - (spot queue not configured)"
fi
echo ""
read -p "Choice [1/2]: " CAPACITY_CHOICE

CAPACITY_TYPE="on-demand"
CAPACITY_LABEL="On-Demand"
JOB_QUEUE_NAME="$ONDEMAND_QUEUE_NAME"
INSTANCE_TYPES="$ONDEMAND_INSTANCE_TYPES"
INSTANCE_TYPE="$ONDEMAND_INSTANCE_TYPE"
PRICE_PER_HOUR="$ONDEMAND_PRICE_PER_HOUR"

case "$CAPACITY_CHOICE" in
  2)
    if [ -n "${BATCH_JOB_QUEUE_NAME_SPOT:-}" ]; then
      CAPACITY_TYPE="spot"
      CAPACITY_LABEL="Spot"
      JOB_QUEUE_NAME="$SPOT_QUEUE_NAME"
      INSTANCE_TYPES="$SPOT_INSTANCE_TYPES"
      INSTANCE_TYPE="$SPOT_INSTANCE_TYPE"
      PRICE_PER_HOUR="$SPOT_PRICE_PER_HOUR"
    else
      echo "⚠️  Spot queue not configured; using On-Demand."
    fi
    ;;
  ""|1)
    ;;
  *)
    echo "⚠️  Invalid choice; using On-Demand."
    ;;
esac

echo "  → Using $CAPACITY_LABEL queue: $JOB_QUEUE_NAME"
echo ""

# Count how many tasks will be submitted
TASK_COUNT=$(echo "$TASK_LIST" | wc -l)

format_cost_display() {
  local instances="$1"
  if [ -z "$PRICE_PER_HOUR" ]; then
    echo "N/A"
    return
  fi
  local raw
  raw=$(echo "scale=4; $PRICE_PER_HOUR * $instances" | bc -l)
  if [ -z "$raw" ]; then
    echo "N/A"
    return
  fi
  printf "\$%.2f/hr" "$raw"
}

if [ -n "$PRICE_PER_HOUR" ]; then
  printf "  Pricing: %s @ $%.4f/hr (%s, %s)\n" "$INSTANCE_TYPE" "$PRICE_PER_HOUR" "$CAPACITY_LABEL" "$AWS_REGION"
else
  echo "  Pricing: unavailable (AWS Pricing API/spot history)"
fi
echo ""

echo "→ Configuring parallelism for $TASK_COUNT task(s)..."
echo ""
echo "How many jobs should run in parallel?"
echo "  (Each job gets: 1 vCPU, 16 GB RAM - matching SLURM)"
echo ""
MAX_INSTANCES=$(( ($TASK_COUNT + 3) / 4 ))
echo "  1) Low parallelism   (~16 jobs)   [~4 instances,  $(format_cost_display 4)]"
echo "  2) Medium parallelism (~32 jobs)  [~8 instances,  $(format_cost_display 8)]"
echo "  3) High parallelism  (~64 jobs)   [~16 instances, $(format_cost_display 16)]"
echo "  4) Maximum throughput (all $TASK_COUNT jobs) [~$MAX_INSTANCES instances, $(format_cost_display "$MAX_INSTANCES")]"
echo "  5) Custom (specify vCPUs)"
echo ""
read -p "Choice [1/2/3/4/5]: " PARALLEL_CHOICE

case "$PARALLEL_CHOICE" in
  1)
    BATCH_MAX_VCPUS=32
    PARALLEL_JOBS=16
    echo "  → Configured for ~16 parallel jobs"
    ;;
  2)
    BATCH_MAX_VCPUS=64
    PARALLEL_JOBS=32
    echo "  → Configured for ~32 parallel jobs"
    ;;
  3)
    BATCH_MAX_VCPUS=128
    PARALLEL_JOBS=64
    echo "  → Configured for ~64 parallel jobs"
    ;;
  4)
    # Calculate vCPUs needed for all tasks
    # r6a.2xlarge has 8 vCPU and 64 GB, can run 4 jobs (RAM limited)
    # So need (tasks / 4) instances, each with 8 vCPUs
    INSTANCES_NEEDED=$(( ($TASK_COUNT + 3) / 4 ))  # Round up
    BATCH_MAX_VCPUS=$(( $INSTANCES_NEEDED * 8 ))
    PARALLEL_JOBS=$TASK_COUNT
    echo "  → Configured for all $TASK_COUNT jobs in parallel"
    ;;
  5)
    read -p "Enter max vCPUs (must be multiple of 8): " BATCH_MAX_VCPUS
    PARALLEL_JOBS=$(( $BATCH_MAX_VCPUS / 2 ))
    echo "  → Configured for ~$PARALLEL_JOBS parallel jobs"
    ;;
  *)
    echo "❌ ERROR: Invalid choice, using default (32 parallel jobs)"
    BATCH_MAX_VCPUS=64
    PARALLEL_JOBS=32
    ;;
esac

echo ""
if [ -n "$INSTANCE_TYPE" ]; then
  echo "  Instance type: $INSTANCE_TYPE (8 vCPU, 64 GB)"
elif [ -n "$INSTANCE_TYPES" ]; then
  echo "  Instance type: $INSTANCE_TYPES (8 vCPU, 64 GB)"
else
  echo "  Instance type: unknown"
fi
echo "  Per job: 1 vCPU, 16 GB (matches SLURM)"
echo "  Max parallel: ~$PARALLEL_JOBS jobs"
echo "  Max vCPUs: $BATCH_MAX_VCPUS"

# Export for use by create_batch_resources.py
export BATCH_MAX_VCPUS

echo ""

# Ctrl+C kill switch is always enabled
DEADMAN_SWITCH="yes"
COMPUTE_CLEANUP_DONE=""
echo "  ✓ Ctrl+C kill switch enabled (will cancel jobs and stop EC2 instances)"
echo ""

# Verify Docker image is configured
echo "→ Checking Docker image configuration..."
if [ -z "$DOCKER_IMAGE" ]; then
  echo "❌ ERROR: DOCKER_IMAGE not set in aws/.env"
  echo "   Run ./aws/setup-aws.sh to configure"
  exit 1
fi
echo "  ✓ Docker image: $DOCKER_IMAGE"

# Only show build instructions if using a custom image
if [[ "$DOCKER_IMAGE" != "ghcr.io/oripress/algotune:latest" ]]; then
  echo ""
  echo "  Note: Using custom image. Make sure it's built and pushed:"
  echo "    $SCRIPT_DIR/build-image.sh"
  echo "    docker tag algotune:latest $DOCKER_IMAGE"
  echo "    docker push $DOCKER_IMAGE"
fi
echo ""

# Create IAM roles (idempotent)
echo "→ Setting up IAM roles..."
bash "$SCRIPT_DIR/create_iam_roles.sh"
echo ""

# Create AWS Batch service-linked role (required for Batch compute environments)
echo "→ Creating AWS Batch service-linked role (if needed)..."
if aws iam get-role --role-name AWSServiceRoleForBatch &>/dev/null; then
  echo "  ✓ Service-linked role already exists"
elif aws iam create-service-linked-role --aws-service-name batch.amazonaws.com &>/dev/null; then
  echo "  ✓ Service-linked role created"
else
  echo "  ❌ ERROR: Service-linked role does not exist and cannot be created"
  echo ""
  echo "  Your IAM user lacks 'iam:CreateServiceLinkedRole' permission."
  echo ""
  echo "  Fix options:"
  echo "  1) Add this IAM policy to your user:"
  echo "     {"
  echo "       \"Effect\": \"Allow\","
  echo "       \"Action\": \"iam:CreateServiceLinkedRole\","
  echo "       \"Resource\": \"arn:aws:iam::*:role/aws-service-role/batch.amazonaws.com/*\","
  echo "       \"Condition\": {"
  echo "         \"StringLike\": {"
  echo "           \"iam:AWSServiceName\": \"batch.amazonaws.com\""
  echo "         }"
  echo "       }"
  echo "     }"
  echo ""
  echo "  2) Or have an admin run this AWS CLI command:"
  echo "     aws iam create-service-linked-role --aws-service-name batch.amazonaws.com"
  echo ""
  exit 1
fi
echo ""

# Create Batch resources (idempotent - will fail gracefully if exists)
echo "→ Setting up AWS Batch resources..."
if python3 "$SCRIPT_DIR/create_batch_resources.py" 2>&1; then
  echo "  ✓ Batch resources ready"
else
  # Check if it's just because they already exist
  if aws batch describe-compute-environments \
      --compute-environments "${BATCH_COMPUTE_ENV_NAME:-AlgoTuneCE}" \
      --region "$AWS_REGION" &>/dev/null; then
    echo "  ✓ Batch resources already exist"
  else
    echo "❌ ERROR: Failed to create Batch resources"
    exit 1
  fi
fi
echo ""

# Submit jobs
echo "→ Submitting jobs to AWS Batch..."
JOB_IDS_FILE="$(mktemp /tmp/algotune_job_ids_XXXXXX)"
SPOT_JOB_IDS_FILE=""
if [ "$CAPACITY_TYPE" = "spot" ]; then
  SPOT_JOB_IDS_FILE="$JOB_IDS_FILE"
fi
if python3 "$SCRIPT_DIR/submit_jobs.py" --model "$MODEL" $TASK_ARGS --s3-bucket "$S3_RESULTS_BUCKET" --job-queue "$JOB_QUEUE_NAME" > "$JOB_IDS_FILE"; then
  JOB_COUNT=$(wc -l < "$JOB_IDS_FILE" || echo "0")
  if [ "$JOB_COUNT" -gt 0 ]; then
    echo "  ✓ Submitted $JOB_COUNT job(s)"
  else
    echo "⚠️  No jobs submitted (check aws/submit_jobs.py configuration)"
    exit 0
  fi
else
  echo "❌ ERROR: Failed to submit jobs"
  exit 1
fi
echo ""

# Shutdown Batch compute environments and instances (safe to call multiple times)
shutdown_compute_resources() {
  set +e

  if [ -n "${COMPUTE_CLEANUP_DONE:-}" ]; then
    return 0
  fi
  COMPUTE_CLEANUP_DONE="yes"

  echo ""
  echo "→ Shutting down AWS Batch compute resources..."

  local ce_names=()
  local -A seen_ce=()

  add_ce() {
    local name="$1"
    if [ -z "$name" ]; then
      return
    fi
    if [ -z "${seen_ce[$name]+x}" ]; then
      seen_ce["$name"]=1
      ce_names+=("$name")
    fi
  }

  add_ce "${BATCH_COMPUTE_ENV_NAME:-}"
  add_ce "${BATCH_COMPUTE_ENV_NAME_SPOT:-}"
  add_ce "${BATCH_COMPUTE_ENV_NAME_ONDEMAND:-}"

  if [ "${#ce_names[@]}" -eq 0 ]; then
    # Fallback: scale down all compute environments
    all_ce_names=$(aws batch describe-compute-environments \
      --region "$AWS_REGION" \
      --query 'computeEnvironments[*].computeEnvironmentName' \
      --output text 2>/dev/null || echo "")
    if [ -n "$all_ce_names" ]; then
      for ce_name in $all_ce_names; do
        add_ce "$ce_name"
      done
    fi
  fi

  if [ "${#ce_names[@]}" -eq 0 ]; then
    echo "  → No compute environments found"
  else
    echo "  → Found compute environments: ${ce_names[*]}"
    for CE_NAME in "${ce_names[@]}"; do
      echo "  → Processing: $CE_NAME"
      if aws batch update-compute-environment \
        --compute-environment "$CE_NAME" \
        --state DISABLED \
        --compute-resources '{"minvCpus":0}' \
        --region "$AWS_REGION" >/dev/null 2>&1; then
        echo "    ✓ Disabled and set minvCpus=0"
      else
        echo "    ⚠️  Warning: Failed to update compute environment"
      fi
    done
  fi

  # Directly terminate any running EC2 instances managed by Batch
  echo "  → Searching for EC2 instances to terminate..."

  # Get current instance ID to avoid terminating ourselves
  CURRENT_INSTANCE=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
  if [ -n "$CURRENT_INSTANCE" ]; then
    echo "  → Protecting current instance: $CURRENT_INSTANCE"
  else
    echo "  → Not running on EC2 (no instance to protect)"
  fi

  INSTANCE_IDS=""
  if [ "${#ce_names[@]}" -gt 0 ]; then
    ce_filter_values=$(IFS=,; echo "${ce_names[*]}")
    INSTANCE_IDS=$(aws ec2 describe-instances \
      --filters "Name=tag:aws:batch:compute-environment,Values=$ce_filter_values" \
                "Name=instance-state-name,Values=running,pending" \
      --query 'Reservations[*].Instances[*].InstanceId' \
      --output text \
      --region "$AWS_REGION" 2>/dev/null || echo "")
  fi

  if [ -z "$INSTANCE_IDS" ]; then
    INSTANCE_IDS=$(aws ec2 describe-instances \
      --filters "Name=tag:AWSBatchServiceTag,Values=batch" \
                "Name=instance-state-name,Values=running,pending" \
      --query 'Reservations[*].Instances[*].InstanceId' \
      --output text \
      --region "$AWS_REGION" 2>/dev/null || echo "")
  fi

  if [ -n "$INSTANCE_IDS" ]; then
    echo "  → Found instances: $INSTANCE_IDS"
  else
    echo "  → No Batch instances found"
  fi

  # Filter out current instance if we're running on EC2
  FILTERED_INSTANCE_IDS=""
  if [ -n "$INSTANCE_IDS" ]; then
    for instance in $INSTANCE_IDS; do
      if [ "$instance" != "$CURRENT_INSTANCE" ]; then
        FILTERED_INSTANCE_IDS="$FILTERED_INSTANCE_IDS $instance"
      else
        echo "  → Skipping current instance: $instance"
      fi
    done
    FILTERED_INSTANCE_IDS=$(echo "$FILTERED_INSTANCE_IDS" | xargs)  # Trim whitespace
  fi

  if [ -n "$FILTERED_INSTANCE_IDS" ]; then
    echo "  → Terminating instances: $FILTERED_INSTANCE_IDS"
    if aws ec2 terminate-instances \
      --instance-ids $FILTERED_INSTANCE_IDS \
      --region "$AWS_REGION" 2>&1; then
      echo "  ✓ EC2 instances terminated successfully"
    else
      echo "  ⚠️  Warning: Failed to terminate some instances"
    fi
  else
    echo "  ✓ No Batch EC2 instances to terminate (excluding current instance)"
  fi

  echo "✓ Compute shutdown complete"
}

# Ctrl+C kill switch for submitted jobs (optional)
cancel_submitted_jobs() {
  set +e

  local job_ids=()
  local -A seen_job_ids=()
  local queues=()
  local -A seen_queues=()

  local job_name_prefix="algotune-"
  local model_suffix="--$(echo "$MODEL" | tr '/:.' '_')"

  add_job_id() {
    local job_id="$1"
    if [ -z "$job_id" ]; then
      return
    fi
    if [ -z "${seen_job_ids[$job_id]+x}" ]; then
      seen_job_ids["$job_id"]=1
      job_ids+=("$job_id")
    fi
  }

  add_queue() {
    local queue="$1"
    if [ -z "$queue" ]; then
      return
    fi
    if [ -z "${seen_queues[$queue]+x}" ]; then
      seen_queues["$queue"]=1
      queues+=("$queue")
    fi
  }

  add_job_ids_from_file() {
    local file="$1"
    if [ -z "$file" ] || [ ! -f "$file" ]; then
      return
    fi
    while IFS= read -r line; do
      line=$(echo "$line" | xargs)
      [ -z "$line" ] && continue
      add_job_id "$line"
    done < "$file"
  }

  echo ""
  echo "→ Canceling queued jobs and terminating running jobs..."
  echo "  → Collecting jobs from files and AWS Batch queues..."

  add_job_ids_from_file "${JOB_IDS_FILE:-}"
  add_job_ids_from_file "${SPOT_JOB_IDS_FILE:-}"
  add_job_ids_from_file "${ONDEMAND_JOB_IDS_FILE:-}"

  add_queue "${JOB_QUEUE_NAME:-}"
  add_queue "${SPOT_QUEUE_NAME:-}"
  add_queue "${ONDEMAND_QUEUE_NAME:-}"

  if [ "${#queues[@]}" -gt 0 ]; then
    for queue in "${queues[@]}"; do
      for status in SUBMITTED PENDING RUNNABLE STARTING RUNNING; do
        listed_pairs=$(aws batch list-jobs \
          --job-queue "$queue" \
          --job-status "$status" \
          --region "$AWS_REGION" \
          --query 'jobSummaryList[*].[jobId,jobName]' \
          --output text 2>/dev/null || echo "")
        if [ -z "$listed_pairs" ]; then
          continue
        fi
        while read -r job_id job_name; do
          [ -z "$job_id" ] && continue
          if [[ "$job_name" != ${job_name_prefix}* ]]; then
            continue
          fi
          if [ -n "$model_suffix" ] && [[ "$job_name" != *"$model_suffix" ]]; then
            continue
          fi
          add_job_id "$job_id"
        done <<< "$listed_pairs"
      done
    done
  fi

  if [ "${#job_ids[@]}" -eq 0 ]; then
    echo "  ⚠️  No matching jobs found to cancel/terminate."
  else
    local total="${#job_ids[@]}"
    echo "  → Processing $total job(s)"

    local chunk_size=100
    local i=0
    local cancel_count=0
    local terminate_count=0
    local completed_count=0

    while [ $i -lt $total ]; do
      local chunk=("${job_ids[@]:i:chunk_size}")
      local job_lines

      job_json=$(aws batch describe-jobs \
        --jobs "${chunk[@]}" \
        --region "$AWS_REGION" \
        --output json 2>/dev/null)
      if [ $? -ne 0 ] || [ -z "$job_json" ]; then
        echo "⚠️  Warning: Failed to fetch job statuses. Attempting cancel/terminate for all in chunk."
        for job_id in "${chunk[@]}"; do
          aws batch cancel-job \
            --job-id "$job_id" \
            --reason "Ctrl+C kill switch" \
            --region "$AWS_REGION" 2>&1 | grep -v "ClientException" || true
          aws batch terminate-job \
            --job-id "$job_id" \
            --reason "Ctrl+C kill switch" \
            --region "$AWS_REGION" 2>&1 | grep -v "ClientException" || true
          cancel_count=$((cancel_count + 1))
        done
        i=$((i + chunk_size))
        continue
      fi

      job_lines=$(printf "%s" "$job_json" | python3 - <<'PY'
import json
import sys

raw = sys.stdin.read().strip()
if not raw:
    sys.exit(0)
payload = json.loads(raw)
for job in payload.get("jobs", []):
    job_id = job.get("jobId", "")
    status = job.get("status", "")
    if job_id and status:
        print(f"{job_id} {status}")
PY
)

      while read -r job_id status; do
        [ -z "$job_id" ] && continue  # Skip empty lines
        case "$status" in
          SUBMITTED|PENDING|RUNNABLE)
            echo "  → Canceling job $job_id (status: $status)"
            if aws batch cancel-job \
              --job-id "$job_id" \
              --reason "Ctrl+C kill switch" \
              --region "$AWS_REGION" 2>&1 | grep -v "ClientException"; then
              cancel_count=$((cancel_count + 1))
            fi
            ;;
          STARTING|RUNNING)
            echo "  → Terminating job $job_id (status: $status)"
            if aws batch terminate-job \
              --job-id "$job_id" \
              --reason "Ctrl+C kill switch" \
              --region "$AWS_REGION" 2>&1 | grep -v "ClientException"; then
              terminate_count=$((terminate_count + 1))
            fi
            ;;
          SUCCEEDED|FAILED)
            completed_count=$((completed_count + 1))
            ;;
          *)
            echo "  → Job $job_id in state $status (no action)"
            ;;
        esac
      done <<< "$job_lines"

      i=$((i + chunk_size))
    done

    echo "✓ Cancellation/termination requests sent"
    echo "  → Canceled: $cancel_count, Terminated: $terminate_count, Already completed: $completed_count"

    # Wait for jobs to actually be canceled/terminated
    echo "→ Waiting for jobs to transition to terminal states (max 120s)..."
    local max_wait=120  # Wait up to 2 minutes
    local wait_count=0
    local jobs_still_active=1
    local last_active_count=0

    while [ $wait_count -lt $max_wait ] && [ $jobs_still_active -eq 1 ]; do
      sleep 1
      wait_count=$((wait_count + 1))

      # Check if any jobs are still in non-terminal states
      jobs_still_active=0
      local check_chunk_size=100
      local check_i=0
      local total_active=0

      while [ $check_i -lt $total ]; do
        local check_chunk=("${job_ids[@]:check_i:check_chunk_size}")

        local check_json=$(aws batch describe-jobs \
          --jobs "${check_chunk[@]}" \
          --region "$AWS_REGION" \
          --output json 2>/dev/null || echo "")

        if [ -n "$check_json" ]; then
          local active_count=$(printf "%s" "$check_json" | python3 - <<'PY'
import json
import sys

raw = sys.stdin.read().strip()
if not raw:
    print("0")
    sys.exit(0)

payload = json.loads(raw)
active = 0
for job in payload.get("jobs", []):
    status = job.get("status", "")
    # Count jobs that are still active (not in terminal state)
    if status in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]:
        active += 1
print(active)
PY
)
          if [ "$active_count" -gt 0 ]; then
            jobs_still_active=1
            total_active=$((total_active + active_count))
          fi
        fi

        check_i=$((check_i + check_chunk_size))
      done

      if [ $jobs_still_active -eq 1 ]; then
        if [ $total_active -ne $last_active_count ] || [ $((wait_count % 5)) -eq 0 ]; then
          echo "  → Still $total_active jobs active after ${wait_count}s..."
          last_active_count=$total_active
        fi
      fi
    done

    if [ $jobs_still_active -eq 0 ]; then
      echo "✓ All jobs transitioned to terminal states"
    else
      echo "⚠️  Timeout: $total_active jobs still active after ${wait_count}s"
      echo "  → Proceeding with compute environment shutdown anyway"
    fi
  fi

  shutdown_compute_resources
  echo ""
  echo "✓ All cleanup actions completed"
}

# Start automatic log downloader in background
echo "→ Starting automatic log downloader..."
echo "  AWS outputs/errors: $ROOT_DIR/aws/outputs and $ROOT_DIR/aws/errors"
echo "  Application logs: $ROOT_DIR/logs (SLURM-compatible)"
echo "  Updating every 60 seconds in the background"
python3 -u "$SCRIPT_DIR/download_logs.py" \
  --job-ids-file "$JOB_IDS_FILE" \
  --watch \
  --interval 60 \
  --cleanup-failed \
  --output-dir "$ROOT_DIR" \
  > /dev/null 2>&1 &
LOG_DOWNLOADER_PID=$!
echo "  ✓ Log downloader started (PID: $LOG_DOWNLOADER_PID)"
echo ""
echo "  Note: You should start seeing logs appear in the logs/ directory."
echo "        If not, the jobs are likely still generating datasets - this can take 1-2 hours."
echo ""

# Setup cleanup on exit
cleanup_downloader() {
  if [ -n "$LOG_DOWNLOADER_PID" ]; then
    kill $LOG_DOWNLOADER_PID 2>/dev/null || true
    echo ""
    echo "✓ Stopped log downloader"
  fi
  for ids_file in "${JOB_IDS_FILE:-}" "${SPOT_JOB_IDS_FILE:-}" "${ONDEMAND_JOB_IDS_FILE:-}"; do
    if [ -n "$ids_file" ] && [ -f "$ids_file" ]; then
      rm -f "$ids_file"
    fi
  done
}

cleanup_on_exit() {
  local exit_code=$?
  if [ "${DEADMAN_SWITCH:-no}" = "yes" ]; then
    if [ $exit_code -ne 0 ] && [ -n "${JOB_IDS_FILE:-}" ]; then
      cancel_submitted_jobs
    else
      shutdown_compute_resources
    fi
  fi
  cleanup_downloader
  return $exit_code
}
trap cleanup_on_exit EXIT

handle_interrupt() {
  if [ "${CLEANUP_IN_PROGRESS:-}" = "yes" ]; then
    echo ""
    echo "⚠️  Cleanup already in progress; exiting now."
    exit 130
  fi
  CLEANUP_IN_PROGRESS="yes"
  trap 'echo ""; echo "⚠️  Second interrupt received; exiting immediately."; exit 130' INT
  if [ "$DEADMAN_SWITCH" = "yes" ]; then
    cancel_submitted_jobs
  else
    echo ""
    echo "⚠️  Interrupt received. Jobs will continue running."
  fi
  exit 130
}
trap handle_interrupt INT

# Automatically monitor and generate HTML
echo "════════════════════════════════════════"
echo "Jobs submitted successfully!"
echo ""
echo "→ Starting real-time job monitor..."
if [ "$DEADMAN_SWITCH" = "yes" ]; then
  echo "  (Press Ctrl+C to stop monitoring, cancel jobs, and stop EC2 instances)"
else
  echo "  (Press Ctrl+C to stop monitoring - jobs will continue running)"
fi
echo ""
LIVE_SPOT_RETRY=false
MONITOR_RETRY_ARGS=()
if [ "$CAPACITY_TYPE" = "spot" ] \
  && [ -n "${BATCH_JOB_QUEUE_NAME_SPOT:-}" ] \
  && [ -n "${BATCH_JOB_QUEUE_NAME_ONDEMAND:-}" ]; then
  LIVE_SPOT_RETRY=true
  MONITOR_RETRY_ARGS=(
    --retry-spot
    --spot-job-ids-file "$SPOT_JOB_IDS_FILE"
    --ondemand-queue "$ONDEMAND_QUEUE_NAME"
    --model "$MODEL"
    --model-display "$MODEL_DISPLAY"
  )
fi

python3 "$SCRIPT_DIR/monitor_jobs.py" \
  --job-ids-file "$JOB_IDS_FILE" \
  --interval 30 \
  --sync-logs \
  --s3-bucket "$S3_RESULTS_BUCKET" \
  --output-dir "$ROOT_DIR" \
  --cleanup-failed \
  "${MONITOR_RETRY_ARGS[@]}"

MONITOR_EXIT_CODE=$?

# If monitor exited with 130 (Ctrl+C), trigger cleanup
if [ $MONITOR_EXIT_CODE -eq 130 ]; then
  if [ "$DEADMAN_SWITCH" = "yes" ]; then
    echo ""
    cancel_submitted_jobs
  fi
  exit 130
fi

# Retry Spot interruptions on On-Demand (only if spot/on-demand queues are configured)
if [ "$LIVE_SPOT_RETRY" = "true" ]; then
  echo "→ Spot retry handled during monitoring."
elif [ "$CAPACITY_TYPE" = "spot" ] \
  && [ -n "${BATCH_JOB_QUEUE_NAME_SPOT:-}" ] \
  && [ -n "${BATCH_JOB_QUEUE_NAME_ONDEMAND:-}" ]; then
  SPOT_RETRY_TASKS_FILE="$(mktemp /tmp/algotune_spot_retry_tasks_XXXXXX)"
  python3 "$SCRIPT_DIR/retry_spot.py" \
    --job-ids-file "$SPOT_JOB_IDS_FILE" \
    --out "$SPOT_RETRY_TASKS_FILE" \
    --model "$MODEL" \
    --model-display "$MODEL_DISPLAY" \
    --s3-bucket "$S3_RESULTS_BUCKET" \
    --output-dir "$ROOT_DIR" || true

  if [ -s "$SPOT_RETRY_TASKS_FILE" ]; then
    RETRY_COUNT=$(wc -l < "$SPOT_RETRY_TASKS_FILE" || echo "0")
    echo ""
    echo "→ Retrying $RETRY_COUNT Spot-interrupted task(s) on On-Demand..."
    TASKS_INPUT=$(tr '\n' ',' < "$SPOT_RETRY_TASKS_FILE" | sed 's/,$//')
    ONDEMAND_JOB_IDS_FILE="$(mktemp /tmp/algotune_job_ids_ondemand_XXXXXX)"
    if python3 "$SCRIPT_DIR/submit_jobs.py" --model "$MODEL" --tasks "$TASKS_INPUT" --s3-bucket "$S3_RESULTS_BUCKET" --job-queue "$ONDEMAND_QUEUE_NAME" > "$ONDEMAND_JOB_IDS_FILE"; then
      JOB_IDS_FILE="$ONDEMAND_JOB_IDS_FILE"
      echo "  ✓ Submitted On-Demand retries"
      echo ""
      echo "→ Monitoring On-Demand retries..."
      python3 "$SCRIPT_DIR/monitor_jobs.py" \
        --job-ids-file "$ONDEMAND_JOB_IDS_FILE" \
        --interval 30 \
        --sync-logs \
        --s3-bucket "$S3_RESULTS_BUCKET" \
        --output-dir "$ROOT_DIR" \
        --cleanup-failed

      MONITOR_EXIT_CODE=$?
      if [ $MONITOR_EXIT_CODE -eq 130 ]; then
        if [ "$DEADMAN_SWITCH" = "yes" ]; then
          echo ""
          cancel_submitted_jobs
        fi
        exit 130
      fi
    else
      echo "❌ ERROR: Failed to submit On-Demand retries"
    fi
  else
    echo "→ No Spot interruptions detected."
  fi
  rm -f "$SPOT_RETRY_TASKS_FILE"
else
  echo "→ Spot retry disabled (Spot/On-Demand queues not configured)."
fi

echo ""
echo "→ Analyzing local logs..."
TASKS_FILE="$(mktemp)"
printf "%s\n" "$TASK_LIST" > "$TASKS_FILE"
python3 "$SCRIPT_DIR/analyze_local_logs.py" \
    --log-dir "$ROOT_DIR/logs" \
    --model "$MODEL_DISPLAY" \
    --tasks-file "$TASKS_FILE" || true
rm -f "$TASKS_FILE"

python3 "$SCRIPT_DIR/download_logs.py" \
  --prune-local \
  --output-dir "$ROOT_DIR" \
  --summary-file "$ROOT_DIR/reports/agent_summary.json" || true

echo "════════════════════════════════════════"
echo "✅ All done!"
echo "   AWS Logs: $ROOT_DIR/aws/outputs and $ROOT_DIR/aws/errors"
echo "   Application Logs: $ROOT_DIR/logs/"
echo ""
