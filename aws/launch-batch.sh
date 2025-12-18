#!/usr/bin/env bash
set -euo pipefail

#######################################################
# AlgoTune AWS Batch Launcher
# One script to launch optimization jobs on AWS Batch
#######################################################

echo "╔════════════════════════════════════════╗"
echo "║  AlgoTune AWS Batch Launcher           ║"
echo "╚════════════════════════════════════════╝"
echo ""

#######################################################
# Interactive Configuration
#######################################################

echo "→ Configuring batch job..."
echo ""

# Prompt for model
echo "Which model do you want to use?"
echo "  Examples:"
echo "    - o4-mini"
echo "    - openrouter/anthropic/claude-3.5-sonnet"
echo "    - openrouter/qwen/qwen3-coder"
echo "    - openrouter/deepseek/deepseek-chat"
echo ""
read -p "Model: " MODEL

if [ -z "$MODEL" ]; then
  echo "❌ ERROR: Model is required"
  exit 1
fi

# Prompt for task selection
echo ""
echo "Which tasks do you want to run?"
echo "  1) All tasks (154 tasks)"
echo "  2) Specific tasks (comma-separated)"
echo ""
read -p "Choice [1/2]: " TASK_CHOICE

case "$TASK_CHOICE" in
  1)
    TASK_ARGS="--all-tasks"
    echo "  → Will run all 154 tasks"
    ;;
  2)
    echo "Enter tasks separated by commas (e.g., svm,pca,kmeans):"
    read -p "Tasks: " TASKS_INPUT
    if [ -z "$TASKS_INPUT" ]; then
      echo "❌ ERROR: Task list cannot be empty"
      exit 1
    fi
    TASK_ARGS="--tasks $TASKS_INPUT"
    echo "  → Will run specified tasks"
    ;;
  *)
    echo "❌ ERROR: Invalid choice"
    exit 1
    ;;
esac

echo ""

# Check for already-completed tasks
echo "→ Checking for already-completed tasks..."
# First, get the task list to check
case "$TASK_CHOICE" in
  1)
    # Get all tasks from AlgoTuneTasks directory
    ALL_TASKS_LIST=$(find AlgoTuneTasks -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | grep -v __pycache__ | sort)
    ;;
  2)
    # For specific tasks, use the provided list
    ALL_TASKS_LIST=$(echo "$TASKS_INPUT" | tr ',' '\n')
    ;;
esac

# Check for completed tasks
SKIP_COMPLETED="no"
COMPLETED_COUNT=$(echo "$ALL_TASKS_LIST" | python3 aws/check_completed.py --model "$MODEL" --quiet | wc -l)
TOTAL_COUNT=$(echo "$ALL_TASKS_LIST" | wc -l)
ALREADY_DONE=$((TOTAL_COUNT - COMPLETED_COUNT))

if [ "$ALREADY_DONE" -gt 0 ]; then
  echo "  Found $ALREADY_DONE completed task(s) for model: $MODEL"
  echo ""
  read -p "Skip already-completed tasks? [Y/n]: " SKIP_CHOICE
  if [[ ! "$SKIP_CHOICE" =~ ^[Nn]$ ]]; then
    SKIP_COMPLETED="yes"
    echo "  ✓ Will skip $ALREADY_DONE completed task(s)"

    # Filter the task list
    FILTERED_TASKS=$(echo "$ALL_TASKS_LIST" | python3 aws/check_completed.py --model "$MODEL" --quiet)
    TASKS_INPUT=$(echo "$FILTERED_TASKS" | tr '\n' ',' | sed 's/,$//')
    TASK_ARGS="--tasks $TASKS_INPUT"

    if [ -z "$TASKS_INPUT" ]; then
      echo "  ✓ All tasks already completed! Nothing to do."
      exit 0
    fi
  else
    echo "  → Will run all tasks (including completed ones)"
  fi
else
  echo "  ✓ No completed tasks found, will run all selected tasks"
fi

echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

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

# Check if Docker image exists in ECR
echo "→ Checking Docker image in ECR..."
if ! aws ecr describe-images \
    --repository-name algotune \
    --region "$AWS_REGION" \
    --image-ids imageTag=latest &>/dev/null; then
  echo "❌ ERROR: Docker image not found in ECR"
  echo "   Run the following to build and push:"
  echo ""
  echo "   aws ecr get-login-password --region $AWS_REGION | \\"
  echo "       docker login --username AWS --password-stdin $ECR_IMAGE_URI"
  echo "   docker build -t algotune:latest -f aws/Dockerfile ."
  echo "   docker tag algotune:latest $ECR_IMAGE_URI"
  echo "   docker push $ECR_IMAGE_URI"
  exit 1
fi
echo "  ✓ Docker image found in ECR"
echo ""

# Create IAM roles (idempotent)
echo "→ Setting up IAM roles..."
bash aws/create_iam_roles.sh
echo ""

# Create Batch resources (idempotent - will fail gracefully if exists)
echo "→ Setting up AWS Batch resources..."
if python3 aws/create_batch_resources.py 2>&1; then
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
JOB_IDS_FILE="batch_job_ids_$(date +%Y%m%d_%H%M%S).txt"
if python3 aws/submit_jobs.py --model "$MODEL" $TASK_ARGS --s3-bucket "$S3_RESULTS_BUCKET" > "$JOB_IDS_FILE"; then
  JOB_COUNT=$(wc -l < "$JOB_IDS_FILE" || echo "0")
  if [ "$JOB_COUNT" -gt 0 ]; then
    echo "  ✓ Submitted $JOB_COUNT job(s)"
    echo "  Job IDs saved to: $JOB_IDS_FILE"
  else
    echo "⚠️  No jobs submitted (check aws/submit_jobs.py configuration)"
    exit 0
  fi
else
  echo "❌ ERROR: Failed to submit jobs"
  exit 1
fi
echo ""

# Ask user if they want to wait for completion
echo "════════════════════════════════════════"
echo "Jobs submitted successfully!"
echo ""
echo "Options:"
echo "  1) Monitor jobs in real-time (updates every 30s)"
echo "  2) Wait for completion and generate HTML"
echo "  3) Exit now (jobs run in background)"
echo ""
read -p "Choice [1/2/3]: " choice

case "$choice" in
  1)
    echo ""
    echo "→ Starting real-time job monitor..."
    echo "  (Press Ctrl+C to exit monitoring)"
    echo ""
    python3 aws/monitor_jobs.py --job-ids-file "$JOB_IDS_FILE" --interval 30
    echo ""

    # After monitoring ends (all jobs done or user exits), ask about HTML generation
    echo ""
    echo "Options:"
    echo "  1) Generate HTML visualization now"
    echo "  2) Exit (generate HTML later)"
    echo ""
    read -p "Choice [1/2]: " html_choice

    if [ "$html_choice" = "1" ]; then
      echo ""
      echo "→ Generating HTML visualization..."
      RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
      python3 aws/generate_batch_html.py \
          --job-ids-file "$JOB_IDS_FILE" \
          --out-dir "$RESULTS_DIR" \
          --s3-bucket "$S3_RESULTS_BUCKET"
      echo ""

      echo "════════════════════════════════════════"
      echo "✅ All done!"
      echo "   HTML Results: $RESULTS_DIR/index.html"
      echo "   Job IDs: $JOB_IDS_FILE"
      echo ""
      echo "To view results, open: $RESULTS_DIR/index.html"
    else
      echo ""
      echo "════════════════════════════════════════"
      echo "To generate HTML visualization later:"
      echo "  python3 aws/generate_batch_html.py --job-ids-file $JOB_IDS_FILE --out-dir results/"
    fi
    ;;

  2)
    echo ""
    echo "→ Waiting for jobs to complete..."
    python3 aws/wait_for_jobs.py --job-ids-file "$JOB_IDS_FILE"
    echo ""

    echo "→ Generating HTML visualization..."
    RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
    python3 aws/generate_batch_html.py \
        --job-ids-file "$JOB_IDS_FILE" \
        --out-dir "$RESULTS_DIR" \
        --s3-bucket "$S3_RESULTS_BUCKET"
    echo ""

    echo "════════════════════════════════════════"
    echo "✅ All done!"
    echo "   HTML Results: $RESULTS_DIR/index.html"
    echo "   Job IDs: $JOB_IDS_FILE"
    echo ""
    echo "To view results, open: $RESULTS_DIR/index.html"
    ;;

  3|*)
    echo ""
    echo "════════════════════════════════════════"
    echo "✅ Jobs running in background on AWS Batch"
    echo ""
    echo "To monitor job progress:"
    echo "  python3 aws/monitor_jobs.py --job-ids-file $JOB_IDS_FILE"
    echo ""
    echo "To wait for completion:"
    echo "  python3 aws/wait_for_jobs.py --job-ids-file $JOB_IDS_FILE"
    echo ""
    echo "To generate HTML visualization after completion:"
    echo "  python3 aws/generate_batch_html.py --job-ids-file $JOB_IDS_FILE --out-dir results/"
    ;;
esac
