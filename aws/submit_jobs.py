#!/usr/bin/env python3
import os
import sys
import argparse
import boto3
import json
import time
import glob
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment from both .env files
# 1. Load API keys from root .env
root_dotenv = Path(__file__).parent.parent / ".env"
if root_dotenv.exists():
    load_dotenv(root_dotenv)

# 2. Load AWS configuration from aws/.env
aws_dotenv = Path(__file__).parent / ".env"
load_dotenv(aws_dotenv)

def setup_s3_lifecycle_policy(s3_client, bucket: str, dataset_retention_days: int = 21):
    """
    Setup S3 lifecycle policy to auto-delete old datasets.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        dataset_retention_days: Number of days to retain datasets (default: 21)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        lifecycle_config = {
            'Rules': [
                {
                    'Id': 'DeleteOldDatasets',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'datasets/'},
                    'Expiration': {'Days': dataset_retention_days}
                }
            ]
        }

        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket,
            LifecycleConfiguration=lifecycle_config
        )

        print(f"✓ S3 lifecycle policy set: datasets/ prefix will expire after {dataset_retention_days} days", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Warning: Could not set S3 lifecycle policy: {e}", file=sys.stderr)
        print(f"         Datasets will not auto-delete. You may need to clean them up manually.", file=sys.stderr)
        return False

def load_generation_config():
    """Load generation.json to get 'n' values for each task."""
    generation_file = Path(__file__).parent.parent / "reports" / "generation.json"

    if not generation_file.exists():
        print(f"WARNING: generation.json not found at {generation_file}", file=sys.stderr)
        print(f"         Will use default target-time-ms=100 for all tasks", file=sys.stderr)
        return {}

    try:
        with open(generation_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded generation config with {len(data)} tasks", file=sys.stderr)
        return data
    except Exception as e:
        print(f"WARNING: Failed to load generation.json: {e}", file=sys.stderr)
        print(f"         Will use default target-time-ms=100 for all tasks", file=sys.stderr)
        return {}


def discover_all_tasks():
    """Discover all available tasks from AlgoTuneTasks/ directory."""
    tasks_root = Path(__file__).parent.parent / "AlgoTuneTasks"
    tasks = []

    if not tasks_root.exists():
        print(f"ERROR: AlgoTuneTasks directory not found at {tasks_root}", file=sys.stderr)
        return []

    for task_dir in sorted(tasks_root.iterdir()):
        if task_dir.is_dir() and (task_dir / "description.txt").exists():
            tasks.append(task_dir.name)

    return tasks

def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit AlgoTune optimization jobs to AWS Batch"
    )

    # Model selection (required)
    parser.add_argument(
        "--model",
        required=True,
        help="Model to use (e.g., o4-mini, openrouter/anthropic/claude-3.5-sonnet)"
    )

    # Task selection (mutually exclusive)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--tasks",
        help="Comma-separated list of tasks (e.g., svm,pca,kmeans)"
    )
    task_group.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run on all available tasks"
    )

    # S3 bucket for results
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_RESULTS_BUCKET", ""),
        help="S3 bucket for results (default: from .env S3_RESULTS_BUCKET)"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load generation config for 'n' values
    generation_config = load_generation_config()

    # Discover available tasks
    all_tasks = discover_all_tasks()

    if not all_tasks:
        print("ERROR: No tasks found in AlgoTuneTasks/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_tasks)} available tasks", file=sys.stderr)

    # Build task list based on arguments
    if args.all_tasks:
        selected_tasks = all_tasks
        print(f"Selected: All {len(selected_tasks)} tasks", file=sys.stderr)
    else:  # args.tasks
        requested = [t.strip() for t in args.tasks.split(",")]
        selected_tasks = []
        for task in requested:
            if task in all_tasks:
                selected_tasks.append(task)
            else:
                print(f"WARNING: Task '{task}' not found, skipping", file=sys.stderr)

        if not selected_tasks:
            print("ERROR: None of the requested tasks were found", file=sys.stderr)
            sys.exit(1)

        print(f"Selected: {len(selected_tasks)} specific tasks", file=sys.stderr)

    # Initialize Batch client
    batch = boto3.client("batch", region_name=os.getenv("AWS_REGION"))

    job_queue = os.getenv("BATCH_JOB_QUEUE_NAME", "AlgoTuneQueue")
    job_def = os.getenv("BATCH_JOB_DEF_NAME", "AlgoTuneJobDef")
    s3_bucket = args.s3_bucket

    # Wait for job queue to be VALID before submission (helps right after updates).
    wait_secs = int(os.getenv("BATCH_QUEUE_WAIT_SECONDS", "180"))
    poll_interval = int(os.getenv("BATCH_QUEUE_POLL_SECONDS", "5"))
    if wait_secs > 0:
        print(f"Waiting up to {wait_secs}s for job queue to be VALID...", file=sys.stderr)
        waited = 0
        while waited <= wait_secs:
            try:
                resp_q = batch.describe_job_queues(jobQueues=[job_queue])
                queues = resp_q.get("jobQueues", [])
                if queues:
                    q = queues[0]
                    if q.get("status") == "VALID" and q.get("state") == "ENABLED":
                        print("Job queue is VALID.", file=sys.stderr)
                        break
                time.sleep(poll_interval)
                waited += poll_interval
            except Exception:
                time.sleep(poll_interval)
                waited += poll_interval

    print(f"", file=sys.stderr)
    print(f"Submitting jobs to queue: {job_queue}", file=sys.stderr)
    print(f"Job definition: {job_def}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    if s3_bucket:
        print(f"S3 results bucket: s3://{s3_bucket}/", file=sys.stderr)
    print(f"", file=sys.stderr)

    # Load config to inject into container so model definitions stay in sync
    config_path = Path(__file__).resolve().parents[1] / "AlgoTuner" / "config" / "config.yaml"
    config_text = ""
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"WARNING: Failed to read config.yaml for injection: {e}", file=sys.stderr)

    config_s3_uri = ""
    if s3_bucket and config_text:
        # Ensure DATA_DIR is defined for AWS runs so dataset discovery works.
        config_text = config_text.rstrip() + "\nDATA_DIR: /app/AlgoTuner/data\n"
        config_key = f"config/config_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.yaml"
        try:
            s3_client = boto3.client('s3', region_name=os.getenv("AWS_REGION"))
            s3_client.put_object(Bucket=s3_bucket, Key=config_key, Body=config_text.encode("utf-8"))
            config_s3_uri = f"s3://{s3_bucket}/{config_key}"
            print(f"Uploaded config to {config_s3_uri}", file=sys.stderr)

            # Setup S3 lifecycle policy for dataset auto-cleanup
            dataset_retention_days = int(os.getenv("DATASET_RETENTION_DAYS", "21"))
            setup_s3_lifecycle_policy(s3_client, s3_bucket, dataset_retention_days)
        except Exception as e:
            print(f"WARNING: Failed to upload config.yaml to S3: {e}", file=sys.stderr)

    # Extract model display name (strip provider prefix like "openrouter/")
    model_display = args.model.split('/')[-1] if '/' in args.model else args.model

    # Submit a job for each selected task
    submitted_count = 0
    failed_count = 0

    for task_name in selected_tasks:
        # Clean up local logs for this task+model before submitting
        # (start fresh, like SLURM does)
        logs_dir = Path(__file__).parent.parent / "logs"
        if logs_dir.exists():
            pattern = str(logs_dir / f"{task_name}_{model_display}_*.log")
            matching_logs = glob.glob(pattern)
            for log_file in matching_logs:
                try:
                    os.remove(log_file)
                    print(f"  → Deleted old log: {Path(log_file).name}", file=sys.stderr)
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not delete {log_file}: {e}", file=sys.stderr)
        # Get target_time_ms and n from generation config if available
        task_config = generation_config.get(task_name, {})
        target_time_ms = task_config.get('target_time_ms', 100)
        n_value = task_config.get('n', 'auto')

        # Prioritize n value over target_time_ms for generation
        if n_value and n_value != 'auto':
            generation_arg = f"--n {n_value}"
            dataset_s3_prefix = f"datasets/{task_name}_n{n_value}"
            print(f"Task {task_name}: n={n_value} (ignoring target_time_ms)", file=sys.stderr)
        else:
            generation_arg = f"--target-time-ms {target_time_ms}"
            dataset_s3_prefix = f"datasets/{task_name}_T{target_time_ms}ms"
            print(f"Task {task_name}: target_time_ms={target_time_ms}", file=sys.stderr)

        # Build the combined command to run both steps
        # Note: rely on the image ENTRYPOINT for shell execution
        # Redirect stdout/stderr to files consistent with SLURM pattern
        config_block = ""
        if config_s3_uri:
            config_block = f"""\
echo '=== SYNC config.yaml ===';
aws s3 cp "{config_s3_uri}" /app/AlgoTuner/config/config.yaml || echo 'ERROR: Failed to sync config.yaml';
export ALGOTUNE_CONFIG_PATH=/app/AlgoTuner/config/config.yaml;
"""

        cmd = f"""\
set -e;
JOB_ID=${{AWS_BATCH_JOB_ID:-unknown}};
mkdir -p /app/aws/outputs /app/aws/errors;
exec 1>/app/aws/outputs/${{JOB_ID}}.out 2>/app/aws/errors/${{JOB_ID}}.err;
{config_block}
export DATA_DIR="/app/AlgoTuner/data";
mkdir -p "$DATA_DIR";

upload_results() {{
  set +e;
  echo '=== FINAL UPLOAD TO S3 ===';
  if [ -d /app/logs ]; then
    echo 'Uploading logs...';
    aws s3 cp --recursive /app/logs/ s3://{s3_bucket}/jobs/$JOB_ID/logs/ || echo 'ERROR: S3 upload failed for logs';
  fi;
  if [ -f /app/reports/summary.json ]; then
    echo 'Uploading summary.json...';
    aws s3 cp /app/reports/summary.json s3://{s3_bucket}/jobs/$JOB_ID/summary.json || echo 'ERROR: S3 upload failed for summary';
  fi;
  if [ -f /app/reports/agent_summary.json ]; then
    echo 'Uploading agent_summary.json...';
    aws s3 cp /app/reports/agent_summary.json s3://{s3_bucket}/jobs/$JOB_ID/agent_summary.json || echo 'ERROR: S3 upload failed for agent_summary';
  fi;
  if [ -f /app/aws/outputs/${{JOB_ID}}.out ]; then
    echo 'Uploading stdout...';
    aws s3 cp /app/aws/outputs/${{JOB_ID}}.out s3://{s3_bucket}/jobs/$JOB_ID/outputs/ || echo 'ERROR: S3 upload failed for stdout';
  fi;
  if [ -f /app/aws/errors/${{JOB_ID}}.err ]; then
    echo 'Uploading stderr...';
    aws s3 cp /app/aws/errors/${{JOB_ID}}.err s3://{s3_bucket}/jobs/$JOB_ID/errors/ || echo 'ERROR: S3 upload failed for stderr';
  fi;
  echo \"Upload complete. Results at s3://{s3_bucket}/jobs/$JOB_ID/\";

  if [ -n \"${{PRESIGNED_STDOUT_URL:-}}\" ] && [ -f /app/aws/outputs/${{JOB_ID}}.out ]; then
    echo 'Uploading stdout via presigned URL...';
    python3 - <<'PY'
import os
import urllib.request

url = os.environ.get("PRESIGNED_STDOUT_URL")
path = os.environ.get("PRESIGNED_STDOUT_PATH") or ("/app/aws/outputs/%s.out" % os.environ.get("AWS_BATCH_JOB_ID", "unknown"))
if url and os.path.exists(path):
    with open(path, "rb") as f:
        req = urllib.request.Request(url, data=f.read(), method="PUT")
        with urllib.request.urlopen(req) as resp:
            print(f"Presigned stdout upload status: {{resp.status}}")
PY
  fi;
  if [ -n \"${{PRESIGNED_STDERR_URL:-}}\" ] && [ -f /app/aws/errors/${{JOB_ID}}.err ]; then
    echo 'Uploading stderr via presigned URL...';
    python3 - <<'PY'
import os
import urllib.request

url = os.environ.get("PRESIGNED_STDERR_URL")
path = os.environ.get("PRESIGNED_STDERR_PATH") or ("/app/aws/errors/%s.err" % os.environ.get("AWS_BATCH_JOB_ID", "unknown"))
if url and os.path.exists(path):
    with open(path, "rb") as f:
        req = urllib.request.Request(url, data=f.read(), method="PUT")
        with urllib.request.urlopen(req) as resp:
            print(f"Presigned stderr upload status: {{resp.status}}")
PY
  fi;
}}
"""

        # Add background log sync if S3 bucket is configured
        if s3_bucket:
            cmd += f"""
echo '=== STARTING BACKGROUND LOG SYNC (every 60s) ===';
(
  while true; do
    sleep 60;
    echo "[$(date)] Syncing logs to S3...";
    aws s3 cp --recursive /app/logs/ s3://{s3_bucket}/jobs/$JOB_ID/logs/ --only-show-errors || echo "Warning: S3 upload failed for logs";
    aws s3 cp /app/aws/outputs/${{JOB_ID}}.out s3://{s3_bucket}/jobs/$JOB_ID/outputs/ --only-show-errors || echo "Warning: S3 upload failed for stdout";
    aws s3 cp /app/aws/errors/${{JOB_ID}}.err s3://{s3_bucket}/jobs/$JOB_ID/errors/ --only-show-errors || echo "Warning: S3 upload failed for stderr";
  done
) &
SYNC_PID=$!;
trap 'kill $SYNC_PID 2>/dev/null || true; upload_results' EXIT;
"""

        # Add S3 dataset caching logic if bucket is configured
        if s3_bucket:
            cmd += f"""
EXIT_CODE=0;
DATASET_S3_PREFIX="{dataset_s3_prefix}";

echo '=== CHECKING S3 FOR CACHED DATASET ===';
DATASET_FOUND=0;
if aws s3 ls s3://{s3_bucket}/$DATASET_S3_PREFIX/ 2>/dev/null | grep -q "_train.jsonl"; then
  echo 'Found cached dataset in S3';

  # Check dataset age and refresh if expiring soon (within 5 days)
  echo '=== CHECKING DATASET AGE ===';
  OLDEST_FILE=$(aws s3api list-objects-v2 --bucket {s3_bucket} --prefix "$DATASET_S3_PREFIX/" --query 'Contents[0].LastModified' --output text 2>/dev/null);
  if [ -n "$OLDEST_FILE" ]; then
    DATASET_AGE_DAYS=$(python3 -c "from datetime import datetime, timezone; import sys; dt=datetime.fromisoformat('$OLDEST_FILE'.replace('Z','+00:00')); age=(datetime.now(timezone.utc)-dt).days; print(age)" 2>/dev/null || echo "999");
    DAYS_UNTIL_EXPIRY=$((21 - DATASET_AGE_DAYS));
    echo "Dataset age: $DATASET_AGE_DAYS days, expires in $DAYS_UNTIL_EXPIRY days";

    if [ $DAYS_UNTIL_EXPIRY -lt 5 ]; then
      echo "⚠️  Dataset expires in <5 days - refreshing to extend lifecycle...";
      # Copy all files to themselves to refresh LastModified date (resets 21-day clock)
      aws s3 ls s3://{s3_bucket}/$DATASET_S3_PREFIX/ --recursive | awk '{{print $4}}' | while read key; do
        aws s3 cp "s3://{s3_bucket}/$key" "s3://{s3_bucket}/$key" --metadata-directive REPLACE --only-show-errors;
      done;
      echo "✓ Dataset lifecycle extended by 21 days";
    else
      echo "✓ Dataset has $DAYS_UNTIL_EXPIRY days remaining - no refresh needed";
    fi;
  fi;

  echo 'Downloading dataset from S3...';
  mkdir -p /app/AlgoTuner/data/{task_name};
  aws s3 sync s3://{s3_bucket}/$DATASET_S3_PREFIX/ /app/AlgoTuner/data/{task_name}/ --only-show-errors;
  if [ $? -eq 0 ]; then
    echo 'Dataset downloaded from S3 successfully';
    DATASET_FOUND=1;
  else
    echo 'Failed to download dataset, will regenerate';
  fi;
else
  echo 'No cached dataset found in S3';
fi;

if [ $DATASET_FOUND -eq 0 ]; then
  echo '=== RUN generate ({task_name}) ===';
  /app/algotune.sh --standalone generate {generation_arg} --tasks {task_name} || EXIT_CODE=$?;

  if [ -d /app/AlgoTuner/data/{task_name} ] && [ $EXIT_CODE -eq 0 ]; then
    echo '=== UPLOADING DATASET TO S3 ===';
    aws s3 sync /app/AlgoTuner/data/{task_name}/ s3://{s3_bucket}/$DATASET_S3_PREFIX/ --only-show-errors;
    if [ $? -eq 0 ]; then
      echo "Dataset uploaded to s3://{s3_bucket}/$DATASET_S3_PREFIX/";
    else
      echo 'Warning: Failed to upload dataset to S3';
    fi;
  fi;
else
  echo 'Skipping generation, using cached dataset from S3';
fi;

echo '=== RUN agent ({args.model} on {task_name}) ===';
/app/algotune.sh --standalone agent {args.model} {task_name} || EXIT_CODE=$?;
echo '=== CLEANUP generated data ===';
rm -rf /app/AlgoTuner/data/{task_name}/* || true;
echo 'Deleted generated datasets for {task_name}';
exit $EXIT_CODE;
"""
        else:
            # No S3 bucket - use original logic without caching
            cmd += f"""
EXIT_CODE=0;
echo '=== RUN generate ({task_name}) ===';
/app/algotune.sh --standalone generate {generation_arg} --tasks {task_name} || EXIT_CODE=$?;
echo '=== RUN agent ({args.model} on {task_name}) ===';
/app/algotune.sh --standalone agent {args.model} {task_name} || EXIT_CODE=$?;
echo '=== CLEANUP generated data ===';
rm -rf /app/AlgoTuner/data/{task_name}/* || true;
echo 'Deleted generated datasets for {task_name}';
exit $EXIT_CODE;
"""

        # Add S3 upload if bucket is configured
        if s3_bucket:
            cmd += ""

        # Build environment variables for the container
        mp_start_method = os.getenv("ALGO_MP_START_METHOD", "forkserver")
        env_vars = [
            {"name": "TASK_NAME", "value": task_name},
            {"name": "MODEL", "value": args.model},
            {"name": "OPENROUTER_API_KEY", "value": os.getenv("OPENROUTER_API_KEY", "")},
            {"name": "AWS_REGION", "value": os.getenv("AWS_REGION", "")},
            {"name": "AWS_DEFAULT_REGION", "value": os.getenv("AWS_REGION", "")},
            {"name": "DATA_DIR", "value": "/app/AlgoTuner/data"},
            {"name": "ALGO_MP_START_METHOD", "value": mp_start_method},
        ]

        dataset_size = task_config.get("dataset_size")
        if n_value and n_value != 'auto':
            env_vars.append({"name": "TASK_N", "value": str(n_value)})
        if target_time_ms:
            env_vars.append({"name": "TASK_TARGET_TIME_MS", "value": str(target_time_ms)})
        if dataset_size:
            env_vars.append({"name": "TASK_DATASET_SIZE", "value": str(dataset_size)})

        for key in ["PRESIGNED_STDOUT_URL", "PRESIGNED_STDERR_URL",
                    "PRESIGNED_STDOUT_PATH", "PRESIGNED_STDERR_PATH"]:
            value = os.getenv(key, "")
            if value:
                env_vars.append({"name": key, "value": value})

        # Add optional API keys if present
        for key in ["OPENAI_API_KEY", "CLAUDE_API_KEY", "DEEPSEEK_API_KEY",
                    "GEMINI_API_KEY", "TOGETHER_API_KEY"]:
            value = os.getenv(key, "")
            if value:
                env_vars.append({"name": key, "value": value})

        # Add S3 bucket if configured
        if s3_bucket:
            env_vars.append({"name": "S3_RESULTS_BUCKET", "value": s3_bucket})

        try:
            # Submit job to AWS Batch
            # Note: Job definition has no entrypoint, so we must provide bash wrapper
            safe_model = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in args.model)
            response = batch.submit_job(
                jobName=f"algotune-{task_name}-{safe_model}",
                jobQueue=job_queue,
                jobDefinition=job_def,
                containerOverrides={
                    "command": [cmd],  # Use image ENTRYPOINT for shell execution
                    "environment": env_vars,
                }
            )

            job_id = response["jobId"]
            print(f"✓ Submitted {task_name} (model: {args.model}) - job id: {job_id}", file=sys.stderr)
            print(job_id)  # Output just the job ID to stdout for capture
            submitted_count += 1

        except Exception as e:
            print(f"✗ Failed to submit {task_name}: {e}", file=sys.stderr)
            failed_count += 1

    print(f"", file=sys.stderr)
    print(f"════════════════════════════════════════", file=sys.stderr)
    print(f"Submitted: {submitted_count} jobs", file=sys.stderr)
    if failed_count > 0:
        print(f"Failed: {failed_count} jobs", file=sys.stderr)
    print(f"════════════════════════════════════════", file=sys.stderr)

    if submitted_count == 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
