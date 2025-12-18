#!/usr/bin/env python3
import os
import sys
import argparse
import boto3
import json
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

    print(f"", file=sys.stderr)
    print(f"Submitting jobs to queue: {job_queue}", file=sys.stderr)
    print(f"Job definition: {job_def}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    if s3_bucket:
        print(f"S3 results bucket: s3://{s3_bucket}/", file=sys.stderr)
    print(f"", file=sys.stderr)

    # Submit a job for each selected task
    submitted_count = 0
    failed_count = 0

    for task_name in selected_tasks:
        # Get target_time_ms from generation config if available
        task_config = generation_config.get(task_name, {})
        target_time_ms = task_config.get('target_time_ms', 100)
        n_value = task_config.get('n', 'auto')

        print(f"Task {task_name}: target_time_ms={target_time_ms}, n={n_value}", file=sys.stderr)

        # Build the combined command to run both steps
        # Redirect stdout/stderr to files consistent with SLURM pattern
        cmd = f"""
bash -lc \"\
set -e;
JOB_ID=${{AWS_BATCH_JOB_ID:-unknown}};
mkdir -p /app/aws/outputs /app/aws/errors;
exec 1>/app/aws/outputs/${{JOB_ID}}.out 2>/app/aws/errors/${{JOB_ID}}.err;
echo '=== RUN generate ({task_name}) ===';
/app/algotune.sh --standalone generate --target-time-ms {target_time_ms} --tasks {task_name};
echo '=== RUN agent ({args.model} on {task_name}) ===';
/app/algotune.sh --standalone agent {args.model} {task_name};
echo '=== CLEANUP generated data ===';
rm -rf /app/AlgoTuner/data/{task_name}/* || true;
echo 'Deleted generated datasets for {task_name}';
"""

        # Add S3 upload if bucket is configured
        if s3_bucket:
            cmd += f"""
echo '=== UPLOADING RESULTS TO S3 ===';
if [ -d /app/logs ]; then
  aws s3 cp /app/logs/ s3://{s3_bucket}/jobs/$JOB_ID/logs/ --recursive --only-show-errors || echo 'Warning: S3 upload failed for logs';
fi;
if [ -f /app/reports/summary.json ]; then
  aws s3 cp /app/reports/summary.json s3://{s3_bucket}/jobs/$JOB_ID/summary.json --only-show-errors || echo 'Warning: S3 upload failed for summary';
fi;
if [ -f /app/aws/outputs/${{JOB_ID}}.out ]; then
  aws s3 cp /app/aws/outputs/${{JOB_ID}}.out s3://{s3_bucket}/jobs/$JOB_ID/outputs/ --only-show-errors || echo 'Warning: S3 upload failed for stdout';
fi;
if [ -f /app/aws/errors/${{JOB_ID}}.err ]; then
  aws s3 cp /app/aws/errors/${{JOB_ID}}.err s3://{s3_bucket}/jobs/$JOB_ID/errors/ --only-show-errors || echo 'Warning: S3 upload failed for stderr';
fi;
echo \"Results uploaded to s3://{s3_bucket}/jobs/$JOB_ID/\";
"""

        # Close the command
        cmd += """
\\"
"""

        # Build environment variables for the container
        env_vars = [
            {"name": "TASK_NAME", "value": task_name},
            {"name": "MODEL", "value": args.model},
            {"name": "OPENROUTER_API_KEY", "value": os.getenv("OPENROUTER_API_KEY", "")},
        ]

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
            response = batch.submit_job(
                jobName=f"algotune-{task_name}-{args.model.replace('/', '-')}",
                jobQueue=job_queue,
                jobDefinition=job_def,
                containerOverrides={
                    "command": ["bash", "-lc", cmd],
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
