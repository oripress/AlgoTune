#!/usr/bin/env python3
"""
Submit dataset-only generation jobs to AWS Batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv


# Load environment from both .env files
root_dotenv = Path(__file__).parent.parent / ".env"
if root_dotenv.exists():
    load_dotenv(root_dotenv)

aws_dotenv = Path(__file__).parent / ".env"
if aws_dotenv.exists():
    load_dotenv(aws_dotenv)


def setup_s3_lifecycle_policy(s3_client, bucket: str, dataset_retention_days: int = 7) -> bool:
    try:
        lifecycle_config = {
            "Rules": [
                {
                    "ID": "DeleteOldDatasets",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "datasets/"},
                    "Expiration": {"Days": dataset_retention_days},
                }
            ]
        }
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket,
            LifecycleConfiguration=lifecycle_config,
        )
        print(
            f"✓ S3 lifecycle policy set: datasets/ prefix will expire after {dataset_retention_days} days",
            file=sys.stderr,
        )
        return True
    except Exception as exc:
        print(
            f"Warning: Could not set S3 lifecycle policy: {exc}",
            file=sys.stderr,
        )
        print(
            "         Datasets will not auto-delete. You may need to clean them up manually.",
            file=sys.stderr,
        )
        return False


def load_generation_config() -> dict:
    generation_file = Path(__file__).parent.parent / "reports" / "generation.json"
    if not generation_file.exists():
        print(f"ERROR: generation.json not found at {generation_file}", file=sys.stderr)
        return {}
    try:
        with open(generation_file) as f:
            return json.load(f)
    except Exception as exc:
        print(f"ERROR: Failed to load generation.json: {exc}", file=sys.stderr)
        return {}


def discover_all_tasks() -> list[str]:
    tasks_root = Path(__file__).parent.parent / "AlgoTuneTasks"
    tasks = []
    if not tasks_root.exists():
        return tasks
    for task_dir in sorted(tasks_root.iterdir()):
        if task_dir.is_dir() and (task_dir / "description.txt").exists():
            tasks.append(task_dir.name)
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit dataset-only jobs to AWS Batch")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tasks", help="Comma-separated list of tasks")
    group.add_argument("--all-tasks", action="store_true", help="Run on all tasks")
    parser.add_argument(
        "--s3-bucket", default=os.getenv("S3_RESULTS_BUCKET", ""), help="S3 bucket for datasets"
    )
    parser.add_argument("--dataset-prefix", default="datasets", help="S3 prefix for datasets")
    parser.add_argument(
        "--job-queue", default=os.getenv("BATCH_JOB_QUEUE_NAME", ""), help="AWS Batch job queue"
    )
    parser.add_argument(
        "--job-def", default=os.getenv("BATCH_JOB_DEF_NAME", ""), help="AWS Batch job definition"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without submitting jobs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_config = load_generation_config()
    if not generation_config:
        sys.exit(1)

    all_tasks = discover_all_tasks()
    if not all_tasks:
        print("ERROR: No tasks found in AlgoTuneTasks/", file=sys.stderr)
        sys.exit(1)

    if args.all_tasks:
        selected_tasks = all_tasks
    else:
        requested = [t.strip() for t in args.tasks.split(",")]
        selected_tasks = [t for t in requested if t in all_tasks]
        missing = [t for t in requested if t not in all_tasks]
        if missing:
            print(f"WARNING: Skipping unknown tasks: {', '.join(missing)}", file=sys.stderr)
        if not selected_tasks:
            print("ERROR: None of the requested tasks were found", file=sys.stderr)
            sys.exit(1)

    if not args.s3_bucket:
        print("ERROR: S3 bucket is required (--s3-bucket or S3_RESULTS_BUCKET)", file=sys.stderr)
        sys.exit(1)
    if not args.job_queue or not args.job_def:
        print(
            "ERROR: job queue/definition missing (set BATCH_JOB_QUEUE_NAME/BATCH_JOB_DEF_NAME)",
            file=sys.stderr,
        )
        sys.exit(1)

    batch = boto3.client("batch", region_name=os.getenv("AWS_REGION"))
    s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    retention_days = int(os.getenv("DATASET_RETENTION_DAYS", "7"))
    setup_s3_lifecycle_policy(s3_client, args.s3_bucket, retention_days)

    submitted_count = 0
    failed_count = 0

    for task_name in selected_tasks:
        task_cfg = generation_config.get(task_name, {})
        target_time_ms = task_cfg.get("target_time_ms")
        n_value = task_cfg.get("n")

        if n_value is None:
            print(f"WARNING: No n value for {task_name}; skipping.", file=sys.stderr)
            continue

        if target_time_ms is None:
            print(f"WARNING: No target_time_ms for {task_name}; skipping.", file=sys.stderr)
            continue

        dataset_prefix = f"{args.dataset_prefix}/{task_name}_n{n_value}"

        cmd = f"""\
set -e;
set -x;
JOB_ID=${{AWS_BATCH_JOB_ID:-unknown}};
mkdir -p /app/aws/outputs /app/aws/errors;
exec 1>/app/aws/outputs/${{JOB_ID}}.out 2>/app/aws/errors/${{JOB_ID}}.err;
export DATA_DIR="/app/AlgoTuner/data";
export TARGET_TIME_MS="{target_time_ms}";
export ALGOTUNE_HF_DISABLE="1";
TASK_NAME="{task_name}";
TASK_DATASET_DIR="$DATA_DIR/$TASK_NAME";
upload_logs() {{
  set +e;
  aws s3 cp /app/aws/outputs/${{JOB_ID}}.out s3://{args.s3_bucket}/jobs/$JOB_ID/outputs/ --only-show-errors || true;
  aws s3 cp /app/aws/errors/${{JOB_ID}}.err s3://{args.s3_bucket}/jobs/$JOB_ID/errors/ --only-show-errors || true;
}}
trap 'upload_logs' EXIT;
mkdir -p "$TASK_DATASET_DIR";
env | sort;
python3 /app/AlgoTuner/scripts/generate_and_annotate.py "$TASK_NAME" --data-dir "$TASK_DATASET_DIR" --k "{n_value}";
aws s3 sync "$TASK_DATASET_DIR/" "s3://{args.s3_bucket}/{dataset_prefix}/" --only-show-errors;
"""

        env_vars = [
            {"name": "TASK_NAME", "value": task_name},
            {"name": "TARGET_TIME_MS", "value": str(target_time_ms)},
            {"name": "TASK_N", "value": str(n_value)},
            {"name": "DATA_DIR", "value": "/app/AlgoTuner/data"},
            {"name": "ALGOTUNE_HF_DISABLE", "value": "1"},
            {"name": "AWS_REGION", "value": os.getenv("AWS_REGION", "")},
            {"name": "AWS_DEFAULT_REGION", "value": os.getenv("AWS_REGION", "")},
            {"name": "S3_RESULTS_BUCKET", "value": args.s3_bucket},
        ]

        if args.dry_run:
            print(f"[DRY RUN] {task_name}: {cmd}", file=sys.stderr)
            continue

        try:
            response = batch.submit_job(
                jobName=f"algotune-dataset-{task_name}",
                jobQueue=args.job_queue,
                jobDefinition=args.job_def,
                containerOverrides={
                    "command": [cmd],
                    "environment": env_vars,
                },
            )
            job_id = response["jobId"]
            print(f"✓ Submitted dataset job for {task_name} - job id: {job_id}", file=sys.stderr)
            print(job_id)
            submitted_count += 1
        except Exception as exc:
            print(f"✗ Failed to submit {task_name}: {exc}", file=sys.stderr)
            failed_count += 1

    print("", file=sys.stderr)
    print("════════════════════════════════════════", file=sys.stderr)
    print(f"Submitted: {submitted_count} jobs", file=sys.stderr)
    if failed_count:
        print(f"Failed: {failed_count} jobs", file=sys.stderr)
    print("════════════════════════════════════════", file=sys.stderr)

    if submitted_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
