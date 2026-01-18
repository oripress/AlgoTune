#!/usr/bin/env python3
"""
Detect Spot interruption failures and prepare retries on On-Demand.
Deletes local/S3 artifacts for interrupted jobs so retries are clean.
"""

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
load_dotenv(aws_dotenv)


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        return model_name
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    return model_name


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_job_ids(path: Path) -> list[str]:
    try:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    except Exception as exc:
        print(f"ERROR: Failed to read job IDs file {path}: {exc}", file=sys.stderr)
        return []


def chunked(items: list[str], size: int = 100) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def collect_reasons(job: dict) -> list[str]:
    reasons = []
    status_reason = job.get("statusReason")
    if status_reason:
        reasons.append(status_reason)
    container_reason = job.get("container", {}).get("reason")
    if container_reason:
        reasons.append(container_reason)
    for attempt in job.get("attempts", []):
        attempt_reason = attempt.get("statusReason")
        if attempt_reason:
            reasons.append(attempt_reason)
        attempt_container_reason = attempt.get("container", {}).get("reason")
        if attempt_container_reason:
            reasons.append(attempt_container_reason)
    return reasons


def is_spot_interruption(job: dict) -> bool:
    if job.get("status") != "FAILED":
        return False
    reasons = " ".join(collect_reasons(job)).lower()
    if not reasons:
        return False
    if "spot" in reasons or "interruption" in reasons:
        return True
    if "host ec2" in reasons and ("terminated" in reasons or "stopped" in reasons):
        return True
    if "instance terminated" in reasons or "instance was terminated" in reasons:
        return True
    if "ec2 instance terminated" in reasons:
        return True
    return False


def get_env_value(job: dict, name: str):
    env_list = job.get("container", {}).get("environment", [])
    for entry in env_list:
        if entry.get("name") == name:
            return entry.get("value")
    return None


def delete_s3_prefix(s3_client, bucket: str, prefix: str) -> int:
    removed = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue
        objects = [{"Key": obj["Key"]} for obj in contents]
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
        removed += len(objects)
    return removed


def cleanup_local(
    project_root: Path,
    job_ids: list[str],
    tasks: list[str],
    model_full: str,
    model_display: str,
) -> None:
    logs_dir = project_root / "logs"
    outputs_dir = project_root / "aws" / "outputs"
    errors_dir = project_root / "aws" / "errors"
    job_map_path = project_root / "reports" / "job_map.json"
    summary_path = project_root / "reports" / "agent_summary.json"
    normalized_model = normalize_model_name(model_full)
    model_keys = [model_full]
    if normalized_model != model_full:
        model_keys.append(normalized_model)

    # Remove logs for these tasks/model
    if logs_dir.exists():
        for task in tasks:
            pattern = f"{task}_{model_display}_*.log"
            for log_path in logs_dir.glob(pattern):
                try:
                    log_path.unlink()
                except Exception:
                    pass

    # Remove outputs/errors for interrupted job IDs
    for job_id in job_ids:
        out_path = outputs_dir / f"batch-{job_id}.out"
        err_path = errors_dir / f"batch-{job_id}.err"
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        if err_path.exists():
            try:
                err_path.unlink()
            except Exception:
                pass

    # Remove job IDs from job_map.json
    job_map = load_json(job_map_path)
    if job_map:
        changed = False
        for job_id in job_ids:
            if job_id in job_map:
                job_map.pop(job_id, None)
                changed = True
        if changed:
            save_json(job_map_path, job_map)

    # Remove failed entries from agent_summary.json for these tasks
    summary = load_json(summary_path)
    if summary:
        changed = False
        for task in tasks:
            models = summary.get(task)
            if not isinstance(models, dict):
                continue
            metrics = None
            for key in model_keys:
                candidate = models.get(key)
                if isinstance(candidate, dict):
                    metrics = candidate
                    break
            if not metrics:
                continue
            speedup = metrics.get("final_speedup")
            if speedup is None or speedup == "N/A":
                for key in model_keys:
                    models.pop(key, None)
                changed = True
                if not models:
                    summary.pop(task, None)
        if changed:
            save_json(summary_path, summary)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare retry for Spot-interrupted AWS Batch jobs."
    )
    parser.add_argument("--job-ids-file", required=True, help="File of Spot job IDs")
    parser.add_argument("--out", required=True, help="Output file for tasks to retry")
    parser.add_argument("--model", required=True, help="Full model name (as used in config)")
    parser.add_argument("--model-display", default="", help="Model display name in log filenames")
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_RESULTS_BUCKET", ""),
        help="S3 bucket for logs (default: from aws/.env)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root directory (default: repo root)",
    )
    args = parser.parse_args()

    job_ids = load_job_ids(Path(args.job_ids_file))
    if not job_ids:
        Path(args.out).write_text("")
        print("No job IDs found to inspect.", file=sys.stderr)
        return 0

    model_display = args.model_display or (args.model.split("/")[-1] if "/" in args.model else args.model)

    region = os.getenv("AWS_REGION", "us-east-1")
    batch = boto3.client("batch", region_name=region)

    interrupted_jobs = []
    tasks_to_retry = []
    for batch_ids in chunked(job_ids, 100):
        resp = batch.describe_jobs(jobs=batch_ids)
        for job in resp.get("jobs", []):
            if not is_spot_interruption(job):
                continue
            job_id = job.get("jobId")
            task_name = get_env_value(job, "TASK_NAME")
            if not task_name:
                print(
                    f"Warning: Could not determine task name for job {job_id}, skipping",
                    file=sys.stderr,
                )
                continue
            interrupted_jobs.append(job_id)
            tasks_to_retry.append(task_name)

    if not interrupted_jobs:
        Path(args.out).write_text("")
        print("No Spot interruptions detected.", file=sys.stderr)
        return 0

    # Deduplicate tasks
    tasks_to_retry = sorted(set(tasks_to_retry))

    project_root = Path(args.output_dir)
    cleanup_local(project_root, interrupted_jobs, tasks_to_retry, args.model, model_display)

    if args.s3_bucket:
        s3 = boto3.client("s3", region_name=region)
        removed = 0
        for job_id in interrupted_jobs:
            removed += delete_s3_prefix(s3, args.s3_bucket, f"jobs/{job_id}/")
        if removed:
            print(f"Removed {removed} S3 object(s) from Spot-interrupted jobs", file=sys.stderr)
    else:
        print("S3 bucket not set; skipping S3 cleanup.", file=sys.stderr)

    Path(args.out).write_text("\n".join(tasks_to_retry) + "\n")
    print(
        f"Spot interruptions: {len(interrupted_jobs)} job(s),"
        f" retrying {len(tasks_to_retry)} task(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
