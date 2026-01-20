#!/usr/bin/env python3
"""
Download logs from S3 for AWS Batch jobs.
Can download logs for specific jobs or all jobs, with optional watch mode.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv


# Load environment from aws/.env
aws_dotenv = Path(__file__).parent / ".env"
load_dotenv(aws_dotenv)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1]

MODEL_RE = re.compile(r"model[:=]\\s*([\\w./-]+)")
TASK_RE = re.compile(r"task_name to raw '([^']+)'")
LOG_NAME_RE = re.compile(r"^(?P<task>.+)_(?P<model>.+)_(?P<stamp>\\d{8}_\\d{6})\\.log$")


def acquire_lock(lock_path: Path, timeout: float = 30.0) -> bool:
    """Acquire a lock via atomic file creation."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.1 + random.random() * 0.2)
        except Exception:
            return False
    return False


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        return model_name
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    return model_name


def normalize_summary(summary: dict) -> dict:
    normalized = {}
    for task, models in summary.items():
        if not isinstance(models, dict):
            continue
        task_entry = normalized.setdefault(task, {})
        for model, metrics in models.items():
            if not isinstance(metrics, dict):
                continue
            task_entry[normalize_model_name(model)] = metrics
    return normalized


def load_agent_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def extract_task_model_from_log(log_path: Path):
    task = None
    model = None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(400):
                line = handle.readline()
                if not line:
                    break
                if task is None:
                    match = TASK_RE.search(line)
                    if match:
                        task = match.group(1)
                if model is None:
                    match = MODEL_RE.search(line)
                    if match:
                        model = match.group(1)
                if task and model:
                    break
    except Exception:
        task = None
        model = None

    if task is None or model is None:
        name = log_path.name
        match = LOG_NAME_RE.match(name)
        if match:
            task = match.group("task")
            model = match.group("model")
    return task, normalize_model_name(model)


def parse_log_name(log_filename: str):
    match = LOG_NAME_RE.match(log_filename)
    if not match:
        return None, None, None
    return match.group("task"), match.group("model"), match.group("stamp")


def extract_task_model_from_text_path(text_path: Path):
    if not text_path.exists():
        return None, None
    try:
        with text_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(400):
                line = handle.readline()
                if not line:
                    break
                task_match = TASK_RE.search(line)
                model_match = MODEL_RE.search(line)
                if task_match or model_match:
                    task = task_match.group(1) if task_match else None
                    model = model_match.group(1) if model_match else None
                    return task, normalize_model_name(model)
    except Exception:
        return None, None
    return None, None


def load_job_map(job_map_path: Path) -> dict:
    if not job_map_path.exists():
        return {}
    try:
        with job_map_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_job_map(job_map_path: Path, job_map: dict) -> None:
    job_map_path.parent.mkdir(parents=True, exist_ok=True)
    with job_map_path.open("w", encoding="utf-8") as handle:
        json.dump(job_map, handle, indent=2, sort_keys=True)


def update_job_map(job_map_path: Path, job_id: str, task: str, model: str) -> None:
    if not task or not model:
        return
    job_map = load_job_map(job_map_path)
    entry = job_map.get(job_id, {})
    entry.update({"task": task, "model": model})
    job_map[job_id] = entry
    save_job_map(job_map_path, job_map)


def cleanup_failed_logs_s3(s3_client, bucket: str, project_root: Path, summary_path: Path) -> int:
    """Delete failed/old job logs from S3 to match local cleanup."""
    summary = load_agent_summary(summary_path)
    if not summary:
        summary = {}

    job_map_path = project_root / "reports" / "job_map.json"
    job_map = load_job_map(job_map_path)

    removed = 0

    # Build map of latest job per task/model
    latest_by_pair = {}
    for job_id, meta in job_map.items():
        if not isinstance(meta, dict):
            continue
        task = meta.get("task")
        model = meta.get("model")
        stamp = meta.get("timestamp")
        if not task or not model:
            continue
        task_entry = summary.get(task)
        if not isinstance(task_entry, dict):
            continue
        model_entry = task_entry.get(model)
        if not isinstance(model_entry, dict):
            continue
        speedup = model_entry.get("final_speedup")
        pair = (task, model)
        if stamp:
            existing = latest_by_pair.get(pair)
            if existing is None or stamp > existing[0]:
                latest_by_pair[pair] = (stamp, job_id, speedup)

    # Delete S3 objects for old/failed jobs
    jobs_to_delete = []
    for job_id, meta in job_map.items():
        if not isinstance(meta, dict):
            continue
        task = meta.get("task")
        model = meta.get("model")
        if not task or not model:
            continue

        pair = (task, model)
        latest = latest_by_pair.get(pair)

        # Delete if not the latest job for this task/model
        if latest and latest[1] != job_id:
            jobs_to_delete.append(job_id)
        # Or delete if latest but failed
        elif latest:
            _, _, speedup = latest
            if speedup is None or speedup == "N/A":
                jobs_to_delete.append(job_id)

    # Delete S3 objects for each job
    for job_id in jobs_to_delete:
        prefix = f"jobs/{job_id}/"
        try:
            # List all objects for this job
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if "Contents" not in response:
                continue

            # Delete all objects for this job
            objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]
            if objects_to_delete:
                s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects_to_delete})
                removed += len(objects_to_delete)
        except Exception as e:
            print(f"Warning: Failed to delete S3 objects for job {job_id}: {e}", file=sys.stderr)

    return removed


def cleanup_failed_logs(project_root: Path, summary_path: Path) -> int:
    summary = load_agent_summary(summary_path)
    if not summary:
        summary = {}

    logs_dir = project_root / "logs"
    aws_outputs_dir = project_root / "aws" / "outputs"
    aws_errors_dir = project_root / "aws" / "errors"
    job_map_path = project_root / "reports" / "job_map.json"
    job_map = load_job_map(job_map_path)

    removed = 0
    log_groups = {}
    if logs_dir.exists():
        for log_path in logs_dir.glob("*.log"):
            task, model = extract_task_model_from_log(log_path)
            if not task or not model:
                continue
            log_groups.setdefault((task, model), []).append(log_path)

        for (task, model), paths in log_groups.items():
            # Keep only the newest log for this task/model regardless of summary state
            newest = max(paths, key=lambda p: p.stat().st_mtime)
            for log_path in paths:
                if log_path == newest:
                    continue
                try:
                    log_path.unlink()
                    removed += 1
                except Exception:
                    continue

    latest_by_pair = {}
    for job_id, meta in job_map.items():
        if not isinstance(meta, dict):
            continue
        task = meta.get("task")
        model = meta.get("model")
        stamp = meta.get("timestamp")
        task_entry = summary.get(task)
        if not isinstance(task_entry, dict):
            continue
        model_entry = task_entry.get(model)
        if not isinstance(model_entry, dict):
            continue
        speedup = model_entry.get("final_speedup")
        pair = (task, model)
        if stamp:
            existing = latest_by_pair.get(pair)
            if existing is None or stamp > existing[0]:
                latest_by_pair[pair] = (stamp, job_id, speedup)

    for job_id, meta in job_map.items():
        if not isinstance(meta, dict):
            continue
        task = meta.get("task")
        model = meta.get("model")
        stamp = meta.get("timestamp")
        if not task or not model:
            continue
        pair = (task, model)
        latest = latest_by_pair.get(pair)
        if latest and latest[1] != job_id:
            # Remove outputs/errors for older jobs of the same task/model
            out_path = aws_outputs_dir / f"batch-{job_id}.out"
            err_path = aws_errors_dir / f"batch-{job_id}.err"
            if out_path.exists():
                try:
                    out_path.unlink()
                    removed += 1
                except Exception:
                    pass
            if err_path.exists():
                try:
                    err_path.unlink()
                    removed += 1
                except Exception:
                    pass
        elif latest:
            # Also remove outputs/errors for the latest job if it is marked failed
            _, _, speedup = latest
            if speedup is None or speedup == "N/A":
                out_path = aws_outputs_dir / f"batch-{job_id}.out"
                err_path = aws_errors_dir / f"batch-{job_id}.err"
                if out_path.exists():
                    try:
                        out_path.unlink()
                        removed += 1
                    except Exception:
                        pass
                if err_path.exists():
                    try:
                        err_path.unlink()
                        removed += 1
                    except Exception:
                        pass

    for output_path in list(aws_outputs_dir.glob("batch-*.out")) + list(
        aws_errors_dir.glob("batch-*.err")
    ):
        job_id = output_path.stem.replace("batch-", "")
        if job_id in job_map:
            continue
        task, model = extract_task_model_from_text_path(output_path)
        if not task or not model:
            continue
        update_job_map(job_map_path, job_id, task, model)
        task_entry = summary.get(task)
        if not isinstance(task_entry, dict):
            continue
        model_entry = task_entry.get(model)
        if not isinstance(model_entry, dict):
            continue
        speedup = model_entry.get("final_speedup")
        if speedup is None or speedup == "N/A":
            out_path = aws_outputs_dir / f"batch-{job_id}.out"
            err_path = aws_errors_dir / f"batch-{job_id}.err"
            if out_path.exists():
                try:
                    out_path.unlink()
                    removed += 1
                except Exception:
                    pass
            if err_path.exists():
                try:
                    err_path.unlink()
                    removed += 1
                except Exception:
                    pass
    return removed


def merge_agent_summary(local_summary_path: Path, new_summary_path: Path) -> None:
    local_summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = local_summary_path.with_suffix(".json.lock")
    if not acquire_lock(lock_path):
        print(f"Warning: Could not acquire summary lock {lock_path}", file=sys.stderr)
        return
    try:
        local_summary = normalize_summary(load_agent_summary(local_summary_path))
        new_summary = normalize_summary(load_agent_summary(new_summary_path))
        if not new_summary:
            return
        for task, models in new_summary.items():
            if not isinstance(models, dict):
                continue
            local_task = local_summary.setdefault(task, {})
            if not isinstance(local_task, dict):
                local_summary[task] = models
                continue
            for model, metrics in models.items():
                local_task[model] = metrics
        temp_path = local_summary_path.with_suffix(local_summary_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(local_summary, handle, indent=2, sort_keys=True)
        os.replace(temp_path, local_summary_path)
    finally:
        release_lock(lock_path)


def download_job_logs(s3_client, bucket, job_id, project_root="."):
    """Download all logs for a specific job ID to SLURM-compatible structure."""
    project_root = Path(project_root)

    # Create directory structure
    aws_outputs_dir = project_root / "aws" / "outputs"
    aws_errors_dir = project_root / "aws" / "errors"
    logs_dir = project_root / "logs"
    reports_dir = project_root / "reports"
    job_map_path = reports_dir / "job_map.json"
    reports_dir = project_root / "reports"

    aws_outputs_dir.mkdir(parents=True, exist_ok=True)
    aws_errors_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"jobs/{job_id}/"
    downloaded = []

    try:
        # List all objects for this job
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" not in response:
            return downloaded

        for obj in response["Contents"]:
            key = obj["Key"]
            # Skip directory markers
            if key.endswith("/"):
                continue

            # Determine local path based on S3 structure
            relative_path = key[len(prefix) :]

            # Map S3 paths to local structure
            if relative_path.startswith("outputs/"):
                # aws/outputs/batch-<jobid>.out
                local_path = aws_outputs_dir / f"batch-{job_id}.out"
            elif relative_path.startswith("errors/"):
                # aws/errors/batch-<jobid>.err
                local_path = aws_errors_dir / f"batch-{job_id}.err"
            elif relative_path.startswith("logs/"):
                # logs/<filename> (SLURM-compatible)
                log_filename = relative_path[5:]  # Remove 'logs/' prefix
                local_path = logs_dir / log_filename
                task, model, stamp = parse_log_name(log_filename)
                if task and model and stamp:
                    job_map = load_job_map(job_map_path)
                    entry = job_map.get(job_id, {})
                    entry.update({"task": task, "model": model, "timestamp": stamp})
                    job_map[job_id] = entry
                    save_job_map(job_map_path, job_map)
            elif relative_path == "agent_summary.json":
                temp_path = reports_dir / f"agent_summary_{job_id}.json"
                s3_client.download_file(bucket, key, str(temp_path))
                merge_agent_summary(reports_dir / "agent_summary.json", temp_path)
                temp_path.unlink(missing_ok=True)
                continue
            elif relative_path == "summary.json":
                reports_dir.mkdir(parents=True, exist_ok=True)
                local_path = reports_dir / f"summary_{job_id}.json"
            else:
                # Other files - skip
                continue

            # Download if not exists or newer
            should_download = True
            if local_path.exists():
                local_mtime = local_path.stat().st_mtime
                s3_mtime = obj["LastModified"].timestamp()
                should_download = s3_mtime > local_mtime

            if should_download:
                s3_client.download_file(bucket, key, str(local_path))
                downloaded.append(str(local_path.relative_to(project_root)))

    except Exception as e:
        print(f"Error downloading logs for job {job_id}: {e}", file=sys.stderr)

    return downloaded


def download_all_logs(s3_client, bucket, project_root="."):
    """Download logs for all jobs in the bucket."""
    try:
        # List all job directories
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix="jobs/", Delimiter="/")

        if "CommonPrefixes" not in response:
            print("No jobs found in S3 bucket", file=sys.stderr)
            return []

        job_ids = [prefix["Prefix"].split("/")[1] for prefix in response["CommonPrefixes"]]

        all_downloaded = []
        for job_id in job_ids:
            downloaded = download_job_logs(s3_client, bucket, job_id, project_root)
            if downloaded:
                all_downloaded.extend(downloaded)

        return all_downloaded

    except Exception as e:
        print(f"Error downloading all logs: {e}", file=sys.stderr)
        return []


def load_job_ids_from_file(filepath):
    """Load job IDs from a file (one per line)."""
    try:
        with open(filepath) as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading job IDs file: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Download logs from S3 for AWS Batch jobs")

    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_RESULTS_BUCKET", ""),
        help="S3 bucket name (default: from .env S3_RESULTS_BUCKET)",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Project root directory (default: repo root)",
    )

    parser.add_argument(
        "--cleanup-failed",
        action="store_true",
        help="Delete logs not marked successful in reports/agent_summary.json",
    )
    parser.add_argument(
        "--prune-local",
        action="store_true",
        help="Prune local logs/outputs/errors without contacting S3",
    )
    parser.add_argument(
        "--summary-file",
        default=str(DEFAULT_OUTPUT_DIR / "reports" / "agent_summary.json"),
        help="Summary file for cleanup (default: reports/agent_summary.json)",
    )

    parser.add_argument("--job-ids-file", help="File containing job IDs (one per line)")

    parser.add_argument("--job-id", help="Download logs for a specific job ID")

    parser.add_argument(
        "--all", action="store_true", help="Download logs for all jobs in the bucket"
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode: continuously download logs every 60 seconds",
    )

    parser.add_argument(
        "--interval", type=int, default=60, help="Interval in seconds for watch mode (default: 60)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.prune_local:
        removed = cleanup_failed_logs(Path(args.output_dir), Path(args.summary_file))
        print(f"Pruned {removed} local file(s)")
        return

    if not args.s3_bucket:
        print(
            "Error: S3 bucket not specified. Use --s3-bucket or set S3_RESULTS_BUCKET in aws/.env",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (args.job_id or args.job_ids_file or args.all):
        print("Error: Must specify --job-id, --job-ids-file, or --all", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Initialize S3 client
    s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))

    # Determine which jobs to download
    if args.job_id:
        job_ids = [args.job_id]
    elif args.job_ids_file:
        job_ids = load_job_ids_from_file(args.job_ids_file)
        if not job_ids:
            print("Error: No job IDs found in file", file=sys.stderr)
            sys.exit(1)
    else:
        job_ids = None  # Download all

    # Download logs
    def download_once():
        if job_ids is None:
            # Download all jobs
            downloaded = download_all_logs(s3_client, args.s3_bucket, args.output_dir)
            if downloaded:
                print(f"Downloaded {len(downloaded)} file(s)")
                for filename in downloaded:
                    print(f"  {filename}")
            else:
                print("No new files to download")
        else:
            # Download specific jobs
            total_downloaded = []
            for job_id in job_ids:
                downloaded = download_job_logs(s3_client, args.s3_bucket, job_id, args.output_dir)
                if downloaded:
                    total_downloaded.extend(downloaded)

            if total_downloaded:
                print(f"Downloaded {len(total_downloaded)} file(s)")
                for filename in total_downloaded:
                    print(f"  {filename}")
            else:
                print("No new files to download")

        # Cleanup after download (if requested)
        if args.cleanup_failed:
            # Clean up S3 first
            removed_s3 = cleanup_failed_logs_s3(
                s3_client, args.s3_bucket, Path(args.output_dir), Path(args.summary_file)
            )
            if removed_s3:
                print(f"Removed {removed_s3} S3 object(s)")
            # Then clean up local files
            removed = cleanup_failed_logs(Path(args.output_dir), Path(args.summary_file))
            if removed:
                print(f"Removed {removed} local file(s)")

    # Run download
    if args.watch:
        print(f"Starting watch mode (interval: {args.interval}s)")
        print(f"Downloading to: {args.output_dir}")
        print(f"S3 bucket: s3://{args.s3_bucket}/jobs/")
        print("Press Ctrl+C to stop\n")

        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Iteration {iteration}")
                download_once()
                print(f"Waiting {args.interval} seconds...\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped watching")
    else:
        download_once()
        print(f"\nLogs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
