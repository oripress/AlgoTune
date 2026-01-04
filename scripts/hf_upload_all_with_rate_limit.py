#!/usr/bin/env python3
"""
Upload all datasets to HuggingFace with automatic rate limit handling.
Waits 1 hour after hitting 128 commits, then resumes.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


# Configuration
WAIT_TIME_SECONDS = 3660  # 61 minutes to be safe
REPO_ID = "oripress/AlgoTune"
S3_BUCKET = "algotune-results-926341122891"
S3_PREFIX = "datasets"


def log(message: str):
    """Print timestamped message"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def wait_for_rate_limit():
    """Wait for rate limit to reset"""
    resume_time = datetime.utcnow() + timedelta(seconds=WAIT_TIME_SECONDS)
    log("=" * 70)
    log("⚠️  RATE LIMIT HIT - Waiting for reset")
    log("=" * 70)
    log(f"Current time: {datetime.utcnow().strftime('%H:%M:%S UTC')}")
    log(f"Will resume at: {resume_time.strftime('%H:%M:%S UTC')}")
    log(f"Waiting {WAIT_TIME_SECONDS // 60} minutes...")

    remaining = WAIT_TIME_SECONDS
    update_interval = 300  # Update every 5 minutes

    while remaining > 0:
        if remaining % update_interval == 0 or remaining < 60:
            mins_left = remaining // 60
            log(f"⏳ {mins_left} minutes remaining...")
        time.sleep(min(60, remaining))
        remaining -= 60

    log("✓ Rate limit should be reset - resuming uploads\n")


def upload_task(task_name: str, report_path: Path) -> dict:
    """
    Upload a single task to HuggingFace.
    Returns dict with 'success' (bool) and 'rate_limited' (bool).
    """
    log(f"Uploading {task_name}...")

    cmd = [
        "python3",
        "scripts/push_hf_datasets.py",
        "--repo-id",
        REPO_ID,
        "--revision",
        "main",
        "--s3-bucket",
        S3_BUCKET,
        "--s3-prefix",
        S3_PREFIX,
        "--report-path",
        str(report_path),
        "--tasks",
        task_name,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Check for rate limit error
        if "429" in result.stderr or "Too Many Requests" in result.stderr:
            log(f"  ⚠️  {task_name}: Rate limit hit")
            return {"success": False, "rate_limited": True}

        # Check for success
        if result.returncode == 0:
            log(f"  ✓ {task_name}: Uploaded successfully")
            return {"success": True, "rate_limited": False}

        # Other error
        log(f"  ✗ {task_name}: Upload failed (exit {result.returncode})")
        error_msg = result.stderr[-200:] if result.stderr else "Unknown error"
        log(f"    Error: {error_msg}")
        return {"success": False, "rate_limited": False}

    except Exception as e:
        log(f"  ✗ {task_name}: Exception: {e}")
        return {"success": False, "rate_limited": False}


def get_pending_tasks(generation_file: Path, completed_file: Path) -> list:
    """Get list of tasks that need to be uploaded"""
    # Load all available tasks from generation.json
    if not generation_file.exists():
        log(f"ERROR: {generation_file} not found")
        return []

    with open(generation_file) as f:
        all_tasks = list(json.load(f).keys())

    # Load already completed tasks
    completed = set()
    if completed_file.exists():
        with open(completed_file) as f:
            completed = set(json.load(f))

    # Return tasks that haven't been completed
    pending = [t for t in all_tasks if t not in completed]
    return pending


def mark_completed(task: str, completed_file: Path):
    """Mark a task as completed"""
    completed = set()
    if completed_file.exists():
        with open(completed_file) as f:
            completed = set(json.load(f))

    completed.add(task)

    with open(completed_file, "w") as f:
        json.dump(sorted(list(completed)), f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Upload all datasets to HF with rate limit handling"
    )
    parser.add_argument("--generation-file", default="reports/generation.json")
    parser.add_argument("--completed-file", default="reports/hf_upload_completed.json")
    parser.add_argument("--report-dir", default="reports/hf_uploads")
    args = parser.parse_args()

    generation_file = Path(args.generation_file)
    completed_file = Path(args.completed_file)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(exist_ok=True, parents=True)

    log("=" * 70)
    log("HuggingFace Upload All Datasets (with auto rate limit handling)")
    log("=" * 70)

    # Get pending tasks
    pending_tasks = get_pending_tasks(generation_file, completed_file)

    if not pending_tasks:
        log("✓ No pending tasks - all datasets already uploaded!")
        return

    log(f"Found {len(pending_tasks)} pending datasets to upload")
    log("")

    # Upload tasks one by one
    uploaded = 0
    failed = 0

    while pending_tasks:
        task = pending_tasks[0]
        report_path = report_dir / f"{task}.json"

        result = upload_task(task, report_path)

        if result["rate_limited"]:
            # Wait for rate limit, then retry this task
            wait_for_rate_limit()
            continue

        if result["success"]:
            uploaded += 1
            mark_completed(task, completed_file)
            pending_tasks.pop(0)
        else:
            failed += 1
            log(f"  Skipping {task} after failure (will not retry)")
            pending_tasks.pop(0)

        # Brief pause between uploads
        if pending_tasks:
            time.sleep(3)

    # Final summary
    log("")
    log("=" * 70)
    log("Upload Summary")
    log("=" * 70)
    log(f"  ✓ Uploaded: {uploaded}")
    log(f"  ✗ Failed: {failed}")
    log(f"  Total: {uploaded + failed}")
    log("=" * 70)


if __name__ == "__main__":
    main()
