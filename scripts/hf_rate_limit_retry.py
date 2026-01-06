#!/usr/bin/env python3
"""
Wait for HuggingFace rate limit to reset, then retry failed uploads.
"""

import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from huggingface_hub import HfApi


# Configuration
WAIT_TIME_SECONDS = 3660  # 61 minutes to be safe
REPO_ID = "oripress/AlgoTune"
README_PATH = Path("/home/ec2-user/AlgoTune/hf_readme.md")
FAILED_TASKS = ["lyapunov_stability", "vertex_cover"]
S3_BUCKET = "algotune-results-926341122891"
S3_PREFIX = "datasets"


def log(message: str):
    """Print timestamped message"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def upload_readme():
    """Upload README.md to HuggingFace"""
    if not README_PATH.exists():
        log("⚠️  README file not found, skipping")
        return True

    token = os.getenv("HF_TOKEN")
    if not token:
        log("✗ HF_TOKEN not found in environment")
        return False

    api = HfApi()
    log(f"Uploading README.md to {REPO_ID}...")

    try:
        api.upload_file(
            path_or_fileobj=str(README_PATH),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
            commit_message="Add AlgoTune dataset README",
        )
        log("✓ README.md uploaded successfully")

        # Delete local file
        README_PATH.unlink()
        log("✓ Deleted local hf_readme.md")
        return True

    except Exception as e:
        log(f"✗ Failed to upload README: {e}")
        return False


def retry_task(task_name: str):
    """Retry uploading a single task to HuggingFace"""
    log(f"Retrying upload for {task_name}...")

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
        f"reports/hf_retry_{task_name}.json",
        "--tasks",
        task_name,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            log(f"✓ {task_name} uploaded successfully")
            return True
        else:
            log(f"✗ {task_name} upload failed (exit {result.returncode})")
            if result.stderr:
                log(f"  Error: {result.stderr[-500:]}")  # Last 500 chars
            return False
    except Exception as e:
        log(f"✗ Exception retrying {task_name}: {e}")
        return False


def main():
    log("=" * 70)
    log("HuggingFace Rate Limit Retry Script")
    log("=" * 70)

    # Calculate wait time
    resume_time = datetime.utcnow() + timedelta(seconds=WAIT_TIME_SECONDS)
    log(f"Current time: {datetime.utcnow().strftime('%H:%M:%S UTC')}")
    log(f"Will resume at: {resume_time.strftime('%H:%M:%S UTC')}")
    log(f"Waiting {WAIT_TIME_SECONDS // 60} minutes for rate limit to reset...")

    # Wait with periodic updates
    remaining = WAIT_TIME_SECONDS
    update_interval = 300  # Update every 5 minutes

    while remaining > 0:
        if remaining % update_interval == 0 or remaining < 60:
            mins_left = remaining // 60
            log(f"⏳ {mins_left} minutes remaining...")
        time.sleep(min(60, remaining))
        remaining -= 60

    log("\n" + "=" * 70)
    log("Rate limit should be reset now - starting retries")
    log("=" * 70 + "\n")

    # Step 1: Upload README
    log("Step 1: Uploading README.md")
    readme_success = upload_readme()

    if not readme_success:
        log("⚠️  README upload failed, but continuing with dataset uploads...")

    time.sleep(5)  # Brief pause between uploads

    # Step 2: Retry failed tasks
    log(f"\nStep 2: Retrying {len(FAILED_TASKS)} failed dataset uploads")
    results = {}

    for task in FAILED_TASKS:
        success = retry_task(task)
        results[task] = success
        time.sleep(5)  # Brief pause between uploads

    # Summary
    log("\n" + "=" * 70)
    log("Retry Summary")
    log("=" * 70)

    if README_PATH.exists() or not readme_success:
        status = "✓ Uploaded" if readme_success else "✗ Failed"
        log(f"  README.md: {status}")

    successful = sum(1 for v in results.values() if v)
    log(f"\nDatasets: {successful}/{len(FAILED_TASKS)} uploaded successfully")
    for task, success in results.items():
        status = "✓" if success else "✗"
        log(f"  {status} {task}")

    log("\n" + "=" * 70)
    log("Retry script completed")
    log("=" * 70)


if __name__ == "__main__":
    main()
