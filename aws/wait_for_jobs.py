#!/usr/bin/env python3
import os
import random
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv


# Load environment from both .env files
# 1. Load API keys from root .env
root_dotenv = Path(__file__).parent.parent / ".env"
if root_dotenv.exists():
    load_dotenv(root_dotenv)

# 2. Load AWS configuration from aws/.env
aws_dotenv = Path(__file__).parent / ".env"
load_dotenv(aws_dotenv)

batch = boto3.client("batch", region_name=os.getenv("AWS_REGION"))

TERMINAL_STATES = {"SUCCEEDED", "FAILED"}


def check_jobs_complete(job_ids):
    """
    Returns a dict job_id -> status.
    """
    statuses = {}
    # Batch.describe_jobs accepts up to 100 IDs at once
    for i in range(0, len(job_ids), 100):
        chunk = job_ids[i : i + 100]
        desc = batch.describe_jobs(jobs=chunk)
        for job in desc["jobs"]:
            statuses[job["jobId"]] = job["status"]
    return statuses


def wait_for_jobs(job_ids, poll_interval=10):
    """
    Polls AWS Batch until all job_ids are in terminal states.
    """
    pending = list(job_ids)
    while pending:
        statuses = check_jobs_complete(pending)
        still_running = []
        print("Current statuses:")
        for jid, status in statuses.items():
            print(f"  {jid}: {status}")
            if status not in TERMINAL_STATES:
                still_running.append(jid)
        if not still_running:
            print("All jobs reached terminal state.")
            return statuses
        # back off a bit before checking again
        delay = poll_interval + random.uniform(0, poll_interval)
        print(f"Waiting ~{delay:.1f}s before next check...")
        time.sleep(delay)
        pending = still_running


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-ids-file", required=True, help="Path to file with AWS Batch job IDs (one per line)"
    )
    args = parser.parse_args()

    with open(args.job_ids_file) as f:
        job_ids = [line.strip() for line in f if line.strip()]

    print(f"Waiting for {len(job_ids)} jobs to finish...")
    final_statuses = wait_for_jobs(job_ids)

    # Print summary
    print("\nFinal statuses:")
    for jid, status in final_statuses.items():
        print(f"{jid}: {status}")

    # Optionally: write statuses to file
    out_file = "job_statuses.txt"
    with open(out_file, "w") as fp:
        for jid, status in final_statuses.items():
            fp.write(f"{jid} {status}\n")

    print(f"Statuses written to {out_file}")
