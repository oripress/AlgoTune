#!/usr/bin/env python3
import os
import boto3
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

LOG_GROUP = "/aws/batch/job"  # default CloudWatch Logs group for AWS Batch jobs

batch = boto3.client("batch", region_name=os.getenv("AWS_REGION"))
logs  = boto3.client("logs",  region_name=os.getenv("AWS_REGION"))

def download_logs_for_job(job_id: str, out_dir: str = "downloaded_logs"):
    """
    Fetches the CloudWatch Logs for a given AWS Batch job and writes them
    to a local file under out_dir/<job_id>.log
    """
    # Describe the job to get its log stream name
    resp = batch.describe_jobs(jobs=[job_id])
    job = resp["jobs"][0]

    stream = job.get("container", {}).get("logStreamName")
    if not stream:
        print(f"[WARN] No log stream available for job {job_id}")
        return

    print(f"[INFO] Fetching logs for job {job_id} from stream {stream}")

    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"{job_id}.log")

    next_token = None
    with open(outfile, "w") as fp:
        while True:
            kwargs = {
                "logGroupName": LOG_GROUP,
                "logStreamName": stream,
                "startFromHead": True,
            }
            if next_token:
                kwargs["nextToken"] = next_token

            # Get up to ~10K events/page from CloudWatch Logs
            resp_logs = logs.get_log_events(**kwargs)
            events = resp_logs.get("events", [])
            for ev in events:
                fp.write(ev["message"] + "\n")

            # Pagination — keep going until nextForwardToken doesn’t change
            token = resp_logs.get("nextForwardToken")
            if token == next_token:
                break
            next_token = token

    print(f"[SUCCESS] Saved logs for {job_id} → {outfile}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-ids-file",
        required=True,
        help="Path to a file with one AWS Batch job ID per line"
    )
    parser.add_argument(
        "--out-dir",
        default="downloaded_logs",
        help="Directory where fetched logs should be written"
    )

    args = parser.parse_args()

    with open(args.job_ids_file) as f:
        job_ids = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Found {len(job_ids)} job IDs to fetch logs for.")
    for jid in job_ids:
        download_logs_for_job(jid, out_dir=args.out_dir)

    print("[DONE] All logs fetched!")
