#!/usr/bin/env python3
"""
Fetch AlgoTune results from S3 bucket.

Downloads logs and summary.json files for completed AWS Batch jobs
from S3 storage.
"""

import argparse
import json
import os
import sys
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


def download_job_results(s3_client, bucket: str, job_id: str, out_dir: Path):
    """
    Download all results for a single job from S3.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        job_id: AWS Batch job ID
        out_dir: Local output directory
    """
    job_dir = out_dir / job_id
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading results for job {job_id}")

    # Download logs directory
    prefix = f"jobs/{job_id}/logs/"
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        log_count = 0
        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                # Extract filename from key
                filename = Path(key).name

                if filename:
                    local_path = logs_dir / filename
                    s3_client.download_file(bucket, key, str(local_path))
                    log_count += 1

        if log_count > 0:
            print(f"  ✓ Downloaded {log_count} log file(s) to {logs_dir}")
        else:
            print(f"  ⚠️  No logs found in S3 for job {job_id}")

    except Exception as e:
        print(f"  ✗ Failed to download logs: {e}")

    # Download summary.json
    summary_key = f"jobs/{job_id}/summary.json"
    summary_path = job_dir / "summary.json"

    try:
        s3_client.download_file(bucket, summary_key, str(summary_path))
        print(f"  ✓ Downloaded summary.json to {summary_path}")
    except s3_client.exceptions.NoSuchKey:
        print(f"  ⚠️  No summary.json found in S3 for job {job_id}")
    except Exception as e:
        print(f"  ✗ Failed to download summary.json: {e}")

    print()


def aggregate_summaries(out_dir: Path):
    """
    Aggregate all summary.json files into a combined report.

    Args:
        out_dir: Directory containing job result subdirectories
    """
    print("[INFO] Aggregating summary files...")

    all_summaries = []

    for job_dir in sorted(out_dir.iterdir()):
        if not job_dir.is_dir():
            continue

        summary_file = job_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                    summary["job_id"] = job_dir.name
                    all_summaries.append(summary)
            except Exception as e:
                print(f"  ⚠️  Failed to read {summary_file}: {e}")

    if not all_summaries:
        print("  ⚠️  No summary files found")
        return

    # Write aggregated report
    aggregate_file = out_dir / "aggregated_results.json"
    with open(aggregate_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"  ✓ Aggregated {len(all_summaries)} summaries to {aggregate_file}")

    # Print summary statistics
    print()
    print("════════════════════════════════════════")
    print("Summary Statistics")
    print("════════════════════════════════════════")
    print(f"Total jobs: {len(all_summaries)}")

    # Count by status if available
    status_counts = {}
    for summary in all_summaries:
        status = summary.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Download AlgoTune results from S3")

    parser.add_argument(
        "--job-ids-file", required=True, help="Path to a file with one AWS Batch job ID per line"
    )

    parser.add_argument(
        "--out-dir",
        default="s3_results",
        help="Directory where fetched results should be written (default: s3_results)",
    )

    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_RESULTS_BUCKET", ""),
        help="S3 bucket name (default: from .env S3_RESULTS_BUCKET)",
    )

    args = parser.parse_args()

    # Validate S3 bucket
    if not args.s3_bucket:
        print(
            "ERROR: S3 bucket not specified. Use --s3-bucket or set S3_RESULTS_BUCKET in .env",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load job IDs from file
    try:
        with open(args.job_ids_file) as f:
            job_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: Job IDs file not found: {args.job_ids_file}", file=sys.stderr)
        sys.exit(1)

    if not job_ids:
        print("ERROR: No job IDs found in file", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(job_ids)} job ID(s) to fetch results for")
    print(f"[INFO] S3 bucket: s3://{args.s3_bucket}/")
    print(f"[INFO] Output directory: {args.out_dir}")
    print()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize S3 client
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))

    # Download results for each job
    for job_id in job_ids:
        download_job_results(s3, args.s3_bucket, job_id, out_dir)

    # Aggregate all summaries
    aggregate_summaries(out_dir)

    print("════════════════════════════════════════")
    print("✅ All results downloaded!")
    print(f"Results saved to: {out_dir.absolute()}")
    print("════════════════════════════════════════")


if __name__ == "__main__":
    main()
