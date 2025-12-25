#!/usr/bin/env python3
"""
Generate HTML visualizations for AWS Batch job results.
Fetches logs from S3 and generates static HTML site like algotune.io.
"""
import os
import sys
import json
import boto3
import argparse
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment from both .env files
# 1. Load API keys from root .env
repo_root = Path(__file__).resolve().parent.parent
root_dotenv = repo_root / ".env"
if root_dotenv.exists():
    load_dotenv(root_dotenv)

# 2. Load AWS configuration from aws/.env
aws_dotenv = repo_root / "aws" / ".env"
if aws_dotenv.exists():
    load_dotenv(aws_dotenv)

def fetch_logs_from_s3(s3_client, bucket, job_ids, temp_logs_dir):
    """
    Download all logs from S3 to a temporary directory in the format
    expected by batch_html_generator (logs/*.log files).
    """
    print(f"Fetching logs from S3 bucket: {bucket}")

    for job_id in job_ids:
        print(f"  Downloading logs for job: {job_id}")

        # Download log files
        prefix = f"jobs/{job_id}/logs/"
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            for page in pages:
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.log'):
                        # Download to temp logs directory
                        filename = Path(key).name
                        local_path = temp_logs_dir / filename

                        response = s3_client.get_object(Bucket=bucket, Key=key)
                        log_content = response['Body'].read().decode('utf-8')

                        with open(local_path, 'w') as f:
                            f.write(log_content)

                        print(f"    Downloaded: {filename}")

        except Exception as e:
            print(f"    Warning: Could not fetch logs for {job_id}: {e}")
            continue

        # Download summary.json (if exists)
        try:
            summary_key = f"jobs/{job_id}/summary.json"
            response = s3_client.get_object(Bucket=bucket, Key=summary_key)
            summary_content = response['Body'].read().decode('utf-8')

            summary_path = temp_logs_dir / f"{job_id}_summary.json"
            with open(summary_path, 'w') as f:
                f.write(summary_content)

            print(f"    Downloaded: summary.json")
        except Exception as e:
            print(f"    Warning: No summary.json for {job_id}")

    print()

def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML visualization for AWS Batch job results"
    )
    parser.add_argument(
        "--job-ids-file",
        required=True,
        help="File containing job IDs (one per line)"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for HTML files"
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_RESULTS_BUCKET"),
        help="S3 bucket name (default: from .env)"
    )

    args = parser.parse_args()

    # Validate S3 bucket
    if not args.s3_bucket:
        print("ERROR: S3 bucket not specified. Use --s3-bucket or set S3_RESULTS_BUCKET in .env")
        sys.exit(1)

    # Load job IDs
    try:
        with open(args.job_ids_file) as f:
            job_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: Job IDs file not found: {args.job_ids_file}")
        sys.exit(1)

    if not job_ids:
        print("ERROR: No job IDs found in file")
        sys.exit(1)

    print(f"Found {len(job_ids)} job IDs")
    print()

    # Initialize S3 client
    s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))

    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_logs_dir = Path(temp_dir) / "logs"
        temp_logs_dir.mkdir()

        # Fetch all logs from S3
        fetch_logs_from_s3(s3, args.s3_bucket, job_ids, temp_logs_dir)

        # Check if we got any logs
        log_files = list(temp_logs_dir.glob("*.log"))
        if not log_files:
            print("ERROR: No log files were downloaded from S3")
            sys.exit(1)

        print(f"Downloaded {len(log_files)} log files")
        print()

        # Now run the HTML generator
        print("Generating HTML visualization...")
        print()

        # Import and configure the batch_html_generator
        sys.path.insert(0, str(Path(__file__).parent))
        import batch_html_generator

        # Override configuration
        batch_html_generator.LOGS_DIR = str(temp_logs_dir) + "/"
        batch_html_generator.OUTPUT_DIR = args.out_dir
        batch_html_generator.PLOTS_DIR = os.path.join(args.out_dir, "assets", "plots")
        os.makedirs(batch_html_generator.PLOTS_DIR, exist_ok=True)

        # Copy styles.css to output directory
        styles_source = Path(__file__).parent / "styles.css"
        styles_dest = Path(args.out_dir) / "styles.css"
        if styles_source.exists():
            shutil.copy2(styles_source, styles_dest)

        # Copy logo assets
        assets_source = Path(__file__).parent / "assets"
        assets_dest = Path(args.out_dir) / "assets"
        if assets_source.exists():
            shutil.copytree(assets_source, assets_dest, dirs_exist_ok=True)

        # Run the main HTML generation
        try:
            batch_html_generator.main()
        except Exception as e:
            print(f"ERROR during HTML generation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print()
    print("=" * 50)
    print("✅ HTML visualization generated successfully!")
    print(f"Output directory: {args.out_dir}")
    print(f"Open in browser: {args.out_dir}/index.html")
    print("=" * 50)

if __name__ == "__main__":
    main()
