#!/usr/bin/env python3
"""
Real-time monitoring for AWS Batch jobs.
Displays job status, cost metrics, and progress.
"""
import os
import sys
import json
import time
import boto3
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import defaultdict

from download_logs import download_job_logs, cleanup_failed_logs

# Load environment from both .env files
root_dotenv = Path(__file__).parent.parent / ".env"
if root_dotenv.exists():
    load_dotenv(root_dotenv)

aws_dotenv = Path(__file__).parent / ".env"
load_dotenv(aws_dotenv)


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def extract_task_from_job_name(job_name):
    """Extract task name from job name (format: algotune-{task}-{timestamp})."""
    match = re.match(r'algotune-(.+?)-\d+', job_name)
    if match:
        return match.group(1)
    return "unknown"


def extract_budget_from_logs(logs_client, log_stream_name):
    """
    Extract budget information from CloudWatch logs.
    Returns (used_budget, total_budget) or (None, None) if not found.

    Note: CloudWatch access is optional. If permissions are missing, returns (None, None).
    """
    if not logs_client or not log_stream_name:
        return None, None

    try:
        response = logs_client.get_log_events(
            logGroupName='/aws/batch/job',
            logStreamName=log_stream_name,
            limit=10000  # Get a large chunk to find budget info
        )

        log_text = '\n'.join([event['message'] for event in response['events']])

        # Look for budget patterns in logs
        # Pattern: "used up $X.XX ... You have $Y.YY remaining"
        budget_pattern = re.compile(
            r"used up\s*\$([0-9]+(?:\.[0-9]+)?)[^$]*You have\s*\$([0-9]+(?:\.[0-9]+)?)\s*remaining",
            re.IGNORECASE
        )

        match = budget_pattern.search(log_text)
        if match:
            used = float(match.group(1))
            remaining = float(match.group(2))
            total = used + remaining
            return used, total

        # Alternative: just look for "used up $X"
        used_pattern = re.compile(r"used up\s*\$([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        match = used_pattern.search(log_text)
        if match:
            used = float(match.group(1))
            return used, None

    except Exception as e:
        # CloudWatch access may fail due to missing permissions - this is OK
        # Budget info is optional, jobs will still work without it
        pass

    return None, None


def get_job_metrics(batch_client, logs_client, job_ids):
    """
    Fetch metrics for all jobs.

    Returns:
        dict with job_id -> {
            'status': str,
            'task_name': str,
            'used_budget': float or None,
            'total_budget': float or None,
            'started_at': datetime or None,
            'stopped_at': datetime or None,
            'duration_seconds': int or None
        }
    """
    metrics = {}

    # Describe all jobs in batches (max 100 per call)
    for i in range(0, len(job_ids), 100):
        batch = job_ids[i:i+100]

        try:
            response = batch_client.describe_jobs(jobs=batch)

            for job in response['jobs']:
                job_id = job['jobId']
                job_name = job['jobName']
                status = job['status']  # SUBMITTED, PENDING, RUNNABLE, STARTING, RUNNING, SUCCEEDED, FAILED
                task_name = extract_task_from_job_name(job_name)

                # Get timing info
                started_at = None
                stopped_at = None
                duration_seconds = None

                if 'startedAt' in job:
                    started_at = datetime.fromtimestamp(job['startedAt'] / 1000)

                if 'stoppedAt' in job:
                    stopped_at = datetime.fromtimestamp(job['stoppedAt'] / 1000)

                if started_at and stopped_at:
                    duration_seconds = int((stopped_at - started_at).total_seconds())
                elif started_at:
                    duration_seconds = int((datetime.now() - started_at).total_seconds())

                # Get budget info from logs if job is running or completed
                used_budget = None
                total_budget = None

                if status in ['RUNNING', 'SUCCEEDED', 'FAILED']:
                    log_stream = job.get('container', {}).get('logStreamName')
                    if log_stream:
                        used_budget, total_budget = extract_budget_from_logs(logs_client, log_stream)

                metrics[job_id] = {
                    'status': status,
                    'task_name': task_name,
                    'used_budget': used_budget,
                    'total_budget': total_budget,
                    'started_at': started_at,
                    'stopped_at': stopped_at,
                    'duration_seconds': duration_seconds
                }

        except Exception as e:
            print(f"Error fetching job metrics: {e}", file=sys.stderr)

    return metrics


def calculate_statistics(metrics):
    """Calculate aggregate statistics from job metrics."""
    stats = {
        'total_jobs': len(metrics),
        'running': 0,
        'succeeded': 0,
        'failed': 0,
        'pending': 0,
        'other': 0,
        'budgets': [],  # List of (used_budget, task_name)
        'median_budget': None,
        'min_budget': None,
        'min_budget_task': None,
        'max_budget': None,
        'max_budget_task': None,
        'total_budget_used': 0.0
    }

    for job_id, info in metrics.items():
        status = info['status']

        if status == 'RUNNING':
            stats['running'] += 1
        elif status == 'SUCCEEDED':
            stats['succeeded'] += 1
        elif status == 'FAILED':
            stats['failed'] += 1
        elif status in ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING']:
            stats['pending'] += 1
        else:
            stats['other'] += 1

        # Track budgets
        if info['used_budget'] is not None:
            used = info['used_budget']
            task = info['task_name']
            stats['budgets'].append((used, task))
            stats['total_budget_used'] += used

    # Calculate budget statistics
    if stats['budgets']:
        stats['budgets'].sort()  # Sort by budget
        n = len(stats['budgets'])

        # Median
        if n % 2 == 0:
            stats['median_budget'] = (stats['budgets'][n//2 - 1][0] + stats['budgets'][n//2][0]) / 2
        else:
            stats['median_budget'] = stats['budgets'][n//2][0]

        # Min and max
        stats['min_budget'], stats['min_budget_task'] = stats['budgets'][0]
        stats['max_budget'], stats['max_budget_task'] = stats['budgets'][-1]

    return stats


def format_duration(seconds):
    """Format duration in seconds to human-readable string."""
    if seconds is None:
        return "N/A"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def display_metrics(stats, refresh_count, last_updated, interval_seconds, start_time=None):
    """Display metrics in a nice format."""
    clear_screen()

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║          AlgoTune AWS Batch Job Monitor                  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Start time and current time
    if start_time:
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = last_updated - start_time
        elapsed_str = format_duration(int(elapsed.total_seconds()))
        print(f"Started:       {start_str} ({elapsed_str} ago)")

    last_str = last_updated.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Last updated:  {last_str}")
    print()

    # Job status summary
    print("─────────────────────────────────────────────────────────────")
    print("JOB STATUS SUMMARY")
    print("─────────────────────────────────────────────────────────────")
    print(f"  Total Jobs:     {stats['total_jobs']}")
    print(f"  Running:        {stats['running']}")
    print(f"  Succeeded:      {stats['succeeded']}")
    print(f"  Failed:         {stats['failed']}")
    print(f"  Pending:        {stats['pending']}")
    print()

    # Show helpful note if jobs are pending and no logs yet
    if stats['pending'] > 0 and stats['running'] == 0 and stats['succeeded'] == 0:
        print("  💡 Logs should appear in the logs/ directory soon.")
        print("     If not, jobs are likely generating datasets (can take 1-2 hours).")
        print()

    # Status bar
    total = stats['total_jobs']
    if total > 0:
        completed = stats['succeeded'] + stats['failed']
        pct_complete = (completed / total) * 100
        bar_width = 50
        filled = int(bar_width * completed / total)
        bar = '█' * filled + '░' * (bar_width - filled)

        print("─────────────────────────────────────────────────────────────")
        print("PROGRESS")
        print("─────────────────────────────────────────────────────────────")
        print(f"  [{bar}] {pct_complete:.1f}%")
        print(f"  {completed} / {total} jobs completed")
        print()

    print("─────────────────────────────────────────────────────────────")
    print("Press Ctrl+C to exit")
    print()


def monitor_jobs(job_ids_file, interval=30, sync_logs=False, s3_bucket="", output_dir=".", cleanup_failed=False, summary_file=""):
    """
    Monitor jobs in real-time.

    Args:
        job_ids_file: Path to file with job IDs (one per line)
        interval: Update interval in seconds
    """
    def load_job_ids():
        try:
            with open(job_ids_file) as f:
                ids = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"ERROR: Job IDs file not found: {job_ids_file}", file=sys.stderr)
            sys.exit(1)
        if not ids:
            print("ERROR: No job IDs found in file", file=sys.stderr)
            sys.exit(1)
        return ids

    job_ids = load_job_ids()

    # Initialize AWS clients
    region = os.getenv('AWS_REGION', 'us-east-1')
    batch_client = boto3.client('batch', region_name=region)
    s3_client = None
    if sync_logs and s3_bucket:
        s3_client = boto3.client('s3', region_name=region)

    # CloudWatch logs client is optional (previously used for budget info)
    # Keeping it as None since we don't display budget stats
    logs_client = None

    print(f"Monitoring {len(job_ids)} jobs...")
    print(f"Update interval: {interval} seconds")
    print()
    time.sleep(2)

    refresh_count = 0
    start_time = datetime.now()

    try:
        while True:
            refresh_count += 1

            # Refresh job IDs each iteration in case file changed
            job_ids = load_job_ids()

            # Fetch metrics
            metrics = get_job_metrics(batch_client, logs_client, job_ids)

            # Calculate statistics
            stats = calculate_statistics(metrics)

            if sync_logs and s3_client and s3_bucket:
                for job_id in job_ids:
                    download_job_logs(s3_client, s3_bucket, job_id, output_dir)
            if cleanup_failed and summary_file:
                cleanup_failed_logs(Path(output_dir), Path(summary_file))

            # Display
            display_metrics(stats, refresh_count, datetime.now(), interval, start_time)

            # Check if all jobs are done
            if stats['running'] == 0 and stats['pending'] == 0:
                print("═══════════════════════════════════════════════════════════")
                print("✅ All jobs completed!")
                print("═══════════════════════════════════════════════════════════")
                break

            # Wait for next update
            time.sleep(interval)

    except KeyboardInterrupt:
        print()
        print()
        print("═══════════════════════════════════════════════════════════")
        print("Monitoring stopped by user")
        print("═══════════════════════════════════════════════════════════")
        # Exit with 130 (conventional exit code for SIGINT/Ctrl+C)
        # This signals to parent script that cleanup is needed
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor AWS Batch jobs in real-time"
    )
    parser.add_argument(
        "--job-ids-file",
        required=True,
        help="File containing job IDs (one per line)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Update interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--sync-logs",
        action="store_true",
        help="Download logs/aws outputs/errors on each refresh"
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_RESULTS_BUCKET", ""),
        help="S3 bucket name (default: from aws/.env)"
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root directory for logs (default: repo root)"
    )
    parser.add_argument(
        "--cleanup-failed",
        action="store_true",
        help="Delete logs not marked successful in reports/agent_summary.json"
    )
    parser.add_argument(
        "--summary-file",
        default=str(Path(__file__).resolve().parents[1] / "reports" / "agent_summary.json"),
        help="Summary file for cleanup (default: reports/agent_summary.json)"
    )

    args = parser.parse_args()

    monitor_jobs(
        args.job_ids_file,
        args.interval,
        sync_logs=args.sync_logs,
        s3_bucket=args.s3_bucket,
        output_dir=args.output_dir,
        cleanup_failed=args.cleanup_failed,
        summary_file=args.summary_file,
    )


if __name__ == "__main__":
    main()
