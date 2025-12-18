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
from datetime import datetime
from collections import defaultdict

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
    """
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


def display_metrics(stats, refresh_count):
    """Display metrics in a nice format."""
    clear_screen()

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║          AlgoTune AWS Batch Job Monitor                  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Current time and refresh count
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Last updated: {now} (refresh #{refresh_count})")
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

    # Budget statistics
    print("─────────────────────────────────────────────────────────────")
    print("BUDGET STATISTICS")
    print("─────────────────────────────────────────────────────────────")

    if stats['budgets']:
        print(f"  Total Spent:    ${stats['total_budget_used']:.2f}")
        print(f"  Median Spent:   ${stats['median_budget']:.2f}")
        print(f"  Lowest Spent:   ${stats['min_budget']:.2f} ({stats['min_budget_task']})")
        print(f"  Highest Spent:  ${stats['max_budget']:.2f} ({stats['max_budget_task']})")
        print(f"  Jobs w/ data:   {len(stats['budgets'])} / {stats['total_jobs']}")
    else:
        print("  No budget data available yet")
        print("  (Budget info appears once jobs start running)")
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


def monitor_jobs(job_ids_file, interval=30):
    """
    Monitor jobs in real-time.

    Args:
        job_ids_file: Path to file with job IDs (one per line)
        interval: Update interval in seconds
    """
    # Load job IDs
    try:
        with open(job_ids_file) as f:
            job_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: Job IDs file not found: {job_ids_file}", file=sys.stderr)
        sys.exit(1)

    if not job_ids:
        print("ERROR: No job IDs found in file", file=sys.stderr)
        sys.exit(1)

    # Initialize AWS clients
    region = os.getenv('AWS_REGION', 'us-east-1')
    batch_client = boto3.client('batch', region_name=region)
    logs_client = boto3.client('logs', region_name=region)

    print(f"Monitoring {len(job_ids)} jobs...")
    print(f"Update interval: {interval} seconds")
    print()
    time.sleep(2)

    refresh_count = 0

    try:
        while True:
            refresh_count += 1

            # Fetch metrics
            metrics = get_job_metrics(batch_client, logs_client, job_ids)

            # Calculate statistics
            stats = calculate_statistics(metrics)

            # Display
            display_metrics(stats, refresh_count)

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
        sys.exit(0)


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

    args = parser.parse_args()

    monitor_jobs(args.job_ids_file, args.interval)


if __name__ == "__main__":
    main()
