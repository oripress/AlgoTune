#!/usr/bin/env python3
"""
Submit dataset generation jobs in parallel and upload each task to HF on completion.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv


TERMINAL_STATES = {"SUCCEEDED", "FAILED"}
LOG_PATH = Path("reports") / "publish_hf_parallel.log"


def log(message: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp} UTC] {message}"
    print(line, file=sys.stderr)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def load_env() -> None:
    root_dotenv = Path(__file__).parent.parent / ".env"
    if root_dotenv.exists():
        load_dotenv(root_dotenv)

    aws_dotenv = Path(__file__).parent.parent / "aws" / ".env"
    if aws_dotenv.exists():
        load_dotenv(aws_dotenv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate datasets in parallel, upload to HF per task"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tasks", help="Comma-separated list of tasks")
    group.add_argument("--all-tasks", action="store_true", help="Run on all tasks")
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_RESULTS_BUCKET", "algotune-results-926341122891"),
    )
    parser.add_argument("--dataset-prefix", default="datasets")
    parser.add_argument("--repo-id", default=os.getenv("ALGOTUNE_HF_DATASET", "oripress/AlgoTune"))
    parser.add_argument("--revision", default=os.getenv("ALGOTUNE_HF_REVISION", "main"))
    parser.add_argument("--poll-interval", type=int, default=20)
    parser.add_argument(
        "--batch-size", type=int, default=6, help="Max tasks to generate/upload per batch"
    )
    parser.add_argument(
        "--shutdown", action="store_true", help="Shutdown the instance after completion"
    )
    parser.add_argument("--report-path", default="reports/hf_generation_parallel.json")
    return parser.parse_args()


def _current_instance_id() -> str:
    try:
        import urllib.request

        with urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-id", timeout=0.2
        ) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception:
        return ""


def cancel_batch_jobs(job_ids: list[str], region: str) -> None:
    if not job_ids:
        return
    batch = boto3.client("batch", region_name=region)

    log("⚠️  Interrupt received. Canceling queued jobs and terminating running jobs...")
    for i in range(0, len(job_ids), 100):
        chunk = job_ids[i : i + 100]
        try:
            desc = batch.describe_jobs(jobs=chunk)
        except Exception as exc:
            log(f"WARNING: Failed to describe jobs: {exc}")
            desc = {"jobs": [{"jobId": j, "status": ""} for j in chunk]}
        for job in desc.get("jobs", []):
            job_id = job.get("jobId")
            status = job.get("status")
            if not job_id:
                continue
            if status in {"SUBMITTED", "PENDING", "RUNNABLE"}:
                batch.cancel_job(jobId=job_id, reason="Ctrl+C kill switch")
            elif status in {"STARTING", "RUNNING"}:
                batch.terminate_job(jobId=job_id, reason="Ctrl+C kill switch")


def discover_all_tasks() -> list[str]:
    tasks_root = Path(__file__).parent.parent / "AlgoTuneTasks"
    tasks = []
    if not tasks_root.exists():
        return tasks
    for task_dir in sorted(tasks_root.iterdir()):
        if task_dir.is_dir() and (task_dir / "description.txt").exists():
            tasks.append(task_dir.name)
    return tasks


def load_generation_tasks() -> list[str]:
    generation_file = Path(__file__).parent.parent / "reports" / "generation.json"
    if not generation_file.exists():
        return []
    try:
        return list(json.loads(generation_file.read_text()).keys())
    except Exception as exc:
        log(f"WARNING: Failed to load generation.json: {exc}")
        return []


def resolve_tasks(args: argparse.Namespace) -> list[str]:
    if args.all_tasks:
        tasks = load_generation_tasks()
        if not tasks:
            tasks = discover_all_tasks()
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    available = set(discover_all_tasks())
    missing = [t for t in tasks if available and t not in available]
    if missing:
        log(f"WARNING: Skipping unknown tasks: {', '.join(missing)}")
    return [t for t in tasks if not available or t in available]


def chunk_tasks(tasks: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        return [tasks]
    return [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]


def submit_jobs(args: argparse.Namespace, tasks: list[str]) -> list[str]:
    if not tasks:
        log("ERROR: No tasks to submit.")
        sys.exit(1)
    cmd = [
        "python3",
        "aws/submit_dataset_jobs.py",
        "--s3-bucket",
        args.s3_bucket,
        "--dataset-prefix",
        args.dataset_prefix,
    ]
    cmd.extend(["--tasks", ",".join(tasks)])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        log("ERROR: Dataset job submission failed.")
        sys.exit(proc.returncode)

    job_ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not job_ids:
        log("ERROR: No job IDs returned from submit_dataset_jobs.py")
        sys.exit(1)
    return job_ids


def describe_jobs(batch_client, job_ids: list[str]) -> dict[str, dict]:
    info: dict[str, dict] = {}
    for i in range(0, len(job_ids), 100):
        chunk = job_ids[i : i + 100]
        desc = batch_client.describe_jobs(jobs=chunk)
        for job in desc["jobs"]:
            info[job["jobId"]] = job
    return info


def push_task(args: argparse.Namespace, task: str, report_path: str) -> int:
    cmd = [
        "python3",
        "scripts/push_hf_datasets.py",
        "--repo-id",
        args.repo_id,
        "--revision",
        args.revision,
        "--s3-bucket",
        args.s3_bucket,
        "--s3-prefix",
        args.dataset_prefix,
        "--report-path",
        report_path,
        "--tasks",
        task,
    ]
    return subprocess.run(cmd, check=False).returncode


def cleanup_local_task(task: str) -> None:
    staging_dir = Path("hf_staging") / task
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)


def maybe_shutdown() -> None:
    log("Shutting down instance...")
    subprocess.run(["shutdown", "-P", "now"], check=False)


def main() -> None:
    load_env()
    args = parse_args()

    if not args.s3_bucket:
        log("ERROR: S3 bucket is required")
        sys.exit(1)

    tasks = resolve_tasks(args)
    if not tasks:
        log("ERROR: No tasks selected.")
        sys.exit(1)

    job_ids: list[str] = []

    def _handle_interrupt(signum, _frame):
        cancel_batch_jobs(job_ids, os.getenv("AWS_REGION", "us-east-1"))
        sys.exit(130)

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    batch = boto3.client("batch", region_name=os.getenv("AWS_REGION"))
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    combined: dict[str, dict] = {}
    exit_code = 0

    task_batches = chunk_tasks(tasks, args.batch_size)

    for batch_idx, task_batch in enumerate(task_batches, start=1):
        job_ids = submit_jobs(args, task_batch)
        log(f"Submitted {len(job_ids)} dataset jobs (batch {batch_idx}/{len(task_batches)}).")

        pending = set(job_ids)
        while pending:
            job_info = describe_jobs(batch, list(pending))
            for job_id, job in job_info.items():
                status = job.get("status")
                if status not in TERMINAL_STATES:
                    continue
                job_name = job.get("jobName", "")
                task = job_name.replace("algotune-dataset-", "", 1)

                combined.setdefault(task, {})
                combined[task]["generation_status"] = status
                combined[task]["job_id"] = job_id

                if status == "FAILED":
                    combined[task]["hf_uploaded"] = False
                    combined[task]["hf_error"] = "generation_failed"
                    exit_code = 1
                    log(f"Generation failed for {task} (job {job_id})")
                else:
                    hf_report_path = str(report_path.with_name(f"hf_upload_report_{task}.json"))
                    hf_exit = push_task(args, task, hf_report_path)
                    if hf_exit != 0:
                        exit_code = 1
                        log(f"HF upload failed for {task} (exit {hf_exit})")
                    if Path(hf_report_path).exists():
                        try:
                            hf_report = json.loads(Path(hf_report_path).read_text())
                            info = hf_report.get(task, {})
                            combined[task]["hf_uploaded"] = info.get("hf_uploaded", False)
                            combined[task]["s3_downloaded_files"] = info.get(
                                "s3_downloaded_files", 0
                            )
                            combined[task]["hf_error"] = info.get("error", "")
                        except Exception as exc:
                            combined[task]["hf_uploaded"] = False
                            combined[task]["hf_error"] = f"hf_report_read_failed: {exc}"
                            log(f"WARNING: Failed to read HF report for {task}: {exc}")
                    cleanup_local_task(task)
                    log(f"Wrote report update for {task} to {report_path}")

                pending.discard(job_id)

            report_path.write_text(json.dumps(combined, indent=2))
            if pending:
                time.sleep(args.poll_interval)

    log(f"Wrote combined report to {report_path}")

    if args.shutdown:
        maybe_shutdown()

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
