#!/usr/bin/env python3
"""
Prepare retry for tasks with N/A results for a specific model.
Deletes local logs/outputs/errors for those task+model entries and removes the
model entry from reports/agent_summary.json. If a job map exists, optionally
cleans S3 artifacts for the matching job IDs. Outputs tasks to retry (one per line).
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        return model_name
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    return model_name


def load_env_file(path: Path) -> dict:
    if not path.exists():
        return {}
    env = {}
    try:
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        return {}
    return env


def resolve_s3_settings(summary_path: Path, bucket_arg: str, region_arg: str) -> tuple[str, str]:
    bucket = bucket_arg or os.getenv("S3_RESULTS_BUCKET", "")
    region = region_arg or os.getenv("AWS_REGION", "")
    if bucket and region:
        return bucket, region
    repo_root = summary_path.resolve().parent.parent
    env_path = repo_root / "aws" / ".env"
    env = load_env_file(env_path)
    if not bucket:
        bucket = env.get("S3_RESULTS_BUCKET", "")
    if not region:
        region = env.get("AWS_REGION", "")
    return bucket, region


def delete_s3_job_artifacts(bucket: str, region: str, job_ids: list[str]) -> None:
    if not bucket or not job_ids:
        return
    try:
        import boto3
    except Exception as exc:
        print(f"WARNING: boto3 unavailable; skipping S3 cleanup: {exc}", file=sys.stderr)
        return
    try:
        client_kwargs = {}
        if region:
            client_kwargs["region_name"] = region
        s3 = boto3.client("s3", **client_kwargs)
    except Exception as exc:
        print(f"WARNING: failed to init S3 client; skipping cleanup: {exc}", file=sys.stderr)
        return

    paginator = s3.get_paginator("list_objects_v2")
    for job_id in job_ids:
        prefix = f"jobs/{job_id}/"
        keys = []
        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if key:
                        keys.append(key)
                    if len(keys) >= 1000:
                        s3.delete_objects(
                            Bucket=bucket,
                            Delete={"Objects": [{"Key": k} for k in keys]},
                        )
                        keys = []
            if keys:
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": k} for k in keys]},
                )
        except Exception as exc:
            print(
                f"WARNING: failed to delete S3 objects for job {job_id}: {exc}",
                file=sys.stderr,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare retry for N/A tasks.")
    parser.add_argument("--model", required=True, help="Full model name in summary.json")
    parser.add_argument(
        "--model-display", required=True, help="Model display name in log filenames"
    )
    parser.add_argument("--logs-dir", required=True, help="Local logs directory")
    parser.add_argument("--job-map", required=True, help="Path to reports/job_map.json")
    parser.add_argument("--summary-file", required=True, help="Path to reports/agent_summary.json")
    parser.add_argument("--out", required=True, help="Output file for tasks to retry")
    parser.add_argument(
        "--s3-bucket",
        default="",
        help="S3 bucket for cleanup (optional, defaults to S3_RESULTS_BUCKET/aws/.env)",
    )
    parser.add_argument(
        "--aws-region",
        default="",
        help="AWS region for cleanup (optional, defaults to AWS_REGION/aws/.env)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_file)
    summary = load_json(summary_path)
    tasks_to_retry = []
    normalized_model = normalize_model_name(args.model)
    model_keys = [args.model]
    if normalized_model != args.model:
        model_keys.append(normalized_model)

    for task_name, models in summary.items():
        if not isinstance(models, dict):
            continue
        metrics = None
        for key in model_keys:
            candidate = models.get(key)
            if isinstance(candidate, dict):
                metrics = candidate
                break
        if not metrics:
            continue
        speedup = metrics.get("final_speedup")
        if speedup is None or speedup == "N/A":
            tasks_to_retry.append(task_name)

    if not tasks_to_retry:
        Path(args.out).write_text("")
        return 0

    tasks_to_retry_set = set(tasks_to_retry)

    # Delete local logs for the model display name
    logs_dir = Path(args.logs_dir)
    if logs_dir.exists():
        for log_path in logs_dir.glob("*.log"):
            name = log_path.name
            if (
                name.startswith(tuple(f"{task}_" for task in tasks_to_retry))
                and args.model_display in name
            ):
                try:
                    log_path.unlink()
                except Exception:
                    pass

    # Delete outputs/errors for matching job_map entries
    job_map_path = Path(args.job_map)
    job_map = load_json(job_map_path) if job_map_path.exists() else {}
    outputs_dir = Path(args.summary_file).parent.parent / "aws" / "outputs"
    errors_dir = Path(args.summary_file).parent.parent / "aws" / "errors"
    job_ids_to_cleanup = set()
    model_candidates = {args.model_display, args.model, normalized_model}

    for job_id, meta in job_map.items():
        if not isinstance(meta, dict):
            continue
        task = meta.get("task")
        model = meta.get("model")
        if task in tasks_to_retry_set and model in model_candidates:
            job_ids_to_cleanup.add(job_id)
            out_path = outputs_dir / f"batch-{job_id}.out"
            err_path = errors_dir / f"batch-{job_id}.err"
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            if err_path.exists():
                try:
                    err_path.unlink()
                except Exception:
                    pass

    if job_ids_to_cleanup:
        bucket, region = resolve_s3_settings(summary_path, args.s3_bucket, args.aws_region)
        if bucket:
            delete_s3_job_artifacts(bucket, region, sorted(job_ids_to_cleanup))

    # Remove model entries from summary for retry tasks
    for task in tasks_to_retry:
        models = summary.get(task)
        if isinstance(models, dict):
            for key in model_keys:
                if key in models:
                    del models[key]
            if not models:
                summary.pop(task, None)

    save_json(summary_path, summary)

    Path(args.out).write_text("\n".join(tasks_to_retry) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
