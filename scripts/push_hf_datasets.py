#!/usr/bin/env python3
"""
Download datasets from S3 and upload them to Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import boto3
from dotenv import load_dotenv


def load_generation_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"generation.json not found at {path}")
    with path.open("r") as f:
        return json.load(f)


def load_env() -> None:
    root_dotenv = Path(__file__).parent.parent / ".env"
    if root_dotenv.exists():
        load_dotenv(root_dotenv)

    aws_dotenv = Path(__file__).parent.parent / "aws" / ".env"
    if aws_dotenv.exists():
        load_dotenv(aws_dotenv)


def retry_on_502(func, max_retries=5, base_delay=5):
    """Retry a function on 502 Bad Gateway errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            if "502" in error_str or "Bad Gateway" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(
                        f"  ⚠️  502 error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push AlgoTune datasets from S3 to HF")
    parser.add_argument(
        "--repo-id", default=os.environ.get("ALGOTUNE_HF_DATASET", "oripress/AlgoTune")
    )
    parser.add_argument("--revision", default=os.environ.get("ALGOTUNE_HF_REVISION", "main"))
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("S3_RESULTS_BUCKET", "algotune-results-926341122891"),
    )
    parser.add_argument("--s3-prefix", default="datasets")
    parser.add_argument(
        "--tasks", help="Comma-separated list of tasks (default: all in generation.json)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    parser.add_argument("--report-path", default="", help="Write per-task upload report JSON")
    parser.add_argument(
        "--force", action="store_true", help="Force upload even if data already exists in HF"
    )
    parser.add_argument(
        "--batch-files",
        type=int,
        default=int(os.environ.get("ALGOTUNE_HF_BATCH_FILES", "0")),
        help="Upload in batches of N files to limit disk usage (0 disables batching)",
    )
    return parser.parse_args()


def iter_s3_objects(s3_client, bucket: str, prefix: str) -> Iterable[dict]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj


def download_prefix(s3_client, bucket: str, prefix: str, dest_dir: Path, task: str) -> int:
    count = 0
    for obj in iter_s3_objects(s3_client, bucket, prefix):
        key = obj["Key"]
        rel = key[len(prefix) :].lstrip("/")
        if not rel:
            continue
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".jsonl"):
            response = s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            with target.open("wb") as f:
                for chunk in iter_filtered_jsonl(body, task):
                    f.write(chunk)
        else:
            s3_client.download_file(bucket, key, str(target))
        count += 1
    return count


def download_keys(
    s3_client,
    bucket: str,
    prefix: str,
    keys: list[str],
    dest_dir: Path,
    task: str,
) -> int:
    count = 0
    for key in keys:
        rel = key[len(prefix) :].lstrip("/")
        if not rel:
            continue
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".jsonl"):
            response = s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            with target.open("wb") as f:
                for chunk in iter_filtered_jsonl(body, task):
                    f.write(chunk)
        else:
            s3_client.download_file(bucket, key, str(target))
        count += 1
    return count


def iter_filtered_jsonl(streaming_body, task: str) -> Iterator[bytes]:
    for line in streaming_body.iter_lines():
        if not line:
            continue
        record = json.loads(line)
        if "median_oracle_time_ms" in record:
            del record["median_oracle_time_ms"]
        yield (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")


def upload_s3_prefix_to_hf(
    s3_client,
    bucket: str,
    prefix: str,
    task: str,
    api,
    repo_id: str,
    revision: str,
    token: str,
    force: bool,
    batch_files: int,
) -> int:
    repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    task_prefix = f"data/{task}/"
    if not force and any(
        path.startswith(task_prefix) and path.endswith(".jsonl") for path in repo_files
    ):
        return 0

    staging_dir = Path("hf_staging") / task
    if batch_files <= 0:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        count = download_prefix(s3_client, bucket, prefix, staging_dir, task)
        retry_on_502(
            lambda: api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                token=token,
                folder_path=staging_dir,
                path_in_repo=f"data/{task}",
                delete_patterns=f"data/{task}/*",
                commit_message=f"Upload dataset for {task}",
            )
        )
        return count

    keys = [obj["Key"] for obj in iter_s3_objects(s3_client, bucket, prefix)]
    count = 0
    for i in range(0, len(keys), batch_files):
        batch = keys[i : i + batch_files]
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
        count += download_keys(s3_client, bucket, prefix, batch, staging_dir, task)
        retry_on_502(
            lambda: api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                token=token,
                folder_path=staging_dir,
                path_in_repo=f"data/{task}",
                delete_patterns=f"data/{task}/*" if i == 0 else None,
                commit_message=f"Upload dataset for {task} (batch {i // batch_files + 1})",
            )
        )
    return count


def main() -> None:
    load_env()
    args = parse_args()

    if not args.s3_bucket:
        print("ERROR: S3 bucket is required for download", file=sys.stderr)
        sys.exit(1)

    generation_path = Path(__file__).parent.parent / "reports" / "generation.json"
    generation_config = load_generation_config(generation_path)

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = list(generation_config.keys())

    report: dict[str, dict] = {}

    if args.dry_run:
        print("Dry run enabled; skipping HF upload.", file=sys.stderr)
        if args.report_path:
            Path(args.report_path).write_text(json.dumps(report, indent=2))
        return

    try:
        from huggingface_hub import create_repo, HfApi
    except Exception as exc:
        print(f"ERROR: huggingface_hub unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    # Try HF_TOKEN first (official), then fall back to old names for backward compat
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        print(
            "ERROR: HF_TOKEN not set (also tried HUGGING_FACE_TOKEN, HUGGINGFACE_TOKEN)",
            file=sys.stderr,
        )
        sys.exit(1)

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )

    api = HfApi(token=token)
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    for task in tasks:
        task_cfg = generation_config.get(task, {})
        n_value = task_cfg.get("n")
        if n_value is None:
            print(f"WARNING: No n value for {task}; skipping.", file=sys.stderr)
            report[task] = {"s3_downloaded_files": 0, "hf_uploaded": False, "error": "missing_n"}
            continue
        prefix = f"{args.s3_prefix}/{task}_n{n_value}"
        print(
            f"Uploading s3://{args.s3_bucket}/{prefix}/ -> {args.repo_id}:data/{task}",
            file=sys.stderr,
        )
        try:
            count = upload_s3_prefix_to_hf(
                s3_client=s3,
                bucket=args.s3_bucket,
                prefix=prefix,
                task=task,
                api=api,
                repo_id=args.repo_id,
                revision=args.revision,
                token=token,
                force=args.force,
                batch_files=args.batch_files,
            )
            if count == 0 and not args.force:
                print(f"Skipping {task} (already present in HF).", file=sys.stderr)
            else:
                print(f"Uploaded {task} to {args.repo_id}:data/{task}", file=sys.stderr)
            report.setdefault(task, {})
            report[task]["hf_uploaded"] = True
            report[task]["s3_downloaded_files"] = count
            report[task]["error"] = report[task].get("error") or ""
        except Exception as exc:
            print(f"ERROR: Failed to upload {task}: {exc}", file=sys.stderr)
            report.setdefault(task, {})
            report[task]["hf_uploaded"] = False
            report[task]["error"] = report[task].get("error") or f"hf_upload_failed: {exc}"

    if args.report_path:
        Path(args.report_path).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
