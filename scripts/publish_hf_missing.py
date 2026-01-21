#!/usr/bin/env python3
"""
Generate + upload missing/incomplete tasks to Hugging Face, using per-task k from reports/generation.json.

Workflow:
  1) List tasks missing/incomplete on HF.
  2) Generate dataset via Docker (public image contains dependencies).
  3) Upload to HF dataset repo.
  4) Remove staging data to save disk space.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

from huggingface_hub import HfApi


def _load_generation_config(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")
    return data


def _list_local_tasks(tasks_root: Path) -> List[str]:
    tasks = []
    for entry in tasks_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        tasks.append(entry.name)
    return sorted(tasks)


def _index_repo_files(repo_files: Iterable[str]) -> Dict[str, List[str]]:
    by_task: Dict[str, List[str]] = {}
    for path in repo_files:
        if not path.startswith("data/"):
            continue
        parts = path.split("/", 2)
        if len(parts) < 2:
            continue
        task = parts[1]
        by_task.setdefault(task, []).append(path)
    return by_task


def _task_complete(task: str, files: List[str], require_npy: bool) -> bool:
    if not files:
        return False
    has_train = any(f.endswith("_train.jsonl") for f in files)
    has_test = any(f.endswith("_test.jsonl") for f in files)
    if not (has_train and has_test):
        return False
    if not require_npy:
        return True
    npy_prefix = f"data/{task}/_npy_data/"
    has_npy = any(
        f.startswith(npy_prefix) and (f.endswith(".npy") or f.endswith(".bytes"))
        for f in files
    )
    return has_npy


def _docker_generate(
    image: str,
    task: str,
    staging_dir: Path,
    target_time_ms: int,
    k_value: int,
    train_size: int,
    test_size: int | None,
    timeout_s: int | None,
) -> None:
    staging_dir = staging_dir.resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "TARGET_TIME_MS": str(target_time_ms),
        "ALGOTUNE_HF_DISABLE": "1",
    }
    container_name = f"algotune_gen_{task}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-w",
        "/data",
    ]
    for key, value in env.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.extend(["-v", f"{staging_dir}:/data"])
    cmd.append(image)
    cmd_str = (
        "python3 /app/AlgoTuner/scripts/generate_and_annotate.py "
        f"{task} --data-dir /data --train-size {train_size} --k {k_value}"
    )
    if test_size is not None:
        cmd_str += f" --test-size {test_size}"
    cmd.append(cmd_str)

    print(f"▶ Generating {task} via Docker: {image}")
    try:
        subprocess.run(cmd, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "stop", container_name], check=False)
        raise RuntimeError(f"Generation timed out for {task} after {timeout_s}s") from exc


def _upload_to_hf(api: HfApi, repo_id: str, task: str, staging_dir: Path) -> None:
    print(f"⬆ Uploading {task} to {repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(staging_dir),
        path_in_repo=f"data/{task}",
        commit_message=f"Add dataset for {task}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate + upload missing HF tasks.")
    parser.add_argument("--repo-id", default="oripress/AlgoTune", help="HF dataset repo id")
    parser.add_argument(
        "--generation-json",
        default="reports/generation.json",
        help="Path to generation.json with per-task k",
    )
    parser.add_argument(
        "--tasks-root",
        default="AlgoTuneTasks",
        help="Root directory of local tasks",
    )
    parser.add_argument(
        "--staging-dir",
        default="hf_staging",
        help="Local staging directory (per-task subdirs are created here)",
    )
    parser.add_argument(
        "--docker-image",
        default=os.environ.get("DOCKER_IMAGE", "ghcr.io/oripress/algotune:latest"),
        help="Docker image used for dataset generation",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit number of tasks to process",
    )
    parser.add_argument(
        "--skip-tasks",
        default="",
        help="Comma-separated list of task names to skip",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=None,
        help="Per-task timeout in seconds for Docker generation",
    )
    parser.add_argument(
        "--require-npy",
        action="store_true",
        help="Require _npy_data files to consider a task complete on HF",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Do not delete staging data after upload",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list tasks to process")
    args = parser.parse_args()

    generation_json = Path(args.generation_json)
    tasks_root = Path(args.tasks_root)
    staging_root = Path(args.staging_dir)

    gen_cfg = _load_generation_config(generation_json)
    local_tasks = _list_local_tasks(tasks_root)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    by_task = _index_repo_files(repo_files)

    skip_tasks = {t.strip() for t in args.skip_tasks.split(",") if t.strip()}
    missing_or_incomplete: List[str] = []
    for task in local_tasks:
        if task in skip_tasks:
            continue
        files = by_task.get(task, [])
        if not _task_complete(task, files, require_npy=args.require_npy):
            missing_or_incomplete.append(task)

    if args.max_tasks is not None:
        missing_or_incomplete = missing_or_incomplete[: args.max_tasks]

    if not missing_or_incomplete:
        print("✓ All tasks appear complete on HF.")
        return

    print(f"Found {len(missing_or_incomplete)} tasks missing/incomplete on HF.")
    if args.dry_run:
        print("\n".join(missing_or_incomplete))
        return

    for task in missing_or_incomplete:
        if task not in gen_cfg:
            print(f"⚠ Skipping {task}: missing in {generation_json}")
            continue
        cfg = gen_cfg[task]
        target_time_ms = int(cfg["target_time_ms"])
        k_value = int(cfg["n"])
        train_size = int(cfg.get("dataset_size", 100))
        test_size = None
        if "test_size" in cfg:
            test_size = int(cfg["test_size"])

        task_staging = staging_root / task
        if task_staging.exists():
            shutil.rmtree(task_staging)
        task_staging.mkdir(parents=True, exist_ok=True)

        try:
            _docker_generate(
                image=args.docker_image,
                task=task,
                staging_dir=task_staging,
                target_time_ms=target_time_ms,
                k_value=k_value,
                train_size=train_size,
                test_size=test_size,
                timeout_s=args.task_timeout,
            )
            _upload_to_hf(api, args.repo_id, task, task_staging)
        except Exception as exc:
            print(f"❌ {task} failed: {exc}")
        finally:
            if not args.keep_staging and task_staging.exists():
                shutil.rmtree(task_staging)


if __name__ == "__main__":
    main()
