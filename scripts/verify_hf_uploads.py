#!/usr/bin/env python3
"""
Verify which tasks are actually uploaded to HuggingFace.
Updates hf_upload_completed.json to only include tasks that are actually present.
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main():
    # Configuration
    repo_id = "oripress/AlgoTune"
    generation_file = Path("reports/generation.json")
    completed_file = Path("reports/hf_upload_completed.json")

    # Get HF token
    token = (
        os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    )
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    # Load all tasks
    if not generation_file.exists():
        print(f"ERROR: {generation_file} not found")
        sys.exit(1)

    with open(generation_file) as f:
        all_tasks = list(json.load(f).keys())

    print(f"Total tasks in generation.json: {len(all_tasks)}")

    # Check what's on HuggingFace
    api = HfApi(token=token)
    print("Fetching file list from HuggingFace...")

    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        # Find which tasks have data on HF
        tasks_on_hf = set()
        for file_path in repo_files:
            # Files are in format: data/task_name/*.jsonl
            if file_path.startswith("data/") and file_path.endswith(".jsonl"):
                parts = file_path.split("/")
                if len(parts) >= 2:
                    task_name = parts[1]
                    tasks_on_hf.add(task_name)

        print(f"Tasks on HuggingFace: {len(tasks_on_hf)}")

        # Find missing tasks
        missing_tasks = set(all_tasks) - tasks_on_hf
        print(f"Missing tasks: {len(missing_tasks)}")

        if missing_tasks:
            print("\nMissing tasks:")
            for task in sorted(missing_tasks)[:20]:
                print(f"  - {task}")
            if len(missing_tasks) > 20:
                print(f"  ... and {len(missing_tasks) - 20} more")

        # Update completed file to only include tasks actually on HF
        verified_completed = sorted([t for t in all_tasks if t in tasks_on_hf])

        with open(completed_file, "w") as f:
            json.dump(verified_completed, f, indent=2)

        print(f"\nUpdated {completed_file}")
        print(f"  Verified completed: {len(verified_completed)}")
        print(f"  Need to upload: {len(missing_tasks)}")

    except Exception as e:
        print(f"ERROR: Failed to check HuggingFace: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
