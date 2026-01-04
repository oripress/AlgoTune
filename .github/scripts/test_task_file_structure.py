#!/usr/bin/env python3
"""
Task File Structure Testing Script

This script validates that each task directory contains exactly 2 files:
1. description.txt
2. task_name.py (where task_name matches the directory name)

Any additional files will cause the test to fail.
__pycache__ directories are ignored.
"""

import os
import sys

def check_task_file_structure():
    tasks_dir = "AlgoTuneTasks"
    errors = []

    # Directories to exclude from checking
    excluded_dirs = {'__pycache__', 'base', 'template', '.git', '.ipynb_checkpoints'}

    # Directories that are ignored
    ignored_dirs = {'__pycache__'}

    for task_dir in sorted(os.listdir(tasks_dir)):
        task_path = os.path.join(tasks_dir, task_dir)

        # Skip if not a directory or if it's excluded
        if not os.path.isdir(task_path) or task_dir in excluded_dirs:
            continue

        # Get all files in the task directory (non-recursive)
        all_items = os.listdir(task_path)

        # Filter out ignored directories and files
        files = []
        for item in all_items:
            item_path = os.path.join(task_path, item)
            if os.path.isdir(item_path):
                if item not in ignored_dirs:
                    # Found a subdirectory that's not ignored
                    errors.append(f"Task '{task_dir}' contains unexpected subdirectory: {item}")
            else:
                files.append(item)

        # Expected files
        expected_files = {
            'description.txt',
            f'{task_dir}.py'
        }

        # Check if we have exactly the expected files
        actual_files = set(files)

        if actual_files != expected_files:
            # Detailed error message
            missing_files = expected_files - actual_files
            extra_files = actual_files - expected_files

            error_msg = f"Task '{task_dir}' has incorrect file structure:"

            if missing_files:
                error_msg += f"\n  - Missing required files: {', '.join(sorted(missing_files))}"

            if extra_files:
                error_msg += f"\n  - Unexpected files: {', '.join(sorted(extra_files))}"
                error_msg += f"\n  - Action required: Delete the unexpected files listed above"

            error_msg += f"\n  - Expected exactly: description.txt and {task_dir}.py"
            error_msg += f"\n  - Found: {', '.join(sorted(actual_files)) if actual_files else '(no files)'}"

            errors.append(error_msg)

    if errors:
        print("Task File Structure Check FAILED!\n")
        print("Errors found:")
        print("-" * 60)
        for error in errors:
            print(error)
            print("-" * 60)
        print(f"\nTotal errors: {len(errors)}")
        sys.exit(1)
    else:
        print("Task File Structure Check PASSED!")
        print("All task directories contain exactly the required files:")
        print("  - description.txt")
        print("  - {task_name}.py")
        sys.exit(0)

if __name__ == "__main__":
    check_task_file_structure()
