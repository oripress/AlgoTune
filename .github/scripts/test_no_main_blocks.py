#!/usr/bin/env python3
"""
No Main Blocks Testing Script

This script validates that no task Python files contain __main__ blocks.
Task files should be importable modules without executable code at the module level.
"""

import os
import sys
import re

def check_no_main_blocks():
    """Check that no task Python files contain __main__ blocks."""
    tasks_dir = "AlgoTuneTasks"
    errors = []

    # Directories to exclude from checking
    excluded_dirs = {'__pycache__', 'base', 'template', '.git', '.ipynb_checkpoints'}

    # Pattern to match __main__ blocks
    main_pattern = re.compile(r'^\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:', re.MULTILINE)

    for task_dir in sorted(os.listdir(tasks_dir)):
        task_path = os.path.join(tasks_dir, task_dir)

        # Skip if not a directory or if it's excluded
        if not os.path.isdir(task_path) or task_dir in excluded_dirs:
            continue

        # Check the main Python file for this task
        python_file = os.path.join(task_path, f"{task_dir}.py")

        if not os.path.exists(python_file):
            # This will be caught by other tests, skip here
            continue

        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Search for __main__ blocks
            matches = main_pattern.findall(content)

            if matches:
                errors.append(f"Task '{task_dir}' contains __main__ block in {task_dir}.py")

        except Exception as e:
            errors.append(f"Error reading task '{task_dir}' file {python_file}: {e}")

    if errors:
        print("No Main Blocks Check FAILED!\n")
        print("Task files should not contain __main__ blocks.")
        print("These files contain __main__ blocks:")
        print("-" * 60)
        for error in errors:
            print(f"  - {error}")
        print("-" * 60)
        print(f"\nTotal violations: {len(errors)}")
        print("\nAction required: Remove all __main__ blocks from task Python files.")
        print("Task files should be importable modules without executable code.")
        sys.exit(1)
    else:
        print("No Main Blocks Check PASSED!")
        print("All task Python files are clean of __main__ blocks.")
        sys.exit(0)

if __name__ == "__main__":
    check_no_main_blocks()
