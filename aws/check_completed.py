#!/usr/bin/env python3
"""
Check which tasks have already been completed for a given model.
Checks reports/agent_summary.json to avoid re-running completed jobs.
"""
import os
import sys
import argparse
import json
from pathlib import Path


def get_completed_tasks(model_name, summary_file="reports/agent_summary.json", include_na=True):
    """
    Check summary file for completed tasks for a given model.

    Args:
        model_name: Name of the model (e.g., "openrouter/openai/o4-mini")
        summary_file: Path to agent_summary.json file
        include_na: Whether to include tasks with "N/A" results as completed

    Returns:
        Tuple of (completed_set, na_set) - sets of completed and N/A task names
    """
    completed = set()
    na_tasks = set()
    summary_path = Path(summary_file)

    if not summary_path.exists():
        return completed, na_tasks

    try:
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"WARNING: Could not read summary file: {e}", file=sys.stderr)
        return completed, na_tasks

    # Check if summary is a dict
    if not isinstance(summary_data, dict):
        return completed, na_tasks

    # Iterate through tasks in summary
    for task_name, models_data in summary_data.items():
        if not isinstance(models_data, dict):
            continue

        # Check if this model has a completed entry for this task
        if model_name in models_data:
            task_result = models_data[model_name]
            if isinstance(task_result, dict):
                speedup = task_result.get('final_speedup')
                if speedup is not None:
                    if speedup == "N/A":
                        na_tasks.add(task_name)
                        if include_na:
                            completed.add(task_name)
                    else:
                        completed.add(task_name)

    return completed, na_tasks


def filter_tasks(all_tasks, model_name, summary_file="reports/agent_summary.json", verbose=False, interactive=False):
    """
    Filter out already-completed tasks.

    Args:
        all_tasks: List of all task names to check
        model_name: Name of the model
        summary_file: Path to agent_summary.json file
        verbose: Print information about skipped tasks
        interactive: Ask user whether to re-run N/A tasks

    Returns:
        List of tasks that haven't been completed yet
    """
    include_na = True

    # Get completed and N/A tasks
    completed, na_tasks = get_completed_tasks(model_name, summary_file, include_na=True)

    # Check which N/A tasks are in our requested task list
    na_in_list = [task for task in all_tasks if task in na_tasks]

    # If interactive and we have N/A tasks, ask whether to re-run them
    if interactive and na_in_list:
        print(f"\n⚠️  Found {len(na_in_list)} tasks with N/A results (failed/crashed):", file=sys.stderr)
        for task in sorted(na_in_list):
            print(f"  - {task}", file=sys.stderr)
        print("", file=sys.stderr)

        # Ask user
        while True:
            response = input("Re-run these N/A tasks? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                include_na = False
                break
            elif response in ['n', 'no', '']:
                include_na = True
                break
            else:
                print("Please answer 'y' or 'n'", file=sys.stderr)

    # Rebuild completed set based on user choice
    if not include_na:
        completed, _ = get_completed_tasks(model_name, summary_file, include_na=False)

    if verbose and completed:
        non_na_completed = completed - na_tasks
        if non_na_completed:
            print(f"Found {len(non_na_completed)} successfully completed tasks for {model_name}:", file=sys.stderr)
            for task in sorted(non_na_completed):
                print(f"  - {task}", file=sys.stderr)
            print("", file=sys.stderr)

        if include_na and na_in_list:
            print(f"Skipping {len(na_in_list)} N/A tasks (use --interactive to change):", file=sys.stderr)
            for task in sorted(na_in_list):
                print(f"  - {task}", file=sys.stderr)
            print("", file=sys.stderr)

    remaining = [task for task in all_tasks if task not in completed]

    if verbose:
        skipped_count = len(all_tasks) - len(remaining)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} already-completed tasks", file=sys.stderr)
            print(f"Running {len(remaining)} remaining tasks", file=sys.stderr)
        else:
            print(f"No completed tasks found, running all {len(remaining)} tasks", file=sys.stderr)
        print("", file=sys.stderr)

    return remaining


def main():
    parser = argparse.ArgumentParser(
        description="Check which tasks have been completed for a model"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., 'openrouter/openai/o4-mini', 'openrouter/anthropic/claude-opus-4-20250514')"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="List of tasks to check (default: read from stdin, one per line)"
    )
    parser.add_argument(
        "--summary-file",
        default="reports/agent_summary.json",
        help="Summary file to check (default: reports/agent_summary.json)"
    )
    parser.add_argument(
        "--list-completed",
        action="store_true",
        help="Just list completed tasks and exit"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print verbose information to stderr"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask whether to re-run N/A (failed) tasks"
    )

    args = parser.parse_args()

    if args.list_completed:
        # Just list completed tasks (including N/A)
        completed, na_tasks = get_completed_tasks(args.model, args.summary_file, include_na=True)
        for task in sorted(completed):
            # Mark N/A tasks in the output
            marker = " (N/A)" if task in na_tasks else ""
            print(f"{task}{marker}")
        return

    # Get tasks from command line or stdin
    if args.tasks:
        all_tasks = args.tasks
    else:
        # Read from stdin
        all_tasks = [line.strip() for line in sys.stdin if line.strip()]

    if not all_tasks:
        print("ERROR: No tasks provided", file=sys.stderr)
        sys.exit(1)

    # Filter and output remaining tasks
    remaining = filter_tasks(
        all_tasks,
        args.model,
        args.summary_file,
        verbose=not args.quiet,
        interactive=args.interactive
    )

    # Output one task per line
    for task in remaining:
        print(task)


if __name__ == "__main__":
    main()
