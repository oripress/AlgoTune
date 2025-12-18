#!/usr/bin/env python3
"""
Check which tasks have already been completed for a given model.
Scans results directories to avoid re-running completed jobs.
"""
import os
import sys
import argparse
from pathlib import Path


def get_completed_tasks(model_name, results_dir="results"):
    """
    Scan results directory for completed tasks for a given model.

    Args:
        model_name: Name of the model (e.g., "o4-mini", "Claude Opus 4")
        results_dir: Root results directory

    Returns:
        Set of completed task names
    """
    completed = set()
    results_path = Path(results_dir)

    if not results_path.exists():
        return completed

    # Check multiple possible model name formats
    # Model names in results might be cleaned up versions
    possible_model_dirs = [
        model_name,
        model_name.replace("/", "_"),
        model_name.replace("openrouter/", ""),
        model_name.replace("anthropic/", ""),
        model_name.replace("deepseek-ai/", ""),
        model_name.replace("deepseek/", ""),
        model_name.replace("gemini/", ""),
        model_name.replace("openai/", ""),
        # Common clean names
        "o4-mini" if "o4-mini" in model_name else None,
        "Claude Opus 4" if "claude-opus-4" in model_name else None,
        "DeepSeek R1" if "deepseek" in model_name.lower() and "r1" in model_name.lower() else None,
        "Qwen3 Coder" if "qwen3-coder" in model_name else None,
    ]

    # Remove None values
    possible_model_dirs = [d for d in possible_model_dirs if d]

    for model_dir_name in possible_model_dirs:
        model_path = results_path / model_dir_name
        if model_path.exists() and model_path.is_dir():
            # List all task directories
            for task_dir in model_path.iterdir():
                if task_dir.is_dir():
                    # Check if task has results (solver.py exists)
                    solver_path = task_dir / "solver.py"
                    if solver_path.exists():
                        completed.add(task_dir.name)

    return completed


def filter_tasks(all_tasks, model_name, results_dir="results", verbose=False):
    """
    Filter out already-completed tasks.

    Args:
        all_tasks: List of all task names to check
        model_name: Name of the model
        results_dir: Root results directory
        verbose: Print information about skipped tasks

    Returns:
        List of tasks that haven't been completed yet
    """
    completed = get_completed_tasks(model_name, results_dir)

    if verbose and completed:
        print(f"Found {len(completed)} completed tasks for {model_name}:", file=sys.stderr)
        for task in sorted(completed):
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
        help="Model name (e.g., 'o4-mini', 'anthropic/claude-opus-4-20250514')"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="List of tasks to check (default: read from stdin, one per line)"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory to scan (default: results)"
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

    args = parser.parse_args()

    if args.list_completed:
        # Just list completed tasks
        completed = get_completed_tasks(args.model, args.results_dir)
        for task in sorted(completed):
            print(task)
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
        args.results_dir,
        verbose=not args.quiet
    )

    # Output one task per line
    for task in remaining:
        print(task)


if __name__ == "__main__":
    main()
