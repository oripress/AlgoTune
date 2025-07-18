#!/usr/bin/env python3
"""
Simple evaluation script for testing SLURM mechanism.
Just writes a placeholder result to test the lock mechanism.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def acquire_lock(lock_file_path, timeout=60):
    """Attempts to acquire an exclusive lock using a lock file."""
    import time
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            print(f"Acquired lock: {lock_file_path}")
            return True
        except FileExistsError:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error acquiring lock {lock_file_path}: {e}")
            break
    print(f"Timeout acquiring lock after {timeout}s: {lock_file_path}")
    return False


def release_lock(lock_file_path):
    """Releases the lock by removing the lock file."""
    try:
        os.remove(lock_file_path)
        print(f"Released lock: {lock_file_path}")
    except FileNotFoundError:
        pass  # Lock was already released


def main():
    parser = argparse.ArgumentParser(description="Simple evaluation test")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default="reports/evaluate_summary.json")
    parser.add_argument("--results-dir", type=Path, default="results")
    parser.add_argument("--generation-file", type=Path, default="reports/generation.json")
    parser.add_argument("--data-dir", type=Path, default="../data")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--standalone", action="store_true")
    
    args = parser.parse_args()
    
    model = args.models[0]
    task = args.tasks[0]
    
    print(f"Evaluating {task} for {model}")
    
    # Simulate some work
    time.sleep(2)
    
    # Generate a fake speedup result
    fake_speedup = 2.5
    
    # Write result to summary file using lock mechanism
    lock_file_path = args.output.with_suffix('.lock')
    
    if not acquire_lock(str(lock_file_path)):
        print(f"Failed to acquire lock for {args.output}")
        sys.exit(1)
    
    try:
        # Load existing summary or create new one
        summary_data = {}
        if args.output.exists():
            try:
                with open(args.output, 'r') as f:
                    summary_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Could not load existing summary file {args.output}, creating new one")
                summary_data = {}
        
        # Add this result to summary
        if task not in summary_data:
            summary_data[task] = {}
        
        summary_data[task][model] = {
            "final_speedup": f"{fake_speedup:.4f}"
        }
        
        # Write updated summary
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(summary_data, f, indent=2, sort_keys=True)
            
        print(f"✅ {task}/{model}: {fake_speedup:.4f}x speedup (FAKE)")
        print(f"Result written to summary: {args.output}")
        
    finally:
        release_lock(str(lock_file_path))


if __name__ == "__main__":
    main()