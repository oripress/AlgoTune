#!/usr/bin/env python3
"""
submit_generate_python.py – SLURM orchestration wrapper for baseline generation.

This script provides SLURM orchestration for baseline generation using the centralized
timing_core module. It can also run in standalone mode for non-SLURM environments.

Usage:
  # SLURM mode (default)
  python3 submit_generate_python.py --target-time-ms 50
  
  # Standalone mode (all tasks in parallel)
  python3 submit_generate_python.py --target-time-ms 50 --standalone
  
  # Standalone mode (sequential - one task at a time)
  python3 submit_generate_python.py --target-time-ms 50 --standalone --sequential
  
  # Single task standalone
  python3 submit_generate_python.py --task svm --target-time-ms 50 --standalone
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Import the centralized timing logic
sys.path.insert(0, str(Path(__file__).parent))
from AlgoTuner.timing_core import run_complete_timing_evaluation
from AlgoTuner.task_lists import get_task_list, get_available_task_lists, load_custom_task_list, filter_existing_tasks

# ---------------------------------------------------------------- constants --
# ENV_FILE_REL is deprecated - use load_unified_config() instead
GENERATE_RUNSCRIPT_REL = "scripts/slurm_jobs/generate_dataset.sh"
EVAL_RUNSCRIPT_REL     = "scripts/slurm_jobs/generate.sh"

REPORT_DIR_REL = "reports"
LOG_DIR_REL    = "tests/logs"
SUMMARY_NAME   = "generation.json"

# ---------------------------------------------------------------- utilities --
def now() -> str:
    return datetime.now().strftime("%F %T")


def cleanup_timing_logs(project_root: Path, target_time_ms: int = None) -> None:
    """Clean up old generation-related logs from ./logs and slurm directories."""
    print(f"🧹 Cleaning up old generation logs...")
    
    # Clean up logs from ./logs directory
    logs_dir = project_root / "logs"
    total_cleaned = 0
    
    if logs_dir.exists():
        # Pattern for timing logs - look for logs with timing script pattern
        timing_log_patterns = [
            "*_timing_script_*.log",
            "*timing*.log", 
            # "gen_*.log",  # Don't delete generation logs
            "eval_*.log"
        ]
        
        old_logs = []
        for pattern in timing_log_patterns:
            old_logs.extend(logs_dir.glob(pattern))
        
        if old_logs:
            print(f"🧹 Cleaning up {len(old_logs)} old generation logs from ./logs...")
            for log_file in old_logs:
                try:
                    log_file.unlink()
                    total_cleaned += 1
                except OSError as e:
                    print(f"Warning: Could not delete {log_file}: {e}")
    
    # Clean up SLURM outputs and errors
    slurm_outputs_dir = project_root / "slurm/outputs"  
    slurm_errors_dir = project_root / "slurm/errors"
    
    for cleanup_dir, desc in [(slurm_outputs_dir, "outputs"), (slurm_errors_dir, "errors")]:
        if cleanup_dir.exists():
            # Pattern for timing-related SLURM files
            timing_slurm_patterns = [
                # "gen_*",  # Don't delete generation scripts
                "eval_*", 
                "*timing*"
            ]
            
            old_slurm_files = []
            for pattern in timing_slurm_patterns:
                old_slurm_files.extend(cleanup_dir.glob(pattern))
            
            if old_slurm_files:
                print(f"🧹 Cleaning up {len(old_slurm_files)} old SLURM generation {desc}...")
                for file in old_slurm_files:
                    try:
                        file.unlink()
                        total_cleaned += 1
                    except OSError as e:
                        print(f"Warning: Could not delete {file}: {e}")
    
    if total_cleaned == 0:
        print("🧹 No old generation logs found to clean up")
    else:
        print(f"🧹 Successfully cleaned up {total_cleaned} old generation log files")


def load_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for p in ("export ", "declare -x "):
            if line.startswith(p):
                line = line[len(p):]
                break
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_unified_config(project_root: Path) -> Dict[str, str]:
    """Load configuration with unified config.env priority, fallback to slurm/run_config.env."""
    config = {}
    
    # Try unified config first
    unified_config = project_root / "config.env"
    if unified_config.exists():
        config.update(load_env_file(unified_config))
        print(f"📋 Loaded unified config from: {unified_config}")
        return config
    
    # Fallback to old location for backward compatibility
    old_config = project_root / "slurm/run_config.env"
    if old_config.exists():
        config.update(load_env_file(old_config))
        print(f"📋 Loaded config from: {old_config} (consider migrating to config.env)")
        return config
    
    print("⚠️  No configuration file found. Expected config.env or slurm/run_config.env")
    return config


def setup_environment(project_root: Path, standalone: bool = False) -> Dict[str, str]:
    """Setup environment variables for both SLURM and standalone modes."""
    env_vars = {}
    
    # Try to load unified config first
    env_vars.update(load_unified_config(project_root))
    
    # Override with standalone-friendly defaults if needed
    if standalone:
        # Use local paths for standalone mode if not set
        if "DATA_DIR" not in env_vars:
            env_vars["DATA_DIR"] = str(project_root / "data")
        if "TEMP_DIR_STORAGE" not in env_vars:
            env_vars["TEMP_DIR_STORAGE"] = "/tmp"
        if "TASKS_ROOT" not in env_vars:
            env_vars["TASKS_ROOT"] = str(project_root / "AlgoTuneTasks")
            
        print(f"{now()}: Using standalone mode with DATA_DIR={env_vars['DATA_DIR']}", file=sys.stderr)
    
    # Apply to current environment
    os.environ.update(env_vars)
    return env_vars


def is_task_complete(entry: Dict, target_time_ms: int = None) -> bool:
    """Check if a task has successful baseline runs for the specified target time."""
    runs = entry.get("baseline_runs", {})
    if len(runs) < 3:  # NUM_EVAL_RUNS
        return False
    
    # Check if all runs are successful (preserve existing valid runs regardless of target time)
    for run_data in runs.values():
        if (
            run_data.get("avg_min_ms") is None
            or run_data.get("std_min_ms") is None
            or run_data.get("success") is False
        ):
            return False
    
    # If we have 3 valid runs, task is complete regardless of target time
    # This preserves existing valid results and prevents unnecessary re-runs
    return True


def has_failed_or_null_runs(entry: Dict) -> bool:
    """Check if a task has any failed runs or null values that require cleanup."""
    runs = entry.get("baseline_runs", {})
    if not runs:
        return False
        
    for run_data in runs.values():
        if (
            run_data.get("avg_min_ms") is None
            or run_data.get("std_min_ms") is None
            or run_data.get("success") is False
        ):
            return True
    return False


def has_invalid_nulls(entry: Dict) -> bool:
    """Check if a task has invalid null values that require cleanup and retry."""
    # Only mark as invalid if target_time_ms is null AND there are no baseline runs
    # This prevents deleting entries that just have different target times
    if entry.get("target_time_ms") is None and not entry.get("baseline_runs"):
        return True
    
    # Don't check baseline runs for null values - preserve all existing data
    # Users can manually clean up if needed
    return False


def cleanup_invalid_task(task: str, summary: Dict, data_dir: Path) -> None:
    """Remove invalid task data and dataset to force regeneration."""
    print(f"{now()}: Cleaning up invalid task '{task}' with null values")
    
    # Remove from summary
    if task in summary:
        del summary[task]
    
    # Remove dataset directory if it exists
    task_dataset_dir = data_dir / task
    if task_dataset_dir.exists():
        shutil.rmtree(task_dataset_dir, ignore_errors=True)
        print(f"{now()}: Removed dataset directory for '{task}'")


def get_valid_tasks(tasks_root: Path) -> List[str]:
    """Get list of valid task names."""
    if not tasks_root.is_dir():
        return []
    return [
        p.name for p in tasks_root.iterdir()
        if p.is_dir() and (p / "description.txt").is_file()
    ]


def update_summary_file(summary_file: Path, results: Dict) -> None:
    """Update the summary file with results from a task evaluation."""
    # Load existing summary
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        except:
            summary = {}
    else:
        summary = {}
    
    # Update with new results
    task_name = results["task_name"]
    summary[task_name] = {
        "target_time_ms": results["target_time_ms"],
        "baseline_runs": results["baseline_runs"]
    }
    
    if "n" in results:
        summary[task_name]["n"] = results["n"]
    if "dataset_size" in results:
        summary[task_name]["dataset_size"] = results["dataset_size"]
    
    # Write back to file
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def run_standalone_mode(target_time_ms: int = None, task_name: str = None, 
                       data_dir: Path = None, project_root: Path = None,
                       sequential: bool = False, task_list: str = None,
                       task_list_file: Path = None, tasks: List[str] = None,
                       lazy: bool = False) -> None:
    """Run timing evaluation in standalone mode (no SLURM)."""
    if project_root is None:
        project_root = Path.cwd()
        
    # Clean up old timing logs first
    cleanup_timing_logs(project_root, target_time_ms)
        
    # Setup environment (will use DATA_DIR from config or default)
    env_vars = setup_environment(project_root, standalone=True)
    
    if data_dir is None:
        data_dir = Path(env_vars["DATA_DIR"])
    
    tasks_root = Path(env_vars.get("TASKS_ROOT", project_root / "AlgoTuneTasks"))
    summary_file = project_root / REPORT_DIR_REL / SUMMARY_NAME
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize summary file if missing or empty
    if not summary_file.exists() or summary_file.stat().st_size == 0:
        summary_file.write_text("{}")
        print(f"{now()}: Initialized {summary_file}")
    
    # Load existing summary to get cached values
    summary = {}
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        except:
            # If JSON is corrupted, reset to empty dict
            summary_file.write_text("{}")
            summary = {}
    
    # Clean up metadata keys from existing entries
    all_valid_tasks = get_valid_tasks(tasks_root)
    for task in all_valid_tasks:
        if task in summary:
            entry = summary[task]
            keys_to_remove = [k for k in entry.keys() if k.startswith(("attempted_at_", "runs_attempted_for_")) 
                             or k in {"current_target_time_ms_attempted", "last_gen_jobid", "last_eval_jobid", "final_status"}]
            for key in keys_to_remove:
                entry.pop(key, None)
    
    # Clean up any tasks with invalid null values
    tasks_to_cleanup = []
    for task, entry in summary.items():
        if has_invalid_nulls(entry):
            tasks_to_cleanup.append(task)
    
    for task in tasks_to_cleanup:
        cleanup_invalid_task(task, summary, data_dir)
    
    # When target_time_ms is provided, clean up wrong target times
    if target_time_ms is not None:
        summary_changed = False
        for task in all_valid_tasks:
            if cleanup_wrong_target_summary_entries(summary, task, target_time_ms):
                summary_changed = True
        
        # Save cleaned summary if any tasks were removed
        if tasks_to_cleanup or summary_changed:
            summary_file.write_text(json.dumps(summary, indent=2))
            if tasks_to_cleanup:
                print(f"{now()}: Cleaned up {len(tasks_to_cleanup)} tasks with invalid null values: {', '.join(tasks_to_cleanup)}", file=sys.stderr)
            if summary_changed:
                print(f"{now()}: Cleaned up summary entries with wrong target times", file=sys.stderr)
    else:
        # When using cached target times, only save cleanup of invalid tasks
        if tasks_to_cleanup:
            summary_file.write_text(json.dumps(summary, indent=2))
            print(f"{now()}: Cleaned up {len(tasks_to_cleanup)} tasks with invalid null values: {', '.join(tasks_to_cleanup)}", file=sys.stderr)
    
    # Determine which tasks to run
    if task_name:
        # Run single task
        tasks_to_run = [task_name]
    elif task_list:
        # Use predefined task list
        try:
            tasks_to_run = get_task_list(task_list)
            tasks_to_run = filter_existing_tasks(tasks_to_run, tasks_root)
            print(f"{now()}: Using predefined task list '{task_list}': {tasks_to_run}")
        except KeyError as e:
            available = ", ".join(get_available_task_lists())
            sys.exit(f"❌ {e}. Available task lists: {available}")
    elif task_list_file:
        # Load custom task list from file
        try:
            tasks_to_run = load_custom_task_list(task_list_file)
            tasks_to_run = filter_existing_tasks(tasks_to_run, tasks_root)
            print(f"{now()}: Using custom task list from {task_list_file}: {tasks_to_run}")
        except FileNotFoundError as e:
            sys.exit(f"❌ {e}")
    elif tasks:
        # Use provided task list
        tasks_to_run = filter_existing_tasks(tasks, tasks_root)
        print(f"{now()}: Using provided task list: {tasks_to_run}")
        if len(tasks_to_run) != len(tasks):
            missing = set(tasks) - set(tasks_to_run)
            print(f"{now()}: Warning: {len(missing)} tasks not found in {tasks_root}: {', '.join(missing)}")
    elif lazy:
        # Lazy mode: require explicit task specification
        print(f"{now()}: ❌ --lazy mode requires explicit task specification")
        print(f"{now()}: Use --task, --tasks, --task-list, or --task-list-file to specify which tasks to run")
        return
    else:
        # Run all incomplete tasks (default behavior)
        tasks_to_run = []
        
        for task in all_valid_tasks:
            entry = summary.get(task, {})
            if not is_task_complete(entry, target_time_ms):
                tasks_to_run.append(task)
    
    if not tasks_to_run:
        print(f"{now()}: No tasks to run - all specified tasks are complete or don't exist")
        return
    
    # When target_time_ms is None, validate that all requested tasks have cached target times
    if target_time_ms is None:
        tasks_with_cached_times = []
        missing_cache = []
        
        for task in tasks_to_run:
            cached_target_time = summary.get(task, {}).get("target_time_ms")
            if cached_target_time is not None:
                tasks_with_cached_times.append((task, cached_target_time))
            else:
                missing_cache.append(task)
        
        if missing_cache:
            print(f"{now()}: ❌ No cached target times found for: {', '.join(missing_cache)}")
            print(f"{now()}: Run these tasks first with --target-time-ms to establish baseline measurements")
            return
        
        print(f"{now()}: Using cached target times for {len(tasks_with_cached_times)} tasks:")
        for task, cached_time in tasks_with_cached_times:
            print(f"{now()}: '{task}' → {cached_time}ms (cached)")
    
    # In lazy mode, run all specified tasks regardless of completion status
    # (datasets will be generated on-demand as needed)
    if lazy:
        mode_info = "lazy (on-demand dataset generation)"
    else:
        # In normal mode, filter out completed tasks when target_time_ms is provided
        if target_time_ms is not None:
            incomplete_tasks = []
            for task in tasks_to_run:
                entry = summary.get(task, {})
                if not is_task_complete(entry, target_time_ms):
                    incomplete_tasks.append(task)
                else:
                    print(f"{now()}: '{task}' already complete - skipping")
            tasks_to_run = incomplete_tasks
            mode_info = "normal (skip completed tasks)"
            
            if not tasks_to_run:
                print(f"{now()}: No incomplete tasks to run - all specified tasks are already complete")
                return
        else:
            # When using cached target times, always run (user explicitly requested these tasks)
            mode_info = "cached target times (run requested tasks)"
    
    mode_desc = "sequential" if sequential else "parallel"
    print(f"{now()}: Running standalone evaluation ({mode_desc}, {mode_info}) for {len(tasks_to_run)} tasks")
    if target_time_ms is not None:
        print(f"{now()}: Target time: {target_time_ms} ms")
    else:
        print(f"{now()}: Target time: Using cached values from generation.json")
    print(f"{now()}: Data directory: {data_dir}")
    print(f"{now()}: Tasks: {', '.join(tasks_to_run)}")
    
    if sequential:
        # Sequential mode: one task at a time
        for i, task in enumerate(tasks_to_run, 1):
            print(f"\n{now()}: [{i}/{len(tasks_to_run)}] Starting evaluation for '{task}'")
            
            # Use provided target_time_ms or get cached target time for this task
            if target_time_ms is not None:
                task_target_time = target_time_ms
            else:
                task_target_time = summary.get(task, {}).get("target_time_ms")
                if task_target_time is None:
                    print(f"{now()}: ❌ No cached target time for '{task}', skipping")
                    continue
            
            # Get cached override_k if available
            override_k = summary.get(task, {}).get("n")
            if override_k:
                print(f"{now()}: Using cached n={override_k} for '{task}'")
            
            print(f"{now()}: Target time for '{task}': {task_target_time}ms")
            
            # Run the evaluation
            try:
                # Clean up any datasets with wrong target times first
                task_dataset_dir = data_dir / task
                cleanup_wrong_target_datasets(task_dataset_dir, task, task_target_time)
                
                result = run_complete_timing_evaluation(
                    task_name=task,
                    target_time_ms=task_target_time,
                    data_dir=data_dir,
                    override_k=override_k
                )
                
                # Update summary file immediately
                update_summary_file(summary_file, result)
                
                # Reload summary for next task (to get updated n values)
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                except:
                    pass
                
                if result["success"]:
                    print(f"{now()}: ✅ [{i}/{len(tasks_to_run)}] '{task}' completed successfully")
                else:
                    print(f"{now()}: ❌ [{i}/{len(tasks_to_run)}] '{task}' failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"{now()}: ❌ [{i}/{len(tasks_to_run)}] '{task}' failed with exception: {e}")
    else:
        # Parallel mode: all tasks at once (original behavior)
        for task in tasks_to_run:
            print(f"\n{now()}: Starting evaluation for '{task}'")
            
            # Use provided target_time_ms or get cached target time for this task
            if target_time_ms is not None:
                task_target_time = target_time_ms
            else:
                task_target_time = summary.get(task, {}).get("target_time_ms")
                if task_target_time is None:
                    print(f"{now()}: ❌ No cached target time for '{task}', skipping")
                    continue
            
            # Get cached override_k if available
            override_k = summary.get(task, {}).get("n")
            
            print(f"{now()}: Target time for '{task}': {task_target_time}ms")
            
            # Run the evaluation
            try:
                # Clean up any datasets with wrong target times first
                task_dataset_dir = data_dir / task
                cleanup_wrong_target_datasets(task_dataset_dir, task, task_target_time)
                
                result = run_complete_timing_evaluation(
                    task_name=task,
                    target_time_ms=task_target_time,
                    data_dir=data_dir,
                    override_k=override_k
                )
                
                # Update summary file
                update_summary_file(summary_file, result)
                
                if result["success"]:
                    print(f"{now()}: ✅ '{task}' completed successfully")
                else:
                    print(f"{now()}: ❌ '{task}' failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"{now()}: ❌ '{task}' failed with exception: {e}")
    
    print(f"\n{now()}: Standalone evaluation complete")
    print(f"Results saved to: {summary_file}")


def cleanup_wrong_target_datasets(dataset_dir: Path, task_name: str, target_time_ms: int) -> None:
    """Remove dataset files that don't match the current target time."""
    if not dataset_dir.exists():
        return
    
    # Find all dataset files for this task
    pattern = f"{task_name}_T*ms_n*_size*_*.jsonl"
    all_files = list(dataset_dir.glob(pattern))
    
    # Filter out files that don't match the current target time
    current_pattern = f"{task_name}_T{target_time_ms}ms_n*_size*_*.jsonl"
    correct_files = set(dataset_dir.glob(current_pattern))
    
    files_to_remove = [f for f in all_files if f not in correct_files]
    
    if files_to_remove:
        print(f"{now()}: Cleaning up {len(files_to_remove)} dataset files with wrong target times for '{task_name}'")
        for file_path in files_to_remove:
            print(f"{now()}: Removing {file_path.name}")
            file_path.unlink()


def cleanup_wrong_target_summary_entries(summary: Dict, task_name: str, target_time_ms: int) -> bool:
    """Clean up summary entries that have wrong target times. Returns True if cleanup was needed."""
    # DISABLED: Don't delete summary entries, only clean up dataset files on disk
    # This preserves all previous runs even if they had different target times
    return False


def slurm_submit(cmd: List[str]) -> str:
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return res.stdout.strip()



def setup_directories(config: Dict[str, str], project_root: Path) -> Tuple[Path, Path]:
    """Setup data and temp directories with fallbacks."""
    data_dir_str = config.get('DATA_DIR', str(project_root / 'data'))
    temp_dir_str = config.get('TEMP_DIR_STORAGE', '/tmp')
    
    # Try to use configured paths first
    data_dir_path = Path(data_dir_str)
    temp_dir_path = Path(temp_dir_str)
    
    # Check if we can actually use these paths
    try:
        data_dir_path.mkdir(parents=True, exist_ok=True)
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = data_dir_path / ".test_write"
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError) as e:
        print(f"⚠️  Cannot access configured paths ({e})")
        print(f"   DATA_DIR: {data_dir_path}")
        print(f"   TEMP_DIR: {temp_dir_path}")
        print("   Falling back to local directories...")
        
        # Fallback to local directories
        data_dir_path = project_root / 'data'
        temp_dir_path = Path('/tmp')
        
        # Ensure local directories exist
        data_dir_path.mkdir(parents=True, exist_ok=True)
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   Using DATA_DIR: {data_dir_path}")
        print(f"   Using TEMP_DIR: {temp_dir_path}")
    
    return data_dir_path, temp_dir_path


def run_slurm_mode(target_time_ms: int, project_root: Path):
    """Run in SLURM mode with individual job submission per task."""
    print("🤖 Running in SLURM mode")
    
    # Clean up old timing logs first
    cleanup_timing_logs(project_root, target_time_ms)
    
    # Load SLURM configuration
    config = load_unified_config(project_root)
    if not config:
        print("❌ Could not load SLURM configuration")
        sys.exit(1)
    
    # Setup directories with fallbacks
    data_dir_path, temp_dir_path = setup_directories(config, project_root)
    
    # Setup SLURM environment
    for sub in ("outputs", "errors"):
        d = project_root / "slurm" / sub
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for the job scripts
    os.environ['DATA_DIR'] = str(data_dir_path)
    os.environ['TEMP_DIR_STORAGE'] = str(temp_dir_path)
    os.environ['TARGET_TIME_MS'] = str(target_time_ms)
    
    # Load tasks and summary
    tasks_root = project_root / "AlgoTuneTasks"
    all_valid_tasks = get_valid_tasks(tasks_root)
    if not all_valid_tasks:
        print("❌ No valid tasks found")
        sys.exit(1)
    
    summary_file = project_root / REPORT_DIR_REL / SUMMARY_NAME
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize summary file if missing or empty
    if not summary_file.exists() or summary_file.stat().st_size == 0:
        summary_file.write_text("{}")
        print(f"{now()}: Initialized {summary_file}")
    
    # Load existing summary to get cached values
    summary = {}
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        except:
            # If JSON is corrupted, reset to empty dict
            summary_file.write_text("{}")
            summary = {}
    
    # Clean up metadata keys from existing entries
    for task in all_valid_tasks:
        if task in summary:
            entry = summary[task]
            keys_to_remove = [k for k in entry.keys() if k.startswith(("attempted_at_", "runs_attempted_for_")) 
                             or k in {"current_target_time_ms_attempted", "last_gen_jobid", "last_eval_jobid", "final_status"}]
            for key in keys_to_remove:
                entry.pop(key, None)
    
    # Clean up any tasks with invalid null values
    tasks_to_cleanup = []
    for task, entry in summary.items():
        if has_invalid_nulls(entry):
            tasks_to_cleanup.append(task)
    
    for task in tasks_to_cleanup:
        cleanup_invalid_task(task, summary, data_dir_path)
    
    # Clean up summary entries with wrong target times for valid tasks
    summary_changed = False
    for task in all_valid_tasks:
        if cleanup_wrong_target_summary_entries(summary, task, target_time_ms):
            summary_changed = True
    
    # Save cleaned summary if any tasks were removed
    if tasks_to_cleanup or summary_changed:
        summary_file.write_text(json.dumps(summary, indent=2))
        if tasks_to_cleanup:
            print(f"{now()}: Cleaned up {len(tasks_to_cleanup)} tasks with invalid null values: {', '.join(tasks_to_cleanup)}", file=sys.stderr)
        if summary_changed:
            print(f"{now()}: Cleaned up summary entries with wrong target times", file=sys.stderr)
    
    # Prune stale tasks
    for stale in set(summary) - set(all_valid_tasks):
        summary.pop(stale, None)
        shutil.rmtree(data_dir_path / stale, ignore_errors=True)
    
    # Clean up orphaned data directories (exist on disk but not in JSON)
    orphaned_dirs = []
    if data_dir_path.exists():
        for data_subdir in data_dir_path.iterdir():
            if data_subdir.is_dir():
                task_name = data_subdir.name
                # If directory exists but no entry in summary, it's orphaned
                if task_name in all_valid_tasks and task_name not in summary:
                    print(f"{now()}: Found orphaned data directory for '{task_name}' - removing", file=sys.stderr)
                    shutil.rmtree(data_subdir, ignore_errors=True)
                    orphaned_dirs.append(task_name)
    
    if orphaned_dirs:
        print(f"{now()}: Cleaned up {len(orphaned_dirs)} orphaned data directories: {', '.join(orphaned_dirs)}", file=sys.stderr)
    
    # Check runscripts exist
    gen_run = project_root / GENERATE_RUNSCRIPT_REL
    eval_run = project_root / EVAL_RUNSCRIPT_REL
    for s in (gen_run, eval_run):
        if not s.exists():
            print(f"❌ Missing runscript {s}")
            sys.exit(1)
    
    # Find tasks that need work
    tasks_to_submit: List[str] = []
    tasks_cleaned_up: List[str] = []
    
    for task in sorted(all_valid_tasks):
        entry = summary.get(task, {})
        
        # Check if task has failed or null runs that need cleanup
        if has_failed_or_null_runs(entry):
            print(f"{now()}: '{task}' has failed/null runs - cleaning up for regeneration", file=sys.stderr)
            # Remove the task entry from summary to force regeneration
            summary.pop(task, None)
            # Remove dataset directory to force regeneration
            shutil.rmtree(data_dir_path / task, ignore_errors=True)
            tasks_cleaned_up.append(task)
            # Add to tasks to submit for regeneration (don't create empty entry yet)
            tasks_to_submit.append(task)
            print(f"{now()}: '{task}' → {target_time_ms} ms", file=sys.stderr)
            continue
            
        if is_task_complete(entry, target_time_ms):
            print(f"{now()}: '{task}' already complete - skipping", file=sys.stderr)
            continue
        
        # Don't create empty entries in summary - let the actual evaluation create the entry
        # when it has real results to add
        
        tasks_to_submit.append(task)
        print(f"{now()}: '{task}' → {target_time_ms} ms", file=sys.stderr)
    
    # Clean up any existing empty entries in the summary
    empty_entries = [task for task, entry in summary.items() if not entry]
    for task in empty_entries:
        summary.pop(task, None)
    
    # Log cleanup summary
    if tasks_cleaned_up:
        print(f"{now()}: Cleaned up {len(tasks_cleaned_up)} tasks with failed/null runs: {', '.join(tasks_cleaned_up)}", file=sys.stderr)
    if empty_entries:
        print(f"{now()}: Removed {len(empty_entries)} empty entries from JSON: {', '.join(empty_entries)}", file=sys.stderr)
    
    # Save updated summary after cleanup
    if tasks_cleaned_up or empty_entries:
        summary_file.write_text(json.dumps(summary, indent=2))
    
    if not tasks_to_submit:
        print("✅ All tasks are already complete")
        return
    
    # Get SLURM partition
    slurm_part = config.get("SLURM_PARTITIONS_DEFAULT", "cpu")
    
    print(f"📊 Target time: {target_time_ms}ms")
    print(f"📁 Data directory: {data_dir_path}")
    print(f"🗂️  Temp directory: {temp_dir_path}")
    print(f"🎯 Tasks to submit: {len(tasks_to_submit)}")
    if tasks_cleaned_up:
        print(f"🧹 Tasks cleaned up: {len(tasks_cleaned_up)}")
    print(f"🏷️  SLURM partition: {slurm_part}")
    
    # Submit SLURM jobs
    print("\n--- Starting SLURM submissions ---")
    job_ids: List[str] = []

    for task in tasks_to_submit:
        short5 = task[:5]
        dataset_dir = data_dir_path / task
        
        # Clean up any datasets with wrong target times first
        cleanup_wrong_target_datasets(dataset_dir, task, target_time_ms)
        
        # Always regenerate to ensure fresh dataset when task is incomplete
        # Remove existing directory to avoid stale files
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Generate dataset then eval
        gen_jn = f"gen_{short5}"
        export_opts = f"ALL,SKIP_DATASET_GEN=0,TARGET_TIME_MS={target_time_ms},AGENT_MODE=0,MULTIPROCESS_START_METHOD=fork"
        prev_n = summary.get(task, {}).get("n")
        if prev_n is not None:
            export_opts += f",OVERRIDE_K={prev_n}"
        gen_jid = slurm_submit([
            "sbatch", "-p", slurm_part, "--parsable",
            f"--job-name={gen_jn}",
            f"--time=12:00:00",
            f"--output=slurm/outputs/{gen_jn}_%j.out",
            f"--error=slurm/outputs/{gen_jn}_%j.err",
            "--export", export_opts,
            str(gen_run), task, str(target_time_ms),
        ])
        job_ids.append(gen_jid)

        eval_jn = f"eval_{short5}"
        eval_jid = slurm_submit([
            "sbatch", "-p", slurm_part, "--parsable",
            f"--job-name={eval_jn}",
            f"--time=12:00:00",
            f"--output=slurm/outputs/{eval_jn}_%j.out",
            f"--error=slurm/errors/{eval_jn}_%j.err",
            f"--array=0-2",  # NUM_EVAL_RUNS-1
            f"--dependency=afterok:{gen_jid}",
            "--export", f"ALL,TARGET_TIME_MS={target_time_ms},AGENT_MODE=0,MULTIPROCESS_START_METHOD=fork",
            str(eval_run), task, str(dataset_dir),
        ])
        job_ids.append(eval_jid)

        print(f" • submit '{task}' → {gen_jn}({gen_jid}) @ {target_time_ms} ms, "
              f"{eval_jn}({eval_jid})", file=sys.stderr)

    print("\n--- Submission complete ---")
    if job_ids:
        print("🚀 Jobs queued:", " ".join(job_ids))
        print(f"👀 Watch with: squeue -u {os.getenv('USER','')}")
        print(f"📋 Summary:    {summary_file}")
    else:
        print("Nothing to queue – all tasks complete.")


# ------------------------------------------------------------------ main -----
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline measurements for algorithmic tasks")
    parser.add_argument("--target-time-ms", type=int, 
                       help="Target time in milliseconds (optional, uses cached values if not provided)")
    parser.add_argument("--standalone", action="store_true",
                       help="Run in standalone mode (no SLURM)")
    parser.add_argument("--sequential", action="store_true",
                       help="In standalone mode, process tasks sequentially (one at a time)")
    parser.add_argument("--task", 
                       help="Specific task to run (standalone mode only)")
    parser.add_argument("--data-dir", type=Path,
                       help="Data directory override (standalone mode only)")
    parser.add_argument("--task-list", 
                       help="Predefined task list to use (standalone mode only)")
    parser.add_argument("--task-list-file", type=Path,
                       help="Custom task list file to use (standalone mode only)")
    parser.add_argument("--tasks", nargs='+',
                       help="List of task names to run (standalone mode only). Example: --tasks svm kmeans")
    parser.add_argument("--lazy", action="store_true",
                       help="Only generate datasets for explicitly requested tasks (standalone mode only)")
    args = parser.parse_args()
    
    target_time_ms = args.target_time_ms
    
    # Detect project root
    script_dir = Path(__file__).resolve().parent
    try:
        project_root = Path(
            subprocess.check_output(
                ["git", "-C", str(script_dir), "rev-parse", "--show-toplevel"], text=True
            ).strip()
        )
    except subprocess.CalledProcessError:
        # Fallback: assume we're in scripts/ subdirectory
        project_root = script_dir.parent
    
    print(f"{now()}: PROJECT_ROOT = {project_root}", file=sys.stderr)
    
    # Handle case when target_time_ms is not provided
    if target_time_ms is None:
        if not args.standalone:
            print("❌ --target-time-ms is required for SLURM mode when using cached values is not yet supported")
            sys.exit(1)
        
        # In standalone mode without target time, we'll use cached values per task
        print(f"{now()}: No target time specified, will use cached values from generation.json", file=sys.stderr)
    
    if args.standalone:
        # Run in standalone mode
        run_standalone_mode(
            target_time_ms=target_time_ms,
            task_name=args.task,
            data_dir=args.data_dir,
            project_root=project_root,
            sequential=args.sequential,
            task_list=args.task_list,
            task_list_file=args.task_list_file,
            tasks=args.tasks,
            lazy=args.lazy
        )
    else:
        # Run in SLURM mode
        if target_time_ms is None:
            print("❌ SLURM mode requires --target-time-ms to be specified")
            sys.exit(1)
        run_slurm_mode(target_time_ms, project_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        import traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)