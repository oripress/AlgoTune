#!/usr/bin/env python3
"""
AlgoTuner/timing_core.py - Centralized timing evaluation logic

This module contains the core logic for running timing evaluations that can be used:
1. Standalone for individual tasks
2. From SLURM orchestration scripts  
3. From other automation tools

The logic is separated from SLURM-specific concerns.
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from AlgoTuner.utils.streaming_json import stream_jsonl
from AlgoTuner.utils.evaluator.loader import load_task
from AlgoTuner.utils.evaluator.main import evaluate_baseline_dataset
from AlgoTuner.utils.timing_config import DEV_RUNS, EVAL_RUNS, DATASET_WARMUPS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:timing_core:%(message)s")
logger = logging.getLogger(__name__)

NUM_EVAL_RUNS = 3

def generate_dataset(task_name: str, target_time_ms: int, data_dir: Path, 
                    override_k: Optional[int] = None) -> bool:
    """
    Generate dataset for a task with specified target time.
    
    Args:
        task_name: Name of the task
        target_time_ms: Target timing in milliseconds
        data_dir: Directory to store dataset
        override_k: Optional override for k parameter
        
    Returns:
        True if generation succeeded, False otherwise
    """
    try:
        # Import the dataset generation logic
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from AlgoTuneTasks.base import generate_and_annotate_main
        
        # Prepare arguments for dataset generation
        gen_args = [
            "--task", task_name,
            "--target-time-ms", str(target_time_ms),
            "--data-dir", str(data_dir),
            "--size", "100"  # Default dataset size
        ]
        
        if override_k is not None:
            gen_args.extend(["--k", str(override_k)])
            
        logger.info(f"Generating dataset for {task_name} with target {target_time_ms}ms")
        
        success = generate_and_annotate_main(gen_args)
        return success
        
    except Exception as e:
        logger.error(f"Dataset generation failed for {task_name}: {e}")
        logger.debug(traceback.format_exc())
        return False


def evaluate_task_timing(task_name: str, target_time_ms: int, data_dir: Path, 
                        run_id: Optional[int] = None, data_subset: str = "train") -> Dict:
    """
    Evaluate timing for a single task.
    
    Args:
        task_name: Name of the task to evaluate
        target_time_ms: Target timing in milliseconds  
        data_dir: Directory containing the dataset
        run_id: Optional run ID for identification
        data_subset: "train" or "test" to determine number of runs
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        "success": False,
        "error": None,
        "avg_min_ms": None,
        "std_min_ms": None,
        "target_time_ms": target_time_ms,
        "run_id": run_id
    }
    
    try:
        logger.info(f"Evaluating {task_name} with target {target_time_ms}ms")
        
        task_instance = load_task(task_name=task_name, data_dir=str(data_dir))
        if task_instance is None:
            raise ValueError(f"load_task returned None for '{task_name}'")

        if hasattr(task_instance, "_target_time_ms"):
            task_instance._target_time_ms = target_time_ms
        elif hasattr(task_instance, "set_target_time"):
            task_instance.set_target_time(target_time_ms)

        train_files = glob.glob(str(data_dir / f"{task_name}_T{target_time_ms}ms_n*_size*_train.jsonl"))
        if not train_files:
            raise FileNotFoundError(f"No train JSONL file found in {data_dir} for {task_name}")
        
        train_jsonl = train_files[0]
        # Use memory-efficient approach - pass JSONL path instead of loading dataset
        logger.info(f"Using memory-efficient JSONL path: {train_jsonl}")
        
        # Create dummy iterator for compatibility, actual data will be streamed from JSONL
        dataset_iterable = iter([])

        # Create temporary file for results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            baseline_times_filepath = tmp_file.name

        try:
            # Determine number of runs based on data subset
            num_runs = DEV_RUNS if data_subset == "train" else EVAL_RUNS
            
            # Run evaluation
            logger.info(f"Running evaluation with {num_runs} runs ({data_subset} dataset), {DATASET_WARMUPS} warmups")
            returned_fp = evaluate_baseline_dataset(
                task_obj=task_instance,
                dataset_iterable=dataset_iterable,
                num_runs=num_runs,
                warmup_runs=DATASET_WARMUPS,
                output_file=baseline_times_filepath,
                jsonl_path=train_jsonl
            )

            # Read results
            with open(returned_fp, "r") as f:
                problem_times_dict = json.load(f)

            # Process timing results
            min_times_ms = [
                float(v) for v in problem_times_dict.values()
                if v is not None and isinstance(v, (int, float)) and float(v) > 0
            ]
            
            if not min_times_ms:
                raise ValueError("No valid positive baseline times found")

            results["avg_min_ms"] = float(np.mean(min_times_ms))
            results["std_min_ms"] = float(np.std(min_times_ms))
            results["success"] = True
            
            logger.info(f"SUCCESS: avg={results['avg_min_ms']:.2f}ms, std={results['std_min_ms']:.2f}ms")
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(baseline_times_filepath)
            except OSError:
                pass
                
    except Exception as e:
        error_msg = f"Evaluation failed: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        results["error"] = error_msg

    return results


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
        logger.info(f"Cleaning up {len(files_to_remove)} dataset files with wrong target times for '{task_name}'")
        for file_path in files_to_remove:
            logger.info(f"Removing {file_path.name}")
            file_path.unlink()


def run_complete_timing_evaluation(task_name: str, target_time_ms: int, 
                                 data_dir: Path, num_runs: int = NUM_EVAL_RUNS,
                                 override_k: Optional[int] = None, 
                                 force_regenerate: bool = False) -> Dict:
    """
    Run complete timing evaluation for a task (generation + multiple eval runs).
    
    Args:
        task_name: Name of the task
        target_time_ms: Target timing in milliseconds
        data_dir: Base data directory 
        num_runs: Number of evaluation runs to perform
        override_k: Optional override for k parameter in generation
        force_regenerate: Force regeneration even if dataset exists
        
    Returns:
        Dictionary with complete results including all runs
    """
    task_data_dir = data_dir / task_name
    task_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up any datasets with wrong target times first
    cleanup_wrong_target_datasets(task_data_dir, task_name, target_time_ms)
    
    # Check if dataset exists
    pattern = str(task_data_dir / f"{task_name}_T{target_time_ms}ms_n*_size*_train.jsonl")
    dataset_exists = bool(glob.glob(pattern))
    
    if not dataset_exists or force_regenerate:
        logger.info(f"Generating dataset for {task_name}")
        if force_regenerate and dataset_exists:
            logger.info("Force regenerate requested, removing existing dataset")
            shutil.rmtree(task_data_dir, ignore_errors=True)
            task_data_dir.mkdir(parents=True, exist_ok=True)
            
        success = generate_dataset(task_name, target_time_ms, task_data_dir, override_k)
        if not success:
            return {
                "task_name": task_name,
                "target_time_ms": target_time_ms,
                "success": False,
                "error": "Dataset generation failed",
                "baseline_runs": {}
            }
    else:
        logger.info(f"Using existing dataset for {task_name}")
    
    # Run evaluation multiple times
    baseline_runs = {}
    all_successful = True
    
    for run_id in range(num_runs):
        logger.info(f"Running evaluation {run_id + 1}/{num_runs} for {task_name}")
        result = evaluate_task_timing(task_name, target_time_ms, task_data_dir, run_id)
        
        baseline_runs[str(run_id)] = {
            "success": result["success"],
            "avg_min_ms": result["avg_min_ms"],
            "std_min_ms": result["std_min_ms"]
        }
        
        if result.get("error"):
            baseline_runs[str(run_id)]["error"] = result["error"]
            
        if not result["success"]:
            all_successful = False
    
    # Extract dataset metadata
    n_val = None
    dataset_size = None
    if dataset_exists or not force_regenerate:
        train_files = glob.glob(pattern)
        if train_files:
            filename = os.path.basename(train_files[0])
            # Extract n and dataset_size from filename like "task_T50ms_n123_size100_train.jsonl"
            import re
            match = re.search(r'_n(\d+)_size(\d+)_', filename)
            if match:
                n_val = int(match.group(1))
                dataset_size = int(match.group(2))
    
    result = {
        "task_name": task_name,
        "target_time_ms": target_time_ms,
        "success": all_successful,
        "baseline_runs": baseline_runs
    }
    
    if n_val is not None:
        result["n"] = n_val
    if dataset_size is not None:
        result["dataset_size"] = dataset_size
        
    return result


def main():
    """CLI interface for standalone timing evaluation."""
    parser = argparse.ArgumentParser(description="Run timing evaluation for algorithmic tasks")
    parser.add_argument("--task", required=True, help="Task name to evaluate")
    parser.add_argument("--target-time-ms", type=int, default=50, 
                       help="Target time in milliseconds")
    parser.add_argument("--data-dir", type=Path, required=True,
                       help="Base directory for datasets")
    parser.add_argument("--num-runs", type=int, default=NUM_EVAL_RUNS,
                       help="Number of evaluation runs")
    parser.add_argument("--override-k", type=int, 
                       help="Override k parameter for dataset generation")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration of dataset even if it exists")
    parser.add_argument("--output", type=Path, 
                       help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Set up temporary CODE_DIR for auxiliary files if not already set
    temp_code_dir = None
    if "CODE_DIR" not in os.environ:
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        temp_code_dir = tempfile.mkdtemp(prefix=f"algotune_timing_{args.task}_{unique_id}_")
        os.environ["CODE_DIR"] = temp_code_dir
        logger.info(f"Created temporary CODE_DIR for auxiliary files: {temp_code_dir}")
        
        # Also set up DaCe and other temp dirs to use the same location
        from AlgoTuner.utils.dace_init import _ensure_dace_config
        _ensure_dace_config()
    
    # Run the evaluation
    result = run_complete_timing_evaluation(
        task_name=args.task,
        target_time_ms=args.target_time_ms,
        data_dir=args.data_dir,
        num_runs=args.num_runs,
        override_k=args.override_k,
        force_regenerate=args.force_regenerate
    )
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {args.output}")
    else:
        print(json.dumps(result, indent=2))
    
    # Clean up temporary CODE_DIR if we created it
    if temp_code_dir and os.path.exists(temp_code_dir):
        try:
            shutil.rmtree(temp_code_dir)
            logger.info(f"Cleaned up temporary CODE_DIR: {temp_code_dir}")
        except OSError as e:
            logger.warning(f"Could not clean up temporary CODE_DIR {temp_code_dir}: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main() 