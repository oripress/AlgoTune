#!/usr/bin/env python3
"""
Generate dataset JSONL for a task and annotate each record with baseline solver times.
Usage:
  python3 scripts/generate_and_annotate.py TASK_NAME --data-dir DATA_DIR [--train-size N] [--test-size N]
                                      [--seed S] [--num-runs R] [--warmup-runs W]
"""

import argparse
import json
from pathlib import Path
import logging
import sys # Import sys for stderr handler
import os
from AlgoTuneTasks.factory import TaskFactory
from AlgoTuner.config.loader import load_config
from AlgoTuner.utils.logger import setup_logging # Import setup_logging
from AlgoTuner.utils.discover_and_list_tasks import discover_and_import_tasks

def main():
    # Call setup_logging instead of basicConfig or dedicated logger setup
    # Pass task_name from args, model can be None or fixed string for this script
    parser = argparse.ArgumentParser(description="Generate dataset using Task.load_dataset (without baseline annotation).")
    parser.add_argument("task_name", help="Name of the registered task to process")
    parser.add_argument("--data-dir", required=True, help="Directory where dataset JSONL files will be stored")
    parser.add_argument("--train-size", type=int, default=None, # Default handled later
                        help="Number of train instances to generate (from config: dataset.train_size)")
    parser.add_argument("--test-size",  type=int, default=None, # Default handled later
                        help="Number of test instances to generate (from config: dataset.test_size)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset creation")
    parser.add_argument("--k", type=int, default=None, help="Use this k value for dataset generation rather than estimating it")
    args = parser.parse_args()
    
    # Setup logging AFTER parsing args to get task_name
    setup_logging(task=args.task_name, model="timing_script") # Use a fixed model name

    target_time_ms_env = os.environ.get("TARGET_TIME_MS")
    if target_time_ms_env is None:
        raise RuntimeError("TARGET_TIME_MS environment variable must be set by the pipeline. No fallback to config is allowed.")
    oracle_time_limit = int(target_time_ms_env)

    temp_task_instance_for_time = TaskFactory(args.task_name, oracle_time_limit=oracle_time_limit, data_dir=None)
    target_time_ms = temp_task_instance_for_time._get_target_time_ms()
    del temp_task_instance_for_time # Clean up temporary instance
    logging.info(f"Determined target_time_ms = {target_time_ms} for pre-check.")

    cfg = load_config()
    ds_cfg = cfg.get('dataset', {})
    train_size = ds_cfg.get('train_size', 100)
    test_size = ds_cfg.get('test_size', 100)

    # --- Clean up datasets with wrong target times BEFORE checking for existing files ---
    task_data_dir = Path(args.data_dir)
    if task_data_dir.exists():
        # Find all dataset files for this task
        pattern = f"{args.task_name}_T*ms_n*_size*_*.jsonl"
        all_files = list(task_data_dir.glob(pattern))
        
        # Filter out files that don't match the current target time
        current_pattern = f"{args.task_name}_T{target_time_ms}ms_n*_size*_*.jsonl"
        correct_files = set(task_data_dir.glob(current_pattern))
        
        files_to_remove = [f for f in all_files if f not in correct_files]
        
        if files_to_remove:
            logging.info(f"Cleaning up {len(files_to_remove)} dataset files with wrong target times")
            for file_path in files_to_remove:
                logging.info(f"Removing {file_path.name}")
                file_path.unlink()
    
    # --- Check for existing precise dataset file BEFORE proceeding --- 
    # Use raw task name for filename matching
    filename_task_name = args.task_name 
    expected_train_filename = f"{filename_task_name}_T{target_time_ms}ms_n*_size{train_size}_train.jsonl"
    # Construct the full path to the directory where the file should be
    # Note: assumes args.data_dir is the task-specific directory, consistent with recent changes
    task_data_dir = Path(args.data_dir)
    logging.info(f"Checking for existing file pattern '{expected_train_filename}' in {task_data_dir}")
    found_files = list(task_data_dir.glob(expected_train_filename))
    if found_files:
        logging.info(f"Found existing dataset file matching T={target_time_ms}ms and size={train_size}: {found_files[0].name}. Skipping generation.")
        sys.exit(0) # Exit successfully
    else:
        logging.info("No exact match found. Proceeding with load/generation logic.")
    # --- End Check --- 

    try:
        discover_and_import_tasks()
        logging.info("Discovered and imported all task modules.") # Use logging (root)
    except Exception as e:
        logging.warning(f"Task auto-discovery failed: {e}") # Use logging (root)

    # Step 1: Instantiate the task object
    task = TaskFactory(
        args.task_name,
        oracle_time_limit=oracle_time_limit,
        data_dir=args.data_dir
    )
    # Override k if provided, so load_dataset uses fixed problem size instead of timing estimation
    if args.k is not None:
        task.k = args.k
    
    # Step 2: Call load_dataset - This will create the dataset files if they don't exist,
    #         using the logic in AlgoTune/tasks/base.py (which now skips annotation).
    #         We don't need the returned iterables here, just the file creation side effect.
    logging.info(f"Ensuring dataset exists for {args.task_name} via task.load_dataset()...") # Use logging (root)
    try:
        _train_gen, _test_gen = task.load_dataset(
            train_size=train_size,
            test_size=test_size,
            random_seed=args.seed,
        )
        logging.info(f"task.load_dataset() called successfully.") # Use logging (root)
        
    except Exception as e:
        logging.error(f"Dataset generation/loading via task.load_dataset failed for task {args.task_name}: {e}", exc_info=True) # Use logging (root)
        sys.exit(1)

        logging.info(f"Completed dataset check/generation for task {args.task_name}") # Use logging (root)

if __name__ == "__main__":
    main() 