#!/usr/bin/env python3
"""
Single task evaluation script for SLURM array jobs.
Evaluates one task for one model and writes result to a temporary file.
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from AlgoTuner.main import acquire_lock, release_lock
    # Try importing evaluation modules - simplified approach
    print("Loading evaluation modules...")
except ImportError as e:
    print(f"Error importing basic modules: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate single task for SLURM array")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--results-dir", type=Path, 
                       default=os.environ.get("EVAL_RESULTS_DIR", "results"),
                       help="Results directory")
    parser.add_argument("--generation-file", type=Path,
                       default=os.environ.get("EVAL_GENERATION_FILE", "reports/generation.json"),
                       help="Generation file")
    parser.add_argument("--data-dir", type=Path,
                       default=os.environ.get("DATA_DIR", "../data"),
                       help="Data directory")
    parser.add_argument("--num-runs", type=int,
                       default=int(os.environ.get("EVAL_NUM_RUNS", "10")),
                       help="Number of runs")
    parser.add_argument("--output-file", type=Path,
                       default=os.environ.get("EVAL_OUTPUT_FILE", "reports/evaluate_summary.json"),
                       help="Output summary file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load generation data
    generation_data = load_generation_data(args.generation_file)
    if not generation_data:
        logging.error("Failed to load generation data")
        sys.exit(1)
    
    # Check if task exists in generation data
    if args.task not in generation_data:
        logging.error(f"Task {args.task} not found in generation data")
        sys.exit(1)
    
    # Set up paths
    code_dir = args.results_dir / args.model / args.task
    if not code_dir.exists():
        logging.error(f"Code directory does not exist: {code_dir}")
        sys.exit(1)
    
    display_model_name = normalize_model_name(args.model)
    
    # Prepare arguments for evaluation
    eval_args = (
        args.task,
        args.model,
        display_model_name,
        str(code_dir),
        generation_data,
        str(args.data_dir),
        args.num_runs
    )
    
    # Run evaluation
    logging.info(f"Starting evaluation: {args.task} for {display_model_name}")
    result = evaluate_single_task(eval_args)
    
    # Write result directly to summary file using lock mechanism (like agent mode)
    lock_file_path = args.output_file.with_suffix('.lock')
    
    if not acquire_lock(str(lock_file_path)):
        logging.error(f"Failed to acquire lock for {args.output_file}")
        sys.exit(1)
    
    try:
        # Load existing summary or create new one
        summary_data = {}
        if args.output_file.exists():
            try:
                with open(args.output_file, 'r') as f:
                    summary_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logging.warning(f"Could not load existing summary file {args.output_file}, creating new one")
                summary_data = {}
        
        # Add this result to summary
        if result.task_name not in summary_data:
            summary_data[result.task_name] = {}
        
        if result.success and result.speedup is not None:
            speedup_str = f"{result.speedup:.4f}"
        else:
            speedup_str = "N/A"
        
        summary_data[result.task_name][result.model_name] = {
            "final_speedup": speedup_str
        }
        
        # Write updated summary
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(summary_data, f, indent=2, sort_keys=True)
            
        if result.success:
            logging.info(f"✅ {args.task}/{display_model_name}: {result.speedup:.4f}x speedup")
        else:
            logging.error(f"❌ {args.task}/{display_model_name}: {result.error_message}")
            
        logging.info(f"Result written to summary: {args.output_file}")
        
    finally:
        release_lock(str(lock_file_path))


if __name__ == "__main__":
    main()