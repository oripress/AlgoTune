import os
import sys
import inspect
import importlib
import argparse
import logging
import traceback
from AlgoTuneTasks.base import Task
from AlgoTuneTasks.factory import TaskFactory

def initialize_solver_from_task(task_instance, output_dir):
    """
    Initialize solver.py with a stubbed Solver class and solve() wrapper.

    Args:
        task_instance (Task): Task instance (not used, but kept for interface compatibility)
        output_dir (str): Directory where solver.py should be created

    Returns:
        str: Path to the created solver.py file
    """
    # Log call stack to help diagnose when this function is being called
    caller_info = traceback.extract_stack()[-2]  # Get the caller's info
    logging.info(f"initialize_solver_from_task called from {caller_info.filename}:{caller_info.lineno}")
    
    # Create solver.py as an empty file ONLY if it doesn't exist or is empty
    solver_path = os.path.join(output_dir, "solver.py")
    
    # Check if the file already exists and has content
    if os.path.exists(solver_path) and os.path.getsize(solver_path) > 0:
        logging.info(f"Solver.py already exists and has content at {solver_path}. Skipping initialization.")
        return solver_path
    
    # Only create an empty file if it doesn't exist or is empty
    logging.info(f"Creating empty solver.py in {solver_path}")
    with open(solver_path, "w") as f:
        pass # Write nothing to create an empty file
        
    logging.info(f"Created empty solver.py in {solver_path}")
    return solver_path

def initialize_solver(task_name, output_dir, oracle_time_limit=10000):
    task_instance = TaskFactory(task_name, oracle_time_limit=oracle_time_limit)
    return initialize_solver_from_task(task_instance, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize solver.py with a task's solve function")
    parser.add_argument("--task", type=str, required=True, help="Name of the task")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where solver.py should be created")
    parser.add_argument("--oracle-time-limit", type=int, default=10000, help="Time limit for the oracle")

    args = parser.parse_args()
    initialize_solver(args.task, args.output_dir, args.oracle_time_limit)