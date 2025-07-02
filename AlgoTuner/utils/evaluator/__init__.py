"""
Evaluator module for running and measuring performance of code solutions.

This module provides tools for evaluating solver functions against problem datasets,
including timing, error handling, and performance comparison.

The evaluator is organized into several sub-modules:
- helpers: Helper functions for type handling and error formatting
- runner: Core evaluation functions
- process_pool: Process pool management
- dataset: Dataset repair and manipulation
- loader: Module reloading and task loading
- main: Main evaluation entry point

Each module has a specific responsibility, making the codebase more maintainable.
"""

import os
import sys
import logging

# Directory where LLM-generated code is stored
CODE_DIR = os.environ.get("CODE_DIR", "llm_src")

# Ensure CODE_DIR is in sys.path
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# Import public API from sub-modules
from AlgoTuner.utils.evaluator.runner import (
    run_evaluation, 
    run_solver_evaluation,
    run_oracle_evaluation, 
)
from AlgoTuner.utils.evaluator.process_pool import ProcessPoolManager, warmup_evaluator
from AlgoTuner.utils.evaluator.dataset import validate_datasets, repair_datasets
from AlgoTuner.utils.evaluator.loader import reload_all_llm_src, load_task
from AlgoTuner.utils.evaluator.main import evaluate_code_on_dataset
from AlgoTuner.utils.evaluator.helpers import prepare_problem_for_solver, make_error_response
from AlgoTuner.utils.casting import cast_input

# Export public API
__all__ = [
    'load_task',
    'run_evaluation',
    'run_solver_evaluation',
    'run_oracle_evaluation',
    'evaluate_code_on_dataset',
    'warmup_evaluator',
    'ProcessPoolManager',
    'reload_all_llm_src',
    'validate_datasets',
    'repair_datasets',
    'prepare_problem_for_solver',
    'make_error_response',
    'cast_input',
] 