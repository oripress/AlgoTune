"""
Evaluator module for running and measuring performance of code solutions.
This module provides tools for evaluating solver functions against problem datasets,
including timing, error handling, and performance comparison.

This file is a compatibility layer that re-exports functionality from the new
modular evaluator structure. New code should import directly from the
utils.evaluator.* modules instead.
"""

import logging

# Re-export all functionality from the new modular structure
from AlgoTuner.utils.evaluator.loader import load_task, reload_all_llm_src
from AlgoTuner.utils.evaluator.runner import (
    DATASET_RUNS, DATASET_WARMUPS, run_dataset, run_dataset_with_retries, run_dataset_with_timeout, run_dataset_with_retries_and_timeout
)
from AlgoTuner.utils.evaluator.process_pool import ProcessPoolManager, warmup_evaluator
from AlgoTuner.utils.evaluator.dataset import validate_datasets, repair_datasets
from AlgoTuner.utils.evaluator.main import evaluate_code_on_dataset
from AlgoTuner.utils.evaluator.helpers import prepare_problem_for_solver, make_error_response
from AlgoTuner.utils.casting import cast_input

# Export a consistent public API
__all__ = [
    'load_task',
    'run_evaluation',
    'run_solver_evaluation',
    'run_oracle_evaluation',
    'run_evaluation_function',
    'evluate_code_on_dataset',
    'warmup_evaluator',
    'ProcessPoolManager',
    'reload_all_llm_src',
    'validate_datasets',
    'repair_datasets',  # Keep for backward compatibility
    'prepare_problem_for_solver',
    'make_error_response',
    'cast_input',
]

logging.debug("Using modular evaluator structure (utils.evaluator.*)")
