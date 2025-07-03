"""
Main evaluator interface and entry point.
"""

import os
import sys
import logging
import time
import traceback
import math
from typing import Dict, Any, Optional, List, Iterable, Tuple, Union
import numpy as np
import builtins
import random
from functools import partial
import multiprocessing
import multiprocessing as mp
from pathlib import Path
import re
from contextlib import redirect_stdout, redirect_stderr
from collections import deque, defaultdict
from threading import Lock
import importlib
import json
import tempfile
import io
import threading
import psutil
from AlgoTuner.utils.evaluator.benchmark_runner import BenchmarkPool

from AlgoTuner.utils.message_writer import MessageWriter
from AlgoTuner.utils.validation import validate_solver_setup
from AlgoTuner.utils.time_management import calculate_timeout
from AlgoTuner.utils.evaluator.loader import load_task, reload_all_llm_src
from AlgoTuner.utils.evaluator.process_pool import ProcessPoolManager
from AlgoTuner.utils.evaluator.runner import run_evaluation, run_oracle_evaluation, execute_and_capture_errors, run_solver_evaluation, ValidationException, _validate_solution
from AlgoTuner.utils.evaluator.scoring import calculate_input_speedup
from AlgoTuner.utils.utils import clean_traceback, format_object_shape
from AlgoTuner.utils.error_utils import extract_error_context, create_standard_error_result
from AlgoTuner.utils.precise_timing import _initialize_timing_system
from AlgoTuner.utils.timing_manager import TimingManager, Phase
from .failure_analyzer import analyze_is_solution_failures, MAX_FAILURES_TO_ANALYZE_PER_TASK
import resource
from AlgoTuner.utils.benchmark import run_benchmark
from AlgoTuner.utils.multiprocessing_utils import _pool_worker_initializer, load_pool_config
from .benchmark_runner import HardBenchmarkFailure
from AlgoTuner.utils.casting import cast_result
from AlgoTuner.utils.solver_loader import locate_solver_file, load_solver_module, get_fresh_solve_callable, get_solve_callable
from AlgoTuner.utils.streaming_json import stream_jsonl
from AlgoTuner.utils.result_utils import validation_already_done
from AlgoTuner.utils.timing_config import DATASET_RUNS, DATASET_WARMUPS, DEV_RUNS, EVAL_RUNS
from AlgoTuner.utils.smart_result_handler import ResultMetadata
from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark, run_isolated_benchmark_with_fetch

from AlgoTuner.config.loader import load_config
from AlgoTuner.utils.timing_config import RUNS, WARMUPS
_config = load_config()
_bench_cfg = _config.get("benchmark", {})

CODE_DIR = os.environ.get("CODE_DIR", "llm_src")

# Define critical error types at module level for broader access
CRITICAL_ERROR_TYPES = {
    'benchmark_error', 
    'solver_exception', 
    'runtime_error', 
    'setup_error', 
    'method_error', 
    'oracle_setup_error', 
    'validation_error', # Treat errors during validation itself as critical
    'is_critical_validation_error', # Added for explicit marking of critical validation errors
    'oom_kill', # OOM kills should be treated as critical but not crash the evaluation
    'memory_error',
    # Excluded: 'timeout', 'invalid_solution' 
}

# Runtime flags for optional monitoring features (defined early to avoid NameError)
# These can be overridden via environment variables if desired.
try:
    _PARENT_MEM_CHECK_EVERY = int(os.getenv("PARENT_MEM_CHECK_EVERY", "0"))
except ValueError:
    _PARENT_MEM_CHECK_EVERY = 0

# --- Disable legacy toggles ---
ENABLE_HEARTBEAT = False  # hard-off, heartbeat thread removed
_PARENT_MEM_CHECK_EVERY = 0  # disable parent memory checks

def run_partial_func(func_tuple):
    func, args = func_tuple
    return func(*args)

def _simple_baseline_evaluation(jsonl_path, index, task_name, data_dir, num_runs, warmup_runs):
    """
    Simple baseline evaluation using dedicated baseline timing (like dataset generation).
    No subprocess overhead, just direct timing like in base.py.
    """
    try:
        from AlgoTuner.utils.evaluator.loader import load_task
        from AlgoTuner.utils.streaming_json import stream_jsonl
        from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark
        
        # Load task ONCE and reuse (avoid reloading for each run)
        task_obj = load_task(task_name, data_dir)
        
        # Load only the specific problem we need (memory-efficient approach)
        import orjson
        import functools
        from AlgoTuner.utils.serialization import dataset_decoder
        import os
        
        problem_instance = None
        actual_base_dir = os.path.dirname(jsonl_path)
        object_hook_for_load = functools.partial(dataset_decoder, base_dir=actual_base_dir)
        
        with open(jsonl_path, 'r') as f:
            current_index = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if current_index == index:
                    try:
                        raw_record = orjson.loads(line)
                        processed_record = object_hook_for_load(raw_record)
                        problem_instance = processed_record.get("problem", processed_record)
                        break
                    except orjson.JSONDecodeError as e:
                        raise RuntimeError(f"JSON Decode Error in {jsonl_path}, line {current_index}: {e}")
                current_index += 1
        
        if problem_instance is None:
            raise IndexError(f"Index {index} out of range in dataset")
        
        # rec is automatically cleaned up here
        
        # Calculate proper timeout based on target time
        target_time_s = 0.1  # Default fallback to 100ms if no target time available
        if hasattr(task_obj, 'target_time_ms') and task_obj.target_time_ms:
            target_time_s = task_obj.target_time_ms / 1000.0
        elif hasattr(task_obj, '_get_target_time_ms'):
            try:
                target_time_s = task_obj._get_target_time_ms() / 1000.0
            except Exception:
                pass
        
        # Each subprocess does: 10x warmup(problem[i+1]) + 10x timed(problem[i])
        # Since we don't have per-problem baselines yet, use 2x target_time as approximation
        timeout_per_subprocess = target_time_s * 2.0 * 10.0
        baseline_timeout_s = max(60.0, timeout_per_subprocess)
        
        # Use isolated benchmark timing (like dataset generation)
        # Get the current CODE_DIR at runtime (not cached module-level variable)
        base_code_dir = os.environ.get("CODE_DIR", data_dir)
        # For isolated benchmark, use the task-specific directory if it exists
        task_code_dir = os.path.join(base_code_dir, "AlgoTuneTasks", task_name)
        if os.path.isdir(task_code_dir):
            current_code_dir = task_code_dir
        else:
            current_code_dir = base_code_dir
        logging.debug(f"BASELINE_DEBUG: Using code_dir='{current_code_dir}' for isolated benchmark")
        
        # Load the dataset to get a different problem for warmup (matching baseline worker behavior)
        from AlgoTuner.utils.dataset_utils import read_jsonl_data
        # Use memory-efficient streaming approach for large datasets
        from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark_with_fetch
        
        # Create fetch info for streaming access
        warmup_idx = (index - 1) % 100  # Use modulo to avoid loading entire dataset
        warmup_fetch_info = {"type": "jsonl_streaming", "path": jsonl_path, "index": warmup_idx}
        timed_fetch_info = {"type": "jsonl_streaming", "path": jsonl_path, "index": index}
        
        logging.debug(f"BASELINE_DEBUG: Using streaming fetch - problem {index} for timing, problem {warmup_idx} for warmup")
        
        result = run_isolated_benchmark_with_fetch(
            task_name=task_name,
            code_dir=current_code_dir,
            warmup_fetch_info=warmup_fetch_info,
            timed_fetch_info=timed_fetch_info,
            num_runs=num_runs,
            timeout_seconds=baseline_timeout_s,
        )
        
        logging.info(f"BASELINE_TIMING_DEBUG: run_isolated_benchmark returned: success={result.get('success')}")
        logging.info(f"BASELINE_TIMING_DEBUG: result keys: {list(result.keys())}")
        logging.info(f"BASELINE_TIMING_DEBUG: min_time_ms={result.get('min_time_ms')}, mean_time_ms={result.get('mean_time_ms')}")
        if result.get("error"):
            logging.info(f"BASELINE_TIMING_DEBUG: error={result.get('error')}")
        
        if result.get("success"):
            min_time_ms = result.get("min_time_ms", 0.0)
            mean_time_ms = result.get("mean_time_ms", min_time_ms)
            
            out = {
                "success": True,
                "result": result.get("result"),
                "min_time_ms": min_time_ms,
                "elapsed_ms": mean_time_ms,
                "mean_ms": mean_time_ms,
                "timeout_occurred": False
            }
            _cleanup_timing_fields(out)
            return out
        else:
            out = {
                "success": False,
                "error": result.get("error", "Baseline timing failed"),
                "min_time_ms": None,
                "elapsed_ms": 0.0,
                "timeout_occurred": result.get("timeout_occurred", False)
            }
            _cleanup_timing_fields(out)
            return out
            
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        return {
            "success": False,
            "error": f"Simple baseline evaluation error: {e}",
            "min_time_ms": None,
            "elapsed_ms": 0.0,
            "timeout_occurred": False
        }

class AttributedList(list):
    """A list subclass that supports attributes."""
    def __init__(self, *args, **kwargs):
        self.__dict__ = {}
        super().__init__(*args)
        for key, value in kwargs.items():
            setattr(self, key, value)

def _return_with_message_writer(
    evaluation_output: Union[Dict[str, Any], List[Dict[str, Any]]],
    failure_analysis_logs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Format evaluation result with MessageWriter.
    Includes captured stdout if present.
    Conditionally appends is_solution failure analysis logs.
    
    Args:
        evaluation_output: Raw evaluation result (dict or list)
        failure_analysis_logs: Optional list of formatted strings from failure analysis
        
    Returns:
        Evaluation result with formatted message
    """
    mw = MessageWriter()
    if isinstance(evaluation_output, dict):
        logging.info(f"DIAG: _return_with_message_writer received evaluation_output. Keys: {list(evaluation_output.keys())}")
        error_val = evaluation_output.get("error")
        context_val = evaluation_output.get("code_context")
    else:
        logging.info("DIAG: _return_with_message_writer received non-dict evaluation_output.")
        error_val = None
        context_val = None
        
    logging.info(f"DIAG: evaluation_output['error'] (first 100 chars): {str(error_val)[:100]}...")
    logging.info(f"DIAG: evaluation_output['code_context'] is present: {bool(context_val)}")
    if context_val:
        logging.info(f"DIAG: evaluation_output['code_context'] length: {len(str(context_val))}")
        logging.info(f"DIAG: evaluation_output['code_context'] first 100 chars: {str(context_val)[:100]}...") 
    
    if isinstance(evaluation_output, dict):
        if "aggregate_metrics" not in evaluation_output:
            evaluation_output["aggregate_metrics"] = {}
            logging.warning("_return_with_message_writer: Added missing 'aggregate_metrics' key before formatting.")
        aggregate_metrics = evaluation_output.get("aggregate_metrics", {})
        analysis_logs = evaluation_output.get("invalid_solution_analysis", failure_analysis_logs)
        if analysis_logs is not None:
            setattr(evaluation_output, "invalid_solution_analysis", analysis_logs)
    else:
        aggregate_metrics = getattr(evaluation_output, 'aggregate_metrics', {})
        # Attempt to include invalid solution analysis from the dict (dataset evaluations)
        analysis_logs = getattr(evaluation_output, 'invalid_solution_analysis', failure_analysis_logs)
        if not analysis_logs:
            analysis_logs = failure_analysis_logs
        if analysis_logs is not None:
            setattr(evaluation_output, "invalid_solution_analysis", analysis_logs)
    
    if isinstance(evaluation_output, dict):
        base_formatted_message = mw.format_evaluation_result_from_raw(evaluation_output)
    else:
        # Convert AttributedList to the new expected format without the old "results" key
        temp_dict = {
            "success": True,
            "aggregate_metrics": aggregate_metrics,
            "evaluation_type": "dataset",
            "invalid_solution_analysis": analysis_logs if analysis_logs else []
        }
        base_formatted_message = mw.format_evaluation_result_from_raw(temp_dict)
    
    final_formatted_message = base_formatted_message

    # --- Append average timings if available ---
    avg_solver_ms = aggregate_metrics.get("avg_solver_time_ms")
    avg_oracle_ms = aggregate_metrics.get("avg_oracle_time_ms")

    # Add timing information if available
    evaluation_successful = True if not isinstance(evaluation_output, dict) else evaluation_output.get("success", True)
    if evaluation_successful and avg_solver_ms is not None and avg_oracle_ms is not None:
        timing_lines = [
            f"Average time for your solve() [on valid examples]: {avg_solver_ms:.3f} ms",
            f"Average time for the baseline solve() [on valid examples]: {avg_oracle_ms:.3f} ms",
            ""
        ]
        final_formatted_message += "\n" + "\n".join(timing_lines)
    else:
        logging.debug("Average timing information not available in aggregate_metrics.")
        
    # --- Append invalid solution analysis if available ---
    # MessageWriter will embed up to three randomly selected invalid examples when
    # the 'invalid_solution_analysis' key is present, so no additional appending
    # is required here to avoid duplicate content.

    # Build the final result with the formatted message
    if isinstance(evaluation_output, dict):
        # Always mark success=True so that external wrappers do not prefix the
        # formatted message with 'Error:'. The aggregate_metrics section still
        # communicates validity and timeout statistics.
        evaluation_output["success"] = True
        # For dictionaries, add the formatted message and return
        evaluation_output["formatted_message"] = final_formatted_message
        return evaluation_output
    else:
        # For lists (AttributedList), create a new dict with the new format
        result = {
            "success": True,
            "formatted_message": final_formatted_message,
            "aggregate_metrics": aggregate_metrics,
            "evaluation_type": "dataset",
            "invalid_solution_analysis": analysis_logs if analysis_logs else []
        }
        if analysis_logs:
            setattr(result, "invalid_solution_analysis", analysis_logs)
        return result


def _make_hashable(solution: Any) -> Any:
    """Converts a potential solution (list, tuple, numpy array, etc.) into a hashable type.
       Primarily converts mutable list-like structures and numpy arrays to nested tuples.
    """
    if isinstance(solution, list):
        return tuple(_make_hashable(item) for item in solution)
    if isinstance(solution, tuple):
        return tuple(_make_hashable(item) for item in solution)
    if isinstance(solution, np.ndarray):
        # Convert numpy array to nested tuple structure
        if solution.ndim == 0:
            return solution.item() # Extract scalar value
        return tuple(_make_hashable(item) for item in solution)
    if isinstance(solution, (int, float, str, bool, complex, bytes)) or solution is None:
        return solution # Already hashable or None
    # Fallback for other types (like custom objects) - may fail if not hashable
    try:
        hash(solution)
        return solution
    except TypeError:
        logging.warning(f"_make_hashable: Could not convert type {type(solution)} to hashable, using repr as fallback.")
        return repr(solution)

def _calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate metrics from evaluation results.
    
    Args:
        results: List of individual problem evaluation results
                Each dict should have at least: success, speedup (if applicable)
                
    Returns:
        Dict containing aggregate metrics like mean_speedup, success_rate, etc.
    """
    if not results:
        return {}
    
    # Initialize counters and accumulators
    num_evaluated = len(results)
    num_valid = 0
    num_invalid = 0
    num_timeouts = 0
    num_errors = 0
    num_inf_speedup = 0
    
    # Time accumulators
    solver_times = []
    oracle_times = []
    solver_times_mutual = []
    oracle_times_mutual = []
    speedups = []
    
    # Process each result individually
    for result in results:
        error_type = result.get('error_type', '')
        is_valid_solution = result.get('is_valid', False)
        timeout_occurred = (error_type == 'timeout') or result.get('timeout_occurred', False)
        
        if is_valid_solution:
            num_valid += 1
            
            # --- Collect speedup for valid solutions --- 
            speedup_val = result.get('speedup')
            if speedup_val is not None:
                if speedup_val == float('inf'):
                    num_inf_speedup += 1
                speedups.append(speedup_val)

            # --- Accumulate times for averages (using correct keys from result dict) ---
            solver_time = result.get('min_time_ms')
            oracle_time_for_avg = result.get('baseline_time_ms')

            if solver_time is not None:
                solver_times.append(solver_time)
            if oracle_time_for_avg is not None:
                oracle_times.append(oracle_time_for_avg)
            if solver_time is not None and oracle_time_for_avg is not None:
                solver_times_mutual.append(solver_time)
                oracle_times_mutual.append(oracle_time_for_avg)
                
        elif timeout_occurred:
            num_timeouts += 1
        elif error_type == 'invalid_solution':
            # Count invalid solutions from is_solution
            num_invalid += 1
        else:
            # Other validation or execution errors
            num_errors += 1
    
    # Calculate average times
    avg_solver_time = None
    avg_oracle_time = None
    avg_solver_mutual = None
    avg_oracle_mutual = None
    if solver_times:
        avg_solver_time = np.mean(solver_times)
    if oracle_times:
        avg_oracle_time = np.mean(oracle_times)
    if solver_times_mutual:
        avg_solver_mutual = np.mean(solver_times_mutual)
    if oracle_times_mutual:
        avg_oracle_mutual = np.mean(oracle_times_mutual)
    
    # Calculate success rate & overall validity
    success_rate = num_valid / num_evaluated if num_evaluated > 0 else 0.0
    overall_valid = num_valid > 0 and num_valid == num_evaluated
    
    # Calculate mean and median speedup, skipping infinite values
    finite_speedups = [s for s in speedups if s is not None and s != float('inf')] # Ensure not None before checking for inf
    
    if finite_speedups:
        mean_speedup = np.mean(finite_speedups)
        median_speedup = np.median(finite_speedups) if finite_speedups else None
    else:
        # No finite speedups. Check if all actual (non-None) speedups were infinite.
        non_none_speedups = [s for s in speedups if s is not None]
        if non_none_speedups and all(s == float('inf') for s in non_none_speedups):
            mean_speedup = float('inf')
            median_speedup = float('inf')
        else: # speedups list was empty, contained only Nones, or a mix not exclusively infinite
            mean_speedup = None
            median_speedup = None
            
    # If not every solution was valid, invalidate speedup metrics
    if not overall_valid:
        logging.info("Not all solutions were valid; setting speedup metrics to None (N/A).")
        mean_speedup = None
        median_speedup = None
    
    # Assemble the aggregate metrics
    metrics = {
        'num_evaluated': num_evaluated,
        'overall_valid': overall_valid,
        'mean_speedup': mean_speedup,
        'median_speedup': median_speedup,
        'success_rate': success_rate,
        'num_valid': num_valid,
        'num_invalid': num_invalid,
        'num_errors': num_errors,
        'num_timeouts': num_timeouts,
        'num_inf_speedup': num_inf_speedup
    }
    
    # Add timing metrics (carefully handling None values)
    if avg_solver_time is not None:
        metrics['avg_solver_time_ms'] = avg_solver_time
    if avg_oracle_time is not None:
        metrics['avg_oracle_time_ms'] = avg_oracle_time
    if avg_solver_mutual is not None:
        metrics['avg_solver_time_on_mutual_valid'] = avg_solver_mutual
    if avg_oracle_mutual is not None:
        metrics['avg_oracle_time_on_mutual_valid'] = avg_oracle_mutual
    
    return metrics


def _calculate_timeout(
    task_instance: Any, 
    config: Dict[str, Any],
    timeout_factor: float = 10.0
) -> float:
    """
    Calculate a reasonable timeout for task evaluation.
    
    Args:
        task_instance: Instance of the task
        config: Configuration dictionary
        timeout_factor: Multiplier for the timeout
        
    Returns:
        Timeout in milliseconds
    """
    # Default timeout (10 seconds)
    default_timeout_ms = 10000
    
    # Try to get the task's timeout from the task instance
    task_timeout_ms = getattr(task_instance, "timeout_ms", None)
    
    # Check if there's a task-specific timeout in the config
    config_timeout_ms = config.get("tasks", {}).get(
        task_instance.__class__.__name__, {}
    ).get("timeout_ms", None)
    
    # Check if there's a global timeout in the config
    global_timeout_ms = config.get("global", {}).get(
        "evaluation", {}
    ).get("timeout_ms", None)
    
    # Use the most specific timeout, with fallbacks
    timeout_ms = (
        task_timeout_ms or 
        config_timeout_ms or 
        global_timeout_ms or 
        default_timeout_ms
    )
    
    # Scale the timeout if running in debug mode
    debug_mode = config.get("global", {}).get("debug", False)
    if debug_mode:
        timeout_ms *= 2  # Double the timeout in debug mode
    
    # Apply the timeout factor
    timeout_ms *= timeout_factor
    
    logging.info(f"Using timeout of {timeout_ms:.2f}ms for evaluation")
    return timeout_ms


def _evaluate_single_problem(
    dataset_item: Dict[str, Any],
    task_instance: Any,
    provided_oracle_time_ms: Optional[float], # If provided, skip baseline measurement
    num_runs: int = 5,
    warmup_runs: int = 3,
) -> Dict[str, Any]:
    """Helper function to evaluate a single problem instance and calculate its score.
    Returns a dict containing results, including 'problem' and 'solver_output'.
    """
    timing = TimingManager()
    problem_data = dataset_item.get('problem')

    # --- Phase timing instrumentation ---
    all_start = time.perf_counter_ns()

    # --- Determine oracle (baseline) time ---
    if provided_oracle_time_ms is not None:
        oracle_time_ms = provided_oracle_time_ms
        logging.info(f"Using provided oracle_time_ms={oracle_time_ms} ms for problem_id={dataset_item.get('id', 'N/A')}")
        # If baseline is provided, we assume it succeeded, but we don't have the result dict.
        baseline_result = {'success': True, 'elapsed_ms': oracle_time_ms} # Mock baseline result
    else:
        logging.debug("Measuring baseline oracle time for problem")
        oracle_start = time.perf_counter_ns() # Start timer here if measuring
        with timing.phase(Phase.ORACLE_RUN):
            baseline_result = run_oracle_evaluation(
                problem=problem_data,
                task_instance=task_instance,
                oracle_time_ms=None,
                capture_output=True,
                needs_casting=True,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                skip_validation=True
            )
        oracle_end = time.perf_counter_ns() # End timer here if measuring
        oracle_ms = (oracle_end - oracle_start) / 1e6
        logging.info(f"Per-problem timing: oracle_phase={oracle_ms:.2f}ms")

        if not baseline_result.get('success', False):
            logging.error("Oracle run failed for problem. Cannot proceed with solver evaluation.")
            # Use CORRECT indentation for the return dictionary
            return { 
                **dataset_item,
                **baseline_result, 
                'is_valid': False,
                'speedup': None,
                'solver_time_ms': None,
                'oracle_time_ms': baseline_result.get('elapsed_ms', 0) # Use measured time if available
            }
    # If we get here, baseline succeeded (either provided or measured)
    oracle_time_ms = baseline_result.get('elapsed_ms', 0) # Get the final baseline time

    # Remove benchmark name generation comments if they exist
    # ... existing code ... 

    # --- Solver evaluation timing ---
    solver_start = time.perf_counter_ns()
    # SOLVER_RUN phase wraps solver benchmarking
    with timing.phase(Phase.SOLVER_RUN, capture_output=False):
        solver_result = run_solver_evaluation(
            problem=problem_data,
            task_instance=task_instance,
            oracle_time_ms=oracle_time_ms, # Pass baseline time for per-problem solver timeout
            capture_output=False, 
            needs_casting=True,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
        )

    solver_end = time.perf_counter_ns()
    solver_ms = (solver_end - solver_start) / 1e6
    logging.info(f"Per-problem timing: solver_phase={solver_ms:.2f}ms")

    # --- ADDED: Define solver success status after solver run --- 
    # Note: solver_produced_result is based only on the solver run now
    solver_produced_result = solver_result.get("success", False) 

    # === NEW: Get the FINAL result for validation ===
    result_to_validate = None
    if solver_produced_result:
        result_to_validate = solver_result.get("result")
        
        # Results are no longer stripped, so use them directly
    else:
        logging.warning("Solver failed to produce a result. Cannot perform validation.")

    # --- Validation timing ---
    validation_start = time.perf_counter_ns()
    # Run validation on the FINAL solver output if available
    logging.debug(f"Attempting validation on final solver output (if successful)...")
    processed_problem = problem_data # Use the problem data available in this scope
    is_valid = False # Default to False
    validation_error_info = {} # Default to empty dict

    if result_to_validate is not None:
        # Check if validation was already done in-process
        if solver_result.get("validation_result") is not None:
            logging.info("Using in-process validation result")
            validation_result = solver_result.get("validation_result")
        else:
            # Always validate first, then strip
            logging.info("Running validation")
            validation_result = _validate_solution(task_instance, processed_problem, result_to_validate)
            
            # Strip result immediately after validation but preserve validation result
            logging.info("Result stripped after validation - keeping only validation status")
            solver_result["result"] = {"stripped_after_validation": True}
            solver_result["validation_result"] = validation_result
        is_valid = validation_result.get('success', False)
        
        # Store error info only if validation actually failed
        if not is_valid:
            logging.warning(f"EVAL_MAIN: Validation failed for problem problem_id={dataset_item.get('k', 'N/A')}. Error: {validation_result.get('error')}")
            # Store validation error details
            validation_error_info = {
                'error': validation_result.get('error'),
                'error_type': validation_result.get('error_type', 'validation_error'),
                'traceback': validation_result.get('traceback'),
                'code_context': validation_result.get('code_context'),
            }
            
            # Check if this is a critical validation error (exception in is_solution)
            if validation_result.get('is_critical_validation_error', False):
                logging.error(f"EVAL_MAIN: Critical validation error detected. Stopping evaluation.")
            elif validation_result.get('error_type') == 'invalid_solution':
                # Non-critical validation failure (is_solution returned False)
                logging.debug(f"EVAL_MAIN: Invalid solution detected. Continuing evaluation.")
                
                # Capture context immediately when an invalid solution is detected
                if not hasattr(task_instance, '_last_is_solution_failure_context'):
                    logging.debug(f"EVAL_MAIN: Capturing is_solution failure context for problem_id={dataset_item.get('id', 'N/A')}")
                    from AlgoTuner.utils.evaluator.failure_analyzer import trace_is_solution_failure
                    failure_context = trace_is_solution_failure(task_instance, processed_problem, result_to_validate)
                
                # Include the is_solution context if available
                if hasattr(task_instance, '_last_is_solution_failure_context'):
                    validation_error_info["is_solution_context"] = task_instance._last_is_solution_failure_context
                    context_length = len(task_instance._last_is_solution_failure_context)
                    logging.debug(f"EVAL_MAIN: Including _last_is_solution_failure_context in validation_error_info (length: {context_length})")
                    # Log first few lines to help with debugging
                    first_lines = task_instance._last_is_solution_failure_context.split('\n')[:3]
                    logging.debug(f"EVAL_MAIN: First lines of context: {first_lines}")
                else:
                    logging.warning("EVAL_MAIN: task_instance still does NOT have _last_is_solution_failure_context attribute!")
                    # Add a placeholder message
                    validation_error_info["is_solution_context"] = "# No context available from is_solution (trace failed)"
            else:
                # Other validation errors
                logging.info(f"EVAL_MAIN: Validation failed for problem {dataset_item.get('k', 'N/A')}. Error: {validation_result.get('error')}")
                is_valid = False
            # --- End Validation Failure Handling ---
        else:
            logging.debug(f"Validation successful for final solver output.")
    elif not solver_produced_result:
        logging.debug("Skipping validation because solver run was unsuccessful.")
        # Keep is_valid as False, validation_error_info as empty

    validation_end = time.perf_counter_ns()
    validation_ms = (validation_end - validation_start) / 1e6
    total_ms = (time.perf_counter_ns() - all_start) / 1e6
    logging.info(f"Per-problem timing: validation_phase={validation_ms:.2f}ms")
    logging.info(f"Per-problem timing: total={total_ms:.2f}ms")

    # COMPREHENSIVE SOLVER TIMING DEBUG: Log all fields in solver_result before extraction
    solver_timing_fields = ["elapsed_ms", "min_time_ms", "mean_time_ms", "stdev_time_ms", "all_times_ms", "min_ns", "median_ns", "mean_ns", "max_ns", "values_ns", "num_runs_executed", "success", "error", "min_time_ms", "mean_ms", "median_ms", "stddev", "values", "runs"]
    solver_timing_debug = {field: solver_result.get(field) for field in solver_timing_fields}
    logging.info(f"SOLVER_TIMING_DEBUG: All timing fields in solver_result before extraction: {solver_timing_debug}")
    
    # Specifically check the elapsed_ms field that will become solver_time_ms
    extracted_solver_time_ms = solver_result.get('elapsed_ms', 0)
    logging.info(f"SOLVER_TIMING_DEBUG: Extracted solver_time_ms = {extracted_solver_time_ms} (type: {type(extracted_solver_time_ms)})")
    
    # --- Result Combination and Scoring --- #
    final_result = {
        **dataset_item, # Include original k, seed, etc.
        'success': solver_result.get('success', False),
        'solver_time_ms': extracted_solver_time_ms, # This is now the minimum time of the solver runs
        'solver_min_time_ms': solver_result.get('min_time_ms'), # Explicit minimum time
        'solver_mean_time_ms': solver_result.get('mean_time_ms'), # Mean time of solver runs
        'solver_stdev_time_ms': solver_result.get('stdev_time_ms'),
        'solver_all_times_ms': solver_result.get('all_times_ms'),

        # Add nanosecond-based metrics from solver_result for PYHELPER
        'min_ns': solver_result.get('min_ns'),
        'median_ns': solver_result.get('median_ns'),
        'mean_ns': solver_result.get('mean_ns'),
        'max_ns': solver_result.get('max_ns'),
        'values_ns': solver_result.get('values_ns'),
        'num_runs_executed': solver_result.get('num_runs_executed'),

        'oracle_time_ms': oracle_time_ms, # From the initial oracle run
        'is_valid': is_valid, # <<< USE THE RESULT FROM WARMUP VALIDATION LOGIC
        'speedup': calculate_input_speedup(solver_result.get('elapsed_ms', 0), oracle_time_ms, is_valid), # <<< USE is_valid here too
        'solver_output': solver_result.get('result'), # <<< RENAME/ENSURE solver output is here
        'problem': problem_data, # <<< ENSURE original problem data is here
        'first_warmup_output': solver_result.get('first_warmup_result'), # Keep the first warmup output for reference
        # Prioritize validation error info if validation failed, else use solver benchmark error info
        'error': validation_error_info.get('error', solver_result.get('error')),
        'error_type': validation_error_info.get('error_type', solver_result.get('error_type')),
        'traceback': validation_error_info.get('traceback', solver_result.get('traceback')),
        'code_context': validation_error_info.get('code_context', solver_result.get('code_context')),
        'stdout': solver_result.get('stdout'), # Include stdout/stderr from solver run
        'stderr': solver_result.get('stderr'),
    }
    # DO NOT POP 'problem' and 'solver_output'
    # final_result.pop('problem', None) 
    # final_result.pop('solution', None) # Solution was never added, but ensure no popping


    # Early stop on solver exceptions or other failures (excluding invalid solutions)
    if not final_result.get("success") and final_result.get("error_type") != "invalid_solution":
        logging.error(f"EVAL_MAIN: Solver exception on problem_id={dataset_item.get('k', 'N/A')}: {final_result.get('error')}. Stopping evaluation early.")
        final_result["stop_reason"] = final_result.get("error_type", "unknown_error")

    return final_result


def evaluate_code_on_dataset(
    task_obj: Any, # Should be Task
    dataset_iterable: Iterable[Dict[str, Any]],
    timeout_multiplier: float = 10.0,
    min_timeout_seconds: float = 10.0,
    max_timeout_seconds: float = 3600.0,
    default_target_solve_time_ms: float = 100.0,
    default_num_eval_runs: Optional[int] = None, # Changed to Optional[int] = None
    default_num_warmup_runs: Optional[int] = None, # Changed to Optional[int] = None
    problem_id_key: str = "id",
    problem_instance_key: str = "problem",
    baseline_times: Optional[Dict[str, float]] = None,
    data_subset: str = "train",  # 'train' or 'test' – determines which split is timed
    test_mode: bool = False,  # If True, limit dataset to 10 samples
) -> Union[List[Dict[str, Any]], Dict[str, Any]]: # MODIFIED Return Type
    """
    Evaluates a task's solve method on a dataset using the new clean architecture.
    """
    from AlgoTuner.utils.evaluator.evaluation_types import RunnerConfig
    from AlgoTuner.utils.evaluator.evaluation_orchestrator import EvaluationOrchestrator
    from AlgoTuner.utils.evaluator.legacy_adapter import LegacyAdapter
    
    logging.info(f"Using optimized evaluation pipeline (test_mode={test_mode})")
    
    # Convert dataset to list if needed
    dataset = list(dataset_iterable)
    
    # Check if we're in test mode and limit dataset size
    if test_mode:
        # Default to 10 samples for test mode unless specified otherwise
        max_samples = 10
        if len(dataset) > max_samples:
            logging.info(f"TEST MODE: Limiting dataset from {len(dataset)} to {max_samples} samples")
            dataset = dataset[:max_samples]
    else:
        logging.info(f"Not in test mode, evaluating all {len(dataset)} samples")
    
    # Get default runs from config if not specified
    from AlgoTuner.utils.timing_config import RUNS, WARMUPS
    num_runs = default_num_eval_runs if default_num_eval_runs is not None else RUNS
    warmup_runs = default_num_warmup_runs if default_num_warmup_runs is not None else WARMUPS
    
    # Create configuration
    config = RunnerConfig(
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        timeout_multiplier=timeout_multiplier,
        capture_output=False,
        validate_in_process=True,
        strip_solutions=True,
        use_isolated_execution=True,
    )
    
    # Create orchestrator
    orchestrator = EvaluationOrchestrator(config)
    
    # Get solver function
    solver_func = getattr(task_obj, 'solve', None)
    if not solver_func:
        raise ValueError("Task object must have a solve method")
    
    # Prepare dataset with proper structure
    prepared_dataset = []
    
    # Log baseline times status
    if baseline_times:
        logging.info(f"Baseline times available: {len(baseline_times)} entries")
        sample_keys = list(baseline_times.keys())[:5]
        logging.info(f"Sample baseline keys: {sample_keys}")
    else:
        logging.warning("No baseline times provided for dataset evaluation!")
    
    for i, item in enumerate(dataset):
        if isinstance(item, dict):
            # IMPORTANT: Always use index-based ID for baseline lookup
            # Baseline generation uses f"problem_{idx+1}" as keys, NOT dataset IDs
            item_id = f"problem_{i+1}"
            # Store original ID from dataset for logging/debugging only
            dataset_id = item.get(problem_id_key) or item.get("k")
            
            # Get baseline time using the ID
            baseline_time = None
            if baseline_times:
                baseline_time = baseline_times.get(item_id)
                
                # If no baseline time found, this is a critical error
                if baseline_time is None:
                    error_msg = (f"CRITICAL ERROR: No baseline time found for {item_id}. "
                               f"Dataset ID was '{dataset_id}'. "
                               f"Available baseline keys: {list(baseline_times.keys())[:10]}...")
                    logging.error(error_msg)
                    raise RuntimeError(error_msg)
            
            problem_data = {
                "problem": item.get(problem_instance_key, item),
                "id": item_id,
                "baseline_time_ms": baseline_time,
            }
            
            # Debug logging
            if i < 5:  # Log first 5 items
                logging.info(f"Problem {i+1}: item_id='{item_id}', baseline_time={baseline_time}")
            # Include other metadata
            for key, value in item.items():
                if key not in [problem_instance_key, problem_id_key]:
                    problem_data[key] = value
        else:
            # For non-dict items, use index-based ID
            item_id = f"problem_{i+1}"
            baseline_time = baseline_times.get(item_id) if baseline_times else None
            
            # Check for missing baseline time
            if baseline_times and baseline_time is None:
                error_msg = (f"CRITICAL ERROR: No baseline time found for problem {i+1} "
                           f"(item_id='{item_id}'). "
                           f"Available baseline keys: {list(baseline_times.keys())[:10]}...")
                logging.error(error_msg)
                raise RuntimeError(error_msg)
                
            problem_data = {
                "problem": item,
                "id": item_id,
                "baseline_time_ms": baseline_time
            }
        
        prepared_dataset.append(problem_data)
    
    # Run evaluation
    dataset_results = orchestrator.evaluate_dataset(
        task_instance=task_obj,
        dataset=prepared_dataset,
        solver_func=solver_func,
        task_name=task_obj.__class__.__name__,
    )
    
    # Convert to legacy format for compatibility
    adapter = LegacyAdapter()
    legacy_results = adapter.adapt_dataset_results(dataset_results)
    
    logging.info(
        f"Evaluation complete. Valid: {dataset_results.metrics.num_valid}/{dataset_results.metrics.num_evaluated}"
    )
    
    return legacy_results



def run_evaluation_on_input(task_instance, problem_input, timeout_ms=10000, command_source="eval_input"):
    """Run solver on a single problem_input string with validation and error handling."""
    logging.info(f"Starting evaluation on provided input... (Source: {command_source})")
    
    tm = TimingManager()

    try:
        # Directly use the problem_input, assuming it's already parsed correctly
        problem = problem_input
            
        # Reload code so latest edits take effect
        reload_start = time.perf_counter_ns()
        try:
            reload_all_llm_src(CODE_DIR)
            logging.info(f"Code reload for eval_input took {(time.perf_counter_ns() - reload_start)/1e6:.2f}ms")
        except Exception as reload_err:
            logging.warning(f"Failed to reload code before eval_input: {reload_err}")
        
        code_dir_str = os.environ.get("CODE_DIR", ".") 
        code_dir_path = Path(code_dir_str)
        is_valid, validation_error = validate_solver_setup(code_dir_path, command_source)
        if not is_valid:
            logging.error(f"Solver validation failed during eval_on_input (after reload): {validation_error}")
            if "elapsed_ms" not in validation_error:
                validation_error["elapsed_ms"] = 0
            return _return_with_message_writer(validation_error)

        # Log the problem input for debugging
        logging.debug(f"Problem input type: {type(problem)}")
        
        # Load a random problem from dataset for warmup
        warmup_problem = None
        try:
            data_dir = os.environ.get("DATA_DIR", None)
            if data_dir and hasattr(task_instance, 'task_name'):
                dataset_pattern = os.path.join(data_dir, f"{task_instance.task_name}_T{int(timeout_ms)}ms_n*_size*_train.jsonl")
                
                import glob
                matching_files = glob.glob(dataset_pattern)
                
                if matching_files:
                    dataset_path = matching_files[0]
                    logging.info(f"Loading random warmup problem from: {dataset_path}")
                    
                    problems = []
                    for idx, problem_data in enumerate(stream_jsonl(dataset_path)):
                        problems.append(problem_data)
                        if idx >= 10:
                            break
                    
                    if problems:
                        import random
                        random_problems = [p for p in problems if p != problem_input]
                        if random_problems:
                            warmup_problem = random.choice(random_problems)
                            logging.info(f"Selected random warmup problem (different from input)")
                        else:
                            warmup_problem = problems[0] if problems else None
                            logging.warning("All dataset problems match input, using first problem for warmup")
                else:
                    logging.warning(f"No dataset file found matching: {dataset_pattern}")
        except Exception as e:
            logging.warning(f"Failed to load random warmup problem: {e}")
            
        # Use unified solver benchmark harness for eval_input
        # Single measurement run with 3 warmups, capturing stdout
        result = run_solver_evaluation(
            problem=problem,
            task_instance=task_instance,
            oracle_time_ms=timeout_ms,
            capture_output=True,
            needs_casting=False,
            num_runs=1,
            warmup_runs=3,
            warmup_problem=warmup_problem
        )
        
        # Logging the unified result structure and timing information
        logging.debug(f"DEBUG run_evaluation_on_input: result keys: {list(result.keys())}")
        
        # Check for timing information
        elapsed_ms = result.get("elapsed_ms")
        if elapsed_ms is not None:
            logging.debug(f"DEBUG run_evaluation_on_input: elapsed_ms={elapsed_ms}")
        else:
            # Look for timing in output_logs
            if "output_logs" in result and isinstance(result["output_logs"], dict):
                output_logs = result["output_logs"]
                if "elapsed_ms" in output_logs:
                    elapsed_ms = output_logs["elapsed_ms"]
                    logging.debug(f"DEBUG run_evaluation_on_input: Found elapsed_ms={elapsed_ms} in output_logs")
                    result["elapsed_ms"] = elapsed_ms
            
            # Look for timing in first_run_result
            if elapsed_ms is None and "first_run_result" in result:
                first_run = result["first_run_result"]
                if isinstance(first_run, dict) and "elapsed_ms" in first_run:
                    elapsed_ms = first_run["elapsed_ms"]
                    logging.debug(f"DEBUG run_evaluation_on_input: Found elapsed_ms={elapsed_ms} in first_run_result")
                    result["elapsed_ms"] = elapsed_ms
        
        # Check if there's an error in execution
        if not result.get("success", False):
            # Handle invalid_solution specially
            if result.get("error_type") == "invalid_solution":
                success_response = {
                    "success": False,
                    "is_valid": False,
                    "result": result.get("result"),
                    "elapsed_ms": result.get("elapsed_ms"),
                    "invalid_solution": True,
                    "error_type": "invalid_solution",  # signal formatter to show stdout
                    "error": "Solution is invalid",
                    "command_source": command_source,
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", "")
                }
                return _return_with_message_writer(success_response)

            # Generic error formatting
            formatted_error = {
                "success": False,
                "is_valid": False,
                "error": result.get("error"),
                "error_type": result.get("error_type"),
                "traceback": result.get("traceback"),
                "code_context": result.get("code_context"),
                "elapsed_ms": result.get("elapsed_ms"),
                "command_source": command_source,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            }
            # Include output_logs and first_run_result for error details
            if "output_logs" in result:
                formatted_error["output_logs"] = result["output_logs"]
            if "first_run_result" in result:
                formatted_error["first_run_result"] = result["first_run_result"]

            return _return_with_message_writer(formatted_error)
        
        # --- Success Path --- 
        initial_solver_result = result # Rename for clarity
        
        # Get relevant info from the initial run
        raw_solver_output = initial_solver_result.get("result")
        elapsed_ms = initial_solver_result.get("elapsed_ms")
        initial_stdout = initial_solver_result.get("stdout", "")
        initial_stderr = initial_solver_result.get("stderr", "")

        # Check if validation was already completed in a previous stage
        if (isinstance(raw_solver_output, dict) and 
            raw_solver_output.get("stripped_after_validation") and
            initial_solver_result.get("validation_result")):
            logging.info("Result was already validated and stripped - using pre-computed validation result")
            validation_result = initial_solver_result.get("validation_result")
            final_result_value = raw_solver_output  # Keep the stripped marker for consistency
        else:
            # Prepare result for casting by default
            value_to_process = raw_solver_output
            # Extract only the solution portion before casting
            if hasattr(task_instance, "extract_solution"):
                try:
                    value_to_process = task_instance.extract_solution(raw_solver_output)
                    logging.debug("Extracted solution portion for casting in eval_input.")
                except Exception:
                    logging.debug("extract_solution failed; using raw solver output for casting.")
            # Skip output casting for eval_input; use raw or extracted result value
            final_result_value = value_to_process
            logging.debug("Skipping output casting for eval_input; using raw result.")
            
            # Perform validation on the extracted result
            validation_result = _validate_solution(task_instance, problem, final_result_value)
        
        # Process validation result
        try:
            if not validation_result.get("success", False):
                logging.warning(f"Validation failed. Type: {validation_result.get('error_type')}, Error: {validation_result.get('error')}")
                # Merge solver result first, then validation details (excluding stdout/stderr to preserve solver output)
                final_evaluation_result = {
                    "is_valid": False,
                    **initial_solver_result,
                    **{k: v for k, v in validation_result.items() if k not in ("stdout", "stderr")},
                    "success": False,
                    "result": final_result_value,
                    "raw_result": raw_solver_output,
                    "elapsed_ms": elapsed_ms,
                    "oracle_time_ms": initial_solver_result.get("oracle_time_ms"),
                }
                # Override stdout/stderr to preserve solver output
                final_evaluation_result["stdout"] = initial_solver_result.get("stdout", "")
                final_evaluation_result["stderr"] = initial_solver_result.get("stderr", "")
                final_evaluation_result["error_type"] = validation_result.get("error_type", "unknown_validation_failure")
                if "problem" not in final_evaluation_result:  # Ensure problem context
                    final_evaluation_result["problem"] = problem
                return _return_with_message_writer(final_evaluation_result)
            else:
                # Validation passed, merge any stdout/stderr from validation
                initial_stdout += validation_result.get("stdout", "")
                initial_stderr += validation_result.get("stderr", "")

        except Exception as val_err:
            # Catch unexpected errors during validation call itself
            logging.error(f"Unexpected error during validation for eval_input: {val_err}", exc_info=True)
            tb_str = traceback.format_exc()
            error_result = create_standard_error_result(
                exception=val_err,
                traceback_str=tb_str,
                error_type_override="validation_error", 
                default_error_msg="Unexpected validation error during eval_input",
                stdout=initial_stdout,
                stderr=initial_stderr
            )
            error_result["command_source"] = command_source
            return _return_with_message_writer(error_result)

        # If we reach here, execution, casting, and validation succeeded
        final_success_result = {
            "is_valid": True,
            "success": True,
            "result": final_result_value,
            "elapsed_ms": elapsed_ms,
            "command_source": command_source,
            "stdout": initial_stdout,
            "stderr": initial_stderr
        }

        # Log the final result keys and timing
        if logging.getLogger().level <= logging.DEBUG:
            logging.debug(f"DEBUG run_evaluation_on_input: final success_result keys: {list(final_success_result.keys())}")
            logging.debug(f"DEBUG run_evaluation_on_input: final elapsed_ms={final_success_result['elapsed_ms']}")

        return _return_with_message_writer(final_success_result)
        
    except ValidationException as ve:
        # ... (handle validation exception - might already return standard dict from runner?) ...
        # If run_evaluation now returns standard dict, this might be simplified
        # Let's assume for now run_evaluation returns the standard dict on failure
        # So we just need to format it.
        # Note: This block might need adjustment depending on how ValidationException is used
        # For now, treat it like any other exception caught here.
        tb_str = traceback.format_exc()
        logging.warning(f"Validation exception during eval_on_input: {ve}")
        error_result = create_standard_error_result(
            exception=ve,
            traceback_str=tb_str, # tb_str might not be super useful here if ve was raised deliberately
            error_type_override="validation_error",
            # Pass problem_input? Might be too large. Pass shape/type?
            default_error_msg=str(ve) 
        )
        error_result["command_source"] = command_source
        error_result["evaluation_stopped"] = True 
        return _return_with_message_writer(error_result)
        
    except Exception as e:
        # --- Refactored Unexpected Error Handling --- 
        tb_str = traceback.format_exc()
        logging.error(f"Unexpected error during run_evaluation_on_input: {e}", exc_info=False)
        logging.debug(f"Raw Traceback for unexpected error:\n{tb_str}")
        error_result = create_standard_error_result(
            exception=e,
            traceback_str=tb_str,
            error_type_override="runtime_error", # Generic type
            default_error_msg="Error during single input evaluation"
        )
        error_result["command_source"] = command_source
        error_result["evaluation_stopped"] = True
        return _return_with_message_writer(error_result)
        # --- End Refactored Handling --- 


def run_oracle_on_input(task_instance, problem_input, timeout_ms=10000, process_pool_manager=None):
    """
    Run the oracle (solution method) on a specific input.
    
    Args:
        task_instance: Instance of the task
        problem_input: Input to evaluate (assumed to be correctly parsed by the caller)
        timeout_ms: Timeout in milliseconds (Note: timeout is not directly handled by execute_and_capture_errors)
        process_pool_manager: Optional process pool manager for parallel execution
        
    Returns:
        Oracle evaluation result dictionary (includes success, result or error details)
    """
    try:
        # Check if task instance has a solve method
        if not hasattr(task_instance, "solve"):
            return {
                "success": False,
                "error": "Task instance does not have a solve method.",
                "error_type": "method_error"
            }
            
        # Directly use the problem_input, assuming it's already parsed correctly
        problem = problem_input

        # Log the problem input for debugging
        logging.debug(f"Oracle Problem input type: {type(problem)}")

        # Call run_oracle_evaluation without runner/benchmark_name
        # Note: run_oracle_evaluation itself will need updating to remove runner/benchmark_name params
        # (caller signature update happens in next step)
        result = run_oracle_evaluation(
            problem=problem_input,  # Pass the input here
            task_instance=task_instance,
            oracle_time_ms=timeout_ms, # Rename from timeout_ms in original signature
            capture_output=True, # Always capture for oracle command
            needs_casting=True   # Assume casting needed
        )
        
        # Format and return the result using the standard writer
        return _return_with_message_writer(result)
        
    except Exception as e:
        # --- Refactored Unexpected Error Handling --- 
        # Handle unexpected errors during oracle setup/call
        tb_str = traceback.format_exc()
        logging.error(f"Unexpected error during run_oracle_on_input: {e}", exc_info=False)
        logging.debug(f"Raw Traceback for oracle error:\n{tb_str}")
        error_result = create_standard_error_result(
            exception=e,
            traceback_str=tb_str,
            error_type_override="oracle_setup_error", 
            default_error_msg="Error during oracle execution"
        )
        return _return_with_message_writer(error_result)
        # --- End Refactored Handling --- 

# Shared helper to run solver on a list of dataset records under the SOLVE_LOOP phase
def evaluate_problems(task_instance, records_iterable: Iterable[Dict], num_runs: int, warmup_runs: int) -> Tuple[List[Dict], Dict, str, int, str, Optional[Dict]]:
    """
    Run solve() on each record from an iterable under SOLVE_LOOP timing and return
    per-problem results, aggregate metrics, the timing report, the count of evaluated problems,
    a stop reason ('completed', 'timeout_error', etc.), and the record that caused the stop (if any).

    Stops immediately if any problem evaluation results in a critical error type.
    """
    timing = TimingManager()
    per_problem = []
    evaluated_count = 0
    stop_reason = 'completed'  # Assume completion unless stopped early
    stopping_record = None     # Store the record that caused the stop

    # Wrap per-problem evaluation in SOLVE_LOOP phase
    with timing.phase(Phase.SOLVE_LOOP):
        for rec in records_iterable:
            # --- Ensure fresh task module & instance per problem to avoid caching ---
            try:
                module_name = task_instance.__class__.__module__
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                    logging.debug(f"evaluate_baseline_dataset: Reloaded module '{module_name}' to clear caches.")
                # Re-instantiate a fresh task object so that any per-instance state is clean
                task_instance = load_task(task_instance.task_name)
            except Exception as reload_err:
                logging.warning(f"evaluate_baseline_dataset: Failed to reload task module '{module_name}': {reload_err}")
            
            # --- FIX: Use consistent index-based key generation --- 
            problem_id = rec.get("id") or rec.get("problem_id") or f"problem_{i+1}"
            # ---------------------------------------------------

            # --- Purge any modules previously imported from this *task* directory ---
            try:
                task_dir = Path(task_instance.get_task_directory()).resolve()
                for _mod_name, _mod in list(sys.modules.items()):
                    _mod_file = getattr(_mod, "__file__", None)
                    if not _mod_file:
                        continue
                    try:
                        _mod_path = Path(_mod_file).resolve()
                        if str(_mod_path).startswith(str(task_dir)):
                            del sys.modules[_mod_name]
                    except Exception:
                        # Ignore issues with non-standard module paths
                        continue
            except Exception as purge_err:
                logging.debug(f"evaluate_baseline_dataset: Error purging modules for task_dir '{task_dir}': {purge_err}")

            res = _evaluate_single_problem(
                dataset_item=rec,
                task_instance=task_instance,
                provided_oracle_time_ms=None,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
            )
            per_problem.append(res)
            evaluated_count += 1

            # Check for critical error types to stop early
            current_error_type = res.get('error_type')
            if current_error_type in CRITICAL_ERROR_TYPES:
                stop_reason = current_error_type # Use the critical error type as the reason
                stopping_record = rec # Store the original record containing the input
                logging.warning(f"Critical evaluation error for problem k={rec.get('k', 'N/A')}. Error type: {stop_reason}. Stopping evaluation early.")
                # Log the specific error message if available
                if res.get('error'):
                     logging.warning(f"Error details: {res.get('error')}")
                break # Exit the loop immediately

    # Calculate aggregate metrics only if evaluation completed normally OR stopped due to non-critical error
    aggregate = {}
    # We calculate aggregates even if stopped, but the final report structure depends on stop_reason later
    aggregate = _calculate_aggregate_metrics(per_problem) 
    # Return the stop reason and stopping record along with other results
    return per_problem, aggregate, timing.report(), evaluated_count, stop_reason, stopping_record

# --- NEW HELPER FUNCTION ---
def _convert_numpy_to_list(obj):
    """Recursively converts numpy arrays within a nested structure to lists."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [_convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert numpy array to list
    # Handle common numpy scalar types (often returned by np.mean, np.std, etc.)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.void): # Handle void types if necessary (e.g., structured arrays)
        # Decide how to handle structured arrays, maybe convert to dict?
        # For now, converting to string representation as a safe default.
        logging.warning(f"Converting numpy void type to string representation: {obj}")
        return repr(obj) 
    return obj
# --- END NEW HELPER FUNCTION ---



# --- NEW: Baseline dataset timing helper ---
def evaluate_baseline_dataset(
    task_obj: Any,
    dataset_iterable: Iterable[Dict[str, Any]],
    num_runs: Optional[int] = None,
    warmup_runs: Optional[int] = None,
    timeout_seconds: float = 120.0,
    output_file: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    test_mode: bool = False,
    max_samples: Optional[int] = None
) -> str:
    """
    Run baseline timing (AGENT_MODE=0) over the dataset once, and dump a JSON map
    of {problem_id: min_time_ms} to disk in TEMP. Returns the JSON file path.

    Uses warmups + runs and fixed timeout per instance.
    """
    # Use the *current* AGENT_MODE.  We no longer override it to "0" so
    # baseline timing is collected in the same environment (process-isolation
    # strategy, PySAT patches, etc.) as the subsequent solver evaluation.
    original_agent_mode = os.environ.get("AGENT_MODE")  # keep for log/restore
    logging.info(
        "BASELINE_EVAL: Running with existing AGENT_MODE=%s (no override)",
        original_agent_mode,
    )
    try:
        logging.info(
            "Starting baseline dataset evaluation for task: %s...",
            task_obj.task_name if hasattr(task_obj, "task_name") else "UnknownTask",
        )
        
        # Guarantee task_obj has a task_name attribute for downstream code (load_dataset, logging)
        if not hasattr(task_obj, "task_name"):
            setattr(task_obj, "task_name", task_obj.__class__.__name__)
        
        return _evaluate_baseline_dataset_impl(task_obj, dataset_iterable, num_runs, warmup_runs, timeout_seconds, output_file, jsonl_path, test_mode, max_samples)
    finally:
        # No changes were made to AGENT_MODE, but keep the restore logic for
        # safety in case future modifications temporarily tweak it inside the
        # try-block.
        if original_agent_mode is None:
            os.environ.pop("AGENT_MODE", None)
        else:
            os.environ["AGENT_MODE"] = original_agent_mode
        logging.info(
            "BASELINE_EVAL: Ensured AGENT_MODE reset to original value (%s)",
            os.environ.get("AGENT_MODE", "None"),
        )


def _safe_compare_data(data1: Any, data2: Any) -> bool:
    """
    Safely compare two data structures that might contain numpy arrays.
    Returns True if they are equal, False otherwise.
    """
    if type(data1) != type(data2):
        return False
    
    if isinstance(data1, np.ndarray):
        return np.array_equal(data1, data2)
    elif isinstance(data1, dict):
        if set(data1.keys()) != set(data2.keys()):
            return False
        return all(_safe_compare_data(data1[k], data2[k]) for k in data1.keys())
    elif isinstance(data1, (list, tuple)):
        if len(data1) != len(data2):
            return False
        return all(_safe_compare_data(a, b) for a, b in zip(data1, data2))
    else:
        # For other types, use regular comparison
        try:
            return data1 == data2
        except ValueError:
            # If comparison still fails, they're not equal
            return False


def _evaluate_baseline_dataset_impl(
    task_obj: Any,
    dataset_iterable: Iterable[Dict[str, Any]],
    num_runs: Optional[int] = None,
    warmup_runs: Optional[int] = None,
    timeout_seconds: float = 120.0,
    output_file: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    test_mode: bool = False,
    max_samples: Optional[int] = None
) -> str:
    """
    Internal implementation of baseline dataset evaluation.
    """

    # Use config values for consistency
    if num_runs is None:
        num_runs = DATASET_RUNS  # From config (benchmark.runs)
    if warmup_runs is None:
        warmup_runs = DATASET_WARMUPS  # From config (benchmark.warmups)
    
    # Use same approach as AGENT_MODE=1: each isolated process does 1 warmup + 1 timed
    actual_num_runs = num_runs  # Use config value
    actual_warmup_runs = warmup_runs  # Not used directly - isolated benchmark does 1 warmup + 1 timed internally
    baseline_times: Dict[str, float] = {}
    
    logging.info(f"BASELINE_EVAL: Using isolated benchmark with {actual_num_runs} runs (1 warmup + 1 timed per process)")
    
    # Always use isolated benchmark approach for consistent performance with dataset generation
    logging.info(f"BASELINE_EVAL: Using isolated benchmark subprocess evaluation")
    
    if jsonl_path:
        # --- ONE-TIME pass to count records *and* capture byte offsets --------
        problem_ids: list[str] = []
        line_offsets: list[int] = []

        with open(jsonl_path, 'rb') as f:
            pos = f.tell()
            while True:
                line = f.readline()
                if not line:
                    break
                if line.strip():  # skip blank lines
                    line_offsets.append(pos)
                    problem_ids.append(f"problem_{len(problem_ids)+1}")
                pos = f.tell()

        problem_count = len(problem_ids)

        # Keep offsets around so we can hand them to workers (constant-time access).
        # Store in closure for later use.
        jsonl_line_offsets = line_offsets  # type: ignore[var-annotated]

        dataset_records = None  # type: ignore
        
        # Check if we're in test mode and should limit samples
        if test_mode and max_samples is None:
            max_samples = 10  # Default for test mode
        
        if max_samples and problem_count > max_samples:
            logging.info(f"BASELINE_EVAL: Limiting baseline evaluation from {problem_count} to {max_samples} samples (test_mode={test_mode})")
            problem_ids = problem_ids[:max_samples]
            line_offsets = line_offsets[:max_samples]
            problem_count = len(problem_ids)
    else:
        # Fallback for in-memory iterables (small datasets).
        dataset_records = list(dataset_iterable)
        problem_count = len(dataset_records)
        
        # Check if we're in test mode and should limit samples
        if test_mode and max_samples is None:
            max_samples = 10  # Default for test mode
        
        if max_samples and problem_count > max_samples:
            logging.info(f"BASELINE_EVAL: Limiting baseline evaluation from {problem_count} to {max_samples} samples (test_mode={test_mode})")
            dataset_records = dataset_records[:max_samples]
            problem_count = len(dataset_records)
        
        problem_ids = [
            f"problem_{idx+1}"  # Always use index-based IDs for consistency
            for idx, rec in enumerate(dataset_records)
        ]
        
        # Log what IDs we're using
        logging.info(f"BASELINE_EVAL: Using index-based problem IDs. First 5: {problem_ids[:5]}")
        if dataset_records and isinstance(dataset_records[0], dict) and ("id" in dataset_records[0] or "problem_id" in dataset_records[0]):
            dataset_ids = [rec.get("id") or rec.get("problem_id") for rec in dataset_records[:5]]
            logging.info(f"BASELINE_EVAL: Note - dataset has native IDs that we're NOT using: {dataset_ids}")

    # Sanitize task name for filesystem/module safety.
    import re
    safe_task_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(getattr(task_obj, "task_name", "task")))
 
    # Get solver directory for isolated benchmark
    dataset_dir = task_obj.data_dir or task_obj.get_task_directory()
    solver_dir = task_obj.get_task_directory()

    # Execute each baseline solve inside the worker by index
    import time
    overall_start_time = time.time()
    
    # Always evaluate all problems - no sampling
    sampled_indices = list(range(len(problem_ids)))
    logging.info(f"BASELINE_EVAL: Evaluating all {len(problem_ids)} problems")
    
    max_overall_time_s = len(sampled_indices) * 300.0  # Max 5 minutes per problem
    
    for eval_idx, idx in enumerate(sampled_indices):
        problem_id = problem_ids[idx]
        # Check overall timeout
        elapsed_overall = time.time() - overall_start_time
        if elapsed_overall > max_overall_time_s:
            logging.debug(f"BASELINE_DEBUG: Overall timeout exceeded ({elapsed_overall:.1f}s > {max_overall_time_s:.1f}s). Stopping baseline evaluation at problem {eval_idx+1}/{len(sampled_indices)}")
            break
            
        # Memory-efficient problem retrieval: fetch problems one at a time to minimize peak memory
        if jsonl_path:
            from AlgoTuner.utils.streaming_json import stream_jsonl

            def _fetch_record_streaming(ix: int):
                """Fetch a single record by index from JSONL stream (memory-efficient)."""
                import orjson
                import functools
                from AlgoTuner.utils.serialization import dataset_decoder
                import os
                
                actual_base_dir = os.path.dirname(jsonl_path)
                object_hook_for_load = functools.partial(dataset_decoder, base_dir=actual_base_dir)
                
                with open(jsonl_path, 'r') as f:
                    current_index = 0
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if current_index == ix:
                            try:
                                raw_record = orjson.loads(line)
                                processed_record = object_hook_for_load(raw_record)
                                return processed_record.get("problem", processed_record)
                            except orjson.JSONDecodeError as e:
                                raise RuntimeError(f"JSON Decode Error in {jsonl_path}, line {current_index}: {e}")
                        current_index += 1
                return None

            # For large datasets, use a memory-efficient approach:
            # Pass the fetch function to isolated_benchmark instead of pre-loading both problems
            warmup_idx = (idx - 1) % problem_count
            # Problem instance for this index (needed by baseline timing helper)
            problem_data = _fetch_record_streaming(idx)
            warmup_data = _fetch_record_streaming(warmup_idx)
            
            # Debug logging to verify problems are different
            logging.debug(f"BASELINE_DEBUG: idx={idx}, warmup_idx={warmup_idx}")
            logging.debug(f"BASELINE_DEBUG: problem_data type: {type(problem_data)}, keys: {list(problem_data.keys()) if isinstance(problem_data, dict) else 'not a dict'}")
            logging.debug(f"BASELINE_DEBUG: warmup_data type: {type(warmup_data)}, keys: {list(warmup_data.keys()) if isinstance(warmup_data, dict) else 'not a dict'}")
            if isinstance(problem_data, dict) and isinstance(warmup_data, dict):
                for key in problem_data.keys():
                    if key in warmup_data:
                        if not _safe_compare_data(problem_data[key], warmup_data[key]):
                            logging.debug(f"BASELINE_DEBUG: Key '{key}' differs: problem={problem_data[key]} vs warmup={warmup_data[key]}")
                        else:
                            logging.debug(f"BASELINE_DEBUG: Key '{key}' is same: {problem_data[key]}")
            if _safe_compare_data(problem_data, warmup_data):
                logging.error(f"BASELINE_DEBUG: WARNING - problem_data and warmup_data are IDENTICAL for idx={idx}!")
                logging.error(f"BASELINE_DEBUG: problem_data = {problem_data}")
                logging.error(f"BASELINE_DEBUG: warmup_data = {warmup_data}")
            else:
                logging.debug(f"BASELINE_DEBUG: Good - problem_data and warmup_data are different")
            
            # Create lightweight fetch functions that will be called inside the worker
            problem_fetch_info = {"type": "jsonl_seek", "path": jsonl_path, "offset": jsonl_line_offsets[idx]}
            warmup_fetch_info = {"type": "jsonl_seek", "path": jsonl_path, "offset": jsonl_line_offsets[warmup_idx]}
            
        else:
            # For small in-memory datasets, use direct access
            problem_instance = dataset_records[idx].get("problem", dataset_records[idx])
            warmup_idx = (idx - 1) % problem_count
            warmup_problem_instance = dataset_records[warmup_idx].get(
                "problem", dataset_records[warmup_idx]
            )
            # Direct-access path – we already hold the instance
            problem_data = problem_instance
            warmup_data = warmup_problem_instance
            
            problem_fetch_info = {"type": "direct", "data": problem_instance}
            warmup_fetch_info = {"type": "direct", "data": warmup_problem_instance}
        
        # Use fixed baseline timeout from config for consistency with solver evaluation
        # This ensures fair comparison between baseline and solver timings
        baseline_timeout_ms = _bench_cfg.get("baseline_timeout", 60000)  # Default 60s if not in config
        baseline_timeout_s = baseline_timeout_ms / 1000.0
        logging.info(f"Running baseline evaluation for problem {idx+1}/{len(sampled_indices)} (id: {problem_id}) with timeout={baseline_timeout_s:.1f}s")
        
        # Add more detailed logging before the call
        logging.debug(f"BASELINE_DEBUG: About to call baseline evaluation for problem {problem_id}")
        
        problem_start_time = time.time()
        try:
            # Use isolated benchmark timing for clean measurements
            logging.debug(f"BASELINE_DEBUG: Running isolated baseline timing for problem {problem_id} ({eval_idx+1}/{len(sampled_indices)})")
            logging.debug(f"BASELINE_DEBUG: Using solver_dir='{solver_dir}' for isolated benchmark (dataset_dir='{dataset_dir}')")
            
            # Use standard evaluation with AGENT_MODE=0 (baseline mode)
            # Always use isolated benchmark for proper timing isolation
            logging.debug(f"BASELINE_DEBUG: Using isolated benchmark for baseline evaluation")
            logging.debug(f"BASELINE_DEBUG: Current process daemon status: {mp.current_process().daemon}")
            logging.debug(f"BASELINE_DEBUG: Process name: {mp.current_process().name}")
            logging.debug(f"BASELINE_DEBUG: About to call run_isolated_benchmark_with_fetch")
            logging.debug(f"BASELINE_DEBUG: task_name='{safe_task_name}', code_dir='{solver_dir}'")
            logging.debug(f"BASELINE_DEBUG: AGENT_MODE={os.environ.get('AGENT_MODE', 'NOT_SET')}")
            
            # Use retry mechanism for baseline evaluation since it should never fail
            result = _evaluate_baseline_with_retry(
                problem_id=problem_id,
                problem_fetch_info=problem_fetch_info,
                warmup_fetch_info=warmup_fetch_info,
                task_obj=task_obj,
                num_runs=actual_num_runs,
                warmup_runs=actual_warmup_runs,
                timeout_seconds=baseline_timeout_s,
                max_retries=3
            )
            
            problem_elapsed = time.time() - problem_start_time
            logging.debug(f"BASELINE_DEBUG: Baseline evaluation completed successfully for problem {problem_id} in {problem_elapsed:.1f}s")
            logging.debug(f"BASELINE_DEBUG: Raw result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            logging.debug(f"BASELINE_DEBUG: Result success: {result.get('success')}")
            logging.debug(f"BASELINE_DEBUG: Result min_time_ms: {result.get('min_time_ms')}")
            logging.debug(f"BASELINE_DEBUG: Result elapsed_ms: {result.get('elapsed_ms')}")
            logging.debug(f"BASELINE_DEBUG: Result min_time_ms: {result.get('min_time_ms')}")
            logging.debug(f"BASELINE_DEBUG: Result mean_ms: {result.get('mean_ms')}")
            if hasattr(result, '__dict__'):
                logging.debug(f"BASELINE_DEBUG: Result dict: {result.__dict__}")
            elif isinstance(result, dict):
                logging.debug(f"BASELINE_DEBUG: Result dict size: {len(result)}, sample keys: {list(result.keys())[:10]}")
        except Exception as e:
            problem_elapsed = time.time() - problem_start_time
            logging.debug(f"BASELINE_DEBUG: Baseline evaluation failed for problem {problem_id} after {problem_elapsed:.1f}s: {e}")
            result = {
                "success": False,
                "error": f"Baseline evaluation failed: {e}",
                "min_time_ms": None,
                "elapsed_ms": 0.0
            }
        
        # Extract timing directly from result (no subprocess wrapping)
        t_ms = result.get("min_time_ms") if result.get("min_time_ms") is not None else result.get("elapsed_ms", 0.0)
        logging.debug(f"BASELINE_DEBUG: Extracted t_ms: {t_ms}")
        
        # BASELINE BUG FIX: Don't store 0.0 times as valid baseline measurements!
        if result.get("success") and t_ms is not None and t_ms > 0.0:
            baseline_times[problem_id] = t_ms
            logging.info(
                "BASELINE_TIME_DEBUG: problem_id=%s min_time_ms=%.6f (stored)",
                problem_id,
                t_ms,
            )
        else:
            # Store None instead of 0.0 to indicate that baseline measurement failed
            baseline_times[problem_id] = None
            success_status = result.get("success", False)
            error_msg = result.get("error", "Unknown error")
            logging.error(f"BASELINE BUG: Problem {problem_id} baseline measurement FAILED! Success: {success_status}, t_ms: {t_ms}, Error: {error_msg}")
            logging.error(f"BASELINE BUG: This will cause speedup calculation to fail with baseline_time_ms=0.0 or None")
            
        # Force garbage collection every 5 problems to prevent memory buildup
        if (eval_idx + 1) % 5 == 0:
            import gc
            gc.collect()
            logging.debug(f"BASELINE_DEBUG: Forced GC after problem {eval_idx+1}/{len(sampled_indices)}")
    # Determine output path in TEMP
    tmpdir = os.environ.get("TEMP_DIR_STORAGE") or os.environ.get("TEMP") or tempfile.gettempdir()
    if output_file is None:
        import uuid
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars for brevity
        file_name = f"{getattr(task_obj, 'task_name', 'baseline')}_times_{unique_id}.json"
        output_file = os.path.join(tmpdir, file_name)
    # Write JSON
    with open(output_file, "w") as f:
        json.dump(baseline_times, f)
    
    # Log summary of baseline evaluation results
    total_problems = len(baseline_times)
    successful_times = [t for t in baseline_times.values() if t is not None and t > 0.0]
    failed_times = [t for t in baseline_times.values() if t is None or t <= 0.0]
    
    logging.info(f"BASELINE_EVAL_SUMMARY: Total problems: {total_problems}")
    logging.info(f"BASELINE_EVAL_SUMMARY: Successful measurements: {len(successful_times)}")
    logging.info(f"BASELINE_EVAL_SUMMARY: Failed measurements: {len(failed_times)}")
    if successful_times:
        logging.info(f"BASELINE_EVAL_SUMMARY: Min successful time: {min(successful_times):.2f}ms")
        logging.info(f"BASELINE_EVAL_SUMMARY: Max successful time: {max(successful_times):.2f}ms")
    
    # No pool cleanup needed for isolated benchmark approach
    logging.info("BASELINE_EVAL: Isolated benchmark evaluation - no pool cleanup needed")
    
    logging.info(f"Baseline timings dumped to {output_file}")
    return output_file






def _eval_worker_target(task_name, data_dir, timed_problem_instance, warmup_problem_instance, num_runs, warmup_runs, timeout_seconds, problem_metadata):
    """
    Worker function to evaluate a single problem instance.
    This function is executed in a separate process.
    """
    # Use a variable to hold the final result dictionary
    result = {}
    
    # Get worker PID for logging
    worker_pid = os.getpid()

    # Create a **bounded** stream to capture (at most 100 kB of) stdout/stderr
    # from the solver.  This prevents pathological `print` statements inside
    # user code from accumulating gigabytes of strings in memory – something
    # that was triggering OOM-kills when the worker later pickled the result.

    class _BoundedStdCapture:
        """File-like object that stores only the first `MAX_CHARS` characters."""

        MAX_CHARS = 100_000  # 100 kB

        def __init__(self):
            self._buf = []  # type: list[str]
            self._count = 0
            self._truncated = False

        # The multiprocessing redirector calls .write(str) for each chunk.
        def write(self, s: str):  # noqa: D401
            if not s:
                return 0
            if self._count >= self.MAX_CHARS:
                # Already full – silently drop further output.
                self._truncated = True
                return len(s)

            remaining = self.MAX_CHARS - self._count
            take = min(len(s), remaining)
            self._buf.append(s[:take])
            self._count += take

            if take < len(s):
                # Hit the cap on this write – mark as truncated.
                self._truncated = True

            return len(s)

        def flush(self):
            # Nothing to flush – we keep everything in memory.
            pass

        def getvalue(self) -> str:  # noqa: D401
            out = "".join(self._buf)
            if self._truncated:
                out += "\n...[output truncated]...\n"
            return out

    f_out = _BoundedStdCapture()

    # Heart-beat logging thread (optional)
    if ENABLE_HEARTBEAT:
        stop_heartbeat = threading.Event()

        def heartbeat_logger():
            """Logs memory usage and process status periodically."""
            pid = os.getpid()
            try:
                p = psutil.Process(pid)
            except psutil.NoSuchProcess:
                return

            start_time = time.time()
            while not stop_heartbeat.is_set():
                try:
                    mem_info = p.memory_info()
                    cpu_times = p.cpu_times()
                    logging.info(
                        f"EVAL_WORKER_HEARTBEAT (PID: {pid}): Uptime="
                        f"{time.time() - start_time:.1f}s RSS={mem_info.rss / (1024**2):.1f}MB "
                        f"CPU={cpu_times.user + cpu_times.system:.1f}s"
                    )
                except Exception:
                    break
                stop_heartbeat.wait(15.0)

        heartbeat_thread = threading.Thread(target=heartbeat_logger, daemon=True)
        heartbeat_thread.start()
    else:
        stop_heartbeat = None
        heartbeat_thread = None

    try:
        # Redirect stdout to capture any prints from the solver
        with redirect_stdout(f_out), redirect_stderr(f_out):
            # Locate and load the solver module dynamically
            # This ensures the worker uses the latest code from CODE_DIR
            try:
                # Use the actual CODE_DIR environment variable, not the data directory
                base_code_dir = os.getenv("CODE_DIR")
                # For isolated benchmark, use the task-specific directory if it exists
                task_code_dir = os.path.join(base_code_dir, "AlgoTuneTasks", task_name)
                if os.path.isdir(task_code_dir):
                    actual_code_dir = task_code_dir
                else:
                    actual_code_dir = base_code_dir
                logging.info(f"EVAL_WORKER (PID: {worker_pid}): Locating solver for task '{task_name}' using CODE_DIR='{actual_code_dir}'")
                solver_file_path = locate_solver_file(task_name=task_name, code_dir=actual_code_dir)
                logging.info(f"EVAL_WORKER (PID: {worker_pid}): Loading solver module from '{solver_file_path}'")
                # ------------------------------------------------------------------
                # Cache-aware solver-module loading
                # ------------------------------------------------------------------
                from pathlib import Path
                cache_key = str(Path(solver_file_path).resolve())
                global _SOLVER_MODULE_CACHE
                if "_SOLVER_MODULE_CACHE" not in globals():
                    _SOLVER_MODULE_CACHE = {}

                solver_module = _SOLVER_MODULE_CACHE.get(cache_key)
                if solver_module is not None:
                    import importlib
                    logging.info(
                        f"EVAL_WORKER (PID: {worker_pid}): Reloading cached solver module '{cache_key}'"
                    )
                    try:
                        solver_module = importlib.reload(solver_module)
                    except Exception as reload_err:
                        logging.warning(
                            f"EVAL_WORKER (PID: {worker_pid}): Reload failed: {reload_err}. Falling back to fresh load."
                        )
                        solver_module = None

                if solver_module is None:
                    logging.info(
                        f"EVAL_WORKER (PID: {worker_pid}): Loading solver module from '{solver_file_path}' (not cached)"
                    )
                    solver_module = load_solver_module(solver_file_path.parent, solver_file_path.name)
                    _SOLVER_MODULE_CACHE[cache_key] = solver_module
                
                # Get a callable that creates a *fresh* Solver instance per call to
                # eradicate any residual per-instance caches inside the object.
                from AlgoTuner.utils.solver_loader import get_fresh_solve_callable
                solve_callable = get_fresh_solve_callable(solver_module)
                logging.info(f"EVAL_WORKER (PID: {worker_pid}): Successfully loaded solver and got solve callable (single instance).")

            except Exception as setup_error:
                tb_str = traceback.format_exc()
                logging.error(f"EVAL_WORKER (PID: {worker_pid}): Solver setup failed: {setup_error}\n{tb_str}")
                return create_standard_error_result(
                    exception=setup_error,
                    traceback_str=tb_str,
                    error_type_override='setup_error',
                    default_error_msg=str(setup_error)
                )

            # Now, run the benchmark. The benchmark_result will contain timing, success, and the raw result.
            logging.error(f"*** EVAL_WORKER_CRITICAL *** (PID: {worker_pid}): About to call run_isolated_benchmark with num_runs={num_runs}")
            logging.info(f"EVAL_WORKER (PID: {worker_pid}): Using isolated benchmark (num_runs={num_runs}, timeout={timeout_seconds}s)")

            # NEW: strict per-process isolation benchmark
            from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark  # Local import
            
            # CRITICAL: Ensure warmup and timed problems are different
            if warmup_problem_instance is None:
                logging.error(f"EVAL_WORKER (PID: {worker_pid}): CRITICAL BUG - warmup_problem_instance is None! This will cause identical warmup/timed problems!")
                # Use timed problem as fallback, but this is a bug that should be fixed upstream
                warmup_obj = timed_problem_instance
            else:
                warmup_obj = warmup_problem_instance
                
            # Log what we're actually using for verification
            logging.info(f"EVAL_WORKER (PID: {worker_pid}): Warmup problem: {type(warmup_obj)} (is None: {warmup_obj is None})")
            logging.info(f"EVAL_WORKER (PID: {worker_pid}): Timed problem: {type(timed_problem_instance)} (is None: {timed_problem_instance is None})")
            logging.info(f"EVAL_WORKER (PID: {worker_pid}): Problems are identical: {warmup_obj is timed_problem_instance}")
            # Additional deep-equality check for debugging
            try:
                content_equal = warmup_obj == timed_problem_instance
            except Exception:
                content_equal = "uncomparable"
            logging.info(f"EVAL_WORKER (PID: {worker_pid}): Problems content-equal (==): {content_equal}")

            # Use memory-efficient fetch approach for large datasets
            from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark_with_fetch
            
            # Get fetch info from problem metadata (passed from parent)
            problem_fetch_info = problem_metadata.get("problem_fetch_info", {"type": "direct", "data": timed_problem_instance})
            warmup_fetch_info = problem_metadata.get("warmup_fetch_info", {"type": "direct", "data": warmup_obj})
            
            baseline_res = run_isolated_benchmark_with_fetch(
                task_name=task_name,
                code_dir=actual_code_dir,
                warmup_fetch_info=warmup_fetch_info,
                timed_fetch_info=problem_fetch_info,
                num_runs=num_runs,
                timeout_seconds=timeout_seconds,
            )
            
            # For validation, run solver once more in main process to get result
            if baseline_res.get("success"):
                try:
                    logging.info(f"EVAL_WORKER (PID: {worker_pid}): Running solver for validation")
                    solver_result_for_validation = solve_callable(timed_problem_instance)
                    baseline_res["result"] = solver_result_for_validation
                    logging.info(f"EVAL_WORKER (PID: {worker_pid}): Successfully captured solver result for validation")
                except Exception as e:
                    logging.warning(f"EVAL_WORKER (PID: {worker_pid}): Failed to get solver result for validation: {e}")
                    baseline_res["result"] = None

            # Normalise keys to match earlier benchmark_result expectations
            benchmark_result = {
                # Coerce to plain bool to avoid subclass issues with multiprocessing
                "success": bool(baseline_res.get("success")),
                # Primary timing fields expected downstream
                "min_time_ms": baseline_res.get("min_time_ms"),
                "elapsed_ms": baseline_res.get("mean_time_ms"),
                # Keep original names so later conversion logic finds them
                "min_time_ms": baseline_res.get("min_time_ms"),
                "mean_ms": baseline_res.get("mean_time_ms"),
                # Provide second-scale variants for any legacy code that multiplies by 1000
                "min": (baseline_res.get("min_time_ms") / 1000.0) if baseline_res.get("min_time_ms") is not None else None,
                "mean": (baseline_res.get("mean_time_ms") / 1000.0) if baseline_res.get("mean_time_ms") is not None else None,
                # Full set of values and counts
                "values_ms": (
                    baseline_res.get("values_ms")
                    if baseline_res.get("values_ms") is not None
                    else [ns / 1e6 for ns in baseline_res.get("values_ns", [])]
                ),
                "runs": baseline_res.get("num_runs_executed"),
                "num_runs_executed": baseline_res.get("num_runs_executed"),
                "result": baseline_res.get("result"),
                "timeout_occurred": baseline_res.get("timeout_occurred"),
                # Propagate error details for downstream formatting
                "error": baseline_res.get("error"),
                "traceback": baseline_res.get("traceback"),
                "code_context": baseline_res.get("code_context"),
                "error_type": baseline_res.get("error_type"),
            }

            logging.error(
                f"EVAL_WORKER_DEBUG: success flag after cast = {benchmark_result['success']} "
                f"(type: {type(benchmark_result['success'])})"
            )

            logging.error(f"*** EVAL_WORKER_CRITICAL *** (PID: {worker_pid}): run_benchmark returned, checking result...")
            
            # COMPREHENSIVE EVAL_WORKER TIMING DEBUG: Log what run_benchmark returned
            eval_worker_timing_fields = ["success", "values", "values_ns", "runs", "num_runs_executed", "mean", "median", "min", "max", "stddev", "mean_ns", "median_ns", "min_ns", "max_ns", "stddev_ns", "mean_ms", "median_ms", "min_time_ms", "max_ms", "stddev_ms", "error", "timeout_occurred", "elapsed_ms"]
            eval_worker_timing_debug = {field: benchmark_result.get(field) for field in eval_worker_timing_fields}
            logging.info(f"EVAL_WORKER_TIMING_DEBUG: run_benchmark returned for '{task_name}': {eval_worker_timing_debug}")

            try:
                # Check if the benchmark itself was successful.
                if benchmark_result.get('success'):
                    logging.info(f"EVAL_WORKER (PID: {worker_pid}): Benchmark successful")

                    # Validate the result from the benchmark
                    logging.info(f"EVAL_WORKER (PID: {worker_pid}): Starting in-process validation...")
                    # Need to get the task instance for validation
                    from AlgoTuner.utils.evaluator.loader import load_task
                    task_obj = load_task(task_name, data_dir)
                    
                    validation_result = _validate_solution(
                        task_obj,
                        timed_problem_instance,
                        benchmark_result.get("result")
                    )
                    logging.info(f"EVAL_WORKER (PID: {worker_pid}): In-process validation completed.")

                    # Summarize the raw result for logging if it's large
                    result_summary = format_object_shape(benchmark_result.get("result"))

                    # The benchmark succeeded. Now, check if the solution was valid.
                    if validation_result.get("success"):
                        logging.info(f"EVAL_WORKER (PID: {worker_pid}): Solution is VALID.")
                        # ** THE FIX IS HERE **
                        # Update the original benchmark result dictionary.
                        # This preserves all timing information (min_time_ms, etc.)
                        # while adding the validation outcome.
                        benchmark_result.update({
                            "valid": True,
                            "result_summary": result_summary,
                            "validation_completed": True,
                        })
                        
                        # --- NEW: ALWAYS STRIP SOLVER RESULT AFTER VALIDATION ---
                        try:
                            solver_result = benchmark_result.get("result")
                            # Build a compact summary that still lets the parent know validation already happened
                            compact_summary = {
                                "type": str(type(solver_result)),
                                "stripped_after_validation": True,
                                "validation_completed": True,
                            }
                            # Add basic shape/length hints if available
                            if hasattr(solver_result, "__len__"):
                                compact_summary["length"] = len(solver_result)
                            if hasattr(solver_result, "shape"):
                                compact_summary["shape"] = str(solver_result.shape)
                                if hasattr(solver_result, "dtype"):
                                    compact_summary["dtype"] = str(solver_result.dtype)
                            benchmark_result["result"] = compact_summary
                            logging.info(
                                f"EVAL_WORKER (PID: {worker_pid}): Replaced solver result with compact summary after validation (size-agnostic)."
                            )
                        except Exception as e:
                            logging.warning(
                                f"EVAL_WORKER (PID: {worker_pid}): Failed to replace solver result with summary: {e}"
                            )
                        result = benchmark_result
                        benchmark_result["validation_result"] = validation_result  # Ensure parent receives validation outcome
                        _cleanup_timing_fields(result)
                    else:
                        # The solution was invalid.
                        logging.warning(f"EVAL_WORKER (PID: {worker_pid}): Solution is INVALID. Error: {validation_result.get('error')}")
                        # ** THE FIX IS ALSO HERE **
                        # Update the benchmark result with failure info, preserving timing.
                        benchmark_result.update({
                            "valid": False,
                            "validation_error": validation_result.get("error"),
                            "validation_traceback": validation_result.get("traceback"),
                            "result_summary": result_summary,
                            "validation_completed": True,
                        })
                        
                        # --- NEW: ALWAYS STRIP INVALID SOLVER RESULT ---
                        try:
                            solver_result = benchmark_result.get("result")
                            compact_summary_invalid = {
                                "type": str(type(solver_result)),
                                "stripped_after_validation": True,
                                "validation_failed": True,
                                "validation_completed": True,
                            }
                            if hasattr(solver_result, "__len__"):
                                compact_summary_invalid["length"] = len(solver_result)
                            if hasattr(solver_result, "shape"):
                                compact_summary_invalid["shape"] = str(solver_result.shape)
                                if hasattr(solver_result, "dtype"):
                                    compact_summary_invalid["dtype"] = str(solver_result.dtype)
                            benchmark_result["result"] = compact_summary_invalid
                            logging.info(
                                f"EVAL_WORKER (PID: {worker_pid}): Replaced invalid solver result with compact summary."
                            )
                        except Exception as e:
                            logging.warning(
                                f"EVAL_WORKER (PID: {worker_pid}): Failed to replace invalid solver result with summary: {e}"
                            )
                         
                         # EVAL_WORKER TIMING CONVERSION DEBUG: Log for invalid solution case
                        pre_conversion_timing_invalid = {
                            'min_time_ms': benchmark_result.get('min_time_ms'),
                            'mean_ms': benchmark_result.get('mean_ms'),
                            'min': benchmark_result.get('min'),
                            'mean': benchmark_result.get('mean'),
                            'elapsed_ms': benchmark_result.get('elapsed_ms')
                        }
                        logging.info(f"EVAL_WORKER_TIMING_CONVERSION (INVALID): Pre-conversion timing fields: {pre_conversion_timing_invalid}")
                        
                        # Ensure timing fields are in expected format (even for invalid solutions)
                        min_time_ms = benchmark_result.get('min_time_ms') or (benchmark_result.get('min') * 1000 if benchmark_result.get('min') else None)
                        mean_ms = benchmark_result.get('mean_ms') or (benchmark_result.get('mean') * 1000 if benchmark_result.get('mean') else None)
                        
                        logging.info(f"EVAL_WORKER_TIMING_CONVERSION (INVALID): Calculated min_time_ms={min_time_ms}, mean_ms={mean_ms}")
                        
                        benchmark_result.update({
                            'min_time_ms': min_time_ms,
                            'mean_time_ms': mean_ms,
                            'elapsed_ms': min_time_ms
                        })
                        
                        # Log after updating
                        post_conversion_timing_invalid = {
                            'min_time_ms': benchmark_result.get('min_time_ms'),
                            'mean_time_ms': benchmark_result.get('mean_time_ms'),
                            'elapsed_ms': benchmark_result.get('elapsed_ms')
                        }
                        logging.info(f"EVAL_WORKER_TIMING_CONVERSION (INVALID): Post-conversion timing fields: {post_conversion_timing_invalid}")
                        
                        result = benchmark_result
                        benchmark_result["validation_result"] = validation_result  # Propagate failed validation details
                        _cleanup_timing_fields(result)
                else:
                    # The benchmark itself failed (e.g., timed out or crashed).
                    # The benchmark_result already contains the error info.
                    logging.warning(f"EVAL_WORKER (PID: {worker_pid}): Benchmark failed. Error: {benchmark_result.get('error')}")
                    result = benchmark_result

            except Exception as e:
                tb_str = traceback.format_exc()
                logging.error(f"EVAL_WORKER (PID: {worker_pid}): Error during result processing or validation: {e}\n{tb_str}")
                result = create_standard_error_result(
                    exception=e,
                    traceback_str=tb_str,
                    error_type_override='runtime_error',
                    default_error_msg=f"Error after benchmark execution: {e}"
                )
                
    except Exception as e:
        # Broad exception handler for any other unexpected errors in the worker
        tb_str = traceback.format_exc()
        logging.error(f"EVAL_WORKER (PID: {worker_pid}): Unexpected top-level error: {e}\n{tb_str}")
        result = create_standard_error_result(
            exception=e,
            traceback_str=tb_str,
            error_type_override='runtime_error',
            default_error_msg=f"Top-level worker error: {e}"
        )
    finally:
        # Stop the heartbeat thread
        if stop_heartbeat is not None:
            stop_heartbeat.set()
            if heartbeat_thread:
                heartbeat_thread.join(timeout=2.0)
        
        # Add captured output to the final result
        if result and isinstance(result, dict):
            # Check if stdout already exists from a benchmark failure
            if 'stdout' not in result:
                result['stdout'] = ""
            if 'stderr' not in result:
                result['stderr'] = ""
                
            # Append any output captured at this level
            captured_output = f_out.getvalue()
            if captured_output:
                # Add a separator to distinguish this output from benchmark output
                separator = "\n--- Output from _eval_worker_target ---\n"
                result['stdout'] += separator + captured_output
        
        logging.info(f"EVAL_WORKER (PID: {worker_pid}): Terminating.")
        # NEW DEBUG: log final result outside of redirect context
        logging.error(
            f"EVAL_WORKER_FINAL_DEBUG_ERROR (PID: {worker_pid}): success={result.get('success')!r}, error_type={result.get('error_type')!r}, error={result.get('error')!r}, valid={result.get('valid')!r}"
        )
        
        # COMPREHENSIVE EVAL_WORKER FINAL DEBUG: Log all timing fields in final result
        if result and isinstance(result, dict):
            timing_fields = ['success', 'min_time_ms', 'mean_ms', 'min_time_ms', 'mean_time_ms', 'elapsed_ms', 'min', 'mean', 'values', 'values_ns', 'runs', 'num_runs_executed', 'mean_ns', 'median_ns', 'min_ns', 'max_ns', 'stddev_ns', 'error', 'timeout_occurred', 'valid']
            timing_debug = {field: result.get(field) for field in timing_fields}
            logging.info(f"EVAL_WORKER_FINAL_DEBUG (PID: {worker_pid}): Final result from _eval_worker_target: {timing_debug}")
        else:
            logging.warning(f"EVAL_WORKER_FINAL_DEBUG (PID: {worker_pid}): Final result is not a dict or is None: {type(result)}")

    return result

# -----------------------------------------------------------------------------
# Parent-side strict-isolation evaluator (used when daemon workers can't spawn)
# -----------------------------------------------------------------------------


def _evaluate_problem_parent_isolated(
    *,
    task_obj,
    problem_instance,
    warmup_problem_instance,
    num_runs: int,
    timeout_seconds: float,
    task_metadata: dict,
):
    """Run the strict per-process isolation benchmark **in the parent** process
    (non-daemon) and perform validation.

    Returns a dict compatible with the existing eval_result schema so the
    downstream aggregation code works unchanged.
    """

    from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark

    base_code_dir = os.environ.get("CODE_DIR", task_obj.get_task_directory())
    # For isolated benchmark, use the task-specific directory if it exists
    task_code_dir = os.path.join(base_code_dir, "AlgoTuneTasks", task_obj.task_name)
    if os.path.isdir(task_code_dir):
        code_dir = task_code_dir
    else:
        code_dir = base_code_dir

    # Debug logging for problem instances
    logging.debug(f"PARENT_ISOLATED: problem_instance type={type(problem_instance)}")
    logging.debug(f"PARENT_ISOLATED: warmup_problem_instance type={type(warmup_problem_instance)}")
    logging.debug(f"PARENT_ISOLATED: Problems are identical object: {problem_instance is warmup_problem_instance}")
    
    # Deep equality check to see if they have identical content
    try:
        deep_equal_flag = problem_instance == warmup_problem_instance
    except Exception:
        deep_equal_flag = "uncomparable"
    logging.debug(f"PARENT_ISOLATED: Problems content-equal (==): {deep_equal_flag}")
    
    if hasattr(problem_instance, 'keys') and hasattr(warmup_problem_instance, 'keys'):
        logging.debug(f"PARENT_ISOLATED: problem_instance keys: {list(problem_instance.keys())}")
        logging.debug(f"PARENT_ISOLATED: warmup_problem_instance keys: {list(warmup_problem_instance.keys())}")
        # Compare actual content if both are dicts
        if 'A' in problem_instance and 'A' in warmup_problem_instance:
            prob_A = problem_instance['A']
            warmup_A = warmup_problem_instance['A']
            logging.debug(f"PARENT_ISOLATED: problem A shape: {getattr(prob_A, 'shape', 'no shape')}")
            logging.debug(f"PARENT_ISOLATED: warmup A shape: {getattr(warmup_A, 'shape', 'no shape')}")
            if hasattr(prob_A, 'shape') and hasattr(warmup_A, 'shape'):
                import numpy as np
                arrays_equal = np.array_equal(prob_A, warmup_A) if hasattr(np, 'array_equal') else (prob_A == warmup_A).all()
                logging.debug(f"PARENT_ISOLATED: A matrices are equal: {arrays_equal}")

    benchmark_result = run_isolated_benchmark(
        task_name=task_obj.task_name,
        code_dir=code_dir,
        warmup_problem=warmup_problem_instance,
        timed_problem=problem_instance,
        num_runs=num_runs,
        timeout_seconds=timeout_seconds,
    )
    
    # For validation, run solver once more in main process to get result
    if benchmark_result.get("success"):
        try:
            logging.debug("PARENT_ISOLATED: Running solver for validation")
            # Load solver and get result for validation
            from AlgoTuner.utils.solver_loader import load_solver_module, get_fresh_solve_callable
            solver_module = load_solver_module(code_dir)
            solve_callable = get_fresh_solve_callable(solver_module)
            solver_result_for_validation = solve_callable(problem_instance)
            benchmark_result["result"] = solver_result_for_validation
            logging.debug("PARENT_ISOLATED: Successfully captured solver result for validation")
        except Exception as e:
            logging.warning(f"PARENT_ISOLATED: Failed to get solver result for validation: {e}")
            benchmark_result["result"] = None

    # Build eval_result in the same shape _eval_worker_target produces.
    success_flag = bool(benchmark_result.get("success"))

    eval_result = {
        "success": success_flag,
        "min_time_ms": benchmark_result.get("min_time_ms"),
        "elapsed_ms": benchmark_result.get("mean_time_ms"),
        "mean_ms": benchmark_result.get("mean_time_ms"),
        "values_ms": [ns / 1e6 for ns in benchmark_result.get("values_ns", [])],
        "num_runs_executed": benchmark_result.get("num_runs_executed"),
        "timeout_occurred": benchmark_result.get("timeout_occurred"),
        "result": benchmark_result.get("result"),
        "problem_metadata": task_metadata,
    }
    
    # Copy error fields when benchmark fails
    if not success_flag:
        eval_result["error"] = benchmark_result.get("error")
        eval_result["error_type"] = benchmark_result.get("error_type")
        eval_result["code_context"] = benchmark_result.get("code_context")

    # Validate solver output (only if benchmark succeeded)
    if success_flag:
        try:
            solver_result = benchmark_result.get("result")
            logging.debug(f"PARENT_ISOLATED: About to validate solver result: {type(solver_result)} (is None: {solver_result is None})")
            if solver_result is not None:
                logging.debug(f"PARENT_ISOLATED: Solver result keys: {list(solver_result.keys()) if hasattr(solver_result, 'keys') else 'not a dict'}")
                if hasattr(solver_result, 'keys') and 'labels' in solver_result:
                    labels = solver_result['labels']
                    logging.debug(f"PARENT_ISOLATED: Found labels in solver result: type={type(labels)}, shape={getattr(labels, 'shape', 'no shape')}")
                    if hasattr(labels, '__len__'):
                        try:
                            logging.debug(f"PARENT_ISOLATED: Labels length: {len(labels)}")
                        except:
                            pass
            
            logging.debug(f"PARENT_ISOLATED: Problem instance type: {type(problem_instance)}")
            if hasattr(problem_instance, 'keys'):
                logging.debug(f"PARENT_ISOLATED: Problem keys: {list(problem_instance.keys())}")
            
            validation_result = _validate_solution(
                task_obj, problem_instance, solver_result
            )
            
            logging.debug(f"PARENT_ISOLATED: Validation result: {validation_result}")
            eval_result["validation_result"] = validation_result

            if not validation_result.get("success", False):
                logging.warning(f"PARENT_ISOLATED: Validation failed - {validation_result.get('error_type', 'unknown')}: {validation_result.get('error', 'no error message')}")
                eval_result["success"] = False
                eval_result["error"] = validation_result.get("error")
                eval_result["error_type"] = validation_result.get("error_type", "invalid_solution")
                # Copy code context if available for better error reporting
                if validation_result.get("code_context"):
                    eval_result["code_context"] = validation_result.get("code_context")
                # Drop potentially huge solver output – not needed when validation failed
                eval_result["result"] = None
            else:
                logging.debug(f"PARENT_ISOLATED: Validation succeeded!")
                # Validation succeeded – we also don't need the raw solver output anymore
                eval_result["result"] = None
        except Exception as _val_err:
            logging.error(f"PARENT_ISOLATED: Validation exception: {_val_err}", exc_info=True)
            eval_result.update(
                {
                    "success": False,
                    "error": f"validation_exception: {_val_err}",
                    "error_type": "validation_runtime_error",
                    "traceback": traceback.format_exc(),
                }
            )

    return eval_result

# ------------------------------------------------------------------
# Utility helpers – timing field normalisation
# ------------------------------------------------------------------

def _cleanup_timing_fields(d: dict) -> None:
    """Remove legacy timing aliases to keep result dictionaries tidy.

    If *both* 'min_time_ms' and 'min_time_ms' exist and store identical
    values, drop the redundant 'min_time_ms'.  The same logic applies to
    'mean_time_ms' vs. 'mean_ms'.  Mutates *d* in-place.
    """

    try:
        # Remove duplicate alias keys (e.g. 'min_ms' that just replicate 'min_time_ms')
        if "min_ms" in d and "min_time_ms" in d and d["min_ms"] == d["min_time_ms"]:
            d.pop("min_ms", None)
        if "mean_ms" in d and "mean_time_ms" in d and d["mean_ms"] == d["mean_time_ms"]:
            d.pop("mean_ms", None)
    except Exception:
        # Never let clean-up raise – best-effort only
        pass

import psutil


def _cleanup_stuck_subprocesses():
    """
    Clean up stuck solver subprocesses and reset multiprocessing state.
    Does NOT restart the main evaluation process - only cleans subprocess machinery.
    """
    logging.warning("Cleaning up stuck solver subprocesses")
    
    try:
        import multiprocessing as mp  # Ensure access to multiprocessing helpers within this function
        current_pid = os.getpid()
        parent = psutil.Process(current_pid)
        
        # Kill only *solver worker* child processes – leave the fork-server and others untouched
        children_killed = 0
        for child in parent.children(recursive=True):
            try:
                # Skip if not marked as solver worker
                safe_skip = False
                try:
                    if child.environ().get("ALGOTUNER_SOLVER_WORKER") != "1":
                        safe_skip = True
                except Exception:
                    # Fallback: skip if cmdline hints it is the fork-server helper
                    try:
                        if "multiprocessing.forkserver" in " ".join(child.cmdline()):
                            safe_skip = True
                    except Exception:
                        pass
                if safe_skip:
                    continue

                child_name = child.name()
                logging.debug(f"Terminating stuck solver worker: PID={child.pid}, name={child_name}")
                try:
                    child.terminate()
                    child.wait(timeout=2)
                except (psutil.TimeoutExpired, psutil.AccessDenied):
                    child.kill()
                    child.wait(timeout=2)
                children_killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process already gone or cannot access – ignore
                continue

        # No longer reset multiprocessing default context – keep existing fork-server when present
        # Force cleanup of any cached contexts
        if hasattr(mp, '_default_context'):
            mp._default_context = None
        # Clear *solver* processes that are still active via mp.active_children()
        if hasattr(mp, 'active_children'):
            for proc in mp.active_children():
                if os.environ.get("ALGOTUNER_SOLVER_WORKER") == "1":
                    proc.terminate()

        # Clean up solver-specific temp files only
        import tempfile, glob, shutil
        temp_dir = tempfile.gettempdir()
        files_cleaned = 0
        for pattern in ('dace_cache_*', 'clarabel_*', 'cvxpy_*'):
            for path in glob.glob(os.path.join(temp_dir, pattern)):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.unlink(path)
                    files_cleaned += 1
                except Exception:
                    pass

        # Force garbage collection
        import gc
        gc.collect()

        logging.info(f"Subprocess cleanup completed - killed {children_killed} processes, cleaned {files_cleaned} temp files")
    except Exception as e:
        logging.warning(f"Subprocess cleanup encountered issues (continuing anyway): {e}")


def _evaluate_baseline_with_retry(
    problem_id: str,
    problem_fetch_info: Dict[str, Any],
    warmup_fetch_info: Dict[str, Any],
    task_obj: Any,
    num_runs: int,
    warmup_runs: int,
    timeout_seconds: float,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Evaluate baseline with automatic subprocess cleanup and retry on failure.
    Baseline evaluation should never fail, so failures trigger subprocess cleanup + retry.
    """
    safe_task_name = getattr(task_obj, 'task_name', 'unknown_task')
    # For baseline evaluation, check for container path first, then fall back to relative path
    container_path = f"/app/AlgoTuneTasks/{safe_task_name}"
    if os.path.exists(container_path):
        solver_dir = container_path
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        solver_dir = os.path.join(project_root, "AlgoTuneTasks", safe_task_name)
    logging.info(f"BASELINE_RETRY: Using solver_dir={solver_dir} for baseline evaluation")
    
    for attempt in range(max_retries):
        try:
            logging.debug(f"BASELINE_RETRY: Attempt {attempt + 1}/{max_retries} for problem {problem_id}")
            
            # Preload problems in parent to eliminate JSONL parsing in benchmark subprocess
            from AlgoTuner.utils.streaming_json import stream_jsonl
            # Load warmup problem
            warmup_record = next(stream_jsonl(warmup_fetch_info["path"]))
            warmup_problem = warmup_record.get("problem", warmup_record)
            # Load timed problem
            timed_record = next(stream_jsonl(problem_fetch_info["path"]))
            timed_problem = timed_record.get("problem", timed_record)
            # Run isolated benchmark directly on loaded problem dicts
            result_tb = run_isolated_benchmark(
                task_name=safe_task_name,
                code_dir=solver_dir,
                warmup_problem=warmup_problem,
                timed_problem=timed_problem,
                num_runs=num_runs,
                timeout_seconds=timeout_seconds,
            )
            
            if result_tb.get("success"):
                if attempt > 0:
                    logging.info(f"BASELINE_RETRY: Problem {problem_id} succeeded on attempt {attempt + 1}")
                return {
                    "success": True,
                    "min_time_ms": result_tb.get("min_time_ms"),
                    "elapsed_ms": result_tb.get("mean_time_ms"),
                    "timeout_occurred": result_tb.get("timeout_occurred", False),
                }
            else:
                error_msg = result_tb.get("error", "isolated baseline timing failed")
                logging.error(f"BASELINE_RETRY: Attempt {attempt + 1} failed for problem {problem_id}: {error_msg}")
                
        except Exception as e:
            logging.error(f"BASELINE_RETRY: Attempt {attempt + 1} exception for problem {problem_id}: {e}")
        
        # If not the last attempt, clean up stuck subprocesses and retry
        if attempt < max_retries - 1:
            _cleanup_stuck_subprocesses()
            import time
            time.sleep(1)  # Brief pause after cleanup
    
    # All attempts failed
    logging.error(f"BASELINE_RETRY: All {max_retries} attempts failed for problem {problem_id}")
    return {
        "success": False,
        "error": f"Baseline evaluation failed after {max_retries} retry attempts",
        "min_time_ms": None,
        "elapsed_ms": 0.0,
        "timeout_occurred": True,
    }


def _calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate metrics from evaluation results.
    
    Args:
        results: List of individual problem evaluation results
                Each dict should have at least: success, speedup (if applicable)
                
    Returns:
        Dict containing aggregate metrics like mean_speedup, success_rate, etc.
    """
    if not results:
        return {}
    
    # Initialize counters and accumulators
    num_evaluated = len(results)
    num_valid = 0
    num_invalid = 0
    num_timeouts = 0
    num_errors = 0
    num_inf_speedup = 0
    
    # Time accumulators
    solver_times = []
    oracle_times = []
    solver_times_mutual = []
    oracle_times_mutual = []
    speedups = []
    
    # Process each result individually
    for result in results:
        error_type = result.get('error_type', '')
        is_valid_solution = result.get('is_valid', False)
        timeout_occurred = (error_type == 'timeout') or result.get('timeout_occurred', False)
        
        if is_valid_solution:
            num_valid += 1
            
            # --- Collect speedup for valid solutions --- 
            speedup_val = result.get('speedup')
            if speedup_val is not None:
                if speedup_val == float('inf'):
                    num_inf_speedup += 1
                speedups.append(speedup_val)

            # --- Accumulate times for averages (using correct keys from result dict) ---
            solver_time = result.get('min_time_ms')
            oracle_time_for_avg = result.get('baseline_time_ms')

            if solver_time is not None:
                solver_times.append(solver_time)
            if oracle_time_for_avg is not None:
                oracle_times.append(oracle_time_for_avg)
            if solver_time is not None and oracle_time_for_avg is not None:
                solver_times_mutual.append(solver_time)
                oracle_times_mutual.append(oracle_time_for_avg)
                
        elif timeout_occurred:
            num_timeouts += 1
        elif error_type == 'invalid_solution':
            # Count invalid solutions from is_solution
            num_invalid += 1
        else:
            # Other validation or execution errors
            num_errors += 1
    
    # Calculate average times
    avg_solver_time = None
    avg_oracle_time = None
    avg_solver_mutual = None
    avg_oracle_mutual = None
    if solver_times:
        avg_solver_time = np.mean(solver_times)
    if oracle_times:
        avg_oracle_time = np.mean(oracle_times)
    if solver_times_mutual:
        avg_solver_mutual = np.mean(solver_times_mutual)
    if oracle_times_mutual:
        avg_oracle_mutual = np.mean(oracle_times_mutual)
    
    # Calculate success rate & overall validity
    success_rate = num_valid / num_evaluated if num_evaluated > 0 else 0.0
    overall_valid = num_valid > 0 and num_valid == num_evaluated
    
    # Calculate mean and median speedup, skipping infinite values
    finite_speedups = [s for s in speedups if s is not None and s != float('inf')] # Ensure not None before checking for inf
    
    if finite_speedups:
        mean_speedup = np.mean(finite_speedups)
        median_speedup = np.median(finite_speedups) if finite_speedups else None
    else:
        # No finite speedups. Check if all actual (non-None) speedups were infinite.
        non_none_speedups = [s for s in speedups if s is not None]
        if non_none_speedups and all(s == float('inf') for s in non_none_speedups):
            mean_speedup = float('inf')
            median_speedup = float('inf')
        else: # speedups list was empty, contained only Nones, or a mix not exclusively infinite
            mean_speedup = None
            median_speedup = None
            
    # If not every solution was valid, invalidate speedup metrics
    if not overall_valid:
        logging.info("Not all solutions were valid; setting speedup metrics to None (N/A).")
        mean_speedup = None
        median_speedup = None
    
    # Assemble the aggregate metrics
    metrics = {
        'num_evaluated': num_evaluated,
        'overall_valid': overall_valid,
        'mean_speedup': mean_speedup,
        'median_speedup': median_speedup,
        'success_rate': success_rate,
        'num_valid': num_valid,
        'num_invalid': num_invalid,
        'num_errors': num_errors,
        'num_timeouts': num_timeouts,
        'num_inf_speedup': num_inf_speedup
    }
    
    # Add timing metrics (carefully handling None values)
    if avg_solver_time is not None:
        metrics['avg_solver_time_ms'] = avg_solver_time
    if avg_oracle_time is not None:
        metrics['avg_oracle_time_ms'] = avg_oracle_time
    if avg_solver_mutual is not None:
        metrics['avg_solver_time_on_mutual_valid'] = avg_solver_mutual
    if avg_oracle_mutual is not None:
        metrics['avg_oracle_time_on_mutual_valid'] = avg_oracle_mutual
    
    return metrics