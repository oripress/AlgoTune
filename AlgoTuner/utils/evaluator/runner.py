"""
Evaluator runner module for executing and timing solver code.
"""


import os
import sys
import logging
import traceback
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, Union
import inspect
import functools
import builtins
import io
import contextlib
import time
import multiprocessing
import math
import statistics
from pathlib import Path
import faulthandler # Ensure faulthandler is imported
import gc

# Import utils that are safe (don't create circular dependencies)
from AlgoTuner.utils.type_inspection import describe_type
from AlgoTuner.utils.utils import clean_traceback, format_object_shape, safe_import

# Import from the new error utility file
from AlgoTuner.utils.error_utils import extract_error_context, create_standard_error_result
from AlgoTuner.utils.solver_loader import load_solver_module, get_solve_callable, get_fresh_solve_callable, locate_solver_file, get_fresh_solve_callable_with_module_reload
from AlgoTuner.utils.dace_config import initialize_dace_for_process

# Directory where LLM-generated code is stored
CODE_DIR = os.environ.get("CODE_DIR", "llm_src")

# Initialize DaCe configuration for the main process
initialize_dace_for_process()

# Removed duplicate helper imports and redundant duplicate imports

# Import from the precise timing module
from AlgoTuner.utils.benchmark import run_benchmark
from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark
from AlgoTuner.config.loader import load_config

# Import from the timing manager
from AlgoTuner.utils.timing_manager import TimingManager, Phase

# Import from the precise timing module  
from AlgoTuner.utils.benchmark_pool import BenchmarkPool, _run_task

# Import from the isolated benchmark utility
from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark  # NEW

# Enable faulthandler to dump tracebacks on serious errors (like segfaults)
faulthandler.enable()

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper to strip out the giant matrix and full solver output before final return
# -----------------------------------------------------------------------------
def _strip_bulky_fields(res: Dict[str, Any]) -> Dict[str, Any]:
    # Remove all fields that may contain large problem data
    res.pop("problem",    None)
    res.pop("raw_result", None)
    res.pop("problem_input", None)
    res.pop("problem_metadata", None)
    res.pop("validation_result", None)
    # Keep only scalar metrics and validation outcomes
    return res

def _get_result_size_info(result: Any) -> str:
    """Helper to get a descriptive string for a result's size."""
    import sys
    import numpy as np
    
    def get_size_mb(res):
        size_bytes = 0
        if isinstance(res, np.ndarray):
            size_bytes = res.nbytes
        elif isinstance(res, list) and res and all(isinstance(x, np.ndarray) for x in res):
            size_bytes = sum(arr.nbytes for arr in res)
        else:
            size_bytes = sys.getsizeof(res) # fallback
        return size_bytes / (1024 * 1024)

    size_mb = get_size_mb(result)
    size_str = f"size={size_mb:.3f}MB"
    
    length = None
    if hasattr(result, '__len__'):
        length = len(result)
        if hasattr(result, 'shape'):  # numpy array
            size_str += f", shape={result.shape}, dtype={result.dtype}"
        else:  # list, tuple, etc.
            size_str += f", length={length}"
    return size_str

# Define a top-level function for pickability
def _wrapped_pickable_function(func, arg):
    """Top-level wrapper function that can be pickled.
    
    Args:
        func: The function to call
        arg: A tuple of (args, kwargs) where args is a tuple of positional arguments
             and kwargs is a dictionary of keyword arguments
    """
    try:
        args, kwargs = arg  # Unpack the arguments
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        import traceback
        import os
        import logging
        from typing import Optional
        import inspect

        # Get the error location
        tb = e.__traceback__
        while tb and tb.tb_next: # Check tb before accessing tb.tb_next
            tb = tb.tb_next
            
        code_context = None # Initialize
        if tb: # Only proceed if traceback exists
            frame = tb.tb_frame
            file_path = frame.f_code.co_filename
            error_line = tb.tb_lineno
            
            # Get the function source
            try:
                lines, start_line = inspect.getsourcelines(frame.f_code)
                
                # Calculate the relative line number within the function
                func_error_line = error_line - start_line
                
                # Get 10 lines before and after, but stay within function bounds
                context_start = max(0, func_error_line - 10)
                context_end = min(len(lines), func_error_line + 11)
                
                code_context = "Error context:\n"
                for i in range(context_start, context_end):
                    line_num = start_line + i
                    marker = ">" if line_num == error_line else " "
                    code_context += f"{marker} {line_num}: {lines[i].rstrip()}\n"
            except Exception as context_e:
                logging.error(f"Error getting code context: {context_e}")
                code_context = None

        # Clean the traceback to only show filenames
        clean_tb = []
        for line in traceback.format_exc().split('\n'):
            if "File " in line:
                # Replace full path with just filename
                parts = line.split('"')
                if len(parts) > 1:
                    filename = os.path.basename(parts[1])
                    line = line.replace(parts[1], filename)
            clean_tb.append(line)
        clean_tb = '\n'.join(clean_tb)

        # Raise with enhanced error information but without the error type prefix
        raise type(e)(
            str(e),
            {
                'traceback': clean_tb,
                'code_context': code_context,
                'source': getattr(func, '__name__', 'unnamed_function')
            }
        ).with_traceback(e.__traceback__)

def execute_and_capture_errors(
    func: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    func_description: str = "function",
    capture_output: bool = False
) -> Dict[str, Any]:
    """
    Execute a function, capturing errors, traceback, code context, and optionally stdout/stderr.
    
    Args:
        func: Function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        func_description: Description of the function for error messages
        capture_output: Whether to capture stdout and stderr
        
    Returns:
        Dict containing:
            - success: bool
            - result: Return value of the function (if successful)
            - error: Error message (if failed)
            - traceback: Traceback string (if failed)
            - error_type: Categorized error type (if failed)
            - code_context: Snippet of code near the error (if failed)
            - stdout: Captured standard output (if capture_output=True)
            - stderr: Captured standard error (if capture_output=True)
    """
    if kwargs is None:
        kwargs = {}
        
    stdout_str = ""
    stderr_str = ""
    
    try:
        # Execute the function, capturing output if requested
        if capture_output:
            with contextlib.redirect_stdout(io.StringIO()) as captured_stdout, \
                 contextlib.redirect_stderr(io.StringIO()) as captured_stderr:
                result = func(*args, **kwargs)
            stdout_str = captured_stdout.getvalue()
            stderr_str = captured_stderr.getvalue()
        else:
            result = func(*args, **kwargs)
            
        return {
            "success": True,
            "result": result,
            "stdout": stdout_str,
            "stderr": stderr_str,
        }
    except Exception as e:
        tb_str = traceback.format_exc()
        logging.error(f"Error executing {func_description}: {e}", exc_info=False) # Log original error simply
        
        # Determine if it's a validation error originating from is_solution
        error_type_override = None
        if func_description == "task.is_solution":
             error_type_override = "validation_error" 

        # Use the new utility to create the standardized result
        error_result = create_standard_error_result(
            exception=e,
            traceback_str=tb_str,
            error_type_override=error_type_override,
            stdout=stdout_str, # Pass captured output
            stderr=stderr_str,
            default_error_msg=f"Error in {func_description}"
        )

        # Add captured output even on error if requested
        error_result["stdout"] = stdout_str
        error_result["stderr"] = stderr_str
        
        return error_result

_config = load_config()
# Central timing configuration
from AlgoTuner.utils.timing_config import RUNS as DEFAULT_RUNNER_RUNS, WARMUPS as DEFAULT_RUNNER_WARMUPS, WARMUP_MULTIPLIER
# For solver and evaluation defaults
TEST_RUNS = DEFAULT_RUNNER_RUNS
TEST_WARMUPS = DEFAULT_RUNNER_WARMUPS
# For dataset/ oracle timing defaults
DATASET_RUNS = DEFAULT_RUNNER_RUNS
DATASET_WARMUPS = DEFAULT_RUNNER_WARMUPS

DEFAULT_WARMUPS = DEFAULT_RUNNER_WARMUPS

# Timeout calculation constants
TARGET_TIME_MULTIPLIER = 10.0  # Multiplier for target/oracle time

def _calculate_timeout_seconds(
    baseline_time_ms: Optional[float],
    num_runs: int,
    warmup_runs: int,
    *,
    min_timeout_s: float = 2.0,
) -> float:
    """Return the timeout in **seconds** following the uniform 10× rule.

    Args:
        baseline_time_ms:  Per-run baseline time in milliseconds (oracle or
            task-provided target).  If *None* or non-positive we just return
            ``min_timeout_s``.
        num_runs:          Number of timed measurement runs that will be
            executed.
        warmup_runs:       Number of warm-up runs that precede the timed ones.
        min_timeout_s:     Lower bound to guard against a zero / negative or
            missing baseline.

    Returns
    -------
    float
        Timeout in seconds.
    """

    if baseline_time_ms is None or baseline_time_ms <= 0:
        return max(min_timeout_s, 0.0)

    per_run_s = baseline_time_ms / 1000.0
    total_runs = warmup_runs + num_runs
    # One uniform 10× multiplier for **every** run.
    calculated = per_run_s * total_runs * 10.0
    return max(min_timeout_s, calculated)

def _run_benchmark(
    func: Callable,
    args: Tuple,
    task_instance: Any,
    oracle_time_ms: Optional[float] = None,
    capture_output: bool = False,
    num_runs: int = DEFAULT_RUNNER_RUNS,
    warmup_runs: int = DEFAULT_RUNNER_WARMUPS,
    working_dir: Optional[str] = None,
    solver_module = None
) -> Dict[str, Any]:
    """
    Internal wrapper to run a benchmark function with specific configurations.
    Calculates timeout based on task's target time (if available) or defaults.
    Uses the main run_benchmark function from utils.benchmark.
    Handles parsing positional/keyword arguments.
    Returns results with times converted to milliseconds.
    """
    func_name = getattr(func, '__name__', 'anonymous')
    logging.info(f"_run_benchmark starting for '{func_name}' (runs={num_runs}, warmups={warmup_runs})")

    try:
        # --- Timeout Calculation --- 
        DEFAULT_TIMEOUT_S = 1.0 # Default fallback timeout
        FIXED_BASELINE_TIMEOUT_S = 60.0 # Fixed timeout for AGENT_MODE=0 - increased for CP-SAT solvers
        MIN_TIMEOUT_S = 10.0  # Enforce 10-second minimum per subprocess

        timeout_s = DEFAULT_TIMEOUT_S # Start with fallback default
        timeout_reason = "Default Fallback"

        agent_mode = os.environ.get("AGENT_MODE", "0")

        if agent_mode == "0":
            timeout_s = FIXED_BASELINE_TIMEOUT_S
            timeout_reason = f"AGENT_MODE=0 Fixed Baseline"
        else: # Agent mode != 0
            # --- MODIFICATION START: Prioritize passed oracle_time_ms with warmup consideration --- 
            if oracle_time_ms is not None and oracle_time_ms > 0:
                oracle_time_s = oracle_time_ms / 1000.0
                
                # Check if we're using isolated benchmark (AGENT_MODE=1 or forced isolation)
                agent_mode = os.environ.get("AGENT_MODE", "0")
                is_isolated = agent_mode != "0" or os.environ.get("ISOLATED_EVAL", "1") == "1"
                
                if is_isolated:
                    # For isolated benchmark each subprocess performs *warmup_runs*
                    # un-timed iterations plus **1** timed iteration.  The warm-up can
                    # be substantially slower than the timed measurement because of
                    # first-touch page faults and library initialisation, so we base
                    # the timeout on ``(1 + WARMUP_MULTIPLIER)`` rather than the
                    # fixed "×2" heuristic.

                    calculated_timeout_s = (1 + WARMUP_MULTIPLIER) * oracle_time_s * TARGET_TIME_MULTIPLIER
                    timeout_s = max(MIN_TIMEOUT_S, calculated_timeout_s)
                    timeout_reason = f"Isolated Benchmark Oracle Time ({oracle_time_ms:.2f}ms/run): (1 warmup + 1 timed) * {TARGET_TIME_MULTIPLIER}x = {calculated_timeout_s:.1f}s"
                else:
                    # Original calculation for non-isolated mode
                    warmup_time_s = warmup_runs * oracle_time_s * WARMUP_MULTIPLIER
                    measurement_time_s = num_runs * oracle_time_s  # 1x baseline for measurements
                    estimated_total_time_s = warmup_time_s + measurement_time_s
                    calculated_timeout_s = estimated_total_time_s * TARGET_TIME_MULTIPLIER
                    timeout_s = max(MIN_TIMEOUT_S, calculated_timeout_s)
                    timeout_reason = f"Per-Problem Oracle Time ({oracle_time_ms:.2f}ms/run): warmup_time={warmup_time_s:.1f}s + measurement_time={measurement_time_s:.1f}s * {TARGET_TIME_MULTIPLIER}x"
                logging.debug(f"Using timeout based on provided oracle_time_ms with warmup consideration: {oracle_time_ms:.2f}ms")
            else:
                # Fallback to task's target time if oracle_time_ms not provided/invalid
                logging.debug(f"oracle_time_ms ({oracle_time_ms}) not usable, falling back to task target time.")
                try:
                    get_target_method = getattr(task_instance, "_get_target_time_ms", None)
                    if callable(get_target_method):
                        target_time_ms = get_target_method()
                        if target_time_ms is not None and target_time_ms > 0:
                            target_time_s = target_time_ms / 1000.0
                            
                            # Check if we're using isolated benchmark (AGENT_MODE=1 or forced isolation)
                            agent_mode = os.environ.get("AGENT_MODE", "0")
                            is_isolated = agent_mode != "0" or os.environ.get("ISOLATED_EVAL", "1") == "1"
                            
                            if is_isolated:
                                # For isolated benchmark: each subprocess does 1 warmup + 1 timed
                                calculated_timeout_s = 2 * target_time_s * TARGET_TIME_MULTIPLIER  # (1 warmup + 1 timed) × 10
                                timeout_s = max(MIN_TIMEOUT_S, calculated_timeout_s)
                                timeout_reason = f"Isolated Benchmark Task Target Time ({target_time_ms:.2f}ms/run): (1 warmup + 1 timed) * {TARGET_TIME_MULTIPLIER}x = {calculated_timeout_s:.1f}s (Fallback)"
                            else:
                                # Original calculation for non-isolated mode
                                warmup_time_s = warmup_runs * target_time_s * WARMUP_MULTIPLIER
                                measurement_time_s = num_runs * target_time_s  # 1x baseline for measurements
                                estimated_total_target_time_s = warmup_time_s + measurement_time_s
                                calculated_timeout_s = estimated_total_target_time_s * TARGET_TIME_MULTIPLIER
                                timeout_s = max(MIN_TIMEOUT_S, calculated_timeout_s)
                                timeout_reason = f"Task Target Time ({target_time_ms:.2f}ms/run): warmup_time={warmup_time_s:.1f}s + measurement_time={measurement_time_s:.1f}s * {TARGET_TIME_MULTIPLIER}x (Fallback)"
                        else:
                            logging.warning(f"Task {getattr(task_instance, 'task_name', 'unknown')} provided invalid target time: {target_time_ms}. Using default timeout.")
                            timeout_reason = "Invalid Target Time Fallback"
                    else:
                        logging.warning(f"Task {getattr(task_instance, 'task_name', 'unknown')} missing _get_target_time_ms method. Using default timeout.")
                        timeout_reason = "Missing Target Time Method Fallback"
                except Exception as e:
                    logging.warning(f"Error getting target time for task {getattr(task_instance, 'task_name', 'unknown')}: {e}. Using default timeout.")
                    timeout_reason = "Get Target Time Error Fallback"
            # --- MODIFICATION END ---

        logging.info(
            (
                "TIMEOUT_DEBUG: func=%s, oracle_time_ms=%s, per_run_s=%.6f, "
                "warmup_runs=%d, num_runs=%d, calculated_timeout=%.3f s, reason=%s"
            ),
            func_name,
            f"{oracle_time_ms:.2f}" if oracle_time_ms is not None else None,
            (oracle_time_ms or 0) / 1000.0 if oracle_time_ms else 0.0,
            warmup_runs,
            num_runs,
            timeout_s,
            timeout_reason,
        )

        # ------------------------------------------------------------------
        # No additional safety floor – adhere strictly to the 10× baseline rule.
        # ------------------------------------------------------------------
        if timeout_s < 10.0:
            logging.debug(
                "TIMEOUT_DEBUG: Raising timeout from %.2fs to 10.00s (minimum floor)",
                timeout_s
            )
            timeout_s = 10.0

        logging.info(f"Using benchmark timeout: {timeout_s:.2f} seconds for {func_name}. Reason: {timeout_reason}")
        # --- End Timeout Calculation ---

        # Prepare args/kwargs (Keep this logic)
        positional_args = args
        keyword_args = {}
        if isinstance(args, tuple) and len(args) > 0:
            if isinstance(args[-1], dict) and len(args) > 1:
                positional_args = args[:-1]
                keyword_args = args[-1]
            else:
                positional_args = args
                keyword_args = {}
        else:
            positional_args = args if isinstance(args, tuple) else (args,)
            keyword_args = {}


        # Call the updated benchmark function, passing the calculated timeout
        # No runner or benchmark_name needed anymore
        # Extract problem from args for isolated benchmark
        problem = positional_args[0] if positional_args else None
        task_instance = getattr(func, '__self__', None)
        task_name = getattr(task_instance, 'task_name', 'unknown_task')
        
        # Get task directory
        if hasattr(task_instance, 'get_task_directory'):
            task_code_dir = task_instance.get_task_directory()
        else:
            task_code_dir = working_dir or "."
        
        # Use isolated benchmark for consistency
        benchmark_result = run_isolated_benchmark(
            task_name=task_name,
            code_dir=task_code_dir,
            warmup_problem=problem,
            timed_problem=problem,
            num_runs=1,
            timeout_seconds=timeout_s,
        )

        # --- IN-PROCESS VALIDATION (before multiprocessing) ---
        # Run validation immediately after solver execution to avoid memory issues
        validation_result = benchmark_result.get("validation_result")

        try:
            if validation_result is not None:
                # Validation was already performed inside the isolated worker.
                logging.info(
                    f"Validation already present from worker for {func_name}: {validation_result.get('success', False)}"
                )
                # Replace potentially bulky solver output with a compact sentinel
                original_type = type(benchmark_result.get("result"))
                benchmark_result["result"] = {
                    "__stripped__": True,
                    "type": str(original_type),
                    "original_type": str(original_type),
                    "validation_completed": True,
                }
            else:
                solver_result = benchmark_result.get("result")
                problem = positional_args[0] if positional_args else None

                # Run validation only when we *have* a concrete solver_result.
                if (solver_result is not None) and problem is not None and hasattr(task_instance, 'is_solution'):
                    logging.info(
                        f"Running in-process validation for {func_name} (result will be stripped afterwards)"
                    )
                    validation_result = _validate_solution(
                        task_instance, problem, solver_result
                    )
                    logging.info(
                        f"In-process validation completed: {validation_result.get('success', False)}"
                    )

                    # Run failure analysis immediately if validation failed (while we have full result)
                    if not validation_result.get("success", False):
                        try:
                            from AlgoTuner.utils.evaluator.failure_analyzer import trace_is_solution_failure
                            logging.info(f"Running immediate failure analysis for {func_name}")
                            trace_is_solution_failure(task_instance, problem, solver_result)
                            logging.info(f"Failure analysis completed, context stored in task_instance")
                        except Exception as e:
                            logging.warning(f"Failure analysis failed for {func_name}: {e}")

                    # Now strip result for both valid and invalid solutions to save memory
                    compact_summary = {
                        "__stripped__": True,
                        "type": str(type(solver_result)),
                        "original_type": str(type(solver_result)),
                        "validation_completed": True,
                    }
                    if hasattr(solver_result, "__len__"):
                        compact_summary["length"] = len(solver_result)
                    if hasattr(solver_result, "shape"):
                        compact_summary["shape"] = str(solver_result.shape)
                        if hasattr(solver_result, "dtype"):
                            compact_summary["dtype"] = str(solver_result.dtype)

                    benchmark_result["result"] = compact_summary
                    benchmark_result["result_summary"] = compact_summary
                    # Always store validation_result
                    benchmark_result["validation_result"] = validation_result
                else:
                    # Either we do not have a result or no is_solution available – mark skipped
                    placeholder_summary = {
                        "__stripped__": True,
                        "type": str(type(benchmark_result.get("result"))),
                        "original_type": str(type(benchmark_result.get("result"))),
                        "validation_skipped": True,
                        "reason": "result_unavailable_for_validation" if benchmark_result.get("result") is None else "no_is_solution_method",
                    }
                    benchmark_result["result"] = placeholder_summary
                    benchmark_result["result_summary"] = placeholder_summary
                    benchmark_result["validation_result"] = {
                        "success": True,
                        "skipped": True,
                        "reason": placeholder_summary["reason"],
                    }
        except Exception as e:
            logging.error(f"In-process validation failed: {e}", exc_info=True)
            validation_result = {
                "success": False,
                "error": f"In-process validation error: {str(e)}",
                "error_type": "validation_wrapper_error",
            }
            benchmark_result["validation_result"] = validation_result
        
        # --- Processing the result from run_benchmark --- 
        # run_benchmark now returns stats in seconds
        
        # Convert seconds results to milliseconds for this function's output format
        factor = 1000.0
        mean_time_ms = None
        median_time_ms = None
        stdev_time_ms = None
        all_times_ms = []
        elapsed_ms = 0 # Default value

        if benchmark_result.get("success"):
            # LOG: Check what benchmark_result contains
            timing_fields = {k: benchmark_result.get(k) for k in ["mean", "min", "max", "stddev", "values"]}
            logging.info(f"TIMING_DEBUG: benchmark_result timing fields for {func_name}: {timing_fields}")
            mean_time_s = benchmark_result.get("mean")
            min_time_s = benchmark_result.get("min") # Get min time
            stdev_time_s = benchmark_result.get("stddev")
            all_times_s = benchmark_result.get("values", []) # Returns values in seconds

            # Convert to ms
            if mean_time_s is not None: mean_time_ms = mean_time_s * factor
            if min_time_s is not None: min_time_ms = min_time_s * factor # Convert min to ms
            if stdev_time_s is not None: stdev_time_ms = stdev_time_s * factor
            all_times_ms = [t * factor for t in all_times_s]
            
            # LOG: Check converted ms values
            ms_values = {"mean_time_ms": mean_time_ms, "min_time_ms": min_time_ms, "stdev_time_ms": stdev_time_ms}
            logging.info(f"TIMING_DEBUG: converted ms values for {func_name}: {ms_values}")

            # Use minimum time (in ms) as the primary elapsed time indicator
            elapsed_ms = min_time_ms
            if elapsed_ms is None:
                 elapsed_ms = mean_time_ms  # fallback to mean
            if elapsed_ms is None:
                 # If still None, fallback to overall mean of all runs
                 if all_times_ms:
                      elapsed_ms = statistics.mean(all_times_ms)
                 else:
                      logging.warning(f"Could not determine elapsed time for {func_name} from benchmark results.")
                      elapsed_ms = 0
            
            # Ensure elapsed_ms is non-negative
            elapsed_ms = max(0, elapsed_ms if elapsed_ms is not None else 0)
            # If median_time_ms was None (e.g., single-run), fallback to elapsed_ms
            if min_time_ms is None:
                min_time_ms = elapsed_ms
            
            result_data = {
                "success": True,
                "result": benchmark_result.get("result"), 
                "stdout": benchmark_result.get("stdout", ""), 
                "stderr": benchmark_result.get("stderr", ""), 
                "min_time_ms": min_time_ms, # Add min_time_ms
                "mean_time_ms": mean_time_ms,
                "stdev_time_ms": stdev_time_ms,
                "all_times_ms": all_times_ms,
                "loops_per_run": 1, 
                "num_runs": benchmark_result.get("runs", 0), 
                "elapsed_ms": elapsed_ms, 
                "error": None,
                "traceback": None,
                "timeout_occurred": benchmark_result.get("timeout_occurred", False), 
                "error_type": None, 
                "first_warmup_result": benchmark_result.get("first_warmup_result"),
                "validation_result": benchmark_result.get("validation_result"),
                "stdout": benchmark_result.get("stdout", ""),
                "stderr": benchmark_result.get("stderr", ""),
            }
        else:
            # Handle failure reported by run_benchmark (could be error, timeout, or OOM)
            original_error_msg = benchmark_result.get("error", "Unknown benchmark failure") # Keep original message
            tb_str = benchmark_result.get("traceback")
            timeout_occurred = benchmark_result.get("timeout_occurred", False)
            oom_detected = benchmark_result.get("oom_detected", False)
            exit_code = benchmark_result.get("exit_code")
            # Determine error type based on what happened
            if oom_detected:
                error_type = "oom_kill"
                logging.error(f"OOM kill detected for {func_name}: Exit code={exit_code}, Error={original_error_msg}")
            elif timeout_occurred:
                error_type = "timeout_error"
                logging.warning(f"Timeout occurred for {func_name}: Error={original_error_msg}")
            else:
                error_type = "benchmark_error"
                logging.warning(f"Benchmark error for {func_name}: Error={original_error_msg}")

            logging.info(f"Benchmark run failed for {func_name}: Type={error_type}, OOM={oom_detected}, Timeout={timeout_occurred}, Error={original_error_msg}")

            # Try to get more context, but prioritize original message for timeout
            error_info = create_standard_error_result(
                exception=None,
                traceback_str=tb_str,
                error_type_override=error_type,
                default_error_msg=original_error_msg, # Pass original msg as default
                stdout=benchmark_result.get("stdout", ""),
                stderr=benchmark_result.get("stderr", "")
            )
            
            # --- FIX: Ensure the original error message is preserved, especially for timeouts and OOM --- 
            # If the error_info doesn't have a meaningful error message, fall back to the original.
            # Also, explicitly use the original message if it was a timeout or OOM, as context extraction is unlikely.
            final_error_msg = error_info.get("error")
            if timeout_occurred or oom_detected or not final_error_msg or str(final_error_msg).strip().lower() == "none":
                final_error_msg = original_error_msg
            # --- END FIX --- 

            # Populate result dictionary, merging the standardized error info
            result_data = {
                "success": False, 
                "result": None, 
                "min_time_ms": None, # Add min_time_ms here as well for consistency on failure
                "mean_time_ms": None,
                "stdev_time_ms": None, 
                "all_times_ms": [], 
                "loops_per_run": 1,
                "num_runs": benchmark_result.get("runs", 0), # Might be > 0 if error occurred mid-run
                "elapsed_ms": 0, # Default elapsed on failure
                "timeout_occurred": timeout_occurred,
                "oom_detected": oom_detected,
                "exit_code": exit_code,
                "first_warmup_result": benchmark_result.get("first_warmup_result"),
                **error_info, # Merge standardized fields first
                "error": final_error_msg, # Explicitly set the final error message
                "stdout": benchmark_result.get("stdout", ""),
                "stderr": benchmark_result.get("stderr", ""),
            }

        return result_data
        
    except Exception as e:
        # Catch errors within this wrapper function itself
        tb_str = traceback.format_exc()
        logging.error(f"Internal error in _run_benchmark wrapper for {func_name}: {e}", exc_info=False)
        # Use create_standard_error_result for consistency
        error_info = create_standard_error_result(
            exception=e,
            traceback_str=tb_str,
            error_type_override="benchmark_wrapper_error",
            default_error_msg=f"Internal wrapper error in _run_benchmark for {func_name}"
        )
        # Also return None for first warmup result in case of wrapper error
        error_info["first_warmup_result"] = None 
        return {
            "success": False, "result": None, "stdout": "", "stderr": "", 
            "min_time_ms": None, # Add min_time_ms here as well for consistency on failure
            "mean_time_ms": None, "stdev_time_ms": None, 
            "all_times_ms": [], "loops_per_run": 1, "num_runs": 0, 
            "elapsed_ms": 0, "timeout_occurred": False,
            "first_warmup_result": None,
            **error_info # Merge standard error fields (error, traceback, error_type)
        }


# Define a custom exception type for validation errors
class ValidationException(Exception):
    """Exception raised when validation fails due to an error during is_solution execution."""
    def __init__(self, message, traceback=None, solution_type=None, solution_shape=None):
        self.message = message
        self.traceback = traceback
        self.solution_type = solution_type 
        self.solution_shape = solution_shape
        super().__init__(self.message)

# Add a utility function to check solution using is_solution directly
def _validate_solution(task_instance: Any, problem: Any, solution: Any) -> Dict[str, Any]:
    """
    Validate a solution against a problem using the task's is_solution method.
    
    Args:
        task_instance: Task instance with an is_solution method
        problem: Problem instance to validate against
        solution: Solution to validate
        
    Returns:
        Dict with validation results including:
            - success: bool indicating if solution is valid
            - error: Optional error message (if any)
            - error_type: Type of error (if any)
            - etc.
    """
    try:
        # Call the task's is_solution method to validate the solution
        task_has_is_solution = hasattr(task_instance, 'is_solution') and callable(getattr(task_instance, 'is_solution'))
        if not task_has_is_solution:
            return {
                'success': False,
                'is_critical_validation_error': True,
                'error': 'Task has no is_solution method',
                'error_type': 'validation_error'
            }
        
        # Call is_solution
        logging.debug(f"[VALIDATION_DEBUG] Calling task.is_solution with problem type={type(problem)}, solution type={type(solution)}")
        if hasattr(solution, 'keys'):
            logging.debug(f"[VALIDATION_DEBUG] Solution is dict with keys: {list(solution.keys())}")
        elif hasattr(solution, '__len__'):
            try:
                logging.debug(f"[VALIDATION_DEBUG] Solution has length: {len(solution)}")
            except:
                pass
        
        is_valid = task_instance.is_solution(problem, solution)
        logging.debug(f"[VALIDATION_DEBUG] task.is_solution returned: {is_valid} (type: {type(is_valid)})")
        
        # Handle the results
        if not is_valid:
            logging.info(f"Solution explicitly marked as invalid by is_solution.")
            
            # Immediately capture is_solution context when is_solution returns False
            try:
                from AlgoTuner.utils.evaluator.failure_analyzer import trace_is_solution_failure
                if not hasattr(task_instance, '_last_is_solution_failure_context'):
                    logging.info("Capturing is_solution failure context in validation phase")
                    trace_is_solution_failure(task_instance, problem, solution)
            except Exception as e:
                logging.warning(f"Failed to capture is_solution failure context: {e}")
            
            # Create the validation result with is_solution context if available
            validation_result = {
                'success': False,
                'error_type': 'invalid_solution'
            }
            
            # Include is_solution failure context if available
            if hasattr(task_instance, '_last_is_solution_failure_context'):
                context = task_instance._last_is_solution_failure_context
                validation_result["code_context"] = context
                logging.info(f"Including is_solution failure context in validation result (length: {len(context)})")
            else:
                logging.warning("No is_solution failure context available after validation failure")
            
            return validation_result
            
        return {
            'success': True
        }
    except Exception as e:
        # === COMPREHENSIVE DIAGNOSTIC LOGGING ===
        logging.debug(f"[VALIDATION_DEBUG] Raw exception occurred: {e}")
        logging.debug(f"[VALIDATION_DEBUG] Exception type: {type(e)}")
        logging.debug(f"[VALIDATION_DEBUG] Exception str(): '{str(e)}'")
        logging.debug(f"[VALIDATION_DEBUG] Exception repr(): {repr(e)}")
        if hasattr(e, 'args'):
            logging.debug(f"[VALIDATION_DEBUG] Exception args: {e.args}")
        
        # Log validation input details
        logging.debug(f"[VALIDATION_DEBUG] Problem type: {type(problem)}")
        logging.debug(f"[VALIDATION_DEBUG] Solution type: {type(solution)}")
        if hasattr(solution, '__len__'):
            try:
                solution_len = len(solution)
                logging.debug(f"[VALIDATION_DEBUG] Solution length: {solution_len}")
            except:
                logging.debug(f"[VALIDATION_DEBUG] Solution length check failed")
        
        if hasattr(solution, 'shape'):
            logging.debug(f"[VALIDATION_DEBUG] Solution shape: {solution.shape}")
        
        # Try to get first few elements/keys to understand structure
        try:
            if hasattr(solution, 'keys'):
                keys = list(solution.keys())[:5]
                logging.debug(f"[VALIDATION_DEBUG] Solution keys (first 5): {keys}")
            elif hasattr(solution, '__getitem__'):
                logging.debug(f"[VALIDATION_DEBUG] Attempting solution[0]...")
                first_item = solution[0]
                logging.debug(f"[VALIDATION_DEBUG] solution[0] = {type(first_item)}: {first_item}")
        except Exception as access_error:
            logging.debug(f"[VALIDATION_DEBUG] Failed to access solution[0]: {type(access_error)}: {access_error}")
        
        # === END DIAGNOSTIC LOGGING ===
        
        logging.error(f"Unexpected error during _validate_solution call: {e}", exc_info=True)
        tb = traceback.format_exc()
        
        # Enhance error message with context about the validation inputs
        enhanced_error_msg = f"Error: {type(e).__name__}: {str(e)}"
        
        # Note: Validation context section removed as requested
        logging.info(f"Validation error: {enhanced_error_msg[:200]}...")
        
        result = create_standard_error_result(
            exception=e,
            traceback_str=tb,
            error_type_override="validation_wrapper_error",
            default_error_msg="Unexpected error calling validation function"
        )
        
        # Override the error message with enhanced version
        result["error"] = enhanced_error_msg
        
        # For validation errors, try to get code context from the task file where error occurred
        # This uses the same mechanism as other error context extraction
        try:
            from AlgoTuner.utils.error_utils import extract_error_context
            error_context = extract_error_context(tb, str(e))
            if error_context.get("code_context_snippet"):
                result["code_context"] = error_context["code_context_snippet"]
                logging.info(f"Added task code context to validation error: {len(error_context['code_context_snippet'])} chars")
            else:
                logging.info("No code context found for validation error")
        except Exception as context_error:
            logging.warning(f"Failed to extract code context for validation error: {context_error}")
        
        result['failed_solution_type'] = str(type(solution))
        result['failed_solution_shape'] = format_object_shape(solution)

        return result


# Rename the existing run_evaluation to run_solver_evaluation to avoid conflicts
# while keeping all the improved error handling
def run_solver_evaluation(
    problem: Any,
    task_instance: Any,
    oracle_time_ms: Optional[float] = None,
    capture_output: bool = False,
    needs_casting: bool = False, # needs_casting is handled in run_evaluation now
    num_runs: int = TEST_RUNS,
    warmup_runs: int = TEST_WARMUPS,
    warmup_problem: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the solver on a problem using _run_benchmark and return the result.
    
    The solver function is run with consistent error handling and detailed diagnostics.
    
    Args:
        problem: The problem to solve (potentially already cast)
        task_instance: The task instance
        oracle_time_ms: Oracle solve time in milliseconds for this problem (used for timeout calculation)
        capture_output: Whether to capture stdout/stderr during benchmark
        needs_casting: Whether the result needs to be cast (DISABLED - casting now skipped)
        num_runs: Number of timed runs to pass to _run_benchmark.
        warmup_runs: Number of warmup runs to pass to _run_benchmark.
        warmup_problem: Optional warmup problem for isolated benchmark
        
    Returns:
        Dict containing benchmark results or error information.
    """
    # Casting is now disabled to avoid Dict[str, Any] errors
    from AlgoTuner.utils.type_inspection import describe_type
    
    # Allow oracle_time_ms to be None for baseline mode fixed timeout
    # if oracle_time_ms is None:
    #     raise ValueError("oracle_time_ms must be provided to run_solver_evaluation")
    
    logging.debug(f"Running solver evaluation on problem type: {type(problem)}")
    
    # Input validation (problem should not be None)
    if problem is None:
        # Return standardized error format
        return create_standard_error_result(
            exception=ValueError("Input problem cannot be None."),
            traceback_str=None,
            error_type_override="input_error",
            default_error_msg="Error: invalid input. Input cannot be None."
        )

    t_start_solver_eval = time.perf_counter_ns()
    t_last = t_start_solver_eval

    # Extract task_name for isolated benchmark - needed for both baseline and agent mode
    task_name = getattr(task_instance, 'task_name', None)
    if not task_name:
        try:
            from AlgoTuneTasks.base import TASK_REGISTRY
            for name, cls in TASK_REGISTRY.items():
                if isinstance(task_instance, cls):
                    task_name = name
                    break
        except Exception:
            pass
    if not task_name:
        task_name = "unknown_task"
    
    # Set task name in environment for isolated benchmark
    os.environ["CURRENT_TASK_NAME"] = task_name

    # Determine solver callable based on AGENT_MODE
    agent_mode = os.environ.get("AGENT_MODE", "0")
    solver_callable = None
    solver_description = "unknown"
    solver_module = None  # Initialize for cache clearing

    if agent_mode == "0":
        # Baseline run: Use the oracle's solve method directly
        logging.info("AGENT_MODE=0 detected, using task_instance.solve for baseline evaluation.")
        if not hasattr(task_instance, "solve") or not callable(getattr(task_instance, "solve")):
             # This should ideally not happen if task loading succeeded, but check defensively
             user_facing_msg = "Task instance is missing a callable 'solve' method for baseline run."
             logging.error(user_facing_msg)
             # Create an error result, similar to how missing Solver.solve is handled
             return create_standard_error_result(
                 exception=AttributeError(user_facing_msg),
                 traceback_str=None,
                 error_type_override="missing_oracle_solve_method",
                 default_error_msg=user_facing_msg
             )
        # Assign the oracle's solve method directly
        solver_callable = task_instance.solve 
        solver_description = "oracle (task_instance.solve)"
        t_now = time.perf_counter_ns()
        logging.info(f"Solver Eval Timing: Baseline setup took {(t_now - t_last) / 1e6:.2f}ms")
        t_last = t_now
        # Use the task's directory as working_dir for baseline
        working_dir = getattr(task_instance, 'data_dir', None) or getattr(task_instance, 'get_task_directory', lambda: None)()
    else:
        # AGENT_MODE=1: Dynamically load solver
        logging.info(f"AGENT_MODE=1 detected, loading solver from CODE_DIR.")
        try:
            from AlgoTuner.utils.solver_loader import locate_solver_file, load_solver_module, get_fresh_solve_callable, get_fresh_solve_callable_with_module_reload
            
            code_dir = os.getenv("CODE_DIR", "llm_src")
            
            t_locate_start = time.perf_counter_ns()
            solver_file_path = locate_solver_file(task_name=task_name, code_dir=code_dir)
            t_locate_end = time.perf_counter_ns()
            logging.info(f"Solver Eval Timing: locate_solver_file took {(t_locate_end - t_locate_start) / 1e6:.2f}ms")
            
            t_load_start = time.perf_counter_ns()
            # Pass the parent directory and the filename separately
            solver_module = load_solver_module(solver_file_path.parent, solver_file_path.name)
            t_load_end = time.perf_counter_ns()
            logging.info(f"Solver Eval Timing: load_solver_module took {(t_load_end - t_load_start) / 1e6:.2f}ms")

            t_get_start = time.perf_counter_ns()
            # Use regular fresh callable - module reloading will happen between timing runs in precise_timing.py
            solver_callable = get_fresh_solve_callable(solver_module)
            t_get_end = time.perf_counter_ns()
            logging.info(f"Solver Eval Timing: get_fresh_solve_callable took {(t_get_end - t_get_start) / 1e6:.2f}ms")

            solver_description = f"solver ({solver_file_path.name})"
            logging.info(f"Successfully loaded solver callable from {solver_file_path}")
            t_now = time.perf_counter_ns()
            logging.info(f"Solver Eval Timing: Dynamic load took {(t_now - t_last) / 1e6:.2f}ms total")
            t_last = t_now
            # Use the solver file's parent as working_dir
            working_dir = str(solver_file_path.parent)
        except Exception as load_error:
            raw_tb = traceback.format_exc()
            err_ctx = extract_error_context(raw_tb, "")
            tb = err_ctx.get("enhanced_error_message", raw_tb)
            if err_ctx.get("code_context_snippet"):
                tb += "\n\nCode Context:\n" + err_ctx["code_context_snippet"]
            user_facing_msg = f"Failed to load solver module: {load_error}"
            logging.error(f"{user_facing_msg}\n{tb}")
            return create_standard_error_result(
                exception=load_error,
                traceback_str=tb,
                error_type_override="solver_load_error",
                default_error_msg=user_facing_msg
            )

    # Check if solver callable was successfully obtained
    if solver_callable is None:
        # This case should ideally be caught by the load_error exception handler,
        # but handle defensively.
        missing_callable_msg = "Solver callable could not be determined (check loading logs)."
        logging.error(missing_callable_msg)
        return create_standard_error_result(
            exception=RuntimeError(missing_callable_msg), # Generic error if logic missed specifics
            traceback_str=None,
            error_type_override="missing_solver_function",
            default_error_msg=missing_callable_msg
        )

    # ---------------------------------------------------------------------
    # Decide which benchmarking scheme to use early to adjust timeout calculation
    # ---------------------------------------------------------------------
    agent_mode = os.environ.get("AGENT_MODE", "0")
    use_isolated = (agent_mode != "0") and (os.environ.get("ISOLATED_EVAL", "1") == "1")

    # ---------------------------------------------------------------------
    # Compute timeout once using the same rule as _run_benchmark so we can
    # pass it to run_benchmark irrespective of the execution branch below.
    # ---------------------------------------------------------------------

    timeout_seconds_unified = _calculate_timeout_seconds(
        oracle_time_ms,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        # Preserve the historical 60 s fallback when no baseline is known.
        min_timeout_s=60.0 if oracle_time_ms is None else 10.0,
    )
    
    # For isolated benchmarking, we need to adjust the timeout calculation
    # because each subprocess does warmup + timed call, but the above calculation
    # assumes all runs are sequential in the same process
    if use_isolated:
        # Each subprocess needs timeout for: 1 warmup + 1 timed call
        # So we need: 2 * baseline_time * 10x_multiplier per subprocess
        if oracle_time_ms is not None and oracle_time_ms > 0:
            per_run_s = oracle_time_ms / 1000.0
            # Strict 10× baseline (warm-up + timed = 2 runs) – no extra validation factor
            timeout_seconds_unified = (
                (1 + WARMUP_MULTIPLIER) * per_run_s * TARGET_TIME_MULTIPLIER
            )

            logging.info(
                "TIMEOUT_DEBUG (isolated): per_run_s=%.4fs, factor=(1+%d)*%g -> %.2fs",
                per_run_s,
                WARMUP_MULTIPLIER,
                TARGET_TIME_MULTIPLIER,
                timeout_seconds_unified,
            )
        else:
            # Fallback timeout for isolated benchmark when no baseline
            timeout_seconds_unified = 60.0
            logging.info(f"Using fallback timeout for isolated benchmark: {timeout_seconds_unified:.2f}s per subprocess")

    # Run the benchmark using the determined callable (either task_instance.solve or Solver().solve)
    logging.info(f"Benchmarking function: {solver_description}")
    # Run benchmark in CODE_DIR so solver artifacts (like .dace, .mps, .sol files) go there
    # Change to CODE_DIR where the solver code is located
    code_dir = os.environ.get("CODE_DIR", "llm_src")
    if code_dir:
        os.makedirs(code_dir, exist_ok=True)
        old_cwd = os.getcwd()
        try:
            os.chdir(code_dir)
            
            warmup_obj = warmup_problem if warmup_problem is not None else problem

            if use_isolated:
                logging.info("Using per-process isolated benchmark scheme (1 warm-up + 1 timed run per process)")
                # For isolated benchmark, use the task-specific directory if it exists
                task_code_dir = os.path.join(code_dir, "AlgoTuneTasks", task_name)
                if os.path.isdir(task_code_dir):
                    isolated_code_dir = task_code_dir
                else:
                    isolated_code_dir = code_dir
                
                benchmark_result = run_isolated_benchmark(
                    task_name=task_name,
                    code_dir=isolated_code_dir,
                    warmup_problem=warmup_obj,
                    timed_problem=problem,
                    num_runs=1,
                    timeout_seconds=timeout_seconds_unified,
                )
                logging.info(f"[RUNNER_DEBUG] run_isolated_benchmark returned: success={benchmark_result.get('success')}, keys={list(benchmark_result.keys())}")
                
                # For validation, run solver once more in main process to get result
                if benchmark_result.get("success"):
                    try:
                        logging.info("[ISOLATED_VALIDATION] Running solver once more for validation")
                        solver_result_for_validation = solver_callable(problem)
                        benchmark_result["result"] = solver_result_for_validation
                        logging.info("[ISOLATED_VALIDATION] Successfully captured solver result for validation")
                    except Exception as e:
                        logging.warning(f"[ISOLATED_VALIDATION] Failed to get solver result for validation: {e}")
                        benchmark_result["result"] = None
            else:
                # Use isolated benchmark for consistency
                task_code_dir = os.path.join(code_dir, "AlgoTuneTasks", task_name)
                if os.path.isdir(task_code_dir):
                    isolated_code_dir = task_code_dir
                else:
                    isolated_code_dir = code_dir
                
                benchmark_result = run_isolated_benchmark(
                    task_name=task_name,
                    code_dir=isolated_code_dir,
                    warmup_problem=problem,
                    timed_problem=problem,
                    num_runs=1,
                    timeout_seconds=timeout_seconds_unified,
                )
        finally:
            os.chdir(old_cwd)
    else:
        use_isolated = (agent_mode != "0") and (os.environ.get("ISOLATED_EVAL", "1") == "1")

        if use_isolated:
            logging.info("Using per-process isolated benchmark scheme (env branch, fixed 5 runs)")
            benchmark_result = run_isolated_benchmark(
                task_name=task_name,
                code_dir=os.getcwd(),
                warmup_problem=warmup_obj,
                timed_problem=problem,
                num_runs=5,
                timeout_seconds=timeout_seconds_unified,
            )
            logging.info(f"[RUNNER_DEBUG] run_isolated_benchmark returned: success={benchmark_result.get('success')}, keys={list(benchmark_result.keys())}")
            
            # For validation, run solver once more in main process to get result
            if benchmark_result.get("success"):
                try:
                    logging.info("[ISOLATED_VALIDATION] Running solver once more for validation")
                    solver_result_for_validation = solver_callable(problem)
                    benchmark_result["result"] = solver_result_for_validation
                    logging.info("[ISOLATED_VALIDATION] Successfully captured solver result for validation")
                except Exception as e:
                    logging.warning(f"[ISOLATED_VALIDATION] Failed to get solver result for validation: {e}")
                    benchmark_result["result"] = None
        else:
            # Use isolated benchmark for consistency
            task_code_dir = os.path.join(code_dir, "AlgoTuneTasks", task_name)
            if os.path.isdir(task_code_dir):
                isolated_code_dir = task_code_dir
            else:
                isolated_code_dir = code_dir
            
            benchmark_result = run_isolated_benchmark(
                task_name=task_name,
                code_dir=isolated_code_dir,
                warmup_problem=problem,
                timed_problem=problem,
                num_runs=1,
                timeout_seconds=timeout_seconds_unified,
            )
    logging.info(f"[RUNNER_DEBUG] About to process benchmark_result")
    t_now = time.perf_counter_ns()
    logging.info(
        f"Solver Eval Timing: run_benchmark took {(t_now - t_last) / 1e6:.2f}ms"
    )
    t_last = t_now

    # --- Final Result Processing ---
    # Combine original problem info (or casted version) with benchmark results
    # Ensure elapsed_ms and other critical fields are present
    
    # Extract timing from new benchmark format
    elapsed_ms = benchmark_result.get("elapsed_ms")  # Check if already exists
    
    timing_fields = ["elapsed_ms", "min_time_ms", "mean_ms", "median_ms", "min", "mean", "median", "stddev", "values", "min_ns", "mean_ns", "median_ns", "values_ns", "runs", "success", "error", "timeout_occurred", "oom_detected"]
    available_timing = {field: benchmark_result.get(field) for field in timing_fields}
    logging.info(f"SOLVER_EVALUATION_TIMING_DEBUG: ALL benchmark_result fields: {available_timing}")
    
    # Log benchmark success status specifically
    benchmark_success = benchmark_result.get("success")
    logging.info(f"SOLVER_EVALUATION_TIMING_DEBUG: benchmark_result success = {benchmark_success} (type: {type(benchmark_success)})")
    
    if elapsed_ms is None:
        # Extract from new benchmark result format - prefer min time for consistency
        min_time_ms = benchmark_result.get("min_time_ms")
        mean_ms = benchmark_result.get("mean_ms")
        if min_time_ms is not None:
            elapsed_ms = min_time_ms
            logging.info(f"Solver evaluation: using min_time_ms={min_time_ms} as elapsed_ms")
        elif mean_ms is not None:
            elapsed_ms = mean_ms
            logging.info(f"Solver evaluation: using mean_ms={mean_ms} as elapsed_ms")
        else:
            # Fallback: convert from nanoseconds if available (more likely to exist)
            min_ns = benchmark_result.get("min_ns")
            mean_ns = benchmark_result.get("mean_ns")
            if min_ns is not None:
                elapsed_ms = min_ns / 1e6  # Convert ns to ms
                logging.info(f"Solver evaluation: converted min_ns={min_ns if min_ns is not None else 'None'} to elapsed_ms={elapsed_ms if elapsed_ms is not None else 'None'}")
            elif mean_ns is not None:
                elapsed_ms = mean_ns / 1e6  # Convert ns to ms
                logging.info(f"Solver evaluation: converted mean_ns={mean_ns if mean_ns is not None else 'None'} to elapsed_ms={elapsed_ms if elapsed_ms is not None else 'None'}")
            else:
                # Last fallback: convert from seconds
                min_s = benchmark_result.get("min")
                mean_s = benchmark_result.get("mean")
                if min_s is not None:
                    elapsed_ms = min_s * 1000.0
                    logging.info(f"Solver evaluation: converted min={min_s}s to elapsed_ms={elapsed_ms}")
                elif mean_s is not None:
                    elapsed_ms = mean_s * 1000.0
                    logging.info(f"Solver evaluation: converted mean={mean_s}s to elapsed_ms={elapsed_ms}")
                else:
                    elapsed_ms = 0.0
                    logging.warning("Solver evaluation: no timing information found, using elapsed_ms=0.0")
    else:
        logging.info(f"Solver evaluation: found existing elapsed_ms={elapsed_ms}")
    
    # Log what elapsed_ms value we're about to use
    logging.info(f"SOLVER_EVALUATION_TIMING_DEBUG: About to set final_result elapsed_ms = {elapsed_ms} (type: {type(elapsed_ms)})")
    
    final_result = {
        **benchmark_result,
        "problem": problem,
        "elapsed_ms": elapsed_ms,
        "min_time_ms": benchmark_result.get("min_time_ms"),
        "oracle_time_ms": oracle_time_ms # Pass through the oracle time used for timeout
    }
    
    # Log what ended up in final_result
    logging.info(f"SOLVER_EVALUATION_TIMING_DEBUG: final_result elapsed_ms = {final_result.get('elapsed_ms')} (type: {type(final_result.get('elapsed_ms'))})")
    
    # Ensure elapsed_ms is float, as 0 could be int
    if not isinstance(final_result["elapsed_ms"], float) and final_result["elapsed_ms"] is not None:
        try:
            final_result["elapsed_ms"] = float(final_result["elapsed_ms"])
            logging.info(f"SOLVER_EVALUATION_TIMING_DEBUG: Converted elapsed_ms to float: {final_result['elapsed_ms']}")
        except (ValueError, TypeError):
            logging.warning(f"Could not convert elapsed_ms ({final_result['elapsed_ms']}) to float, defaulting to 0.0")
            final_result["elapsed_ms"] = 0.0
    elif final_result["elapsed_ms"] is None: # If get("elapsed_ms") returned None
         logging.warning(f"SOLVER_EVALUATION_TIMING_DEBUG: elapsed_ms was None, setting to 0.0")
         final_result["elapsed_ms"] = 0.0

    logging.info(f"SOLVER_EVALUATION_TIMING_DEBUG: FINAL elapsed_ms = {final_result['elapsed_ms']} (type: {type(final_result['elapsed_ms'])})")
    logging.info(f"Solver Eval Timing: Total run_solver_evaluation took {(time.perf_counter_ns() - t_start_solver_eval) / 1e6:.2f}ms. Using elapsed_ms from benchmark_result: {final_result['elapsed_ms']}")
    return _strip_bulky_fields(final_result)

# Now create the run_evaluation function that matches the original interface
# and routes to our enhanced implementation
def run_evaluation(
    problem: Any,
    task_instance: Any,
    capture_output: bool = False,
    needs_casting: bool = False,
    num_runs: int = TEST_RUNS,
    warmup_runs: int = TEST_WARMUPS
) -> Dict[str, Any]:
    """
    Run evaluation including input casting, solver execution, output extraction,
    output casting (optional), and validation. Calculates oracle time dynamically.
    
    Args:
        problem: The raw problem instance
        task_instance: The task instance
        capture_output: Whether to capture stdout/stderr during solver execution
        needs_casting: Whether to cast input and result types (DISABLED - casting now skipped)
        num_runs: Number of timed runs for the solver benchmark.
        warmup_runs: Number of warmup runs for the solver benchmark.
        
    Returns:
        Dict with evaluation results, including success status, result/error info,
        solver timing, calculated oracle timing, and potentially captured output.
    """
    processed_problem = problem
    input_casting_error_result = None

    # ===== SKIP INPUT CASTING (BEFORE ORACLE RUN) =====
    # Casting is disabled to avoid Dict[str, Any] errors
    # if needs_casting:
    #    try:
    #        processed_problem = cast_input(problem, task_instance)
    # ===== END INPUT CASTING =====

    # ===== START ORACLE BENCHMARKING =====
    logging.info("Starting oracle benchmark...")
    # Use DATASET runs/warmups for oracle benchmark consistency with dataset creation
    # Oracle callable that matches agent overhead (new Task instance each call)
    def _fresh_oracle_callable(problem_inner):
        try:
            # Create new instance with same parameters as original
            inst = task_instance.__class__(
                n=getattr(task_instance, 'n', None),
                dataset_size=getattr(task_instance, 'dataset_size', None),
                target_time_ms=getattr(task_instance, 'target_time_ms', None),
                data_dir=getattr(task_instance, 'data_dir', None)
            )
            # Copy essential attributes
            if hasattr(task_instance, 'task_name'):
                inst.task_name = task_instance.task_name
            if hasattr(task_instance, 'k'):
                inst.k = task_instance.k
        except Exception:
            # Fallback: reuse original instance (no new instantiation)
            inst = task_instance
        try:
            return inst.solve(problem_inner)
        finally:
            if inst is not task_instance:  # Only delete if it's a new instance
                del inst
            gc.collect()

    # Override: if a custom solver.py exists in CODE_DIR, use it for oracle evaluation
    try:
        from AlgoTuner.utils.solver_loader import locate_solver_file, load_solver_module, get_fresh_solve_callable
        solver_file_path = locate_solver_file(task_name=task_name, code_dir=code_dir)
        solver_module = load_solver_module(solver_file_path.parent, solver_file_path.name)
        oracle_callable_to_use = get_fresh_solve_callable(solver_module)
        logging.info(f"Oracle override: using solver from {solver_file_path}")
    except Exception:
        # Fallback to default Task-based callable
        agent_mode = os.environ.get("AGENT_MODE", "0")
        oracle_callable_to_use = task_instance.solve if agent_mode == "0" else _fresh_oracle_callable

    # ------------------------------------------------------------------
    #  FAST-PATH TIMING
    #  For reference (oracle) evaluation we do not need the heavyweight
    #  cache-clearing / memory-monitor / sub-process machinery.  When the
    #  caller requests ≤3 measurement runs and ≤1 warm-up run we instead
    #  perform a minimal timing loop around the callable.  This slashes the
    #  overhead from ~150 ms to <0.5 ms, yielding the sub-millisecond numbers
    #  we expect for trivial problems.
    # ------------------------------------------------------------------

    # DISABLED: Always use isolated benchmark for consistency with solver evaluation
    fast_path = False  # was: (num_runs <= 3 and warmup_runs <= 1 and not capture_output)

    if fast_path:
        logging.info("Oracle Eval: using lightweight timing fast-path")

        # Optional single warm-up (not timed)
        if warmup_runs:
            try:
                oracle_callable_to_use(processed_problem)
            except Exception as warm_err:
                tb = traceback.format_exc()
                return create_standard_error_result(
                    exception=warm_err,
                    traceback_str=tb,
                    error_type_override="oracle_warmup_error",
                    default_error_msg=str(warm_err)
                )

        times_ns: list[int] = []
        result_value = None
        for _ in range(num_runs):
            t0 = time.perf_counter_ns()
            result_value = oracle_callable_to_use(processed_problem)
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)

        # Basic statistics
        min_ns = min(times_ns)
        mean_ns = statistics.mean(times_ns)

        benchmark_result = {
            "success": True,
            "values_ns": times_ns,
            "num_runs_executed": num_runs,
            "min_ns": min_ns,
            "mean_ns": mean_ns,
            "min_time_ms": min_ns / 1e6,
            "mean_ms": mean_ns / 1e6,
            "elapsed_ms": min_ns / 1e6,
            "result": result_value,
            "stdout": "",
            "stderr": "",
        }
    else:
        # Benchmark the oracle's solve method using full machinery
        try:
            solution_method = task_instance.solve  # noqa: F841  # may be useful for future hooks
            timing = TimingManager()
            with timing.phase(Phase.ORACLE_RUN, capture_output=capture_output):
                # Use isolated benchmark for consistency
                task_code_dir = os.path.join(code_dir, "AlgoTuneTasks", task_name)
                if os.path.isdir(task_code_dir):
                    isolated_code_dir = task_code_dir
                else:
                    isolated_code_dir = code_dir
                
                benchmark_result = run_isolated_benchmark(
                    task_name=task_name,
                    code_dir=isolated_code_dir,
                    warmup_problem=processed_problem,
                    timed_problem=processed_problem,
                    num_runs=num_runs,
                    timeout_seconds=oracle_time_ms / 1000.0 if oracle_time_ms else 60.0,
                )

            # For validation, run oracle once more in main process to get result if using isolated benchmark
            logging.info(f"[ISOLATED_VALIDATION_DEBUG] benchmark_result success: {benchmark_result.get('success')}, has result key: {'result' in benchmark_result}, keys: {list(benchmark_result.keys())}")
            if benchmark_result.get("success") and "result" not in benchmark_result:
                try:
                    logging.info("[ISOLATED_VALIDATION] Running oracle once more for validation")
                    # Capture stdout/stderr for isolated benchmark
                    import sys
                    from io import StringIO
                    
                    # Capture stdout/stderr during oracle execution
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    captured_stdout = StringIO()
                    captured_stderr = StringIO()
                    
                    try:
                        sys.stdout = captured_stdout
                        sys.stderr = captured_stderr
                        oracle_result_for_validation = oracle_callable_to_use(processed_problem)
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    # Add captured output to benchmark_result
                    benchmark_result["result"] = oracle_result_for_validation
                    benchmark_result["stdout"] = captured_stdout.getvalue()
                    benchmark_result["stderr"] = captured_stderr.getvalue()
                    
                    logging.info(f"[ISOLATED_VALIDATION] Successfully captured oracle result for validation: {type(oracle_result_for_validation)}")
                    logging.info(f"[ISOLATED_VALIDATION] Captured stdout: {len(benchmark_result['stdout'])} chars, stderr: {len(benchmark_result['stderr'])} chars")
                except Exception as e:
                    logging.warning(f"[ISOLATED_VALIDATION] Failed to get oracle result for validation: {e}")
                    benchmark_result["result"] = None
                    benchmark_result["stdout"] = ""
                    benchmark_result["stderr"] = ""

            # Timing after successful benchmark (inside try block)
            t_now = time.perf_counter_ns()
            logging.info(
                f"Oracle Eval Timing: run_benchmark took {(t_now - t_last) / 1e6:.2f}ms"
            )
            t_last = t_now

        except Exception as bench_exc:
            raw_tb = traceback.format_exc()
            err_ctx = extract_error_context(raw_tb, "")
            tb = err_ctx.get("enhanced_error_message", raw_tb)
            if err_ctx.get("code_context_snippet"):
                tb += "\n\nCode Context:\n" + err_ctx["code_context_snippet"]
            logging.error(
                f"Oracle Eval: run_benchmark failed with {type(bench_exc).__name__}: {bench_exc}\n{tb}"
            )
            return create_standard_error_result(
                exception=bench_exc,
                traceback_str=tb,
                error_type_override="oracle_benchmark_error",
                default_error_msg=str(bench_exc),
            )

    # --- Final Result Processing ---
    # Combine original problem info (or casted version) with benchmark results
    # Ensure elapsed_ms and other critical fields are present
    
    # Extract timing from new benchmark format
    elapsed_ms = benchmark_result.get("elapsed_ms")  # Check if already exists
    
    timing_fields = ["elapsed_ms", "min_time_ms", "mean_ms", "median_ms", "min", "mean", "min_ns", "mean_ns"]
    available_timing = {field: benchmark_result.get(field) for field in timing_fields}
    logging.info(f"TIMING DEBUG: Oracle evaluation benchmark_result timing fields: {available_timing}")
    
    if elapsed_ms is None:
        # Extract from new benchmark result format - prefer min time for consistency
        min_time_ms = benchmark_result.get("min_time_ms")
        mean_ms = benchmark_result.get("mean_ms") 
        median_ms = benchmark_result.get("median_ms")
        
        # Use min time if available, otherwise mean, otherwise median
        if min_time_ms is not None:
            elapsed_ms = min_time_ms
            logging.info(f"Oracle benchmark: using min_time_ms={min_time_ms} as elapsed_ms")
        elif mean_ms is not None:
            elapsed_ms = mean_ms
            logging.info(f"Oracle benchmark: using mean_ms={mean_ms} as elapsed_ms")
        elif median_ms is not None:
            elapsed_ms = median_ms
            logging.info(f"Oracle benchmark: using median_ms={median_ms} as elapsed_ms")
        else:
            # Fallback: convert from nanoseconds if available (more likely to exist)
            min_ns = benchmark_result.get("min_ns")
            mean_ns = benchmark_result.get("mean_ns")
            if min_ns is not None:
                elapsed_ms = min_ns / 1e6  # Convert ns to ms
                logging.info(f"Oracle benchmark: converted min_ns={min_ns if min_ns is not None else 'None'} to elapsed_ms={elapsed_ms if elapsed_ms is not None else 'None'}")
            elif mean_ns is not None:
                elapsed_ms = mean_ns / 1e6  # Convert ns to ms
                logging.info(f"Oracle benchmark: converted mean_ns={mean_ns if mean_ns is not None else 'None'} to elapsed_ms={elapsed_ms if elapsed_ms is not None else 'None'}")
            else:
                # Last fallback: convert from seconds if available
                min_s = benchmark_result.get("min")
                mean_s = benchmark_result.get("mean")
                if min_s is not None:
                    elapsed_ms = min_s * 1000.0
                    logging.info(f"Oracle benchmark: converted min={min_s}s to elapsed_ms={elapsed_ms}")
                elif mean_s is not None:
                    elapsed_ms = mean_s * 1000.0
                    logging.info(f"Oracle benchmark: converted mean={mean_s}s to elapsed_ms={elapsed_ms}")
                else:
                    elapsed_ms = None
                    logging.warning("Oracle benchmark: no timing information found in any expected format")
    else:
        logging.info(f"Oracle benchmark: found existing elapsed_ms={elapsed_ms}")
            
    # Ensure elapsed_ms is included in benchmark_result for downstream processing
    if elapsed_ms is not None:
        benchmark_result["elapsed_ms"] = elapsed_ms
            
    calculated_oracle_time_ms = elapsed_ms
    logging.info(f"Oracle benchmark final elapsed_ms: {elapsed_ms}")
    if benchmark_result.get("success") and (elapsed_ms is None or elapsed_ms == 0):
         logging.warning("Oracle benchmark succeeded but elapsed_ms is None or 0. Performance measurement might be inaccurate.")

    # If benchmark itself failed (e.g., timeout, error in solve)
    if not benchmark_result.get("success", False):
        # Ensure elapsed_ms is included even on failure
        if "elapsed_ms" not in benchmark_result:
             benchmark_result["elapsed_ms"] = 0 # Or potentially a failed timing value if available
        return benchmark_result

    # If benchmark succeeded, proceed with optional casting & validation
    # Results are no longer stripped, so use them directly
    raw_oracle_output = benchmark_result.get("result")
    final_oracle_result = raw_oracle_output

    # === SKIPPING Oracle Output Casting ===
    # Assume oracle output is already in the correct format for its own is_solution.
    # The final_oracle_result remains the raw_oracle_output.
    logging.debug(f"Skipping output casting for oracle result.")
    # Log the time since the last step (benchmark)
    t_now = time.perf_counter_ns()
    logging.info(f"Oracle Eval Timing: Output Casting step skipped (took {(t_now - t_last) / 1e6:.2f}ms)")
    t_last = t_now

    # === Validation ===
    validation_result = None
    try:
        validation_result = _validate_solution(task_instance, processed_problem, final_oracle_result)
        t_now = time.perf_counter_ns()
        logging.info(f"Oracle Eval Timing: Validation took {(t_now - t_last) / 1e6:.2f}ms")
        t_last = t_now
    except Exception as e:
        # Safeguard catch block
        logging.error(f"Unexpected error during oracle _validate_solution call: {e}", exc_info=True)
        tb = traceback.format_exc()
        validation_result = create_standard_error_result(
            exception=e,
            traceback_str=tb,
            error_type_override="validation_wrapper_error",
            default_error_msg="Unexpected error calling validation function for oracle"
        )
        # Augment with context
        validation_result['failed_solution_type'] = str(type(final_oracle_result))
        validation_result['failed_solution_shape'] = format_object_shape(final_oracle_result)

    # === Final Result Combination ===
    # Handle case where validation was skipped or failed
    if not validation_result or not validation_result.get("success", False):
        # If validation failed (and wasn't skipped)
        if not validation_result.get("validation_skipped", False):
            logging.warning(f"Oracle validation failed. Type: {validation_result.get('error_type')}, Error: {validation_result.get('error')}")
            # Merge benchmark info with validation failure info
            final_evaluation_result = {
                **benchmark_result,  # preserve benchmark metrics including min_time_ms
                **validation_result,
                "success": False,
                "result": final_oracle_result,
                "raw_result": raw_oracle_output,
                "elapsed_ms": benchmark_result.get("elapsed_ms", 0),
                "oracle_time_ms": calculated_oracle_time_ms
            }
            final_evaluation_result["error_type"] = validation_result.get("error_type", "unknown_oracle_validation_failure")
            if "problem" not in final_evaluation_result:
                final_evaluation_result["problem"] = processed_problem
            return final_evaluation_result
        else:
             # Validation was skipped, treat as success for this function's return
             # Combine benchmark results with skip marker
             successful_evaluation_result = {
                 **benchmark_result,
                 "result": final_oracle_result, # Use the potentially cast value
                 "validation_passed": True,
                 "validation_skipped": True,
                 # Merge stdout/stderr from validation if captured (won't be if skipped)
                 "stdout": benchmark_result.get("stdout", ""),
                 "stderr": benchmark_result.get("stderr", ""),
             }
             # Log accurately when skipped
             logging.info("Oracle successful (validation skipped).")
             return successful_evaluation_result

    # If validation succeeded (and wasn't skipped)
    logging.info("Oracle successful and solution validated.")
    successful_evaluation_result = {
        **benchmark_result,  # preserve benchmark metrics including min_time_ms
        "result": final_oracle_result,
        "validation_passed": True,
        # Merge stdout/stderr from validation if captured
        "stdout": benchmark_result.get("stdout", "") + validation_result.get("stdout", ""),
        "stderr": benchmark_result.get("stderr", "") + validation_result.get("stderr", ""),
        "oracle_time_ms": calculated_oracle_time_ms
    }
    if "problem" not in successful_evaluation_result:
        successful_evaluation_result["problem"] = processed_problem
        
    logging.info(f"Oracle Eval Timing: Total run_oracle_evaluation took {(time.perf_counter_ns() - t_start_solver_eval) / 1e6:.2f}ms")
    return successful_evaluation_result

def run_oracle_evaluation(
    problem: Any,
    task_instance: Any,
    oracle_time_ms: Optional[float] = None, # Rename timeout_ms to align
    capture_output: bool = False,
    needs_casting: bool = False,
    num_runs: int = DATASET_RUNS,
    warmup_runs: int = DATASET_WARMUPS,
    skip_validation: bool = False, # <<< ADDED PARAMETER
) -> Dict[str, Any]:
    """
    Run evaluation of the oracle solution method (task_instance.solve).

    Includes benchmarking, optional casting, and validation.
    
    Args:
        problem: Problem instance (raw)
        task_instance: Task instance containing the 'solve' method
        oracle_time_ms: Reference time for timeout calculation in benchmark
        capture_output: Whether to capture stdout/stderr during benchmark
        needs_casting: Whether to cast input (before solve) and result (after solve)
        num_runs: Number of timed runs for the benchmark.
        warmup_runs: Number of warmup runs for the benchmark.
        skip_validation: If True, skip the final validation step. # <<< ADDED DOC
        
    Returns:
        Evaluation results dictionary.
    """
    # Check for oracle solve method
    if not hasattr(task_instance, "solve"):
        return create_standard_error_result(
            exception=AttributeError("Task instance does not have a solve method."),
            traceback_str=None,
            error_type_override="attribute_error",
            default_error_msg="Task instance does not have a solve method."
        )

    # Ensure code_dir and task_name are defined early for later use
    code_dir = os.environ.get("CODE_DIR", ".")
    task_name = getattr(task_instance, "task_name", task_instance.__class__.__name__)

    processed_problem = problem
    t_start_oracle_eval = time.perf_counter_ns()
    t_last = t_start_oracle_eval

    # Prepare an oracle callable that mimics agent overhead (new instance each call)
    def _fresh_oracle_callable(problem_inner):
        try:
            # Create new instance with same parameters as original
            inst = task_instance.__class__(
                n=getattr(task_instance, 'n', None),
                dataset_size=getattr(task_instance, 'dataset_size', None),
                target_time_ms=getattr(task_instance, 'target_time_ms', None),
                data_dir=getattr(task_instance, 'data_dir', None)
            )
            # Copy essential attributes
            if hasattr(task_instance, 'task_name'):
                inst.task_name = task_instance.task_name
            if hasattr(task_instance, 'k'):
                inst.k = task_instance.k
        except Exception:
            # Fallback: reuse original instance (no new instantiation)
            inst = task_instance
        try:
            return inst.solve(problem_inner)
        finally:
            if inst is not task_instance:  # Only delete if it's a new instance
                del inst
            gc.collect()

    # Override: if a custom solver.py exists in CODE_DIR, use it for oracle evaluation
    try:
        from AlgoTuner.utils.solver_loader import locate_solver_file, load_solver_module, get_fresh_solve_callable
        solver_file_path = locate_solver_file(task_name=task_name, code_dir=code_dir)
        solver_module = load_solver_module(solver_file_path.parent, solver_file_path.name)
        oracle_callable_to_use = get_fresh_solve_callable(solver_module)
        logging.info(f"Oracle override: using solver from {solver_file_path}")
    except Exception:
        # Fallback to default Task-based callable
        agent_mode = os.environ.get("AGENT_MODE", "0")
        oracle_callable_to_use = task_instance.solve if agent_mode == "0" else _fresh_oracle_callable

    # ------------------------------------------------------------------
    #  FAST-PATH TIMING
    #  For reference (oracle) evaluation we do not need the heavyweight
    #  cache-clearing / memory-monitor / sub-process machinery.  When the
    #  caller requests ≤3 measurement runs and ≤1 warm-up run we instead
    #  perform a minimal timing loop around the callable.  This slashes the
    #  overhead from ~150 ms to <0.5 ms, yielding the sub-millisecond numbers
    #  we expect for trivial problems.
    # ------------------------------------------------------------------

    # DISABLED: Always use isolated benchmark for consistency with solver evaluation
    fast_path = False  # was: (num_runs <= 3 and warmup_runs <= 1 and not capture_output)

    if fast_path:
        logging.info("Oracle Eval: using lightweight timing fast-path")

        # Optional single warm-up (not timed)
        if warmup_runs:
            try:
                oracle_callable_to_use(processed_problem)
            except Exception as warm_err:
                tb = traceback.format_exc()
                return create_standard_error_result(
                    exception=warm_err,
                    traceback_str=tb,
                    error_type_override="oracle_warmup_error",
                    default_error_msg=str(warm_err)
                )

        times_ns: list[int] = []
        result_value = None
        for _ in range(num_runs):
            t0 = time.perf_counter_ns()
            result_value = oracle_callable_to_use(processed_problem)
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)

        # Basic statistics
        min_ns = min(times_ns)
        mean_ns = statistics.mean(times_ns)

        benchmark_result = {
            "success": True,
            "values_ns": times_ns,
            "num_runs_executed": num_runs,
            "min_ns": min_ns,
            "mean_ns": mean_ns,
            "min_time_ms": min_ns / 1e6,
            "mean_ms": mean_ns / 1e6,
            "elapsed_ms": min_ns / 1e6,
            "result": result_value,
            "stdout": "",
            "stderr": "",
        }
    else:
        # Benchmark the oracle's solve method using full machinery
        try:
            solution_method = task_instance.solve
            timing = TimingManager()
            with timing.phase(Phase.ORACLE_RUN, capture_output=capture_output):
                # Use isolated benchmark for consistency
                task_code_dir = os.path.join(code_dir, "AlgoTuneTasks", task_name)
                if os.path.isdir(task_code_dir):
                    isolated_code_dir = task_code_dir
                else:
                    isolated_code_dir = code_dir
                
                # Use fixed baseline timeout from config for consistency
                from AlgoTuner.config.loader import load_config
                _config = load_config()
                baseline_timeout_ms = _config.get("benchmark", {}).get("baseline_timeout", 60000)
                
                benchmark_result = run_isolated_benchmark(
                    task_name=task_name,
                    code_dir=isolated_code_dir,
                    warmup_problem=processed_problem,
                    timed_problem=processed_problem,
                    num_runs=1,
                    timeout_seconds=baseline_timeout_ms / 1000.0,  # Fixed timeout from config
                )

            # For validation, run oracle once more in main process to get result if using isolated benchmark
            logging.info(f"[ISOLATED_VALIDATION_DEBUG] benchmark_result success: {benchmark_result.get('success')}, has result key: {'result' in benchmark_result}, keys: {list(benchmark_result.keys())}")
            if benchmark_result.get("success") and "result" not in benchmark_result:
                try:
                    logging.info("[ISOLATED_VALIDATION] Running oracle once more for validation")
                    # Capture stdout/stderr for isolated benchmark
                    import sys
                    from io import StringIO
                    
                    # Capture stdout/stderr during oracle execution
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    captured_stdout = StringIO()
                    captured_stderr = StringIO()
                    
                    try:
                        sys.stdout = captured_stdout
                        sys.stderr = captured_stderr
                        oracle_result_for_validation = oracle_callable_to_use(processed_problem)
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    # Add captured output to benchmark_result
                    benchmark_result["result"] = oracle_result_for_validation
                    benchmark_result["stdout"] = captured_stdout.getvalue()
                    benchmark_result["stderr"] = captured_stderr.getvalue()
                    
                    logging.info(f"[ISOLATED_VALIDATION] Successfully captured oracle result for validation: {type(oracle_result_for_validation)}")
                    logging.info(f"[ISOLATED_VALIDATION] Captured stdout: {len(benchmark_result['stdout'])} chars, stderr: {len(benchmark_result['stderr'])} chars")
                except Exception as e:
                    logging.warning(f"[ISOLATED_VALIDATION] Failed to get oracle result for validation: {e}")
                    benchmark_result["result"] = None
                    benchmark_result["stdout"] = ""
                    benchmark_result["stderr"] = ""

            # Timing after successful benchmark (inside try block)
            t_now = time.perf_counter_ns()
            logging.info(
                f"Oracle Eval Timing: run_benchmark took {(t_now - t_last) / 1e6:.2f}ms"
            )
            t_last = t_now

        except Exception as bench_exc:
            raw_tb = traceback.format_exc()
            err_ctx = extract_error_context(raw_tb, "")
            tb = err_ctx.get("enhanced_error_message", raw_tb)
            if err_ctx.get("code_context_snippet"):
                tb += "\n\nCode Context:\n" + err_ctx["code_context_snippet"]
            logging.error(
                f"Oracle Eval: run_benchmark failed with {type(bench_exc).__name__}: {bench_exc}\n{tb}"
            )
            return create_standard_error_result(
                exception=bench_exc,
                traceback_str=tb,
                error_type_override="oracle_benchmark_error",
                default_error_msg=str(bench_exc),
            )

    # --- Final Result Processing ---
    # Combine original problem info (or casted version) with benchmark results
    # Ensure elapsed_ms and other critical fields are present
    
    # Extract timing from new benchmark format
    elapsed_ms = benchmark_result.get("elapsed_ms")  # Check if already exists
    
    timing_fields = ["elapsed_ms", "min_time_ms", "mean_ms", "median_ms", "min", "mean", "min_ns", "mean_ns"]
    available_timing = {field: benchmark_result.get(field) for field in timing_fields}
    logging.info(f"TIMING DEBUG: Oracle evaluation benchmark_result timing fields: {available_timing}")
    
    if elapsed_ms is None:
        # Extract from new benchmark result format - prefer min time for consistency
        min_time_ms = benchmark_result.get("min_time_ms")
        mean_ms = benchmark_result.get("mean_ms") 
        median_ms = benchmark_result.get("median_ms")
        
        # Use min time if available, otherwise mean, otherwise median
        if min_time_ms is not None:
            elapsed_ms = min_time_ms
            logging.info(f"Oracle benchmark: using min_time_ms={min_time_ms} as elapsed_ms")
        elif mean_ms is not None:
            elapsed_ms = mean_ms
            logging.info(f"Oracle benchmark: using mean_ms={mean_ms} as elapsed_ms")
        elif median_ms is not None:
            elapsed_ms = median_ms
            logging.info(f"Oracle benchmark: using median_ms={median_ms} as elapsed_ms")
        else:
            # Fallback: convert from nanoseconds if available (more likely to exist)
            min_ns = benchmark_result.get("min_ns")
            mean_ns = benchmark_result.get("mean_ns")
            if min_ns is not None:
                elapsed_ms = min_ns / 1e6  # Convert ns to ms
                logging.info(f"Oracle benchmark: converted min_ns={min_ns if min_ns is not None else 'None'} to elapsed_ms={elapsed_ms if elapsed_ms is not None else 'None'}")
            elif mean_ns is not None:
                elapsed_ms = mean_ns / 1e6  # Convert ns to ms
                logging.info(f"Oracle benchmark: converted mean_ns={mean_ns if mean_ns is not None else 'None'} to elapsed_ms={elapsed_ms if elapsed_ms is not None else 'None'}")
            else:
                # Last fallback: convert from seconds if available
                min_s = benchmark_result.get("min")
                mean_s = benchmark_result.get("mean")
                if min_s is not None:
                    elapsed_ms = min_s * 1000.0
                    logging.info(f"Oracle benchmark: converted min={min_s}s to elapsed_ms={elapsed_ms}")
                elif mean_s is not None:
                    elapsed_ms = mean_s * 1000.0
                    logging.info(f"Oracle benchmark: converted mean={mean_s}s to elapsed_ms={elapsed_ms}")
                else:
                    elapsed_ms = None
                    logging.warning("Oracle benchmark: no timing information found in any expected format")
    else:
        logging.info(f"Oracle benchmark: found existing elapsed_ms={elapsed_ms}")
            
    # Ensure elapsed_ms is included in benchmark_result for downstream processing
    if elapsed_ms is not None:
        benchmark_result["elapsed_ms"] = elapsed_ms
            
    calculated_oracle_time_ms = elapsed_ms
    logging.info(f"Oracle benchmark final elapsed_ms: {elapsed_ms}")
    if benchmark_result.get("success") and (elapsed_ms is None or elapsed_ms == 0):
         logging.warning("Oracle benchmark succeeded but elapsed_ms is None or 0. Performance measurement might be inaccurate.")

    # If benchmark itself failed (e.g., timeout, error in solve)
    if not benchmark_result.get("success", False):
        # Ensure elapsed_ms is included even on failure
        if "elapsed_ms" not in benchmark_result:
             benchmark_result["elapsed_ms"] = 0 # Or potentially a failed timing value if available
        return benchmark_result

    # If benchmark succeeded, proceed with optional casting & validation
    # Results are no longer stripped, so use them directly
    raw_oracle_output = benchmark_result.get("result")
    final_oracle_result = raw_oracle_output

    # === SKIPPING Oracle Output Casting ===
    # Assume oracle output is already in the correct format for its own is_solution.
    # The final_oracle_result remains the raw_oracle_output.
    logging.debug(f"Skipping output casting for oracle result.")
    # Log the time since the last step (benchmark)
    t_now = time.perf_counter_ns()
    logging.info(f"Oracle Eval Timing: Output Casting step skipped (took {(t_now - t_last) / 1e6:.2f}ms)")
    t_last = t_now

    # === Validation ===
    validation_result = None
    if not skip_validation:
        try:
            validation_result = _validate_solution(task_instance, processed_problem, final_oracle_result)
            t_now = time.perf_counter_ns()
            logging.info(f"Oracle Eval Timing: Validation took {(t_now - t_last) / 1e6:.2f}ms")
            t_last = t_now
        except Exception as e:
            # Safeguard catch block
            logging.error(f"Unexpected error during oracle _validate_solution call: {e}", exc_info=True)
            tb = traceback.format_exc()
            validation_result = create_standard_error_result(
                exception=e,
                traceback_str=tb,
                error_type_override="validation_wrapper_error",
                default_error_msg="Unexpected error calling validation function for oracle"
            )
            # Augment with context
            validation_result['failed_solution_type'] = str(type(final_oracle_result))
            validation_result['failed_solution_shape'] = format_object_shape(final_oracle_result)
    else:
         validation_result = {"success": True, "validation_skipped": True}
         t_now = time.perf_counter_ns()
         logging.info(f"Oracle Eval Timing: Validation skipped (took {(t_now - t_last) / 1e6:.2f}ms)")
         t_last = t_now

    # === Final Result Combination ===
    # Handle case where validation was skipped or failed
    if not validation_result or not validation_result.get("success", False):
        # If validation failed (and wasn't skipped)
        if not validation_result.get("validation_skipped", False):
            logging.warning(f"Oracle validation failed. Type: {validation_result.get('error_type')}, Error: {validation_result.get('error')}")
            # Merge benchmark info with validation failure info
            final_evaluation_result = {
                **benchmark_result,  # preserve benchmark metrics including min_time_ms
                **validation_result,
                "success": False,
                "result": final_oracle_result,
                "raw_result": raw_oracle_output,
                "elapsed_ms": benchmark_result.get("elapsed_ms", 0),
                "oracle_time_ms": calculated_oracle_time_ms
            }
            final_evaluation_result["error_type"] = validation_result.get("error_type", "unknown_oracle_validation_failure")
            if "problem" not in final_evaluation_result:
                final_evaluation_result["problem"] = processed_problem
            return final_evaluation_result
        else:
             # Validation was skipped, treat as success for this function's return
             # Combine benchmark results with skip marker
             successful_evaluation_result = {
                 **benchmark_result,
                 "result": final_oracle_result, # Use the potentially cast value
                 "validation_passed": True,
                 "validation_skipped": True,
                 # Merge stdout/stderr from validation if captured (won't be if skipped)
                 "stdout": benchmark_result.get("stdout", ""),
                 "stderr": benchmark_result.get("stderr", ""),
             }
             # Log accurately when skipped
             logging.info("Oracle successful (validation skipped).")
             return successful_evaluation_result

    # If validation succeeded (and wasn't skipped)
    logging.info("Oracle successful and solution validated.")
    successful_evaluation_result = {
        **benchmark_result,  # preserve benchmark metrics including min_time_ms
        "result": final_oracle_result,
        "validation_passed": True,
        # Merge stdout/stderr from validation if captured
        "stdout": benchmark_result.get("stdout", "") + validation_result.get("stdout", ""),
        "stderr": benchmark_result.get("stderr", "") + validation_result.get("stderr", ""),
        "oracle_time_ms": calculated_oracle_time_ms
    }
    if "problem" not in successful_evaluation_result:
        successful_evaluation_result["problem"] = processed_problem
        
    logging.info(f"Oracle Eval Timing: Total run_oracle_evaluation took {(time.perf_counter_ns() - t_start_oracle_eval) / 1e6:.2f}ms")
    return successful_evaluation_result

