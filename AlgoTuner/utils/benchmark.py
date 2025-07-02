# -*- coding: utf-8 -*-
"""
Benchmark Runner Interface Module

This module provides the primary interface (`run_benchmark`) for benchmarking functions.
It uses isolated per-run processes following the pattern:
  ┌──── parent (harness) ────┐
  │ for each of K runs:      │
  │   1. spawn worker proc   │
  │   2. worker does:        │      # all inside the child
  │        warm-up()         │      #   JIT, load libs, etc.
  │        tic               │
  │        timed_call()      │
  │        toc               │
  │   3. harvest time/result │
  │   4. worker exits → RAM, │
  │      globals, __pycache__│
  │      and lru_caches die  │
  └──────────────────────────┘
"""

import sys
import time
import gc
import logging
import traceback
import faulthandler
import os
import platform
import contextlib
from typing import Any, Callable, Dict, Optional, List

from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark
from AlgoTuner.utils.precise_timing import time_execution_ns
from AlgoTuner.utils.error_utils import create_standard_error_result, extract_error_context
from AlgoTuner.utils.time_management import time_limit, TimeoutError

faulthandler.enable()

logger = logging.getLogger(__name__)

def run_benchmark(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_runs: int = 5,      # Number of measurement runs
    num_warmups: int = 3,    # Number of warmup runs
    capture_output: bool = False,
    timeout_seconds: float = 60.0,
    working_dir: str = None,
    force_subprocess: bool = None,  # New parameter
    use_signal_timeout: bool = None,  # New parameter
) -> Dict[str, Any]:
    """
    Run a benchmark for a given function with isolated per-run processes.

    This function uses the isolated multiprocessing approach where each measurement
    spawns a fresh process: warmup → timed call → exit. This eliminates JIT/cache
    effects between runs for cleaner timing measurements.

    Args:
        func: The function to benchmark.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
        num_runs: Number of measurement runs for the benchmark.
        num_warmups: Number of warmup runs for the benchmark.
        capture_output: Whether to capture stdout/stderr from the benchmarked function.
        timeout_seconds: Maximum time allowed for the entire benchmark process (including warmups).
        working_dir: Directory to set as the working directory during all runs (for caches/artifacts).
        force_subprocess: If True, always use isolated. If False, always use direct. If None (default), auto-decide.
        use_signal_timeout: If True, use signal-based timeouts on Unix. If None (default), auto-decide based on platform.

    Returns:
        A dictionary containing benchmark results:
            - success (bool): True if benchmarking completed without errors or timeout.
            - result (Any): The return value from the last successful execution of func.
            - runs (int): Number of successful measurement runs executed.
            - values (List[float]): Raw timing values in seconds for each successful run.
            - mean (float): Mean execution time in seconds.
            - median (float): Median execution time in seconds.
            - stddev (float): Standard deviation of execution times in seconds.
            - min (float): Minimum execution time in seconds.
            - max (float): Maximum execution time in seconds.
            - error (Optional[str]): Error message if benchmarking failed.
            - traceback (Optional[str]): Traceback if benchmarking failed.
            - timeout_occurred (bool): True if the benchmark timed out.
            - input_params (Dict): Parameters used for this benchmark run.
            - ci_low (float): Lower bound of the 95% confidence interval in seconds.
            - ci_high (float): Upper bound of the 95% confidence interval in seconds.
    """
    if kwargs is None:
        kwargs = {}

    func_name = getattr(func, '__name__', 'anonymous_function')
    
    logger.debug(f"*** RUN_BENCHMARK_ENTRY *** func={func_name}, num_runs={num_runs}, num_warmups={num_warmups}, force_subprocess={force_subprocess}")
    
    # Determine execution strategy
    should_use_isolated = _should_use_isolated_benchmark(
        timeout_seconds=timeout_seconds,
        force_subprocess=force_subprocess
    )
    
    should_use_signal_timeout = _should_use_signal_timeout(
        use_signal_timeout=use_signal_timeout
    )
    
    execution_mode = "isolated" if should_use_isolated else "direct"
    timeout_mode = "signal" if should_use_signal_timeout else "threading"
    
    # Log which execution path is chosen
    logger.debug(f"*** RUN_BENCHMARK_PATH *** {func_name}: should_use_isolated={should_use_isolated}, execution_mode={execution_mode}")
    
    logger.info(
        f"run_benchmark starting for '{func_name}' "
        f"(runs={num_runs}, warmups={num_warmups}, timeout={timeout_seconds}s) - "
        f"Execution: {execution_mode}, Timeout: {timeout_mode}"
    )

    # Prepare input parameters log
    input_params = {
        "func_name": func_name,
        "num_runs_requested": num_runs,
        "num_warmups": num_warmups,
        "timeout_seconds": timeout_seconds,
        "working_dir": working_dir,
        "execution_mode": execution_mode,
        "timeout_mode": timeout_mode,
    }
    
    # Route to appropriate execution method
    if should_use_isolated:
        logger.debug(f"*** TAKING ISOLATED PATH *** for {func_name}")
        return _run_benchmark_isolated(
            func=func,
            args=args,
            kwargs=kwargs,
            num_runs=num_runs,
            num_warmups=num_warmups,
            capture_output=capture_output,
            timeout_seconds=timeout_seconds,
            working_dir=working_dir,
            input_params=input_params
        )
    else:
        logger.debug(f"*** TAKING DIRECT PATH *** for {func_name}")
        return _run_benchmark_direct(
            func=func,
            args=args,
            kwargs=kwargs,
            num_runs=num_runs,
            num_warmups=num_warmups,
            capture_output=capture_output,
            timeout_seconds=timeout_seconds,
            working_dir=working_dir,
            input_params=input_params,
            use_signal_timeout=should_use_signal_timeout,
        )


# Signal-based timeout implementation for Unix systems
class _SignalTimeoutError(Exception):
    """Exception raised when a signal-based timeout occurs."""
    pass


@contextlib.contextmanager
def _signal_timeout(seconds: float):
    """
    Context manager for signal-based timeouts on Unix systems.
    More reliable than threading-based timeouts for C extensions and NumPy operations.
    
    Args:
        seconds: Timeout in seconds
        
    Raises:
        _SignalTimeoutError: If the operation times out
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise _SignalTimeoutError(f"Operation timed out after {seconds} seconds (signal)")
    
    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))  # SIGALRM only accepts integer seconds
    
    try:
        yield
    finally:
        # Cancel the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _should_use_isolated_benchmark(
    timeout_seconds: float,
    force_subprocess: Optional[bool] = None
) -> bool:
    """
    Determine if isolated per-run benchmark should be used for clean timing measurements.
    
    Args:
        timeout_seconds: The timeout value being used
        force_subprocess: Override for subprocess usage
        
    Returns:
        True if isolated per-run benchmark should be used
    """
    if force_subprocess is not None:
        return force_subprocess
    
    # Check if we're already in a daemon process - if so, can't create subprocesses
    try:
        import multiprocessing as mp
        current_process = mp.current_process()
        if current_process.daemon:
            logger.debug("Running in daemon process - cannot create subprocesses, using direct execution")
            return False
    except Exception as e:
        logger.debug(f"Could not check daemon status: {e}")
    
    # Always use isolated benchmark for agent mode to get clean timing measurements
    agent_mode = os.environ.get("AGENT_MODE", "0")
    if agent_mode != "0":
        return True
        
    # For baseline mode, check if isolated timing is explicitly requested
    if os.environ.get("ISOLATED_EVAL", "1") == "1":
        return True
        
    return False


def _create_distinct_problem(problem: Any) -> Any:
    """
    Create a problem with distinct computational content to prevent cache pollution.
    
    This function modifies the problem data slightly to ensure it has different
    computational content while still being solvable by the same algorithm.
    
    Args:
        problem: The original problem instance
        
    Returns:
        A modified copy of the problem with distinct computational content
    """
    import copy
    import numpy as np
    
    if problem is None:
        return None
    
    # Deep copy to start with
    distinct_problem = copy.deepcopy(problem)
    
    try:
        # Strategy 1: If it's a dict/object with numerical arrays, add small noise
        if hasattr(distinct_problem, 'items') or isinstance(distinct_problem, dict):
            for key, value in (distinct_problem.items() if hasattr(distinct_problem, 'items') else 
                             dict(distinct_problem).items() if isinstance(distinct_problem, dict) else []):
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):  # numpy-like array
                    # Add tiny numerical noise (1e-12) to make content different but computationally equivalent
                    if np.issubdtype(value.dtype, np.floating):
                        noise = np.random.RandomState(42).normal(0, 1e-12, value.shape).astype(value.dtype)
                        distinct_problem[key] = value + noise
                    elif np.issubdtype(value.dtype, np.integer):
                        # For integers, modify the last element by +1 (usually safe for most algorithms)
                        if value.size > 0:
                            modified_value = value.copy()
                            flat_view = modified_value.flat
                            flat_view[-1] = flat_view[-1] + 1
                            distinct_problem[key] = modified_value
        
        # Strategy 2: If it's a direct numpy array
        elif hasattr(distinct_problem, 'shape') and hasattr(distinct_problem, 'dtype'):
            if np.issubdtype(distinct_problem.dtype, np.floating):
                noise = np.random.RandomState(42).normal(0, 1e-12, distinct_problem.shape).astype(distinct_problem.dtype)
                distinct_problem = distinct_problem + noise
            elif np.issubdtype(distinct_problem.dtype, np.integer) and distinct_problem.size > 0:
                modified_array = distinct_problem.copy()
                flat_view = modified_array.flat
                flat_view[-1] = flat_view[-1] + 1
                distinct_problem = modified_array
        
        # Strategy 3: If it's a list/tuple of arrays
        elif isinstance(distinct_problem, (list, tuple)):
            modified_list = []
            for i, item in enumerate(distinct_problem):
                if hasattr(item, 'shape') and hasattr(item, 'dtype'):
                    if np.issubdtype(item.dtype, np.floating):
                        noise = np.random.RandomState(42 + i).normal(0, 1e-12, item.shape).astype(item.dtype)
                        modified_list.append(item + noise)
                    elif np.issubdtype(item.dtype, np.integer) and item.size > 0:
                        modified_item = item.copy()
                        flat_view = modified_item.flat
                        flat_view[-1] = flat_view[-1] + 1
                        modified_list.append(modified_item)
                    else:
                        modified_list.append(item)
                else:
                    modified_list.append(item)
            distinct_problem = type(distinct_problem)(modified_list)
        
        # Strategy 4: If it's a string, append a tiny suffix
        elif isinstance(distinct_problem, str):
            distinct_problem = distinct_problem + "_warmup_variant"
        
        logger.debug(f"Created distinct problem: original_type={type(problem)}, distinct_type={type(distinct_problem)}")
        
    except Exception as e:
        logger.warning(f"Failed to create distinct problem, using deep copy fallback: {e}")
        # Fallback to deep copy with warning
        distinct_problem = copy.deepcopy(problem)
    
    return distinct_problem


def _should_use_signal_timeout(use_signal_timeout: Optional[bool] = None) -> bool:
    """
    Determine if signal-based timeouts should be used on Unix systems.
    
    Args:
        use_signal_timeout: Override for signal timeout usage
        
    Returns:
        True if signal-based timeouts should be used
    """
    if use_signal_timeout is not None:
        return use_signal_timeout
        
    # Only use signals on Unix-like systems
    if platform.system() in ("Linux", "Darwin", "Unix"):
        # Check if we're in a context where signals work (not in subprocess)
        try:
            import signal
            # Test if we can set a signal handler
            old_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.signal(signal.SIGALRM, old_handler)
            return True
        except (ValueError, OSError):
            # Signal handling not available (might be in thread/subprocess)
            return False
    
    return False


def _run_benchmark_isolated(
    func: Callable[..., Any],
    args: tuple,
    kwargs: Dict[str, Any],
    num_runs: int,
    num_warmups: int,
    capture_output: bool,
    timeout_seconds: float,
    working_dir: Optional[str],
    input_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run benchmark using isolated per-run processes for clean timing measurements.
    Each run spawns a fresh process: warmup → timed call → exit.
    """
    func_name = getattr(func, '__name__', 'anonymous_function')
    logger.info(f"Running '{func_name}' with isolated per-run processes (timeout {timeout_seconds}s)")
    
    try:
        # Check for daemon process issue before attempting subprocess creation
        import multiprocessing as mp
        current_process = mp.current_process()
        if current_process.daemon:
            logger.warning(f"Cannot create subprocesses from daemon process for '{func_name}', falling back to direct execution")
            # Fall back to direct execution with signal timeout if possible
            should_use_signal = _should_use_signal_timeout(use_signal_timeout=None)
            return _run_benchmark_direct(
                func=func,
                args=args,
                kwargs=kwargs,
                num_runs=num_runs,
                num_warmups=num_warmups,
                capture_output=capture_output,
                timeout_seconds=timeout_seconds,
                working_dir=working_dir,
                input_params=input_params,
                use_signal_timeout=should_use_signal,
            )
        
        # Extract task name from the environment or function context
        # This is needed for run_isolated_benchmark
        task_name = os.environ.get("CURRENT_TASK_NAME", "unknown_task")
        
        # If task name is unknown, try to extract from function context
        if task_name == "unknown_task" and func_name == "generate_problem":
            # For generate_problem, the function is a method of a task object
            # Try to get the task name from the method's instance
            if hasattr(func, '__self__'):
                task_instance = func.__self__
                # Try to get task_name attribute
                if hasattr(task_instance, 'task_name'):
                    task_name = task_instance.task_name
                    logger.debug(f"[benchmark] Extracted task name from function instance: {task_name}")
                # Try to get from class name if no task_name attribute
                elif hasattr(task_instance, '__class__'):
                    class_name = task_instance.__class__.__name__
                    # If it's a Task class, the task name might be the class name
                    if class_name.endswith('Task'):
                        task_name = class_name[:-4].lower()  # Remove 'Task' suffix and lowercase
                        logger.debug(f"[benchmark] Extracted task name from class name: {task_name}")
                    # Try to find it in TASK_REGISTRY
                    else:
                        try:
                            from AlgoTuneTasks.base import TASK_REGISTRY
                            for name, cls in TASK_REGISTRY.items():
                                if isinstance(task_instance, cls):
                                    task_name = name
                                    logger.debug(f"[benchmark] Found task name in registry: {task_name}")
                                    break
                        except Exception as e:
                            logger.debug(f"[benchmark] Could not check TASK_REGISTRY: {e}")
        
        logger.debug(f"[benchmark] Final task name for isolated benchmark: {task_name}")
        code_dir = working_dir or os.getcwd()
        
        # Create warmup and timed problems
        # CRITICAL: Different computational content to prevent cache pollution
        import copy
        base_problem = args[0] if args else None
        
        # Create distinct problems with different computational content
        timed_problem = copy.deepcopy(base_problem) if base_problem is not None else None
        warmup_problem = _create_distinct_problem(base_problem) if base_problem is not None else None
        
        if warmup_problem is None:
            logger.error(f"No problem argument found for isolated benchmark of '{func_name}'")
            return create_standard_error_result(
                exception=ValueError("No problem argument provided"),
                traceback_str=None,
                error_type_override="missing_problem_argument",
                default_error_msg="Isolated benchmark requires a problem argument"
            )
        
        # Special handling for generate_problem functions - they don't use the solver-based isolated benchmark
        if func_name == "generate_problem":
            logger.debug(f"[benchmark] Using direct execution for generate_problem function")
            # For generate_problem, call the function directly with its arguments
            # This avoids the solver-loading logic that doesn't apply to generation functions
            return _run_benchmark_direct(
                func=func,
                args=args,
                kwargs=kwargs,
                num_runs=num_runs,
                num_warmups=num_warmups,
                capture_output=capture_output,
                timeout_seconds=timeout_seconds,
                working_dir=working_dir,
                input_params=input_params,
                use_signal_timeout=_should_use_signal_timeout(),
            )
        
        # Use the isolated benchmark utility for solver functions
        result = run_isolated_benchmark(
            task_name=task_name,
            code_dir=code_dir,
            warmup_problem=warmup_problem,
            timed_problem=timed_problem,
            num_runs=num_runs,
            timeout_seconds=timeout_seconds
        )
        
        # Log the result for debugging
        timing_fields = ["success", "values_ns", "min_ns", "mean_ns", "min_time_ms", "mean_ms", "elapsed_ms", "num_runs_executed", "error", "timeout_occurred"]
        isolated_timing_debug = {field: result.get(field) for field in timing_fields}
        logger.info(f"BENCHMARK_ISOLATED_TIMING_DEBUG: run_isolated_benchmark returned: {isolated_timing_debug}")
        
        # Ensure compatibility with expected benchmark result format
        if result.get("success"):
            # Convert from isolated benchmark format to standard benchmark format
            standard_result = {
                "success": True,
                "result": result.get("result"),
                "runs": result.get("num_runs_executed", 0),
                "num_runs_executed": result.get("num_runs_executed", 0),
                "values_ns": result.get("values_ns", []),
                "min_ns": result.get("min_ns"),
                "mean_ns": result.get("mean_ns"),
                "min_time_ms": result.get("min_time_ms"),
                "mean_time_ms": result.get("mean_ms"),
                "elapsed_ms": result.get("elapsed_ms"),
                "timeout_occurred": result.get("timeout_occurred", False),
                "error": result.get("error"),
                "traceback": result.get("traceback"),
                "input_params": input_params,
                "stdout": "",
                "stderr": ""
            }
            
            # Convert values to seconds for compatibility
            if standard_result["values_ns"]:
                standard_result["values"] = [ns / 1e9 for ns in standard_result["values_ns"]]
                standard_result["min"] = standard_result["min_ns"] / 1e9 if standard_result["min_ns"] else None
                standard_result["mean"] = standard_result["mean_ns"] / 1e9 if standard_result["mean_ns"] else None
            else:
                standard_result["values"] = []
                standard_result["min"] = None
                standard_result["mean"] = None
                
            return standard_result
        else:
            # Handle failure case
            return {
                "success": False,
                "result": None,
                "runs": 0,
                "num_runs_executed": 0,
                "values": [],
                "values_ns": [],
                "error": result.get("error", "Isolated benchmark failed"),
                "traceback": result.get("traceback"),
                "timeout_occurred": result.get("timeout_occurred", False),
                "input_params": input_params,
                "stdout": "",
                "stderr": ""
            }
        
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Error in isolated benchmark for '{func_name}': {e}", exc_info=False)
        
        return create_standard_error_result(
            exception=e,
            traceback_str=tb_str,
            error_type_override="isolated_benchmark_error",
            default_error_msg=f"Isolated benchmark failed: {e}"
        )


def _run_benchmark_direct(
    func: Callable[..., Any],
    args: tuple,
    kwargs: Dict[str, Any],
    num_runs: int,
    num_warmups: int,
    capture_output: bool,
    timeout_seconds: float,
    working_dir: Optional[str],
    input_params: Dict[str, Any],
    use_signal_timeout: bool = False,
) -> Dict[str, Any]:
    """
    Run benchmark directly in current process with improved timeout handling.
    """
    func_name = getattr(func, '__name__', 'anonymous_function')
    
    timeout_method = "signal" if use_signal_timeout else "threading"
    logger.info(f"Running '{func_name}' directly with {timeout_method} timeout {timeout_seconds}s")

    # --- Call time_execution_ns with timeout enforcement ---
    timing_result = None # Initialize
    try:
        logger.debug(f"*** DIRECT_PATH_TIMING *** About to call time_execution_ns for {func_name} with runs={num_runs}, warmups={num_warmups}")
        logger.debug(f"[BENCHMARK_EXECUTION] *** CALLING time_execution_ns *** with {timeout_seconds}s {timeout_method} timeout for '{func_name}'")
        
        # Choose timeout method
        if use_signal_timeout:
            with _signal_timeout(timeout_seconds):
                timing_result = time_execution_ns(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    num_runs=num_runs,
                    warmup_runs=num_warmups,
                    capture_output=capture_output,
                    working_dir=working_dir,
                )
        else:
            with time_limit(timeout_seconds):
                timing_result = time_execution_ns(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    num_runs=num_runs,
                    warmup_runs=num_warmups,
                    capture_output=capture_output,
                    working_dir=working_dir,
                )
        logger.debug(f"[BENCHMARK_EXECUTION] *** time_execution_ns COMPLETED *** for '{func_name}'. Success: {timing_result.get('success')}")
        
        timing_result_fields = ["success", "values_ns", "mean_ns", "min_ns", "values", "mean", "min", "num_runs_executed", "error"]
        timing_result_debug = {field: timing_result.get(field) for field in timing_result_fields} if timing_result else None
        logger.debug(f"*** TIME_EXECUTION_NS_RESULT *** {func_name}: {timing_result_debug}")

    except (TimeoutError, _SignalTimeoutError) as timeout_err:
        timeout_type = "signal" if isinstance(timeout_err, _SignalTimeoutError) else "threading"
        logger.warning(f"Benchmark {timeout_type} timeout for '{func_name}' after {timeout_seconds}s")
        timing_result = {
            "success": False,
            "timeout_occurred": True,
            "error": f"Benchmark timed out after {timeout_seconds} seconds ({timeout_type})",
            "error_type": "timeout",
            "traceback": str(timeout_err),
            "input_params": input_params
        }
    except Exception as direct_timing_err:
        tb_str = traceback.format_exc()
        logger.error(f"Error calling time_execution_ns directly for '{func_name}': {direct_timing_err}", exc_info=False)
        # Create a standard error result if the call itself fails
        timing_result = create_standard_error_result(
            exception=direct_timing_err,
            traceback_str=tb_str,
            error_type_override="direct_timing_error",
            default_error_msg=f"Error calling time_execution_ns: {direct_timing_err}"
        )
        # Ensure necessary keys for final processing exist
        timing_result.setdefault("success", False)
        timing_result.setdefault("result", None)
        timing_result.setdefault("first_warmup_result", None)
        timing_result.setdefault("num_runs_executed", 0)
        timing_result.setdefault("values_ns", [])
        timing_result.setdefault("stdout", "")
        timing_result.setdefault("stderr", "")

    # --- Process and Format Results ---
    # Initialize with defaults, especially for failure cases
    final_results = {
        "success": False,
        "result": None,
        "first_warmup_result": None,
        "runs": 0, # Will be num_runs_executed
        "values": [], # Will store seconds; raw ns values stored as 'values_ns'
        "mean": None,   # in seconds
        "median": None, # in seconds
        "stddev": None, # in seconds
        "min": None,    # in seconds
        "max": None,    # in seconds

        # Add keys for direct ns and ms values from time_execution_ns
        "mean_ns": None,
        "median_ns": None,
        "min_ns": None,
        "max_ns": None,
        "stddev_ns": None,
        "values_ns": [], # List of raw ns timings
        "num_runs_executed": 0, # Explicitly store this

        "min_time_ms": None,
        "mean_time_ms": None,
        "max_time_ms": None,
        "stddev_time_ms": None,

        "error": None,
        "traceback": None,
        "timeout_occurred": False, # Timeout is not handled here
        "ci_low": None, # in seconds
        "ci_high": None, # in seconds
        "input_params": input_params,
        "stdout": "",
        "stderr": "",
        "code_context": None, # Ensure this key exists
    }

    # Merge the results from time_execution_ns if it ran
    if timing_result:
        # Basic success/failure and output related fields
        final_results.update({
            "success": timing_result.get("success", False),
            "result": timing_result.get("result"),
            "first_warmup_result": timing_result.get("first_warmup_result"),
            "runs": timing_result.get("num_runs_executed", 0), # For 'runs' key
            "num_runs_executed": timing_result.get("num_runs_executed", 0), # Explicit key
            "error": timing_result.get("error"),
            "traceback": timing_result.get("traceback"),
            "stdout": timing_result.get("stdout", ""),
            "stderr": timing_result.get("stderr", ""),
            "code_context": timing_result.get("code_context"), # Propagate code_context
        })

        # Propagate raw ns and ms stats directly from timing_result
        ns_keys_to_propagate = ["mean_ns", "median_ns", "min_ns", "max_ns", "stddev_ns", "ci_low_ns", "ci_high_ns", "values_ns"]
        for key in ns_keys_to_propagate:
            if timing_result.get(key) is not None:
                final_results[key] = timing_result[key]
        
        # First try to propagate ms values if they exist
        ms_keys_to_propagate = ["mean_time_ms", "min_time_ms", "max_time_ms", "stddev_time_ms"]
        for key in ms_keys_to_propagate:
            if timing_result.get(key) is not None:
                final_results[key] = timing_result[key]
        
        # If ms values don't exist, convert from ns values
        ns_to_ms_conversions = [
            ("mean_ns", "mean_time_ms"),
            ("median_ns", "median_time_ms"), 
            ("min_ns", "min_time_ms"),
            ("max_ns", "max_time_ms"),
            ("stddev_ns", "stddev_time_ms")
        ]
        
        for ns_key, ms_key in ns_to_ms_conversions:
            ns_value = final_results.get(ns_key)
            ms_value = final_results.get(ms_key)
            
            if ms_value is None and ns_value is not None:
                try:
                    converted_ms = ns_value / 1e6
                    final_results[ms_key] = converted_ms
                except (TypeError, ZeroDivisionError) as e:
                    logger.warning(f"Could not convert {ns_key} to {ms_key}: {e}")
                    final_results[ms_key] = None

        # Convert primary stats to seconds for backward compatibility
        values_ns_list = final_results.get("values_ns")
        
        if values_ns_list and isinstance(values_ns_list, list):
            try:
                final_results["values"] = [ns / 1e9 for ns in values_ns_list]
                
                # Populate second-based stat keys if their ns counterparts exist in final_results
                if final_results.get("mean_ns") is not None: final_results["mean"] = final_results["mean_ns"] / 1e9
                if final_results.get("median_ns") is not None: final_results["median"] = final_results["median_ns"] / 1e9
                if final_results.get("min_ns") is not None: final_results["min"] = final_results["min_ns"] / 1e9
                if final_results.get("max_ns") is not None: final_results["max"] = final_results["max_ns"] / 1e9
                if final_results.get("stddev_ns") is not None: final_results["stddev"] = final_results["stddev_ns"] / 1e9
                if final_results.get("ci_low_ns") is not None: final_results["ci_low"] = final_results["ci_low_ns"] / 1e9
                if final_results.get("ci_high_ns") is not None: final_results["ci_high"] = final_results["ci_high_ns"] / 1e9

            except TypeError as e:
                logger.error(f"Error converting timing results from ns to seconds: {e}")
                final_results["success"] = False
                error_msg = f"Failed to convert ns results: {e}"
                final_results["error"] = f"{final_results.get('error', '')}; {error_msg}".lstrip('; ')

    # Compatibility shim for downstream code
    if final_results.get("min_time_ms") is not None:
        logger.info("BENCHMARK_DEBUG: Using min_time_ms for elapsed time")
        final_results.setdefault("elapsed_ms", final_results["min_time_ms"])
    fallback_elapsed = final_results.get("min_time_ms") or final_results.get("mean_time_ms")
    if final_results.get("max_time_ms") is not None:
        final_results["max"] = final_results["max_time_ms"] / 1e3

    return final_results