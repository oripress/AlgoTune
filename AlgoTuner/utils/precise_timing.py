"""
Precise Timing Module

This module provides precise, consistent timing functions for measuring function execution time.
It isolates the function being timed from external factors as much as possible and
ensures consistent timing across different parts of the codebase.
"""


import time
import gc
import io
import logging
import traceback
import contextlib
import os
import threading
import platform
import random
import sys
import signal
import statistics
import math
import tempfile
import subprocess
import pickle
import resource
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, List
from contextlib import redirect_stdout, redirect_stderr, nullcontext
import multiprocessing as mp
import numpy as np
from AlgoTuner.utils.error_utils import create_standard_error_result
from AlgoTuner.utils.solver_loader import with_working_dir
from AlgoTuner.utils.dace_config import initialize_dace_for_process
from AlgoTuner.utils.blas_utils import set_blas_threads, log_current_blas_threads, log_cpu_affinity, log_thread_env
import builtins


T = TypeVar('T')

_timing_overhead_ns = None
_capture_overhead_ns = None
_timing_system_initialized = False
_last_calibration_ns = 0
_RECAL_INTERVAL_NS = int(60 * 1e9)
_TRIM_FRACTION = 0.1

logging.basicConfig(level=logging.INFO)

MEMORY_MONITORING = False

def _initialize_timing_system():
    """
    No-op stub: skip heavy system calibration.
    """
    global _timing_system_initialized
    if _timing_system_initialized:
        return
    _timing_system_initialized = True


def _check_system_load():
    """
    Check if the system is under high load.

    Returns:
        bool: True if the system is under high load, False otherwise
    """
    try:
        try:
            import psutil
        except ImportError:
            logging.debug("psutil not available for system load check")
            return False

        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 80:
            logging.warning(f"High CPU load detected: {cpu_percent}%")
            return True

        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logging.warning(f"High memory usage detected: {memory.percent}%")
            return True

        return False
    except Exception as e:
        logging.debug(f"Could not check system load: {e}")
        return False

def _handle_memory_error(error, func, current_run, total_runs):
    """
    Handle memory error with full context information.
    
    Args:
        error: The MemoryError exception
        func: The function that was being executed
        current_run: The current run number (0-indexed)
        total_runs: Total number of runs requested
        
    Returns:
        Dictionary with error information matching standard timing result format
    """
    import traceback
    from AlgoTuner.utils.error_utils import create_standard_error_result
    
    func_name = getattr(func, '__name__', str(func))
    tb_str = traceback.format_exc()
    error_result = create_standard_error_result(
        exception=error,
        traceback_str=tb_str,
        elapsed_ms=0,
        default_error_msg=f'Memory limit exceeded during run {current_run+1}/{total_runs} of {func_name}'
    )
    timing_fields = {
        'values_ns': [],
        'num_runs_executed': current_run,
        'result': None,
        'first_warmup_result': None,
        'mean_ns': None,
        'median_ns': None,
        'min_ns': None,
        'max_ns': None,
        'stddev_ns': None,
        'mean_time_ms': None,
        'median_time_ms': None,
        'min_time_ms': None,
        'max_time_ms': None,
        'stddev_time_ms': None,
        'ci_low_time_ms': None,
        'ci_high_time_ms': None,
        'function': func_name,
        'context': f'Measurement phase, run {current_run+1} of {total_runs}'
    }
    error_result.update(timing_fields)
    
    logger = logging.getLogger(__name__)
    logger.error(f"Memory limit exceeded in {func_name}")
    logger.debug(f"Memory error details: {error_result.get('error', 'Unknown error')}")
    
    return error_result

def _calculate_confidence_interval(times, confidence=0.95):
    """
    Calculate the confidence interval for a list of timing measurements.

    Args:
        times: List of timing measurements
        confidence: Confidence level (default: 0.95 for 95% confidence)

    Returns:
        Tuple of (mean, margin_of_error, relative_error)
    """
    if len(times) < 2:
        return statistics.mean(times) if times else 0, 0, 0

    try:
        import scipy.stats

        mean = statistics.mean(times)
        stdev = statistics.stdev(times)

        t_value = scipy.stats.t.ppf((1 + confidence) / 2, len(times) - 1)
        margin_of_error = t_value * (stdev / math.sqrt(len(times)))
        relative_error = (margin_of_error / mean) if mean > 0 else 0

        return mean, margin_of_error, relative_error
    except ImportError:
        mean = statistics.mean(times)
        stdev = statistics.stdev(times)

        t_value = 1.96
        margin_of_error = t_value * (stdev / math.sqrt(len(times)))
        relative_error = (margin_of_error / mean) if mean > 0 else 0

        return mean, margin_of_error, relative_error
    except Exception as e:
        logging.debug(f"Could not calculate confidence interval: {e}")
        return statistics.mean(times) if times else 0, 0, 0

class TimeoutError(Exception):
    """Exception raised when a function execution times out."""
    pass

@contextlib.contextmanager
def memory_monitor_context():
    """
    Context manager that monitors memory during execution using process-level monitoring.
    No threads are created - uses the new process-level memory monitoring system.
    """
    try:
        from AlgoTuner.utils.process_monitor import check_worker_memory
        memory_error = check_worker_memory()
        if memory_error:
            raise memory_error
        _orig_open = builtins.open

        def _no_write_open(file, mode='r', *args, **kwargs):
            if any(ch in mode for ch in 'wax+'):
                raise PermissionError("File writes are forbidden during timing runs")
            return _orig_open(file, mode, *args, **kwargs)

        builtins.open = _no_write_open

        yield
        memory_error = check_worker_memory()
        if memory_error:
            raise memory_error
            
    except ImportError:
        logging.debug("Process monitor not available, skipping memory monitoring")
        import resource

        _orig_open = builtins.open

        def _no_write_open(file, mode='r', *args, **kwargs):
            if any(ch in mode for ch in 'wax+'):
                raise PermissionError("File writes are forbidden during timing runs")
            return _orig_open(file, mode, *args, **kwargs)

        builtins.open = _no_write_open

        try:
            yield
        finally:
            builtins.open = _orig_open

    finally:
        try:
            builtins.open = _orig_open  # type: ignore[attr-defined]
        except Exception:
            pass

@contextlib.contextmanager
def time_limit(seconds: float):
    """
    Context manager for timing out operations.
    Uses signal-based timeouts on Unix systems, graceful fallback on others.
    Does not create threads to avoid "can't start new thread" errors.

    Args:
        seconds: Timeout in seconds

    Raises:
        TimeoutError: If the operation times out
    """
    import signal
    import platform
    
    use_signal = (
        hasattr(signal, 'SIGALRM') and 
        platform.system() != 'Windows' and
        threading.current_thread() is threading.main_thread()
    )
    
    if use_signal:
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, int(seconds)))
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        logging.debug(f"time_limit: Skipping timeout enforcement (signal not available or not main thread)")
        yield

def _preallocate_memory(size_mb: int = 50):
    """
    Pre-allocate memory to reduce the chance of memory allocation during timing.

    Args:
        size_mb: Size of memory to pre-allocate in MB
    """
    try:
        size_bytes = size_mb * 1024 * 1024
        chunk_size = 1024 * 1024
        chunks = []

        for _ in range(size_mb):
            chunks.append(bytearray(chunk_size))

        for chunk in chunks:
            chunk[0] = 1
            chunk[-1] = 1

        chunks = None
        gc.collect()
    except Exception as e:
        logging.debug(f"Memory pre-allocation failed: {e}")

def _warmup_function(func: Callable, *args, **kwargs):
    """
    Warm up a function by running it a few times to trigger JIT compilation.

    Args:
        func: The function to warm up
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The captured stdout and stderr from the last warmup run, or empty strings if no output.
    """
    stdout_content = ""
    stderr_content = ""

    try:
        for i in range(3):
            try:
                capture = {}
                with robust_capture_output() as capture:
                    func(*args, **kwargs)

                if i == 2:
                    stdout_content = capture['stdout']
                    stderr_content = capture['stderr']

                    logging.debug(f"Warmup captured stdout ({len(stdout_content)} chars): {stdout_content[:100]}...")
                    logging.debug(f"Warmup captured stderr ({len(stderr_content)} chars): {stderr_content[:100]}...")
            except Exception as e:
                logging.debug(f"Exception during warmup run {i}: {str(e)}")
                pass
    except Exception as e:
        logging.debug(f"Function warmup failed: {e}")

    return stdout_content, stderr_content

class MemoryMonitor:
    """
    A class to monitor memory usage over time.

    This class provides methods to track memory usage at different points in time
    and calculate statistics about memory usage patterns.
    """

    def __init__(self):
        """Initialize the memory monitor."""
        self.snapshots = []
        self.labels = []

    def take_snapshot(self, label=None):
        """
        Take a snapshot of the current memory usage.

        Args:
            label: Optional label for the snapshot

        Returns:
            The memory usage at the time of the snapshot
        """
        gc.collect()

        try:
            try:
                import psutil
            except ImportError:
                memory = {'rss': 0}
                self.snapshots.append(memory)
                self.labels.append(label or f"Snapshot {len(self.snapshots)}")
                return memory
                
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory = {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'shared': getattr(memory_info, 'shared', 0),
                'text': getattr(memory_info, 'text', 0),
                'data': getattr(memory_info, 'data', 0),
                'uss': getattr(process.memory_full_info(), 'uss', 0)
            }

        except (ImportError, Exception) as e:
            logging.warning(f"Could not get detailed memory usage: {e}")
            try:
                 memory = {'rss': process.memory_info().rss}
            except Exception:
                 memory = {'rss': 0}


        self.snapshots.append(memory)
        self.labels.append(label or f"Snapshot {len(self.snapshots)}")

        return memory

    def get_snapshots(self):
        """
        Get all snapshots.

        Returns:
            List of (label, memory) tuples
        """
        return list(zip(self.labels, self.snapshots))

    def get_memory_growth(self):
        """
        Calculate memory growth between snapshots.

        Returns:
            List of (label, growth) tuples
        """
        if len(self.snapshots) < 2:
            return []

        growth = []
        for i in range(1, len(self.snapshots)):
            diff = {}
            for key in self.snapshots[i-1]:
                if key in self.snapshots[i]:
                    diff[key] = self.snapshots[i][key] - self.snapshots[i-1][key]

            growth.append((f"{self.labels[i-1]} -> {self.labels[i]}", diff))

        return growth

    def print_report(self):
        """Print a report of memory usage over time."""
        if not self.snapshots:
            print("No memory snapshots taken.")
            return

        print("\nMemory Usage Report:")
        print("====================")

        print("\nSnapshots:")
        for i, (label, snapshot) in enumerate(self.get_snapshots()):
            print(f"\n{i+1}. {label}:")
            for key, value in snapshot.items():
                if isinstance(value, int) and value > 1024*1024:
                    print(f"  {key}: {value / (1024*1024):.2f} MB")
                else:
                    print(f"  {key}: {value}")

        growth = self.get_memory_growth()
        if growth:
            print("\nMemory Growth:")
            for label, diff in growth:
                print(f"\n{label}:")
                for key, value in diff.items():
                    if isinstance(value, int) and abs(value) > 1024*1024:
                        sign = "+" if value > 0 else ""
                        print(f"  {key}: {sign}{value / (1024*1024):.2f} MB")
                    elif isinstance(value, int) and value != 0:
                        sign = "+" if value > 0 else ""
                        print(f"  {key}: {sign}{value} bytes")


def measure_method_call_overhead(iterations: int = 1000000) -> float:
    """
    Measure the overhead of method calls compared to function calls.

    Args:
        iterations: Number of iterations to measure (default: 1,000,000)

    Returns:
        The overhead per call in milliseconds
    """
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    try:
        def empty_func():
            pass

        start = time.perf_counter_ns()
        for _ in range(iterations):
            empty_func()
        func_time = time.perf_counter_ns() - start

        class TestClass:
            def empty_method(self):
                pass

        test_obj = TestClass()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            test_obj.empty_method()
        method_time = time.perf_counter_ns() - start

        overhead_per_call_ns = (method_time - func_time) / iterations
        return overhead_per_call_ns / 1e6

    finally:
        if gc_was_enabled:
            gc.enable()

@contextlib.contextmanager
def robust_capture_output():
    """
    A more robust context manager for capturing output.
    Uses StringIO buffers and temporary redirection to avoid file descriptor issues.

    This context manager safely captures stdout and stderr and provides the captured
    content as a dictionary through the yielded object.

    Usage:
        with robust_capture_output() as captured:
            # Run code that produces output
            print("Hello world")

        # Access captured output
        stdout = captured['stdout']  # Contains "Hello world\n"
        stderr = captured['stderr']  # Empty string if no errors
    """
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    captured = {'stdout': '', 'stderr': ''}

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        yield captured

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        captured['stdout'] = stdout_buffer.getvalue()
        captured['stderr'] = stderr_buffer.getvalue()

        if captured['stdout']:
            logging.debug(f"robust_capture_output captured stdout: '{captured['stdout'][:100]}...'")
        if captured['stderr']:
            logging.debug(f"robust_capture_output captured stderr: '{captured['stderr'][:100]}...'")

        stdout_buffer.close()
        stderr_buffer.close()

def time_execution_ns(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_runs: int = 5,
    warmup_runs: int = 3,
    capture_output: bool = False,
    working_dir: str = None,
    is_baseline: bool = False,
    solver_module = None
) -> Dict[str, Any]:
    """
    Times a function using time.perf_counter_ns with warmups and GC control.

    Args:
        func: The function to time.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        num_runs: Number of timed measurement runs.
        warmup_runs: Number of untimed warmup runs.
        capture_output: Whether to capture stdout/stderr from func.
        working_dir: Optional working directory for the function calls
        is_baseline: If True, uses more relaxed memory limits for baseline algorithms
        solver_module: Optional module containing the solver for cache clearing

    Returns:
        A dictionary containing timing results and statistics in nanoseconds.
    """
    global os
    if kwargs is None:
        kwargs = {}

    try:
        n_thr = set_blas_threads()
        log_current_blas_threads(f"[time_execution_ns:{func.__name__}] ")
        log_cpu_affinity(f"[time_execution_ns:{func.__name__}] ")
        log_thread_env(f"[time_execution_ns:{func.__name__}] ")
    except Exception as _blas_e:
        logging.debug(f"time_execution_ns: could not configure BLAS threads – {_blas_e}")

    _initialize_timing_system()
    overhead_ns = (_timing_overhead_ns or 0) + (_capture_overhead_ns or 0)
    func_name = getattr(func, '__name__', 'anonymous')
    logger = logging.getLogger(__name__)
    logging.error(f"TIME_EXECUTION_NS_ENTRY_DEBUG: func='{func_name}', requested_warmups={warmup_runs}, requested_runs={num_runs}, capture_output={capture_output}")
    logging.info(f"time_execution_ns ENTER: func='{func_name}', requested_warmups={warmup_runs}, requested_runs={num_runs}, capture_output={capture_output}")
    
    initial_module_state = None

    try:
        import resource
        memory_limit_gb = 14
        memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        
        current_limit = resource.getrlimit(resource.RLIMIT_AS)
        if current_limit[0] == resource.RLIM_INFINITY or current_limit[0] > memory_limit_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
            logging.debug(f"Set {memory_limit_gb}GB memory limit for {func_name}")
        else:
            current_gb = current_limit[0] / (1024**3)
            logging.debug(f"Using existing {current_gb:.1f}GB memory limit for {func_name}")
            
    except (OSError, ValueError) as e:
        logging.warning(f"Could not set memory limit for {func_name}: {e}")

    results = {
        "success": False,
        "values_ns": [],
        "num_runs_executed": 0,
        "error": None,
        "traceback": None,
        "result": None,
        "first_warmup_result": None,
        "stdout": "",
        "stderr": "",
        "mean_ns": None, "median_ns": None, "stddev_ns": None,
        "min_ns": None, "max_ns": None, "ci_low_ns": None, "ci_high_ns": None,
        "mean_time_ms": None,
        "median_time_ms": None,
        "min_time_ms": None,
        "max_time_ms": None,
        "stddev_time_ms": None,
        "ci_low_time_ms": None,
        "ci_high_time_ms": None
    }

    _collect_overhead = os.environ.get("TIMING_OVERHEAD_DEBUG", "0") == "1"
    if _collect_overhead:
        results["pre_overhead_ns"] = []   # time from loop entry until immediately before start_ns
        results["post_overhead_ns"] = []  # time from end_ns until loop exit

    all_stdout = io.StringIO()
    all_stderr = io.StringIO()
    # Determine capture context based on the flag
    # This context manager will handle redirecting stdout/stderr if capture_output is True
    # or do nothing if capture_output is False.

    # --- Warmup Phase ---
    warmup_success = True
    logging.info(f"time_execution_ns: Starting warmup phase for '{func_name}' ({warmup_runs} iterations).")
    for i in range(warmup_runs):
        logging.debug(f"time_execution_ns: Warmup run {i+1}/{warmup_runs} for '{func_name}' starting...")
        current_warmup_stdout = io.StringIO()
        current_warmup_stderr = io.StringIO()
        warmup_capture_ctx = redirect_stdout(current_warmup_stdout) if capture_output else nullcontext()
        warmup_error_ctx = redirect_stderr(current_warmup_stderr) if capture_output else nullcontext()
        try:
            # BEGIN pre-section timing --------------------------
            if _collect_overhead:
                _t_pre_start = time.perf_counter_ns()
            # Check memory usage before execution to prevent OOM kills
            if MEMORY_MONITORING:
                try:
                    # Use process-level helper if available
                    from AlgoTuner.utils.process_monitor import check_worker_memory
                    mem_err = check_worker_memory()
                    if mem_err:
                        raise mem_err
                except ImportError:
                    # Lightweight fallback using psutil
                    try:
                        import psutil, os
                        if psutil.virtual_memory().percent > 90:
                            logging.warning("System memory usage high (>90%) – continuing (RLIMIT_AS enforced)")
                        # No further action; RLIMIT_AS already provides the hard cap
                    except ImportError:
                        pass
            if _collect_overhead:
                _t_pre_end = time.perf_counter_ns()
            logging.debug(f"time_execution_ns: About to execute warmup run {i+1} for '{func_name}'")
            current_run_result = None
            if working_dir:
                logging.debug(f"time_execution_ns: Using working directory {working_dir} for warmup run {i+1}")
                with with_working_dir(working_dir), warmup_capture_ctx, warmup_error_ctx, memory_monitor_context():
                    current_run_result = func(*args, **kwargs)
            else:
                with warmup_capture_ctx, warmup_error_ctx, memory_monitor_context():
                    current_run_result = func(*args, **kwargs)

            # Do not store the warmup result to prevent holding onto large objects.
            # if i == 0:
            #     results["first_warmup_result"] = current_run_result

            logging.debug(f"time_execution_ns: Warmup run {i+1}/{warmup_runs} for '{func_name}' completed successfully.")
        except Exception as e:
            tb_str = traceback.format_exc()
            logging.warning(f"time_execution_ns: Warmup run {i+1}/{warmup_runs} for '{func_name}' FAILED. Error: {e}")
            logging.debug(f"time_execution_ns: Warmup failure traceback: {tb_str}")
            if results["error"] is None: # Store first error
                results["error"] = f"Error: {str(e)}"
                results["traceback"] = tb_str
            warmup_success = False
            break # Exit warmup loop on first error
        finally:
            # Explicitly free memory from the warmup run - no monitoring, just cleanup
            if 'current_run_result' in locals() and current_run_result is not None:
                del current_run_result
                gc.collect()

            # Don't capture output during warmup to avoid duplicates - only capture from measurement runs
            # if capture_output and i == 0:
            #     all_stdout.write(current_warmup_stdout.getvalue())
            #     all_stderr.write(current_warmup_stderr.getvalue())
            current_warmup_stdout.close()
            current_warmup_stderr.close()
            
            
    logging.info(f"time_execution_ns: Warmup phase for '{func_name}' finished. Warmup success: {warmup_success}. Error (if any): {results['error']}")

    if warmup_success:
        logging.info(f"time_execution_ns: Starting measurement runs for '{func_name}' (requested: {num_runs}).")
        try:
            current_result = None
            for i in range(num_runs):
                if current_result is not None:
                    del current_result
                    current_result = None
                    gc.collect()

                logging.debug(f"time_execution_ns: Measurement run {i+1}/{num_runs} for '{func_name}' starting...")
                current_run_stdout = io.StringIO()
                current_run_stderr = io.StringIO()
                run_capture_ctx = redirect_stdout(current_run_stdout) if capture_output else nullcontext()
                run_error_ctx = redirect_stderr(current_run_stderr) if capture_output else nullcontext()
                try:
                    if _collect_overhead:
                        _t_pre_start = time.perf_counter_ns()
                    if MEMORY_MONITORING:
                        try:
                            import psutil, os
                            if psutil.virtual_memory().percent > 90:
                                logging.warning("System memory usage high (>90%) – continuing (RLIMIT_AS enforced)")
                        except ImportError:
                            pass
                    if _collect_overhead:
                        _t_pre_end = time.perf_counter_ns()
                    logging.debug(f"time_execution_ns: About to execute measurement run {i+1} for '{func_name}'")
                    if working_dir:
                        logging.debug(f"time_execution_ns: Using working directory {working_dir} for measurement run {i+1}")
                        with with_working_dir(working_dir), run_capture_ctx, run_error_ctx, memory_monitor_context():
                            start_ns = time.perf_counter_ns()
                            current_result = func(*args, **kwargs)
                            end_ns = time.perf_counter_ns()
                    else:
                        with run_capture_ctx, run_error_ctx, memory_monitor_context():
                            start_ns = time.perf_counter_ns()
                            current_result = func(*args, **kwargs)
                            end_ns = time.perf_counter_ns()
                    if _collect_overhead:
                        _t_post_end = time.perf_counter_ns()
                        results["pre_overhead_ns"].append(max(0, _t_pre_end - _t_pre_start))
                        results["post_overhead_ns"].append(max(0, _t_post_end - end_ns))
                    
                    raw_duration_ns = end_ns - start_ns
                    run_duration_ns = max(0, raw_duration_ns - overhead_ns)
                    results["values_ns"].append(run_duration_ns)
                    results["num_runs_executed"] += 1
                    
                    logging.debug(f"time_execution_ns: Measurement run {i+1}/{num_runs} for '{func_name}' successful. Duration: {run_duration_ns/1e6:.6f} ms.")
                    
                    try:
                        import psutil
                        current_process = psutil.Process()
                        rss_gb = current_process.memory_info().rss / (1024**3)
                    except ImportError:
                        pass
                    except MemoryError:
                        raise
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logging.warning(f"time_execution_ns: Measurement run {i+1}/{num_runs} for '{func_name}' FAILED. Error: {e}")
                    logging.debug(f"time_execution_ns: Measurement failure traceback: {tb_str}")
                    if results["error"] is None: 
                        results["error"] = f"Error: {str(e)}"
                        results["traceback"] = tb_str
                    # Decide whether to break or continue. Current logic implies continuing to try all runs.
                    # If we break here, num_runs_executed will be less than num_runs.
                    # break # Uncomment to stop on first measurement error
                finally:
                    # Capture output from first measurement run to show actual solver execution output
                    if capture_output and i == 0:
                        all_stdout.write(current_run_stdout.getvalue())
                        all_stderr.write(current_run_stderr.getvalue())
                    current_run_stdout.close()
                    current_run_stderr.close()
                    
                    # If it's not the last run, clean up the result to save memory - no monitoring
                    if i < num_runs - 1 and 'current_result' in locals() and current_result is not None:
                        del current_result
                        current_result = None
                        gc.collect()

                
            # After loop, store the last result if it exists
            if current_result is not None:
                results["result"] = current_result

        except MemoryError as e:
            # Handle memory limit exceeded - return error result immediately
            return _handle_memory_error(e, func, i if 'i' in locals() else 0, num_runs)
    else:
        logging.warning(f"time_execution_ns: Skipping measurement runs for '{func_name}' due to warmup failure.")

    # --- Calculate statistics and determine final success ---
    logging.info(f"time_execution_ns: Finished measurement phase for '{func_name}'. Total runs executed: {results['num_runs_executed']}/{num_runs}.")
    
    if results["num_runs_executed"] > 0:
        logging.info(f"time_execution_ns: Calculating statistics for '{func_name}' with {results['num_runs_executed']} successful runs")
        # LOG: Check values_ns before statistics calculation
        logging.info(f"TIMING_DEBUG: values_ns list for '{func_name}': {results['values_ns']} (length: {len(results['values_ns'])})")
        logging.info(f"TIMING_DEBUG: values_ns data types: {[type(v) for v in results['values_ns']]}")
        # Calculate statistics using the standard statistics module only
        import statistics as _stats

        try:
            vals = results["values_ns"]

            results["mean_ns"] = float(_stats.mean(vals))
            results["median_ns"] = float(_stats.median(vals))
            results["min_ns"] = float(min(vals))
            results["max_ns"] = float(max(vals))

            if len(vals) > 1:
                results["stddev_ns"] = float(_stats.stdev(vals))
            else:
                results["stddev_ns"] = 0.0

            # Convert to milliseconds
            results["mean_time_ms"] = results["mean_ns"] / 1e6 if results["mean_ns"] is not None else None
            results["median_time_ms"] = results["median_ns"] / 1e6 if results["median_ns"] is not None else None
            results["min_time_ms"] = results["min_ns"] / 1e6 if results["min_ns"] is not None else None
            results["max_time_ms"] = results["max_ns"] / 1e6 if results["max_ns"] is not None else None
            results["ci_low_time_ms"] = results["ci_low_ns"] / 1e6 if results["ci_low_ns"] is not None else None
            results["ci_high_time_ms"] = results["ci_high_ns"] / 1e6 if results["ci_high_ns"] is not None else None

            logging.info(
                f"time_execution_ns: Stats for '{func_name}' (ms): "
                f"Mean={results['mean_time_ms']:.3f}, Median={results['median_time_ms']:.3f}, "
                f"Min={results['min_time_ms']:.3f}, Max={results['max_time_ms']:.3f}"
            )
        except Exception as stat_exc:
            logging.error(
                f"TIMING_DEBUG: Statistics calculation FAILED for '{func_name}'. Error: {stat_exc}. values_ns: {results['values_ns']}"
            )
            logging.error(f"TIMING_DEBUG: Full traceback: {traceback.format_exc()}")
            logging.warning(
                f"time_execution_ns: Could not calculate statistics for '{func_name}'. Error: {stat_exc}"
            )
            # Ensure stats are None if calculation failed
            results["mean_ns"] = results["median_ns"] = results["min_ns"] = results["max_ns"] = results["stddev_ns"] = None
            results["mean_time_ms"] = results["median_time_ms"] = results["min_time_ms"] = results["max_time_ms"] = results["ci_low_time_ms"] = results["ci_high_time_ms"] = None

        if results["num_runs_executed"] == num_runs:
            results["success"] = True # Full success only if all requested runs completed
            logging.info(f"time_execution_ns: All {num_runs} measurement runs were successful for '{func_name}'. Setting success=True.")
        else:
            results["success"] = False # Partial success
            logging.warning(f"time_execution_ns: Only {results['num_runs_executed']}/{num_runs} measurement runs were successful for '{func_name}'. Setting success=False.")
            if results["error"] is None:
                 results["error"] = f"Only {results['num_runs_executed']}/{num_runs} measurement runs succeeded."
    else: # No runs executed successfully (either warmup failed or all measurement runs failed)
        results["success"] = False
        logging.warning(f"time_execution_ns: No measurement runs were successful for '{func_name}'. Setting success=False.")
        if results["error"] is None: # Ensure error message exists
            if not warmup_success:
                results["error"] = results.get("error", "Warmup failed, leading to no measurement runs.") # Preserve original warmup error if any
            else:
                results["error"] = "All measurement runs failed but no specific error was captured."
        # Ensure ms stats are also None here if no runs
        results["mean_time_ms"] = results["median_time_ms"] = results["min_time_ms"] = results["max_time_ms"] = results["ci_low_time_ms"] = results["ci_high_time_ms"] = None
    
    # Store captured stdout/stderr
    if capture_output:
        results["stdout"] = all_stdout.getvalue()
        results["stderr"] = all_stderr.getvalue()
    all_stdout.close()
    all_stderr.close()

    # --- SAFE IN-PROCESS VALIDATION + STRIPPING (prevents validation issues) ---
    # Apply the same pattern as in AlgoTuner/utils/evaluator/runner.py lines 367-408
    # This ensures validation runs FIRST before any result stripping occurs
    if results.get("success") and results.get("result") is not None:
        try:
            solver_result = results.get("result")
            
            # Try to perform in-process validation if we can detect solver context
            # This requires access to task_instance and problem, which may not be available
            # In time_execution_ns context. We'll implement size-based stripping for now.
            logging.info(f"time_execution_ns: Checking result size for potential stripping to avoid memory issues")
            
            # Use the same size estimation logic as runner.py
            import sys
            import numpy as np
            
            def get_size_mb(res):
                size_bytes = 0
                if isinstance(res, np.ndarray):
                    size_bytes = res.nbytes
                elif isinstance(res, list) and res and all(isinstance(x, np.ndarray) for x in res):
                    size_bytes = sum(arr.nbytes for arr in res)
                else:
                    size_bytes = sys.getsizeof(res)
                return size_bytes / (1024 * 1024)
                
            result_size_mb = get_size_mb(solver_result)

            # Always strip solver results after validation - no reason to keep actual outputs
            logging.info(f"time_execution_ns: Always replacing solver result with summary (size: {result_size_mb:.2f}MB) to prevent memory waste.")
            
            # Create the same type of summary as runner.py
            result_summary = {
                "type": str(type(solver_result)),
                "size_mb": round(result_size_mb, 2),
                "stripped_in_timing": True,  # Flag to indicate this was stripped in timing phase
            }
            
            # Add shape/length info if available
            if hasattr(solver_result, '__len__'):
                result_summary["length"] = len(solver_result)
                if hasattr(solver_result, 'shape'):  # numpy array
                    result_summary["shape"] = str(solver_result.shape)
                    if hasattr(solver_result, 'dtype'):
                        result_summary["dtype"] = str(solver_result.dtype)
            
            # Add LazyOuterMemmap specific info
            if hasattr(solver_result, 'filename'):
                result_summary["lazy_outer_memmap"] = True
                result_summary["filename"] = solver_result.filename
                result_summary["shape"] = str(solver_result.shape)
                result_summary["dtype"] = str(solver_result.dtype)
            
            results["result"] = result_summary
            logging.info(f"time_execution_ns: Replaced solver result with summary: {result_summary}")
                
        except Exception as e:
            logging.warning(f"time_execution_ns: Failed to perform result size check/stripping: {e}")
            # Continue with original result if stripping fails
    
    # Remove the old result stripping logic (lines that follow)
    # Strip out large solver results to prevent MemoryError during multiprocessing communication
    # Only keep timing data, not actual solver outputs which can be huge numpy arrays
    
    # COMPREHENSIVE TIME_EXECUTION_NS FINAL TIMING DEBUG: Log all timing fields before return
    timing_exit_fields = ["success", "values_ns", "num_runs_executed", "mean_ns", "median_ns", "stddev_ns", "min_ns", "max_ns", "mean_time_ms", "median_time_ms", "stddev_time_ms", "min_time_ms", "max_time_ms", "error", "traceback"]
    timing_exit_debug = {field: results.get(field) for field in timing_exit_fields}
    logging.info(f"TIME_EXECUTION_NS_EXIT_DEBUG: Final results from time_execution_ns for '{func_name}': {timing_exit_debug}")
    
    # Specific check for timing values that should be converted to benchmark format
    if results.get("success") and results.get("values_ns"):
        logging.info(f"TIME_EXECUTION_NS_EXIT_DEBUG: SUCCESS CASE - values_ns has {len(results['values_ns'])} values, min_ns={results.get('min_ns')}, mean_ns={results.get('mean_ns')}")
        logging.info(f"TIME_EXECUTION_NS_EXIT_DEBUG: SUCCESS CASE - min_time_ms={results.get('min_time_ms')}, mean_time_ms={results.get('mean_time_ms')}")
    else:
        logging.warning(f"TIME_EXECUTION_NS_EXIT_DEBUG: FAILURE CASE - success={results.get('success')}, values_ns length={len(results.get('values_ns', []))}, error={results.get('error')}")
    
    logging.info(f"time_execution_ns EXIT: func='{func_name}'. Final success: {results['success']}, Runs executed: {results['num_runs_executed']}/{num_runs}. Error: {results['error']}")

    # Attach aggregated overhead stats if collected
    if _collect_overhead and results.get("pre_overhead_ns"):
        try:
            results["pre_overhead_stats_ns"] = {
                "mean": _stats.mean(results["pre_overhead_ns"]),
                "min": min(results["pre_overhead_ns"]),
                "max": max(results["pre_overhead_ns"])
            }
            results["post_overhead_stats_ns"] = {
                "mean": _stats.mean(results["post_overhead_ns"]),
                "min": min(results["post_overhead_ns"]),
                "max": max(results["post_overhead_ns"])
            }
            logging.info(
                f"OVERHEAD_DEBUG '{func_name}': pre-mean={results['pre_overhead_stats_ns']['mean']/1e6:.3f}ms, "
                f"post-mean={results['post_overhead_stats_ns']['mean']/1e6:.3f}ms"
            )
        except Exception:
            pass

    # Cache cheating detection disabled

    return results
