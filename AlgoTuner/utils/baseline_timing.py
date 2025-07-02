"""
Baseline Timing Module

Simplified timing infrastructure specifically for baseline evaluation.
No memory monitoring, no complex context managers, no threads.
Designed to avoid deadlocks and timeouts that occur in the main timing system.
"""

import time
import statistics
import logging
import traceback
import gc
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# NEW: standardise BLAS thread usage for baseline timing
try:
    from AlgoTuner.utils.blas_utils import set_blas_threads, log_current_blas_threads, log_cpu_affinity, log_thread_env
    n_thr = set_blas_threads()  # uses env or cpu_count()
    log_current_blas_threads("[baseline_timing] ")
    log_cpu_affinity("[baseline_timing] ")
    log_thread_env("[baseline_timing] ")
    logger.info(f"baseline_timing: BLAS threads set to {n_thr}")
except Exception as _blas_e:
    logger.debug(f"baseline_timing: could not configure BLAS threads – {_blas_e}")

MEMORY_MONITORING = False  # disable per-run psutil checks for speed

def time_baseline_function(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_runs: int = 10,
    warmup_runs: int = 3,
    solver_module = None,
) -> Dict[str, Any]:
    """
    Baseline timing wrapper that now simply delegates to
    `AlgoTuner.utils.precise_timing.time_execution_ns` so that all timing
    paths (agent mode, baseline mode, dataset generation, k-search, …)
    share the exact same code base.

    The public signature and the legacy return-value format are preserved
    so that existing callers in `evaluator.main` continue to work without
    modification.
    """
    from AlgoTuner.utils.precise_timing import time_execution_ns

    if kwargs is None:
        kwargs = {}

    # Run the canonical timing routine
    timing_result = time_execution_ns(
        func=func,
        args=args,
        kwargs=kwargs,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        capture_output=False,  # Baseline eval does not need stdout capture
        working_dir=None,
        solver_module=solver_module,
    )

    # Map to the historical baseline_timing result structure
    values_ns = timing_result.get("values_ns", [])
    return {
        "success": timing_result.get("success", False),
        "min_time_ms": timing_result.get("min_time_ms"),
        "mean_time_ms": timing_result.get("mean_time_ms"),
        "median_time_ms": timing_result.get("median_time_ms"),
        "values_ms": [v / 1e6 for v in values_ns],
        "num_runs_executed": timing_result.get("num_runs_executed", 0),
        "result": timing_result.get("result"),
        "error": timing_result.get("error"),
        "traceback": timing_result.get("traceback"),
        "code_context": timing_result.get("code_context"),
        "timeout_occurred": timing_result.get("timeout_occurred", False),
    }


def time_baseline_with_timeout(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_runs: int = 5,
    warmup_runs: int = 3,
    timeout_seconds: float = 30.0
) -> Dict[str, Any]:
    """
    Time baseline function with a simple timeout mechanism.
    Uses signal-based timeout on Unix systems.
    
    Args:
        func: The function to time
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs
        timeout_seconds: Maximum time allowed for entire timing process
        
    Returns:
        Same as time_baseline_function, with timeout_occurred flag
    """
    import signal
    import platform
    
    if kwargs is None:
        kwargs = {}
    
    func_name = getattr(func, '__name__', 'anonymous')
    logger.info(f"baseline_timing: Starting timing with {timeout_seconds}s timeout for '{func_name}'")
    
    logger.warning(f"baseline_timing: Signal timeout disabled in multiprocessing context, relying on pool timeout")
    
    # Add start time for progress tracking
    start_time = time.time()
    logger.info(f"baseline_timing: Starting time_baseline_function at {start_time}")
    
    try:
        result = time_baseline_function(func, args, kwargs, num_runs, warmup_runs)
        elapsed = time.time() - start_time
        logger.info(f"baseline_timing: time_baseline_function completed successfully in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time  
        logger.error(f"baseline_timing: time_baseline_function failed after {elapsed:.2f}s: {e}")
        return {
            "success": False,
            "min_time_ms": None,
            "mean_time_ms": None,
            "median_time_ms": None,
            "values_ms": [],
            "num_runs_executed": 0,
            "result": None,
            "error": str(e),
            "timeout_occurred": False
        }