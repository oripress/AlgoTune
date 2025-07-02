import multiprocessing
from multiprocessing import Pool, cpu_count
from AlgoTuner.utils.precise_timing import _initialize_timing_system, time_execution_ns
import atexit
import os  # needed for setting threading environment variables
import logging
import time
import pickle
import statistics
from AlgoTuner.utils.discover_and_list_tasks import discover_and_import_tasks

# Storage for pool-based timing calibration
_pickle_overhead_ns = None

# Initialize each worker process with enforced single-threaded BLAS and timing calibration
def _worker_initializer():
    # Disable GC entirely in worker for consistent timing
    import gc as _gc
    if _gc.isenabled():
        _gc.disable()
    # Pre-import heavy modules to warm code caches (NumPy, tasks, evaluator, benchmark)
    import importlib
    modules_to_preload = [
        'numpy', 'json', 'config.loader', 'tasks.base',
        'utils.serialization', 'utils.casting',
        'utils.evaluator.main', 'utils.benchmark',
        'utils.k_search', 'tasks.registry'
    ]
    for m in modules_to_preload:
        try:
            importlib.import_module(m)
        except ImportError:
            pass
    # Load all tasks to ensure task modules are imported
    try:
        discover_and_import_tasks()
    except ImportError:
        pass
    # Now initialize timing system (calibration, pinning, etc.)
    _initialize_timing_system()
    # Skip pickle overhead calibration to reduce startup time
    global _pickle_overhead_ns
    _pickle_overhead_ns = 0
    # Pin each worker to a distinct physical core to avoid contention
    try:
        import multiprocessing as _mp, psutil
        proc = _mp.current_process()
        # Identity is a tuple like (i,), so extract worker index
        worker_id = proc._identity[0] if hasattr(proc, '_identity') and proc._identity else 1
        # Determine number of physical cores
        num_phys = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
        core_id = (worker_id - 1) % num_phys
        psutil.Process(os.getpid()).cpu_affinity([core_id])
        logging.debug(f"BenchmarkPool worker {worker_id} pinned to physical core {core_id}")
    except Exception as e:
        logging.debug(f"Could not set per-worker CPU affinity: {e}")

# Task wrapper for Pool
def _run_task(task_args):
    func, func_args, func_kwargs, num_runs, warmup_runs = task_args
    # Execute timing internally
    result = time_execution_ns(
        func=func,
        args=func_args,
        kwargs=func_kwargs,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
    )
    # Subtract pickle/unpickle overhead from measured durations
    if _pickle_overhead_ns and result.get("values_ns"):
        vals = [max(0, v - _pickle_overhead_ns) for v in result["values_ns"]]
        result["values_ns"] = vals
        try:
            result["mean_ns"] = statistics.mean(vals) if vals else result.get("mean_ns")
            result["median_ns"] = statistics.median(vals) if vals else result.get("median_ns")
            result["stddev_ns"] = statistics.stdev(vals) if len(vals) > 1 else result.get("stddev_ns")
            result["min_ns"] = min(vals) if vals else result.get("min_ns")
            result["max_ns"] = max(vals) if vals else result.get("max_ns")
        except Exception:
            pass
    return result

class BenchmarkPool:
    """
    A singleton-style pool of worker processes to perform timing tasks repeatedly.
    
    DEPRECATED: This class is deprecated in favor of ProcessPoolManager.
    Use ProcessPoolManager.get_pool() for new code.
    """
    _pool = None

    @classmethod
    def start(cls, processes=None):
        if cls._pool is None:
            logging.warning("BenchmarkPool is deprecated. Consider using ProcessPoolManager instead.")
            # Use forkserver context for better isolation
            ctx = multiprocessing.get_context('forkserver')
            # Start pool without initializer to avoid long startup
            cls._pool = ctx.Pool(processes=processes or cpu_count())

    @classmethod
    def stop(cls):
        if cls._pool is not None:
            cls._pool.close()
            cls._pool.join()
            cls._pool = None

    @classmethod
    def run(cls, func, args=(), kwargs=None, num_runs=5, warmup_runs=3):
        if cls._pool is None:
            cls.start()
        task = (func, args, kwargs or {}, num_runs, warmup_runs)
        return cls._pool.apply(_run_task, (task,))

# Ensure the pool is stopped at program exit to clean up worker processes
atexit.register(BenchmarkPool.stop) 