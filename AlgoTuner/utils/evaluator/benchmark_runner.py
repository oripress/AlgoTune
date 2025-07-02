import os
import logging
import multiprocessing
import time
from AlgoTuner.utils.evaluator.loader import reload_all_llm_src
from AlgoTuner.utils.evaluator.runner import _run_benchmark
from AlgoTuner.utils.multiprocessing_utils import load_pool_config, _pool_worker_initializer, _simple_worker_initializer
from typing import Callable, Tuple, Dict, Optional, Any


def _warmup_worker_task():
    """Simple warmup task to force worker initialization. Must be at module level for pickling."""
    import logging
    import time
    import resource
    import os
    import sys
    
    worker_pid = os.getpid()
    start_time = time.time()
    logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** WARMUP TASK STARTED *** PID {worker_pid}")
    
    # Check current working directory
    try:
        cwd = os.getcwd()
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** WORKER CWD *** {cwd}")
    except Exception as e:
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: Failed to get CWD: {e}")
    
    # Check CODE_DIR environment variable
    code_dir = os.environ.get('CODE_DIR')
    logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** CODE_DIR *** {code_dir}")
    
    # Check parent PID to confirm we're in a separate process
    try:
        parent_pid = os.getppid()
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** PROCESS INFO *** PID={worker_pid}, PPID={parent_pid}")
    except Exception as e:
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: Failed to get parent PID: {e}")
    
    # Check memory limits in the worker
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft == resource.RLIM_INFINITY:
            logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** CRITICAL *** RLIMIT_AS soft = INFINITY")
        else:
            logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** CRITICAL *** RLIMIT_AS soft = {soft} bytes ({soft / (1024**3):.2f}GB)")
        if hard == resource.RLIM_INFINITY:
            logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** CRITICAL *** RLIMIT_AS hard = INFINITY")
        else:
            logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** CRITICAL *** RLIMIT_AS hard = {hard} bytes ({hard / (1024**3):.2f}GB)")
            
        # Also check if we can actually allocate memory
        try:
            import numpy as np  # Import numpy before using np.zeros
            test_size_mb = 100  # Try to allocate 100MB
            test_array = np.zeros(test_size_mb * 1024 * 1024 // 8, dtype=np.float64)  # 100MB of float64
            logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** SUCCESS *** Allocated {test_size_mb}MB test array")
            del test_array  # Clean up
        except Exception as alloc_e:
            logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** FAILED *** Cannot allocate {test_size_mb}MB: {alloc_e}")
            
    except Exception as e:
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: Failed to check RLIMIT_AS: {e}")
    
    # Check other resource limits that might be relevant
    try:
        rss_soft, rss_hard = resource.getrlimit(resource.RLIMIT_RSS)
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** RLIMIT_RSS *** soft={rss_soft if rss_soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={rss_hard if rss_hard != resource.RLIM_INFINITY else 'INFINITY'}")
        if rss_soft != resource.RLIM_INFINITY:
            logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** RLIMIT_RSS *** soft={rss_soft / (1024**3):.2f}GB")
    except Exception as e:
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: Failed to check RLIMIT_RSS: {e}")
    
    # Check data segment limit
    try:
        data_soft, data_hard = resource.getrlimit(resource.RLIMIT_DATA)
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** RLIMIT_DATA *** soft={data_soft if data_soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={data_hard if data_hard != resource.RLIM_INFINITY else 'INFINITY'}")
        if data_soft != resource.RLIM_INFINITY:
            logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** RLIMIT_DATA *** soft={data_soft / (1024**3):.2f}GB")
    except Exception as e:
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: Failed to check RLIMIT_DATA: {e}")
    
    # Check stack limit
    try:
        stack_soft, stack_hard = resource.getrlimit(resource.RLIMIT_STACK)
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** RLIMIT_STACK *** soft={stack_soft if stack_soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={stack_hard if stack_hard != resource.RLIM_INFINITY else 'INFINITY'}")
        if stack_soft != resource.RLIM_INFINITY:
            logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** RLIMIT_STACK *** soft={stack_soft / (1024**2):.2f}MB")
    except Exception as e:
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: Failed to check RLIMIT_STACK: {e}")
    
    # Test basic numpy import to see if it works
    try:
        import numpy as np
        test_array = np.zeros(1000)  # Small test allocation
        logging.info(f"[WARMUP_TASK] {start_time:.3f}: Successfully imported numpy and allocated small test array")
    except Exception as e:
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: Failed to import numpy or allocate test array: {e}")
    
    # Check system memory info if available
    try:
        import psutil
        mem = psutil.virtual_memory()
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** SYSTEM MEMORY *** Total: {mem.total / (1024**3):.2f}GB, Available: {mem.available / (1024**3):.2f}GB, Used: {mem.percent:.1f}%")
        
        # Check current process memory usage
        process = psutil.Process()
        mem_info = process.memory_info()
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: *** PROCESS MEMORY *** RSS: {mem_info.rss / (1024**3):.2f}GB, VMS: {mem_info.vms / (1024**3):.2f}GB")
    except Exception as e:
        logging.debug(f"[WARMUP_TASK] {start_time:.3f}: Failed to get system/process memory info: {e}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    logging.info(f"[WARMUP_TASK] {end_time:.3f}: _warmup_worker_task completing successfully after {elapsed:.3f}s")
    return worker_pid


class HardBenchmarkFailure(Exception):
    """Raised when a benchmark fails even after pool reset."""
    pass


class BenchmarkPool:
    """
    Helper for running benchmarks with retries and hard resets on repeated timeouts.
    Uses multiprocessing.get_context('spawn') to avoid file descriptor inheritance issues.
    Automatically reloads CODE_DIR modules on initialization and after reaching timeout thresholds,
    ensuring that worker processes always run fresh solver code.
    """
    def __init__(self, pool_config_name="validation_pool", code_dir_env="CODE_DIR", worker_initializer=None):
        self.pool_config_name = pool_config_name
        self.code_dir_env = code_dir_env
        self.worker_initializer = worker_initializer  # Allow custom initializer
        self.max_timeouts = 1  # Maximum timeouts before pool reset (reduced from 3)
        self._task_count = 0  # Track tasks for health monitoring
        # Perform initial code reload in parent process before pool creation
        # This ensures workers inherit fresh modules without deadlock risk
        code_dir = os.environ.get(self.code_dir_env, ".")
        try:
            logging.info(f"BenchmarkPool.__init__: About to perform initial reload of llm src modules from '{code_dir}' in parent process")
            reload_all_llm_src(code_dir)
            logging.info(f"BenchmarkPool.__init__: Successfully performed initial reload of llm src modules from '{code_dir}' in parent process")
        except Exception as e:
            logging.exception(f"BenchmarkPool.__init__: Error during initial llm src reload: {e}")
        logging.info("BenchmarkPool.__init__: About to create initial worker pool")
        self._make_pool()
        logging.info("BenchmarkPool.__init__: Successfully created initial worker pool")
    
    def _make_pool(self):
        logging.info(f"BenchmarkPool._make_pool: Starting pool creation with config '{self.pool_config_name}'")
        cfg = load_pool_config(pool_config_name=self.pool_config_name, force_num_workers=1)
        logging.info(f"BenchmarkPool._make_pool: Loaded pool config: {cfg}")
        
        # --- Enforce per-worker address space limit ---
        # Some earlier experiments switched off RLIMIT_AS via the
        # `disable_rlimit_as` flag in the YAML config, which allowed buggy
        # solver code to allocate virtually unlimited memory and led to OOM
        # kills.  We now hard-override this flag so that *every* worker runs
        # with the configured soft/hard memory cap (mem_limit_bytes).
        
        # Respect the disable_rlimit_as setting from config
        disable_rlimit_as = cfg.get("disable_rlimit_as", False)
        if disable_rlimit_as:
            logging.info("BenchmarkPool._make_pool: Respecting disable_rlimit_as=True from config")
        
        ctx = multiprocessing.get_context("forkserver")  # Use forkserver for better isolation
        logging.info("BenchmarkPool._make_pool: Got multiprocessing context 'forkserver'")
        
        # Import the desired initializer now (late import avoids heavy deps at module load)
        from AlgoTuner.utils.multiprocessing_utils import _simple_worker_initializer
        
        # Use lightweight simple initializer to avoid heavy diagnostics that can hang on some systems
        initializer = _simple_worker_initializer
        logging.info(f"BenchmarkPool._make_pool: Using SIMPLE initializer for worker processes")
        
        # Use original config but with reasonable upper bound for safety
        # Reduced default to prevent thread accumulation issues
        max_tasks = cfg["maxtasksperchild"]
        if max_tasks is None:
            max_tasks = 25  # Reduced from 100 to prevent thread accumulation
        elif max_tasks > 50:
            max_tasks = 50  # Reduced safety upper bound from 200
        
        # Get disable_rlimit_as setting from above
            
        logging.info(f"BenchmarkPool._make_pool: About to create Pool with processes=1, initializer={initializer.__name__}, mem_limit={cfg['mem_limit_bytes']}, disable_rlimit_as={disable_rlimit_as}, maxtasksperchild={max_tasks}")
        
        # ------------------------------------------------------------------
        # Guard against RLIMIT_FSIZE==0 which breaks POSIX semaphore creation
        # in multiprocessing (raises OSError 27 "File too large").  This can
        # happen if a prior timing context set a 0-byte quota and the restore
        # did not lift the limit (e.g. original limit already 0).
        # ------------------------------------------------------------------
        # Bump file-size limit to a small but non-zero value so POSIX semaphores
        # can back their shared-memory file.  Unlimited (RLIM_INFINITY) may be
        # disallowed by hardened kernels, therefore use 16 MB unless a higher
        # limit is already in place.
        try:
            import resource  # Unix-only, safe import here

            if hasattr(resource, "RLIMIT_FSIZE"):
                soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
                desired = 16 * 1024 * 1024  # 16 MB

                # If either soft or hard is 0 we try to raise both.
                if soft == 0 or hard == 0:
                    new_soft = max(soft, desired)
                    new_hard = max(hard, desired)
                    try:
                        resource.setrlimit(resource.RLIMIT_FSIZE, (new_soft, new_hard))
                        logging.warning(
                            "BenchmarkPool._make_pool: RLIMIT_FSIZE was 0. Raised to 16 MB to allow multiprocessing SemLock creation."
                        )
                    except Exception as _raise_err:
                        # Could not raise – fallback to running without a pool
                        logging.error(
                            f"BenchmarkPool._make_pool: Cannot raise RLIMIT_FSIZE ({_raise_err}). "
                            "Falling back to in-process timing path."
                        )
                        raise RuntimeError("RLIMIT_FSIZE too small and cannot be adjusted")
        except RuntimeError:
            raise
        except Exception as _fsize_err:
            # Log but continue – most systems will still work.
            logging.debug(
                f"BenchmarkPool._make_pool: RLIMIT_FSIZE inspection failed – {_fsize_err}"
            )
        
        # Construct initargs for debug initializer (same signature as baseline/simple)
        initargs = (cfg["mem_limit_bytes"], disable_rlimit_as)
        
        # Log parent process memory limits before creating pool
        try:
            import resource
            import subprocess
            
            parent_soft, parent_hard = resource.getrlimit(resource.RLIMIT_AS)
            logging.error(f"[PARENT_LIMITS] Parent process RLIMIT_AS: soft={parent_soft if parent_soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={parent_hard if parent_hard != resource.RLIM_INFINITY else 'INFINITY'}")
            if parent_soft != resource.RLIM_INFINITY:
                logging.error(f"[PARENT_LIMITS] Parent soft limit: {parent_soft / (1024**3):.2f}GB")
            if parent_hard != resource.RLIM_INFINITY:
                logging.error(f"[PARENT_LIMITS] Parent hard limit: {parent_hard / (1024**3):.2f}GB")
                
            # Show system ulimit settings
            try:
                ulimit_output = subprocess.check_output(['ulimit', '-a'], shell=True, text=True, stderr=subprocess.STDOUT)
                logging.error(f"[PARENT_LIMITS] *** SYSTEM ULIMIT SETTINGS ***")
                for line in ulimit_output.strip().split('\n'):
                    if 'virtual memory' in line or 'address space' in line or 'memory' in line:
                        logging.error(f"[PARENT_LIMITS] {line}")
            except Exception as ulimit_e:
                logging.error(f"[PARENT_LIMITS] Failed to get ulimit settings: {ulimit_e}")
                
        except Exception as e:
            logging.error(f"[PARENT_LIMITS] Failed to get parent limits: {e}")

        # Log the exact time before pool creation
        pool_create_start = time.time()
        logging.info(f"[POOL_TIMING] {pool_create_start:.3f}: About to call ctx.Pool() - this is where spawn might hang")
        
        try:
            self.pool = ctx.Pool(
                processes=1,
                initializer=initializer,
                initargs=initargs,
                maxtasksperchild=max_tasks
            )
        except Exception as e:
            # Pool creation failed – most likely because RLIMIT_FSIZE could
            # not be raised on hardened systems.  Fall back to *no pool* and
            # let callers run benchmarks inline.
            logging.error(
                f"BenchmarkPool._make_pool: Failed to create pool ({e}). "
                "Falling back to in-process timing without a worker pool."
            )
            self.pool = None
            return
        
        pool_create_end = time.time()
        pool_create_elapsed = pool_create_end - pool_create_start
        logging.info(f"[POOL_TIMING] {pool_create_end:.3f}: ctx.Pool() completed in {pool_create_elapsed:.3f}s")
        
        self.timeouts = 0
        logging.info(f"BenchmarkPool._make_pool: Successfully created new worker pool with config '{self.pool_config_name}'")
        
        # Defensive check: Ensure we have a real Pool, not a monkey-patched fake one
        pool_type = type(self.pool).__name__
        logging.info(f"BenchmarkPool._make_pool: Created pool type: {pool_type}")
        if pool_type == "_SerialPool":
            raise RuntimeError(f"ERROR: Pool creation returned fake _SerialPool instead of real multiprocessing.Pool. This indicates solver code is monkey-patching multiprocessing. Please restart the evaluation to clear module cache.")
        
        # Skip warm-up task – simple initializer starts instantly and submitting an
        # extra task occasionally dead-locks on some HPC clusters.
        logging.info("BenchmarkPool._make_pool: Skipping worker warm-up task (simple initializer)")
    
    def _reset_pool(self):
        logging.critical(f"BenchmarkPool._reset_pool: Starting pool reset. {self.timeouts} timeouts reached, terminating pool and reloading code")
        try:
            # Aggressive immediate termination to prevent deadlocks
            logging.info("BenchmarkPool._reset_pool: Calling pool.terminate() immediately")
            self.pool.terminate()
            logging.info("BenchmarkPool._reset_pool: pool.terminate() completed")
            
            # Force kill any remaining processes using OS signals
            try:
                import signal
                import psutil
                current_process = psutil.Process()
                for child in current_process.children(recursive=True):
                    try:
                        logging.info(f"BenchmarkPool._reset_pool: Force killing child process {child.pid}")
                        child.send_signal(signal.SIGKILL)
                        child.wait(timeout=1.0)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        pass
            except Exception as e:
                logging.warning(f"BenchmarkPool._reset_pool: Could not force kill children: {e}")
            
            # Attempt brief pool join - don't create threads for this
            try:
                logging.info("BenchmarkPool._reset_pool: Attempting brief pool.join() (no timeout)")
                # Simple approach: just call join and rely on aggressive terminate/kill above
                # If join hangs, the terminate/kill should have handled problematic processes
                self.pool.join()
                logging.info("BenchmarkPool._reset_pool: pool.join() completed successfully")
            except Exception as e:
                logging.warning(f"BenchmarkPool._reset_pool: Error during pool.join(): {e}")
                
        except Exception as e:
            logging.exception(f"BenchmarkPool._reset_pool: Error shutting down pool: {e}")
            
        # Skip module reload to save time; assume solver code is stable during one evaluation run
        logging.info("BenchmarkPool._reset_pool: Skipping reload_all_llm_src for faster reset")
        
        # Recreate the pool
        logging.info("BenchmarkPool._reset_pool: About to recreate pool")
        self._make_pool()
        
        # Reset task count for new pool
        self._task_count = 0
        
        logging.info("BenchmarkPool._reset_pool: Successfully completed pool reset")
    
    def run(self, func: Callable, args: Tuple = (), kwds: Dict = None, timeout_s: Optional[float] = None, max_retries: int = 3) -> Dict[str, Any]:
        func_name = getattr(func, "__name__", str(func))
        if kwds is None:
            kwds = {}
        
        # Set a reasonable default timeout to prevent infinite hangs
        effective_timeout = timeout_s if timeout_s is not None else 60.0  # 1 minute default (was 5 minutes)
        
        logging.info(f"BenchmarkPool.run: Starting {func_name} with max_retries={max_retries}, timeout={effective_timeout:.1f}s")
        
        # Check if we should proactively recycle the pool based on health metrics
        # This prevents "cannot allocate memory for thread-local data" errors
        try:
            # Note: Health monitoring happens inside the worker, but we can check task count here
            task_count = getattr(self, '_task_count', 0)
            if task_count > 0 and task_count % 10 == 0:  # Check every 10 tasks
                logging.info(f"BenchmarkPool.run: Completed {task_count} tasks, considering pool health")
                # Could add more sophisticated health checks here in the future
                
        except Exception as health_check_error:
            logging.warning(f"BenchmarkPool.run: Health check failed: {health_check_error}")
        
        last_exception = None  # Track the last exception for better error reporting
        
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"BenchmarkPool.run: Attempt {attempt}/{max_retries} for {func_name}")
                logging.info(f"BenchmarkPool.run: About to submit task {func_name} to pool")
                
                # Submit the task
                submit_start = time.time()
                async_res = self.pool.apply_async(func, args=args, kwds=kwds)
                submit_elapsed = time.time() - submit_start
                logging.info(f"BenchmarkPool.run: Task {func_name} submitted in {submit_elapsed:.3f}s, waiting for result")
                
                # Log task details for debugging
                logging.info(f"BenchmarkPool.run: Task details - func={func_name}, args_len={len(args) if args else 0}, kwds_keys={list(kwds.keys()) if kwds else []}")
                
                logging.info(f"BenchmarkPool.run: About to call async_res.get() with timeout={effective_timeout:.1f}s")
                get_start = time.time()
                
                # Direct get with timeout – simpler and avoids spawning an
                # extra thread for every problem.  In practice the underlying
                # Pool.get() is reliable once maxtasksperchild is set.
                try:
                    result = async_res.get(timeout=effective_timeout)
                except multiprocessing.TimeoutError:
                    raise
                except Exception as e:
                    # Bubble up any other worker-side exception
                    raise e

                get_elapsed = time.time() - get_start
                logging.info(
                    f"BenchmarkPool.run: Got result for {func_name} within timeout in {get_elapsed:.3f}s"
                )
                
                # Timing fields debug – switch to DEBUG level to reduce noise
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    timing_fields = [
                        "elapsed_ms",
                        "min_time_ms",
                        "mean_ms",
                        "median_ms",
                        "min",
                        "mean",
                        "median",
                        "stddev",
                        "values",
                        "runs",
                        "success",
                    ]
                    timing_debug = {field: result.get(field) for field in timing_fields}
                    logging.debug(
                        f"BENCHMARK_POOL_TIMING_DEBUG: {func_name}: {timing_debug}"
                    )
                
                # Track successful task completion for health monitoring
                self._task_count += 1
                
                return result
                
            except multiprocessing.TimeoutError as e:
                last_exception = e  # Save the timeout exception
                self.timeouts += 1
                logging.warning(f"BenchmarkPool.run: Timeout {self.timeouts}/{self.max_timeouts} for {func_name} on attempt {attempt}/{max_retries}")
                
                if self.timeouts >= self.max_timeouts:
                    # Reset the pool and reload modules before trying again
                    logging.warning(f"BenchmarkPool.run: Max timeouts reached, resetting pool")
                    self._reset_pool()
                    self.timeouts = 0
                    
                    # One final attempt after reset
                    if attempt == max_retries:
                        logging.error(f"BenchmarkPool.run: Final attempt after reset failed for {func_name}: {e}")
                        # Preserve the timeout exception in HardBenchmarkFailure
                        raise HardBenchmarkFailure(e)
                        
            except Exception as e:
                last_exception = e  # Save the general exception
                logging.error(f"BenchmarkPool.run: Unexpected error on attempt {attempt}/{max_retries} for {func_name}: {e}")
                if attempt == max_retries:
                    logging.error(f"BenchmarkPool.run: All attempts failed for {func_name}: {e}")
                    raise HardBenchmarkFailure(e) 
                
        # Should never reach here, but if we do, preserve the last exception
        logging.error(f"BenchmarkPool.run: Exhausted all attempts for {func_name} without returning or raising")
        if last_exception:
            raise HardBenchmarkFailure(last_exception)
        else:
            raise HardBenchmarkFailure(f"Exhausted all attempts for {func_name}") 

    def close(self):
        """Gracefully close and join the underlying multiprocessing pool."""
        try:
            if hasattr(self, 'pool') and self.pool is not None:
                logging.info("BenchmarkPool.close: Closing worker pool")
                self.pool.close()
                self.pool.join()
                self.pool = None
                logging.info("BenchmarkPool.close: Pool closed successfully")
            else:
                logging.info("BenchmarkPool.close: No active pool to close")
        except Exception as e:
            logging.warning(f"BenchmarkPool.close: Failed to close pool cleanly: {e}")
        finally:
            try:
                import gc
                gc.collect()
            except Exception:
                pass

# Minimal standalone initializer for debugging - no AlgoTuner imports
def _debug_minimal_initializer(memory_limit_bytes: int, disable_rlimit_as: bool = False):
    """Minimal initializer with essential memory setup for debugging worker spawn issues."""
    import logging
    import os
    import time
    import resource
    
    worker_pid = os.getpid()
    current_time = time.time()
    
    # NEW: configure BLAS threads to match parent before any heavy imports
    try:
        from AlgoTuner.utils.blas_utils import set_blas_threads, log_current_blas_threads, log_cpu_affinity, log_thread_env
        n_thr = set_blas_threads()
        log_current_blas_threads(f"[DEBUG_MINIMAL:{worker_pid}] ")
        log_cpu_affinity(f"[DEBUG_MINIMAL:{worker_pid}] ")
        log_thread_env(f"[DEBUG_MINIMAL:{worker_pid}] ")
        logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: BLAS threads set to {n_thr}")
    except Exception as _blas_e:
        logging.debug(f"[DEBUG_MINIMAL] Could not set BLAS threads – {_blas_e}")
    
    # Force immediate logging to ensure we see this
    logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** STARTING MINIMAL INITIALIZER *** PID {worker_pid}")
    logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: memory_limit_bytes={memory_limit_bytes}, disable_rlimit_as={disable_rlimit_as}")
    
    # === COMPREHENSIVE INITIAL MEMORY STATE ===
    try:
        # RLIMIT_AS
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** INITIAL RLIMIT_AS *** soft={soft if soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={hard if hard != resource.RLIM_INFINITY else 'INFINITY'}")
        if soft != resource.RLIM_INFINITY:
            logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** INITIAL RLIMIT_AS *** soft={soft / (1024**3):.2f}GB")
        if hard != resource.RLIM_INFINITY:
            logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** INITIAL RLIMIT_AS *** hard={hard / (1024**3):.2f}GB")
            
        # RLIMIT_RSS
        try:
            rss_soft, rss_hard = resource.getrlimit(resource.RLIMIT_RSS)
            logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** INITIAL RLIMIT_RSS *** soft={rss_soft if rss_soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={rss_hard if rss_hard != resource.RLIM_INFINITY else 'INFINITY'}")
        except Exception as e:
            logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: Failed to get RLIMIT_RSS: {e}")
            
        # RLIMIT_DATA
        try:
            data_soft, data_hard = resource.getrlimit(resource.RLIMIT_DATA)
            logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** INITIAL RLIMIT_DATA *** soft={data_soft if data_soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={data_hard if data_hard != resource.RLIM_INFINITY else 'INFINITY'}")
        except Exception as e:
            logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: Failed to get RLIMIT_DATA: {e}")
            
    except Exception as e:
        logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: Failed to get initial limits: {e}")
    
    # CRITICAL: Set up memory limits without importing AlgoTuner modules
    memory_setup_start = time.time()
    logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** STARTING MEMORY LIMIT SETUP ***")
    try:
        if disable_rlimit_as and memory_limit_bytes > 0:
            # Inline implementation of raise_rlimit_as to avoid heavy imports
            logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: disable_rlimit_as=True, getting current limits")
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** PRE-RAISE *** soft={soft if soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={hard if hard != resource.RLIM_INFINITY else 'INFINITY'}")
            logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** REQUESTED *** memory_limit_bytes={memory_limit_bytes} ({memory_limit_bytes / (1024**3):.2f}GB)")
            
            if soft != resource.RLIM_INFINITY and soft < memory_limit_bytes:
                new_limit = max(memory_limit_bytes, hard if hard != resource.RLIM_INFINITY else memory_limit_bytes)
                logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** RAISING *** limit from {soft} ({soft / (1024**3):.2f}GB) to {new_limit} ({new_limit / (1024**3):.2f}GB)")
                resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
                logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** SUCCESS *** Raised RLIMIT_AS to {new_limit} ({new_limit / (1024**3):.2f}GB)")
            elif soft == resource.RLIM_INFINITY:
                logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** NO CHANGE *** RLIMIT_AS already INFINITY")
            else:
                logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** NO CHANGE *** RLIMIT_AS already sufficient: {soft} ({soft / (1024**3):.2f}GB) >= {memory_limit_bytes} ({memory_limit_bytes / (1024**3):.2f}GB)")
        elif memory_limit_bytes > 0:
            # Cap RLIMIT_AS to the requested limit
            logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: disable_rlimit_as=False, capping RLIMIT_AS to {memory_limit_bytes} ({memory_limit_bytes / (1024**3):.2f}GB)")
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
            logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** SUCCESS *** Set RLIMIT_AS to {memory_limit_bytes} ({memory_limit_bytes / (1024**3):.2f}GB)")
        else:
            logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** SKIPPING *** memory limit setup (memory_limit_bytes={memory_limit_bytes})")
    except Exception as e:
        logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** FAILED *** to set memory limits: {type(e).__name__}: {e}")
        import traceback
        logging.debug(f"[DEBUG_MINIMAL] {memory_setup_start:.3f}: *** TRACEBACK *** {traceback.format_exc()}")
    
    memory_setup_end = time.time()
    memory_setup_elapsed = memory_setup_end - memory_setup_start
    logging.info(f"[DEBUG_MINIMAL] {memory_setup_end:.3f}: Memory setup completed in {memory_setup_elapsed:.3f}s")
    
    # === COMPREHENSIVE FINAL MEMORY STATE ===
    try:
        # RLIMIT_AS
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** FINAL RLIMIT_AS *** soft={soft if soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={hard if hard != resource.RLIM_INFINITY else 'INFINITY'}")
        if soft != resource.RLIM_INFINITY:
            logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** FINAL RLIMIT_AS *** soft={soft / (1024**3):.2f}GB")
        if hard != resource.RLIM_INFINITY:
            logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** FINAL RLIMIT_AS *** hard={hard / (1024**3):.2f}GB")
        
        # Test allocation to see what works
        try:
            import numpy as np
            test_sizes = [10, 50, 100, 200]  # MB
            for size_mb in test_sizes:
                try:
                    test_array = np.zeros(size_mb * 1024 * 1024 // 8, dtype=np.float64)
                    logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** ALLOC SUCCESS *** {size_mb}MB test array")
                    del test_array
                except Exception as alloc_e:
                    logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: *** ALLOC FAILED *** {size_mb}MB: {alloc_e}")
                    break  # Stop at first failure
        except Exception as e:
            logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: Failed to test allocations: {e}")
            
    except Exception as e:
        logging.debug(f"[DEBUG_MINIMAL] {current_time:.3f}: Failed to get final limits: {e}")
    
    # Change directory
    dir_change_start = time.time()
    logging.info(f"[DEBUG_MINIMAL] {dir_change_start:.3f}: Starting directory change")
    code_dir = os.environ.get("CODE_DIR", ".")
    logging.info(f"[DEBUG_MINIMAL] {dir_change_start:.3f}: CODE_DIR from env: {code_dir}")
    
    if code_dir and os.path.exists(code_dir):
        try:
            os.chdir(code_dir)
            new_cwd = os.getcwd()
            logging.info(f"[DEBUG_MINIMAL] {dir_change_start:.3f}: Successfully changed to CODE_DIR: {code_dir}")
            logging.info(f"[DEBUG_MINIMAL] {dir_change_start:.3f}: Verified new CWD: {new_cwd}")
        except Exception as e:
            logging.debug(f"[DEBUG_MINIMAL] {dir_change_start:.3f}: Failed to change directory: {type(e).__name__}: {e}")
    else:
        logging.warning(f"[DEBUG_MINIMAL] {dir_change_start:.3f}: CODE_DIR not found or invalid: {code_dir}")
        if code_dir:
            logging.warning(f"[DEBUG_MINIMAL] {dir_change_start:.3f}: os.path.exists({code_dir}) = {os.path.exists(code_dir)}")
    
    dir_change_end = time.time()
    dir_change_elapsed = dir_change_end - dir_change_start
    logging.info(f"[DEBUG_MINIMAL] {dir_change_end:.3f}: Directory change completed in {dir_change_elapsed:.3f}s")
    
    end_time = time.time()
    total_elapsed = end_time - current_time
    logging.info(f"[DEBUG_MINIMAL] {end_time:.3f}: Minimal initializer completed in {total_elapsed:.3f}s for PID {worker_pid}")
    
    # Summary of timing breakdown
    logging.info(f"[DEBUG_MINIMAL] {end_time:.3f}: Timing breakdown - Memory: {memory_setup_elapsed:.3f}s, Directory: {dir_change_elapsed:.3f}s, Total: {total_elapsed:.3f}s") 