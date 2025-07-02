"""
utils/multiprocessing_utils.py
==============================

Utilities for managing multiprocessing Pools consistently.
"""
import logging
import os
import multiprocessing
import signal
import time

try:
    import resource # Unix-specific
    RESOURCE_AVAILABLE = hasattr(os, 'fork')
except ImportError:
    RESOURCE_AVAILABLE = False


def _pool_worker_initializer(memory_limit_bytes: int, disable_rlimit_as: bool = False):
    """Initializer for pool workers to set memory limits and suppress logs."""
    try:
        worker_pid = os.getpid()
        start_time = time.time()
        
        # Preserve any FileHandler(s) that were configured in the parent so that
        # worker-side logs continue to flow into the same log file.  Remove all
        # other handlers (e.g. StreamHandlers inherited from the parent) to
        # avoid duplicate console output in every subprocess.

        root_logger = logging.getLogger()

        # Snapshot existing handlers *before* we touch them.
        _orig_handlers = root_logger.handlers[:]

        # Separate them by type.
        _file_handlers = [h for h in _orig_handlers if isinstance(h, logging.FileHandler)]
        _other_handlers = [h for h in _orig_handlers if not isinstance(h, logging.FileHandler)]

        # Remove the non-file handlers.
        for handler in _other_handlers:
            try:
                handler.close()
            except Exception:
                pass
            root_logger.removeHandler(handler)

        # Re-attach preserved FileHandlers (they are already closed above, so no
        # need to close them again; just add back so that log records are still
        # written to the same file).
        for fh in _file_handlers:
            root_logger.addHandler(fh)

        # Finally, add a StreamHandler that is local to the worker.  This gives
        # immediate visibility when running interactively without duplicating
        # every parent log line.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(levelname)s - WORKER %(process)d - %(message)s'))
        root_logger.addHandler(console_handler)

        # Ensure root logger has at least INFO level (file handler may have its
        # own level).
        root_logger.setLevel(logging.INFO)
        
        logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): Starting initialization at {time.time():.3f}. Received mem_limit_bytes: {memory_limit_bytes}, disable_rlimit_as: {disable_rlimit_as}")
        
        # Set memory limit (if possible and not disabled)
        if RESOURCE_AVAILABLE and memory_limit_bytes > 0 and not disable_rlimit_as:
            try:
                logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): About to set RLIMIT_AS at {time.time():.3f}")
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
                logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): Successfully set RLIMIT_AS to {memory_limit_bytes / (1024**3):.2f} GB at {time.time():.3f}")
            except Exception as e:
                logging.warning(f"POOL_WORKER_INIT (PID: {worker_pid}): Failed to set memory limit: {e}")
        elif disable_rlimit_as:
            logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): RLIMIT_AS disabled by configuration")
        else:
            logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): RLIMIT_AS not set (RESOURCE_AVAILABLE={RESOURCE_AVAILABLE}, mem_limit_bytes={memory_limit_bytes}).")

        # Set working directory and initialize DaCe
        code_dir = os.environ.get('CODE_DIR')
        logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): CODE_DIR environment variable: {code_dir} at {time.time():.3f}")
        if code_dir:
            try:
                logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): About to change working directory to '{code_dir}' at {time.time():.3f}")
                os.chdir(code_dir)
                logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): Changed working directory to '{code_dir}' at {time.time():.3f}")
            except Exception as e:
                logging.warning(f"POOL_WORKER_INIT (PID: {worker_pid}): Failed to change working directory to '{code_dir}': {e}")
            
            # Initialize DaCe configuration
            logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): About to initialize DaCe for process at {time.time():.3f}")
            try:
                from AlgoTuner.utils.dace_config import initialize_dace_for_process
                initialize_dace_for_process()
                logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): Successfully initialized DaCe for process at {time.time():.3f}")
            except Exception as e:
                logging.warning(f"POOL_WORKER_INIT (PID: {worker_pid}): Failed to initialize DaCe: {e}")
            
            # Set Numba cache dir for isolation
            os.environ['NUMBA_CACHE_DIR'] = code_dir
        
        # Module reloading is now handled in the parent process before worker creation
        logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): Skipping module reload in worker (handled in parent process) at {time.time():.3f}")
        
        # Just ensure the CODE_DIR is available for module loading
        if code_dir:
            logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): CODE_DIR '{code_dir}' is available for imports")

        # Optional: Suppress non-critical logs from worker processes
        
        elapsed = time.time() - start_time
        logging.debug(f"POOL_WORKER_INIT (PID: {worker_pid}): Worker initialization completed successfully in {elapsed:.3f}s at {time.time():.3f}")
        
    except Exception as e:
        logging.error(f"POOL_WORKER_INIT (PID: {worker_pid if 'worker_pid' in locals() else 'unknown'}): Worker initialization failed with exception: {e}")
        raise


def _simple_worker_initializer(memory_limit_bytes: int, disable_rlimit_as: bool = False):
    """Simplified worker initializer that avoids module reloading deadlocks."""
    import logging
    import os
    import time
    
    worker_pid = os.getpid()
    logging.info(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Starting simple initialization, disable_rlimit_as={disable_rlimit_as}")
    
    # Set memory limits for recursive algorithms - account for 16GB SLURM limit with container overhead  
    # Must leave room for parent process + container overhead, but allow recursive algorithms
    target_limit_gb = 14.0  # Allow 14GB for solver algorithms
    target_limit_bytes = int(target_limit_gb * 1024**3)
    
    # Use the smaller of provided limit or 14GB
    effective_limit = min(memory_limit_bytes, target_limit_bytes) if memory_limit_bytes > 0 else target_limit_bytes
    effective_limit_gb = effective_limit / (1024**3)
    
    # Initialize monitoring / helper subsystems
    try:
        # Thread and health helpers are always useful
        from AlgoTuner.utils.thread_manager import get_worker_thread_manager
        from AlgoTuner.utils.worker_health import get_worker_health_monitor

        if not disable_rlimit_as:
            # Only connect the memory monitor (which sets RLIMIT_AS) when the flag allows it
            from AlgoTuner.utils.process_monitor import init_worker_memory_monitor
            memory_monitor = init_worker_memory_monitor(effective_limit_gb)
            logging.info(
                f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Initialized process-level memory monitor with {effective_limit_gb:.2f}GB limit"
            )
        else:
            memory_monitor = None  # noqa: F841  # variable kept for symmetry/debugging
            logging.info(
                f"SIMPLE_WORKER_INIT (PID: {worker_pid}): RLIMIT_AS disabled; skipping memory monitor"
            )

        # Initialize thread manager for tracking
        thread_manager = get_worker_thread_manager()
        logging.info(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Initialized thread manager")

        # Initialize health monitor
        health_monitor = get_worker_health_monitor()
        logging.info(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Initialized worker health monitor")

    except Exception as e:
        logging.warning(
            f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Failed to initialize monitoring/helpers, falling back to resource limits: {e}"
        )
        
        # Fallback to old resource limit approach (only if not disabled)
        if not disable_rlimit_as:
            try:
                import resource
                # Set both virtual memory (RLIMIT_AS) and RSS memory (RLIMIT_RSS) if available
                resource.setrlimit(resource.RLIMIT_AS, (effective_limit, effective_limit))
                logging.info(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Set virtual memory limit to {effective_limit_gb:.2f} GB")
                
                # Also try to set RSS limit (not all systems support this)
                try:
                    resource.setrlimit(resource.RLIMIT_RSS, (effective_limit, effective_limit))
                    logging.info(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Set RSS memory limit to {effective_limit_gb:.2f} GB")
                except (AttributeError, OSError) as e:
                    logging.info(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): RSS limit not supported: {e}")
                    
            except (ImportError, Exception) as e:
                logging.warning(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Failed to set resource limits: {e}")
        else:
            logging.info(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): RLIMIT_AS disabled by configuration")
    
    # The old thread-based memory monitoring has been replaced with process-level monitoring
    # This eliminates thread accumulation issues that were causing worker hangs
    
    # Set working directory
    code_dir = os.environ.get('CODE_DIR')
    if code_dir and os.path.exists(code_dir):
        try:
            os.chdir(code_dir)
            logging.info(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Changed working directory to '{code_dir}'")
        except Exception as e:
            logging.warning(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Failed to change working directory: {e}")
    
    # ---------- Configure libraries to avoid enormous debug prints ----------
    try:
        import numpy as _np  # Local import to keep init lightweight
        # Limit numpy printout to a small, summary form.  Large arrays will be
        # shown in truncated form like "[ 1.  2.  3. ...]".
        _np.set_printoptions(threshold=200, edgeitems=3, linewidth=120, suppress=True)
        logging.info(
            f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Configured numpy print options to prevent huge matrix dumps"
        )
    except Exception as _np_err:
        logging.debug(
            f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Could not set numpy printoptions: {_np_err}"
        )

    # ---------- Replace built-in print with a logging-backed stub ----------
    try:
        import builtins as _bi

        def _print_to_log(*args, **kwargs):  # noqa: D401
            sep = kwargs.get("sep", " ")
            msg = sep.join(str(a) for a in args)
            logging.info(msg)

        _bi.print = _print_to_log  # type: ignore[assignment]
        logging.info(
            f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Overrode built-in print to forward to logging.info"
        )
    except Exception as _patch_err:
        logging.debug(
            f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Could not override print: {_patch_err}"
        )

    # ---------- Mute Numba's internal IR dump loggers ----------
    try:
        import logging as _logging

        # Clean environment of debug variables
        for _var in ("NUMBA_DEBUG", "NUMBA_DUMP_IR", "NUMBA_DUMP_CFG", "NUMBA_DUMP_OPT_STATS"):
            os.environ.pop(_var, None)

        # Numba logging removed
    except Exception:
        pass

    logging.info(f"SIMPLE_WORKER_INIT (PID: {worker_pid}): Simple initialization completed")


def _baseline_worker_initializer(memory_limit_bytes: int, disable_rlimit_as: bool = False):
    """Simplified initializer for baseline evaluation workers."""
    import time
    start_time = time.time()
    
    worker_pid = os.getpid()
    
    # === COMPREHENSIVE DIAGNOSTIC LOGGING ===
    logging.info(f"[TIMING] {time.time():.3f}: Starting _baseline_worker_initializer for PID {worker_pid}")
    logging.info(f"[MEMORY_DEBUG] Starting _baseline_worker_initializer for PID {worker_pid}")
    logging.info(f"[MEMORY_DEBUG] Received memory_limit_bytes: {memory_limit_bytes} ({memory_limit_bytes / (1024**3):.2f}GB)")
    logging.info(f"[MEMORY_DEBUG] Received disable_rlimit_as: {disable_rlimit_as}")
    # === END INITIAL LOGGING ===
    
    logging.info(f"[TIMING] {time.time():.3f}: About to reset logging")
    
    # Reset logging to prevent deadlocks from inherited handlers, but preserve
    # parent handlers so worker messages still propagate to the main log file.
    import logging
    root_logger = logging.getLogger()

    parent_handlers = list(root_logger.handlers)  # copy before mutation

    # Remove existing handlers (avoid duplicated messages / forked locks)
    for h in parent_handlers:
        root_logger.removeHandler(h)

    # Console handler for quick per-worker diagnostics
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s - BASELINE_WORKER %(process)d - %(message)s')
    )
    root_logger.addHandler(console_handler)

    # Re-attach parent handlers so messages show up in the main log file
    for h in parent_handlers:
        root_logger.addHandler(h)

    root_logger.setLevel(logging.INFO)

    logging.info(f"[TIMING] {time.time():.3f}: Logging reset complete")

    logging.info(
        f"BASELINE_WORKER_INIT (PID: {worker_pid}): Initializing baseline worker. "
        f"Received mem_limit_bytes: {memory_limit_bytes}, disable_rlimit_as: {disable_rlimit_as}"
    )

    # === COMPREHENSIVE MEMORY LIMIT DIAGNOSTIC LOGGING ===
    logging.info(f"[TIMING] {time.time():.3f}: Starting memory limit analysis")
    logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: Starting memory limit analysis")
    
    # Diagnostic: Show current limits before adjustment
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: INITIAL RLIMIT_AS state:")
        if soft == resource.RLIM_INFINITY:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}:   soft = INFINITY")
        else:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}:   soft = {soft} bytes ({soft / (1024**3):.2f}GB)")
        if hard == resource.RLIM_INFINITY:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}:   hard = INFINITY")
        else:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}:   hard = {hard} bytes ({hard / (1024**3):.2f}GB)")
            
        logging.info(
            f"BASELINE_WORKER_INIT (PID: {worker_pid}): Pre-adjust RLIMIT_AS: "
            f"soft={soft if soft == resource.RLIM_INFINITY else soft / (1024**3):.2f}GB, "
            f"hard={hard if hard == resource.RLIM_INFINITY else hard / (1024**3):.2f}GB"
        )
    except Exception as e:
        logging.debug(f"BASELINE_WORKER_INIT (PID: {worker_pid}): Could not query RLIMIT_AS: {e}")
        logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: Failed to get initial RLIMIT_AS: {e}")

    logging.info(f"[TIMING] {time.time():.3f}: About to set memory limits")

    # === MEMORY LIMIT SETTING LOGIC WITH COMPREHENSIVE LOGGING ===
    try:
        import resource
        if disable_rlimit_as:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: disable_rlimit_as=True, will call raise_rlimit_as({memory_limit_bytes})")
            # Only *raise* the limit if it's lower than memory_limit_bytes
            logging.info(f"[TIMING] {time.time():.3f}: About to import raise_rlimit_as")
  
            from AlgoTuner.utils.resource_utils import raise_rlimit_as
            logging.info(f"[TIMING] {time.time():.3f}: raise_rlimit_as imported")
            if memory_limit_bytes > 0:
                logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: About to call raise_rlimit_as with {memory_limit_bytes} bytes")
                logging.info(f"[TIMING] {time.time():.3f}: About to call raise_rlimit_as")
                
                # Log limits BEFORE raise_rlimit_as
                try:
                    pre_soft, pre_hard = resource.getrlimit(resource.RLIMIT_AS)
                    logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: PRE-raise_rlimit_as: soft={pre_soft if pre_soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={pre_hard if pre_hard != resource.RLIM_INFINITY else 'INFINITY'}")
                except Exception as e:
                    logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: Failed to get PRE-raise_rlimit_as limits: {e}")
                
                raise_rlimit_as(memory_limit_bytes)
                
                # Log limits AFTER raise_rlimit_as
                try:
                    post_soft, post_hard = resource.getrlimit(resource.RLIMIT_AS)
                    logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: POST-raise_rlimit_as: soft={post_soft if post_soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={post_hard if post_hard != resource.RLIM_INFINITY else 'INFINITY'}")
                    if post_soft != resource.RLIM_INFINITY:
                        logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: POST-raise_rlimit_as: soft={post_soft / (1024**3):.2f}GB, hard={post_hard / (1024**3) if post_hard != resource.RLIM_INFINITY else 'INFINITY'}GB")
                except Exception as e:
                    logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: Failed to get POST-raise_rlimit_as limits: {e}")
                
                logging.info(f"[TIMING] {time.time():.3f}: raise_rlimit_as completed")
                logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: raise_rlimit_as completed")
            else:
                logging.warning(f"[MEMORY_DEBUG] Worker {worker_pid}: Skipping raise_rlimit_as because memory_limit_bytes <= 0")
        else:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: disable_rlimit_as=False, will explicitly cap RLIMIT_AS")
            # Explicitly cap RLIMIT_AS to the requested pool limit
            if memory_limit_bytes > 0:
                logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: About to set RLIMIT_AS to {memory_limit_bytes} bytes")
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
                logging.info(
                    f"BASELINE_WORKER_INIT (PID: {worker_pid}): Set RLIMIT_AS to {memory_limit_bytes / (1024**3):.2f}GB"
                )
                logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: Successfully set RLIMIT_AS")
            else:
                logging.warning(f"[MEMORY_DEBUG] Worker {worker_pid}: Skipping RLIMIT_AS setting because memory_limit_bytes <= 0")
    except Exception as e:
        logging.warning(
            f"BASELINE_WORKER_INIT (PID: {worker_pid}): Could not adjust RLIMIT_AS: {e}"
        )
        logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: RLIMIT_AS adjustment failed: {type(e).__name__}: {e}")
        import traceback
        logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: RLIMIT_AS adjustment traceback: {traceback.format_exc()}")

    logging.info(f"[TIMING] {time.time():.3f}: Memory limits set, checking final state")

    # After adjustment, log the effective limits for confirmation
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: FINAL RLIMIT_AS state:")
        if soft == resource.RLIM_INFINITY:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}:   soft = INFINITY")
        else:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}:   soft = {soft} bytes ({soft / (1024**3):.2f}GB)")
        if hard == resource.RLIM_INFINITY:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}:   hard = INFINITY")
        else:
            logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}:   hard = {hard} bytes ({hard / (1024**3):.2f}GB)")
            
        logging.info(
            f"BASELINE_WORKER_INIT (PID: {worker_pid}): Post-adjust RLIMIT_AS: "
            f"soft={soft if soft == resource.RLIM_INFINITY else soft / (1024**3):.2f}GB, "
            f"hard={hard if hard == resource.RLIM_INFINITY else hard / (1024**3):.2f}GB"
        )
        
        # === CRITICAL ANALYSIS ===
        if memory_limit_bytes > 0:
            expected_gb = memory_limit_bytes / (1024**3)
            if soft != resource.RLIM_INFINITY and soft < memory_limit_bytes:
                logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: *** CRITICAL *** RLIMIT_AS soft limit ({soft / (1024**3):.2f}GB) is LOWER than requested ({expected_gb:.2f}GB)!")
            elif soft == resource.RLIM_INFINITY:
                logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: RLIMIT_AS is INFINITY (unlimited)")
            else:
                logging.info(f"[MEMORY_DEBUG] Worker {worker_pid}: RLIMIT_AS appears correctly set")
    except Exception:
        logging.error(f"[MEMORY_DEBUG] Worker {worker_pid}: Failed to get final RLIMIT_AS state")
        pass

    # === END COMPREHENSIVE MEMORY LOGGING ===

    logging.info(f"[TIMING] {time.time():.3f}: About to set working directory")

    # Set working directory
    code_dir = os.environ.get('CODE_DIR')
    if code_dir:
        try:
            os.chdir(code_dir)
            logging.info(f"BASELINE_WORKER_INIT (PID: {worker_pid}): Changed working directory to '{code_dir}'")
        except Exception as e:
            logging.warning(f"BASELINE_WORKER_INIT (PID: {worker_pid}): Failed to change working directory: {e}")
    
    elapsed = time.time() - start_time
    logging.info(f"[TIMING] {time.time():.3f}: Baseline worker initialization complete in {elapsed:.3f}s")
    logging.info(f"BASELINE_WORKER_INIT (PID: {worker_pid}): Baseline worker initialization complete")


def load_pool_config(pool_config_name: str = "validation_pool", force_num_workers: int = None) -> dict:
    """
    Loads multiprocessing pool configuration from the benchmark config.

    Args:
        pool_config_name: The key in config.yaml under 'benchmark' (e.g., 'validation_pool', 'evaluation_pool').
        force_num_workers: If not None, this value will override the num_workers from the config.

    Returns:
        A dictionary with 'num_workers', 'maxtasksperchild', 'mem_limit_bytes', 'disable_rlimit_as'.
    """
    from AlgoTuner.config.loader import load_config
    cfg = load_config()
    pool_cfg = cfg.get("benchmark", {}).get(pool_config_name, {})
    logging.info(f"POOL_CONFIG: Loading from 'benchmark.{pool_config_name}': {pool_cfg}")

    # Determine num_workers
    if force_num_workers is not None:
        num_workers = max(1, force_num_workers)
        logging.info(f"POOL_CONFIG: num_workers forced to {num_workers}.")
    else:
        configured_num_workers = pool_cfg.get("num_workers")
        if isinstance(configured_num_workers, int) and configured_num_workers > 0:
            num_workers = configured_num_workers
        else: # Default if not set, null, or invalid
            default_num_workers = max(1, (os.cpu_count() or 1) // 2)
            slurm_cpus_env = os.environ.get('SLURM_CPUS_PER_TASK')
            if slurm_cpus_env:
                try:
                    default_num_workers = max(1, int(slurm_cpus_env) // 2)
                    logging.info(f"POOL_CONFIG: Default num_workers from SLURM_CPUS_PER_TASK: {default_num_workers}")
                except ValueError:
                    logging.warning(f"POOL_CONFIG: Could not parse SLURM_CPUS_PER_TASK ('{slurm_cpus_env}'), using CPU-based default.")
            num_workers = default_num_workers
        logging.info(f"POOL_CONFIG: Effective num_workers = {num_workers} (configured: {pool_cfg.get('num_workers')})")

    # Determine maxtasksperchild
    maxtasks = pool_cfg.get("maxtasksperchild")
    if maxtasks is not None:
        if isinstance(maxtasks, int) and maxtasks > 0:
            pass # Use valid integer
        else:
            logging.warning(f"POOL_CONFIG: Invalid maxtasksperchild '{maxtasks}', using None (long-lived workers).")
            maxtasks = None
    else:
        maxtasks = None # Default if not in config or explicitly null
    logging.info(f"POOL_CONFIG: Effective maxtasksperchild = {maxtasks if maxtasks is not None else 'None (long-lived)'}")

    # Determine memory_limit_gb_per_worker
    default_mem_gb = 14.0  # Original default

    configured_mem_gb = pool_cfg.get("memory_limit_gb_per_worker", default_mem_gb)
    if not isinstance(configured_mem_gb, (int, float)) or configured_mem_gb <= 0:
        actual_mem_gb = default_mem_gb
        logging.warning(
            f"POOL_CONFIG: Invalid memory_limit_gb_per_worker '{configured_mem_gb}', using default {default_mem_gb}GB."
        )
    else:
        actual_mem_gb = configured_mem_gb
    
    # ------------------------------------------------------------------
    # Dynamically adjust the requested limit so we never EXCEED the
    # container / job hard-limit (otherwise setrlimit will fail and leave
    # the process unlimited, which later triggers an OOM-kill).
    # ------------------------------------------------------------------
    try:
        import resource

        soft_cur, hard_cur = resource.getrlimit(resource.RLIMIT_AS)

        # Convert to GB for logging convenience
        hard_cur_gb = None if hard_cur == resource.RLIM_INFINITY else hard_cur / 1024**3

        requested_bytes = int(actual_mem_gb * 1024**3)

        # If the hard limit is finite and *lower* than what the YAML asked
        # for, respect the hard limit – otherwise setrlimit will fail later.
        if hard_cur != resource.RLIM_INFINITY and requested_bytes > hard_cur:
            logging.warning(
                "POOL_CONFIG: Requested %.2fGB per-worker exceeds the parent hard RLIMIT_AS of %.2fGB. "
                "Clamping to the hard limit.",
                actual_mem_gb,
                hard_cur_gb,
            )
            requested_bytes = hard_cur  # keep exactly the hard limit
            actual_mem_gb = hard_cur / 1024**3

        mem_limit_bytes = requested_bytes if RESOURCE_AVAILABLE and requested_bytes > 0 else -1

    except Exception as _rl_err:
        # Fallback – keep original computation if RLIMIT query fails
        logging.debug(f"POOL_CONFIG: Could not query RLIMIT_AS ({_rl_err}), keeping requested %.2fGB", actual_mem_gb)
        mem_limit_bytes = int(actual_mem_gb * 1024**3) if RESOURCE_AVAILABLE and actual_mem_gb > 0 else -1

    logging.info(f"POOL_CONFIG: Effective memory_limit_per_worker={actual_mem_gb:.2f}GB (bytes: {mem_limit_bytes})")

    # Determine disable_rlimit_as setting
    disable_rlimit_as = pool_cfg.get("disable_rlimit_as", False)
    logging.info(f"POOL_CONFIG: disable_rlimit_as = {disable_rlimit_as}")

    return {
        "num_workers": num_workers,
        "maxtasksperchild": maxtasks,
        "mem_limit_bytes": mem_limit_bytes,
        "disable_rlimit_as": disable_rlimit_as
    } 