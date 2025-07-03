import os
import time
import pickle
import statistics
import multiprocessing as mp
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import importlib.util
import gc
import random

from AlgoTuner.utils.error_utils import extract_error_context
from AlgoTuner.utils.timing_config import WARMUPS

# Configuration constants
VALIDATION_OVERHEAD_FACTOR = 150.0  # Account for validation overhead (120s timeout + buffer)

# -----------------------------------------------------------------------------
# Isolated benchmark utilities
# -----------------------------------------------------------------------------

# Don't set global start method - let each context be created explicitly
# to avoid conflicts with other parts of the codebase


def materialize_mmap_objects(obj):
    """Materialize mmap objects to make them picklable.
    
    This function recursively traverses data structures and converts:
    - numpy arrays with mmap_mode to regular arrays
    - mmap.mmap objects to bytes
    
    Args:
        obj: Any Python object or data structure
        
    Returns:
        The same structure with all mmap objects materialized
    """
    import numpy as np
    import mmap
    
    if isinstance(obj, np.ndarray) and hasattr(obj, 'base') and isinstance(obj.base, mmap.mmap):
        # This is a memory-mapped numpy array
        return np.array(obj, copy=True)
    elif isinstance(obj, mmap.mmap):
        # This is a raw mmap object - read it into bytes
        obj.seek(0)
        return obj.read()
    elif hasattr(obj, '__array__'):
        # Other array-like objects
        arr = np.asarray(obj)
        if hasattr(arr, 'base') and isinstance(arr.base, mmap.mmap):
            return np.array(arr, copy=True)
        return arr
    elif isinstance(obj, dict):
        return {k: materialize_mmap_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Preserve list type
        return [materialize_mmap_objects(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively materialise tuple elements first
        processed = tuple(materialize_mmap_objects(item) for item in obj)

        # If this is a NamedTuple (detected via _fields attribute) try to rebuild the
        # same NamedTuple class.  This keeps downstream type-checking happy while
        # avoiding the earlier bug where the generator was passed as a single arg.
        if hasattr(obj, "_fields"):
            try:
                return obj.__class__(*processed)
            except TypeError:
                # Fallback: return plain tuple if reconstruction fails for any reason
                return processed
        return processed
    else:
        return obj


def deep_materialize_fast(obj):
    """Force materialization of lazy objects without unnecessary copies.
    
    This function recursively traverses data structures and forces any objects
    with __array__ methods (like lazy arrays) to materialize their data.
    
    Args:
        obj: Any Python object or data structure
        
    Returns:
        The same structure with all lazy arrays materialized
    """
    import numpy as np
    if hasattr(obj, '__array__'):
        return np.asarray(obj)
    elif isinstance(obj, dict):
        return {k: deep_materialize_fast(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_materialize_fast(item) for item in obj]
    elif isinstance(obj, tuple):
        processed = tuple(deep_materialize_fast(item) for item in obj)
        if hasattr(obj, "_fields"):
            try:
                return obj.__class__(*processed)
            except TypeError:
                return processed
        return processed
    else:
        return obj


def _extract_clean_error_with_context(error_message: str) -> str:
    """
    Extract a clean error message with code context using the standard error_utils.
    
    Args:
        error_message: Full error message with traceback
        
    Returns:
        Clean error message with code context formatted like the rest of the codebase
    """
    try:
        # Check if the error message already contains code context to avoid duplication
        if "Code Context:" in error_message:
            # Already processed, return as-is
            return error_message
            
        # Use the standard extract_error_context function
        error_context = extract_error_context(error_message, "")
        
        enhanced_message = error_context.get("enhanced_error_message", "").strip()
        code_context = error_context.get("code_context_snippet", None)
        
        # Build the final message
        if code_context:
            return f"{enhanced_message}\n\nCode Context:\n{code_context}"
        else:
            return enhanced_message or error_message
            
    except Exception:
        # Fallback to original message
        return error_message


def _fork_run_worker(
    task_name: str,
    code_dir: str,
    tmp_dir: str,
    warmup_payload,
    timed_payload,
    payload_is_pickled: bool,
    ret_dict,
):
    """Worker executed in a fresh interpreter via ``spawn``.

    Parameters
    ----------
    task_name : str
        Name of the current task (unused for now but may be handy for logging).
    code_dir : str
        Directory that contains the user-supplied ``solver.py``.
    tmp_dir : str
        Temporary directory for the worker process.
    warmup_payload : bytes or object
        Pickled warm-up problem instance or the problem itself if not pickled.
    timed_payload : bytes or object
        Pickled timed problem instance or the problem itself if not pickled.
    payload_is_pickled : bool
        True if warmup_payload and timed_payload are pickled, False otherwise.
    ret_dict : multiprocessing.Manager().dict
        Return channel for the measured wall-clock time (ns) and the solver output.
    """
    import traceback  # Local import – tiny overhead but avoids fork safety issues.
    import os  # Local import needed since there's a later local import that shadows the module-level import
    os.environ["ALGOTUNER_SOLVER_WORKER"] = "1"  # Mark this process as a solver worker
    import sys  # For debug output
    
    # Fix for pysat threading issues in multiprocessing workers
    worker_logger = logging.getLogger("isolated_worker")

    # ------------------------------------------------------------------
    # Memory safety: cap RLIMIT_AS inside every isolated benchmark worker
    # ------------------------------------------------------------------
    # Check if RLIMIT_AS should be disabled from config
    disable_rlimit_as = False
    try:
        from AlgoTuner.config.loader import load_config
        config = load_config()
        disable_rlimit_as = config.get("benchmark", {}).get("validation_pool", {}).get("disable_rlimit_as", False)
        if disable_rlimit_as:
            # Set environment variable to skip RLIMIT_AS in ProcessMemoryMonitor
            os.environ['SKIP_RLIMIT_AS'] = '1'
            worker_logger.debug(
                "RLIMIT_AS disabled by configuration in isolated benchmark worker (%d)",
                os.getpid(),
            )
    except Exception as config_err:
        worker_logger.debug(
            "Could not load config to check disable_rlimit_as in worker (%d): %s",
            os.getpid(),
            config_err,
        )
    
    if not disable_rlimit_as:
        try:
            # Import lazily so we don't pay the cost unless the worker actually
            # starts executing (they do in a fresh interpreter).
            from AlgoTuner.utils.process_monitor import init_worker_memory_monitor

            # Use the same 14 GB limit that the evaluation pool applies.  This is
            # well below the 16 GB SLURM allocation and leaves ~2 GB head-room for
            # the parent process and shared pages.
            _mem_mon = init_worker_memory_monitor(14.0)  # noqa: WPS437 (unused var)
            worker_logger.debug(
                "ProcessMemoryMonitor initialised – RLIMIT_AS capped at 14 GB in isolated benchmark worker (%d)",
                os.getpid(),
            )
        except Exception as _mm_err:  # noqa: BLE001
            # Never fail the benchmark because we couldn't set the limit – the
            # container / cgroup limit will still apply as a safety net.
            worker_logger.warning(
                "Could not initialise ProcessMemoryMonitor in isolated benchmark worker (%d): %s",
                os.getpid(),
                _mm_err,
            )

    # Apply centralised PySAT fixes (idempotent, lightweight)
    try:
        if importlib.util.find_spec("pysat") is not None:
            # PySAT present – apply lightweight patches.
            from AlgoTuner.utils.pysat_fix import apply_pysat_fixes

            apply_pysat_fixes()
            worker_logger.debug("PYSAT_FIX: patches applied in isolated worker")

            # Verify that the MainThread patch sticks (optional, inexpensive)
            try:
                from pysat._utils import MainThread  # type: ignore
                worker_logger.debug("PYSAT_FIX: MainThread.check() = %s", MainThread.check())
            except Exception as verify_exc:  # noqa: BLE001
                worker_logger.warning("PYSAT_FIX: Verification failed – %s", verify_exc)
        else:
            worker_logger.debug("PYSAT_FIX: PySAT not found – skipping patch application")
    except Exception as exc:  # noqa: BLE001
        # Importing pysat or applying patches can be very costly when the
        # shared library is missing / incompatible; failing fast avoids a
        # multi-second delay that can cause false time-outs in tiny benchmarks.
        worker_logger.warning("PYSAT_FIX: skipped due to import error – %s", exc)

    try:
        # ------------------------------------------------------------------
        # 1) Re-establish environment inside the fresh interpreter
        #    • point __pycache__ → tmp_dir via PYTHONPYCACHEPREFIX
        #    • change CWD to tmp_dir so any scratch files live there
        # ------------------------------------------------------------------
        os.environ.setdefault("CODE_DIR", code_dir)
        os.environ["CURRENT_TASK_NAME"] = task_name  # Ensure task name is available
        os.environ["PYTHONPYCACHEPREFIX"] = tmp_dir  # redirect .pyc files
        # Removed legacy cache-clearing flag – no longer used
        # Clean environment of debug variables
        for _var in ("NUMBA_DEBUG", "NUMBA_DUMP_IR", "NUMBA_DUMP_CFG", "NUMBA_DUMP_OPT_STATS"):
            os.environ.pop(_var, None)

        # Make the tmp_dir the current working directory so that any ad-hoc
        # writes (e.g. plots, temporary files) vanish afterwards.
        os.chdir(tmp_dir)

        code_dir_path = Path(code_dir)

        # ------------------------------------------------------------------
        # 2) Import solver module and obtain *fresh* solve callable
        # ------------------------------------------------------------------
        # Import after env vars set so loader honours them
        from AlgoTuner.utils.solver_loader import load_solver_module, get_fresh_solve_callable

        # Prefer task-specific implementation '<task_name>.py'.  If it is absent, fall back
        # to a generic 'solver.py'.  This avoids misleading log noise when every task ships
        # its reference implementation inside the task package.

        alt_filename = f"{task_name}.py"
        alt_file_path = code_dir_path / alt_filename
        
        # Debug logging - convert to proper debug level
        logging.debug(f"[isolated_benchmark] task_name='{task_name}', code_dir='{code_dir}'")
        logging.debug(f"[isolated_benchmark] Looking for '{alt_filename}' in '{code_dir_path}'")

        if alt_file_path.is_file():
            solver_module = load_solver_module(code_dir_path, solver_filename=alt_filename)
        else:
            solver_file = code_dir_path / "solver.py"
            if solver_file.is_file():
                logging.debug(
                    f"[isolated_benchmark] '{alt_filename}' not found in '{code_dir_path}'. "
                    f"Falling back to '{solver_file.name}'."
                )
                solver_module = load_solver_module(code_dir_path)
            else:
                # Try to auto-detect task from code_dir path structure
                logging.warning(f"[isolated_benchmark] Neither '{alt_filename}' nor 'solver.py' found in {code_dir_path}")
                logging.debug(f"[isolated_benchmark] Attempting auto-detection...")
                
                # Get task name from environment if parameter is wrong
                env_task_name = os.environ.get('CURRENT_TASK_NAME', task_name)
                if env_task_name != task_name:
                    logging.debug(f"[isolated_benchmark] Using task name from env: '{env_task_name}' instead of param: '{task_name}'")
                    task_name = env_task_name
                    alt_filename = f"{task_name}.py"
                
                # First check if code_dir already points to the task directory
                # (e.g., code_dir is already /app/AlgoTuneTasks/sha256_hashing)
                if code_dir_path.name == task_name and (code_dir_path / alt_filename).is_file():
                    # We're already in the task directory
                    logging.debug(f"[isolated_benchmark] code_dir already points to task directory: {code_dir_path}")
                    solver_module = load_solver_module(code_dir_path, solver_filename=alt_filename)
                else:
                    # Try common task directories
                    possible_task_dirs = [
                        code_dir_path / "AlgoTuneTasks" / task_name,
                        Path("/app/AlgoTuneTasks") / task_name,
                        Path("/app") / "AlgoTuneTasks" / task_name,
                    ]
                
                    logging.debug(f"[isolated_benchmark] Trying directories: {[str(d) for d in possible_task_dirs]}")
                    
                    for possible_dir in possible_task_dirs:
                        possible_task_file = possible_dir / f"{task_name}.py"
                        logging.debug(f"[isolated_benchmark] Checking: {possible_task_file}")
                        if possible_task_file.is_file():
                            logging.debug(f"[isolated_benchmark] Found task file at: {possible_task_file}")
                            solver_module = load_solver_module(possible_dir, f"{task_name}.py")
                            break
                    else:
                        # List what's actually in /app to debug - only in debug mode
                        try:
                            logging.debug(f"[isolated_benchmark] Current working directory: {os.getcwd()}")
                            logging.debug(f"[isolated_benchmark] Environment CODE_DIR: {os.environ.get('CODE_DIR', 'NOT_SET')}")
                            logging.debug(f"[isolated_benchmark] Environment CURRENT_TASK_NAME: {os.environ.get('CURRENT_TASK_NAME', 'NOT_SET')}")
                        
                            # Only show detailed directory contents in debug mode
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                app_contents = list(Path("/app").iterdir())
                                logging.debug(f"[isolated_benchmark] /app contents: {[str(p) for p in app_contents[:10]]}")
                                if Path("/app/AlgoTuneTasks").exists():
                                    algo_contents = list(Path("/app/AlgoTuneTasks").iterdir())
                                    logging.debug(f"[isolated_benchmark] /app/AlgoTuneTasks contents: {[str(p) for p in algo_contents]}")
                        except Exception as e:
                            logging.debug(f"[isolated_benchmark] Could not list /app contents: {e}")
                        
                        raise FileNotFoundError(
                            f"Neither '{alt_filename}' nor 'solver.py' found in {code_dir_path} or auto-detected paths"
                        )

        # ------------------------------------------------------------------
        # Ensure the loaded module exposes a `Solver` class compatible with
        # the loader utilities.  Many baseline reference implementations only
        # provide a stand-alone `solve` function or embed the logic inside a
        # Task subclass.  Wrap these cases on-the-fly so the timing utilities
        # continue to work without modification elsewhere.
        # ------------------------------------------------------------------

        # Check if Solver exists and is not the PySAT Solver class
        existing_solver = getattr(solver_module, "Solver", None)
        is_pysat_solver = (existing_solver is not None and 
                          getattr(existing_solver, "__module__", "").startswith("pysat"))
        
        if is_pysat_solver:
            logging.debug(
                f"[isolated_benchmark] Detected PySAT Solver class (module: {existing_solver.__module__}), "
                "will look for Task subclass instead"
            )
        
        if existing_solver is None or is_pysat_solver:
            # Case A – module-level `solve` function
            if hasattr(solver_module, "solve") and callable(solver_module.solve):
                logging.debug(
                    "[isolated_benchmark] Using top-level solve() function directly"
                )

                # Use the module's solve function directly
                solve = solver_module.solve

            else:
                # Case B – find first class with a callable `solve`
                fallback_cls = None
                for _name, _obj in vars(solver_module).items():
                    if not isinstance(_obj, type):
                        continue
                    # Only consider classes defined in *this* module to avoid picking
                    # imported base classes like `Task` that raise NotImplementedError.
                    if getattr(_obj, "__module__", None) != solver_module.__name__:
                        continue
                    # Lazy-import Task to avoid heavy dependency cost when not needed
                    from AlgoTuneTasks.base import Task  # local import for reflection
                    solve_attr = getattr(_obj, "solve", None)
                    if callable(solve_attr):
                        # Ensure the method is actually overridden (not inherited from Task)
                        import inspect
                        try:
                            if solve_attr is getattr(Task, "solve", None):
                                continue  # skip abstract base implementation
                        except Exception:
                            pass
                        fallback_cls = _obj
                        break

                if fallback_cls is not None:
                    logging.debug(
                        f"[isolated_benchmark] Auto-wrapping {fallback_cls.__name__} into solve callable"
                    )

                    # Create solve callable directly without modifying module namespace
                    def solve(problem):
                        task_instance = fallback_cls()
                        result = task_instance.solve(problem)
                        del task_instance
                        return result
                else:
                    raise AttributeError(
                        "No 'Solver' class, top-level solve(), or Task subclass with solve() found."
                    )
        else:
            # Normal case - get solve callable from Solver class
            solve = get_fresh_solve_callable(solver_module)

        # ------------------------------------------------------------------
        # 3) Deserialize problems
        # ------------------------------------------------------------------
        if payload_is_pickled:
            warmup_problem = pickle.loads(warmup_payload)
            timed_problem = pickle.loads(timed_payload)
        else:
            warmup_problem = warmup_payload
            timed_problem = timed_payload
        
        # Log payload sizes for debugging data format/size impact
        try:
            if isinstance(warmup_problem, dict) and 'plaintext' in warmup_problem:
                size = len(warmup_problem['plaintext'])
                worker_logger.info(f"[ISOLATED_WORKER_DEBUG] Task {task_name} payload size = {size} bytes")
        except Exception as e:
            worker_logger.warning(f"[ISOLATED_WORKER_DEBUG] Unable to determine payload size: {e}")
        
        # EXTENSIVE Debug logging to find the caching bug
        logging.debug(f"[isolated_worker] Problems are different objects: {warmup_problem is not timed_problem}")
        
        # Check if problems are actually different
        if isinstance(warmup_problem, dict) and isinstance(timed_problem, dict):
            logging.debug(f"[isolated_worker] Warmup problem keys: {list(warmup_problem.keys())}")
            logging.debug(f"[isolated_worker] Timed problem keys: {list(timed_problem.keys())}")
            
            # Check each key in detail
            problems_identical = True
            for key in warmup_problem.keys():
                if key not in timed_problem:
                    problems_identical = False
                    logging.debug(f"[isolated_worker] Key '{key}' missing in timed_problem")
                    break
                
                warmup_val = warmup_problem[key]
                timed_val = timed_problem[key]
                
                # Safe equality check that handles NumPy arrays properly
                try:
                    # Try to use NumPy array_equal for arrays, fall back to != for other types
                    import numpy as np
                    if hasattr(warmup_val, 'shape') and hasattr(timed_val, 'shape'):
                        # Both are array-like, use numpy comparison
                        values_equal = np.array_equal(warmup_val, timed_val)
                    else:
                        # Non-array types, use regular comparison
                        values_equal = warmup_val == timed_val
                        
                    if not values_equal:
                        problems_identical = False
                        logging.debug(f"[isolated_worker] Key '{key}' differs: warmup={type(warmup_val)} vs timed={type(timed_val)}")
                        if hasattr(warmup_val, 'shape'):
                            logging.debug(f"[isolated_worker] Shapes: warmup={getattr(warmup_val, 'shape', 'no shape')} vs timed={getattr(timed_val, 'shape', 'no shape')}")
                        else:
                            # For small sequences log exact values; guard against objects where
                            # __len__ raises (e.g., SciPy sparse matrices).
                            try:
                                if hasattr(warmup_val, '__len__') and len(warmup_val) < 10:
                                    logging.debug(f"[isolated_worker] Values: warmup={warmup_val} vs timed={timed_val}")
                            except TypeError:
                                # Length is ambiguous (e.g., sparse matrix); skip detailed value logging.
                                pass
                        break
                    else:
                        logging.debug(f"[isolated_worker] Key '{key}' is IDENTICAL")
                except (ValueError, TypeError, AttributeError) as e:
                    # Different shapes, types, or comparison error - definitely not identical
                    problems_identical = False
                    logging.debug(f"[isolated_worker] Key '{key}' comparison failed ({type(e).__name__}): warmup={type(warmup_val)} vs timed={type(timed_val)} - treating as different")
                    if hasattr(warmup_val, 'shape') and hasattr(timed_val, 'shape'):
                        logging.debug(f"[isolated_worker] Shapes: warmup={warmup_val.shape} vs timed={timed_val.shape}")
                    break
            
            if problems_identical:
                logging.error(f"[isolated_worker] *** CRITICAL BUG DETECTED ***")
                logging.error(f"[isolated_worker] Warmup and timed problems are COMPLETELY IDENTICAL!")
                logging.error(f"[isolated_worker] This explains the impossible 0.05ms timing!")
                logging.error(f"[isolated_worker] Solver is hitting cache from warmup run!")
            else:
                logging.info(f"[isolated_worker] ✓ GOOD: Warmup and timed problems are different (cache bug fixed)")
        else:
            logging.debug(f"[isolated_worker] Problem types: warmup={type(warmup_problem)}, timed={type(timed_problem)}")

        # ------------------------------------------------------------------
        # 4) Warmup calls - use config WARMUPS
        # ------------------------------------------------------------------
        logging.debug(f"[isolated_bm child] About to start {WARMUPS} warmup calls")
        
        # Diagnostic – verify PySAT patch (if PySAT present)
        try:
            from pysat._utils import MainThread  # type: ignore
            worker_logger.debug(
                "PYSAT_FIX: Before warmup – MainThread.check() = %s", MainThread.check()
            )
        except Exception:
            pass
        
        total_warmup_ns = 0
        warmup_result = None
        
        # Standard: 1 warmup per process
        t_w0 = time.perf_counter_ns()
        warmup_result = solve(warmup_problem)
        warmup_result = deep_materialize_fast(warmup_result)  # Force materialization
        single_warmup_ns = time.perf_counter_ns() - t_w0
        total_warmup_ns += single_warmup_ns
        logging.debug(f"[isolated_bm child] Warmup completed: {single_warmup_ns/1e6:.3f}ms")
        
        # Check if warmup returned None
        if warmup_result is None:
            raise ValueError("Solver returned None during warmup instead of a valid result dictionary")
            
        warmup_ns = total_warmup_ns
        logging.debug(f"[isolated_bm child] Warmup completed, total time: {warmup_ns/1e6:.3f}ms, result type: {type(warmup_result)}")
        
        # Check warmup result details
        if isinstance(warmup_result, dict) and 'total_cost' in warmup_result:
            logging.debug(f"[isolated_bm child] Warmup total cost: {warmup_result['total_cost']}")

        # ------------------------------------------------------------------
        # 5) Light cache clearing between warmup and timed
        # ------------------------------------------------------------------
        gc.collect()
        
        logging.debug(f"[isolated_bm child] Cache clearing completed")
        
        # ------------------------------------------------------------------
        # 6) Timed call
        # ------------------------------------------------------------------
        logging.debug(f"[isolated_bm child] About to start timed call")
        
        # Diagnostic – verify PySAT patch (if PySAT present)
        try:
            from pysat._utils import MainThread  # type: ignore
            worker_logger.debug(
                "PYSAT_FIX: Before solve – MainThread.check() = %s", MainThread.check()
            )
        except Exception:
            pass
        
        # Import StringIO and redirect_stdout for stdout capture
        from io import StringIO
        from contextlib import redirect_stdout
        
        # Capture stdout during timed execution
        captured_stdout = StringIO()
        
        t0 = time.perf_counter_ns()
        with redirect_stdout(captured_stdout):
            timed_result = solve(timed_problem)
        timed_result = deep_materialize_fast(timed_result)  # Force materialization
        timed_ns = time.perf_counter_ns() - t0
        
        # Get captured stdout
        stdout_content = captured_stdout.getvalue()
        
        logging.debug(f"[isolated_bm child] Timed call completed, result type: {type(timed_result)}")
        
        # If solver returned None treat as failure
        if timed_result is None:
            raise ValueError("Solver returned None instead of a valid result dictionary")
        
        # ------------------------------------------------------------------
        # 7) Marshal timing results back to parent (no validation field)
        # ------------------------------------------------------------------
        ret_dict["success"] = True
        ret_dict["warmup_ns"] = int(warmup_ns)
        ret_dict["timed_ns"] = int(timed_ns)
        ret_dict["stdout"] = stdout_content  # Captured stdout
        
        # Pickle the result for validation - we already have it, no need to run again!
        try:
            ret_dict["out_pickle"] = pickle.dumps(timed_result)
        except Exception as e:
            logging.warning(f"[isolated_worker] Failed to pickle result: {e}")
            # If pickling fails, at least pass a flag so we know we had a result
            ret_dict["out_pickle"] = b""
            ret_dict["had_result"] = True
        
        # ------------------------------------------------------------------
        # End of worker
        # ------------------------------------------------------------------

    except Exception as exc:  # pragma: no cover – we just forward error info
        ret_dict["success"] = False
        ret_dict["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


# -----------------------------------------------------------------------------
# Public helper – one warm-up + one timed call per fresh process, repeated K times
# -----------------------------------------------------------------------------

def run_isolated_benchmark(
    *,
    task_name: str,
    code_dir: str,
    warmup_problem: Any,
    timed_problem: Any,
    num_runs: int = 5,
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    """Benchmark *problem* using the strict isolation scheme requested by the
    user: every timing measurement happens in a dedicated interpreter.

    Note: timeout_seconds should be calculated as the timeout PER SUBPROCESS,
    not for all runs combined. Each subprocess performs warmup + timed call.

    Returns a dictionary that mirrors the most important keys of the existing
    ``utils.benchmark.run_benchmark`` contract so that downstream code can stay
    unchanged.
    """

    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Detect if we are already inside a **daemon** multiprocessing worker.
    # In that case `spawn`-ing further children would throw
    # "daemonic processes are not allowed to have children". We therefore
    # fall back to an **in-process** benchmark that still performs a warm-up
    # followed by a timed run – without the per-measurement interpreter
    # isolation, but at least it completes instead of crashing.
    # ------------------------------------------------------------------

    if mp.current_process().daemon:
        logger.warning(
            f"run_isolated_benchmark called from a daemonic process – falling back to in-process timing. "
            f"Process: {mp.current_process().name}, PID: {os.getpid()}"
        )

        from AlgoTuner.utils.solver_loader import load_solver_module, get_fresh_solve_callable_with_module_reload

        # Removed legacy cache-clearing flag – no longer used

        # Numba logging removed

        # Prefer '<task_name>.py' when available, matching main worker logic.
        code_dir_path = Path(code_dir)
        alt_filename = f"{task_name}.py"
        alt_file_path = code_dir_path / alt_filename

        if alt_file_path.is_file():
            solver_module = load_solver_module(code_dir_path, solver_filename=alt_filename)
        else:
            solver_module = load_solver_module(code_dir_path)

        # Use module reload version to ensure complete isolation between runs
        solve = get_fresh_solve_callable_with_module_reload(solver_module)

        times_ns: List[int] = []
        last_result: Optional[Any] = None

        for _ in range(num_runs):
            # Warm-up timing - standard: 1 warmup
            try:
                t_w0 = time.perf_counter_ns()
                warmup_res = solve(warmup_problem)
                warmup_res = deep_materialize_fast(warmup_res)  # Force materialization
                warmup_ns = time.perf_counter_ns() - t_w0
            except Exception as exc:
                logger.warning(f"[isolated_benchmark/fallback] Warm-up failed: {exc}")
                continue

            t0 = time.perf_counter_ns()
            try:
                out = solve(timed_problem)
                out = deep_materialize_fast(out)  # Force materialization
                # Check if solver returned None
                if out is None:
                    raise ValueError("Solver returned None instead of a valid result dictionary")
            except Exception as exc:
                logger.warning(f"[isolated_benchmark/fallback] Timed call failed: {exc}")
                continue

            elapsed_ns = time.perf_counter_ns() - t0
            times_ns.append(elapsed_ns)
            last_result = out
            logger.debug(
                f"[isolated_bm fallback] run warmup={warmup_ns/1e6:.3f}ms timed={elapsed_ns/1e6:.3f}ms"
            )

        if not times_ns:
            return {
                "success": False,
                "error": "All in-process fallback runs failed",
                "timeout_occurred": False,
                "runs": num_runs,
            }

        min_ns = min(times_ns)
        mean_ns = statistics.mean(times_ns)

        return {
            "success": True,
            "values_ns": times_ns,
            "num_runs_executed": len(times_ns),
            "min_ns": min_ns,
            "mean_ns": mean_ns,
            "min_time_ms": min_ns / 1e6,
            "mean_time_ms": mean_ns / 1e6,
            "elapsed_ms": min_ns / 1e6,
            "result": last_result,
            "timeout_occurred": False,
        }

    # ------------------------------------------------------------------
    # Normal (non-daemon) path – spawn one process per measurement
    # ------------------------------------------------------------------

    # Use 'forkserver' for thread-safe process creation - each run gets its own process
    ctx = mp.get_context("forkserver")
    logging.debug(f"[isolated_benchmark] Using 'forkserver' multiprocessing context for thread-safe per-run isolation")

    run_results: List[Dict[str, float]] = []
    last_result: Optional[Any] = None

    # We're using forkserver which requires pickling
    FORCE_PICKLE = True

    if FORCE_PICKLE:
        # Materialize any mmap objects before pickling
        warmup_problem_materialized = materialize_mmap_objects(warmup_problem)
        timed_problem_materialized = materialize_mmap_objects(timed_problem)
        warmup_payload = pickle.dumps(warmup_problem_materialized)
        timed_payload = pickle.dumps(timed_problem_materialized)
        payload_is_pickled = True
    else:
        warmup_payload = warmup_problem
        timed_payload = timed_problem
        payload_is_pickled = False

    def _run_with_manager(ctx):
        """Inner function that uses the manager context."""
        # Initialize local variables
        run_results = []
        last_result = None
        
        # Load manager refresh interval from config
        try:
            from AlgoTuner.config.config import load_config
            config = load_config()
            MANAGER_REFRESH_INTERVAL = config.get("benchmark", {}).get("manager_refresh_interval", 50)
            logger.debug(f"[isolated_benchmark] Using manager_refresh_interval={MANAGER_REFRESH_INTERVAL} from config")
        except Exception as e:
            MANAGER_REFRESH_INTERVAL = 50
            logger.debug(f"[isolated_benchmark] Failed to load manager_refresh_interval from config: {e}. Using default: {MANAGER_REFRESH_INTERVAL}")
        
        # Track Manager usage
        manager_usage = 0
        mgr = None
        
        try:
            for idx in range(num_runs):
                # Create or refresh Manager if needed
                if mgr is None or manager_usage >= MANAGER_REFRESH_INTERVAL:
                    if mgr is not None:
                        logger.debug(f"[isolated_benchmark] Refreshing Manager after {manager_usage} uses")
                        try:
                            mgr.shutdown()
                        except Exception as e:
                            logger.warning(f"[isolated_benchmark] Error shutting down Manager: {e}")
                        del mgr
                        gc.collect()  # Force cleanup of Manager resources
                    
                    logger.debug(f"[isolated_benchmark] Creating new Manager for run {idx+1}/{num_runs}")
                    mgr = ctx.Manager()
                    manager_usage = 0
                # Each run: one fork does warmup+timed sequentially, then dies
                with tempfile.TemporaryDirectory() as tmp_dir:
                    ret = mgr.dict()
                    proc = ctx.Process(
                        target=_fork_run_worker,
                        args=(task_name, code_dir, tmp_dir, warmup_payload, timed_payload, payload_is_pickled, ret),
                        daemon=False,
                    )
                    proc.start()
                    proc.join(timeout_seconds)

                    if proc.is_alive():
                        timeout_error = f"Process timed out after {timeout_seconds}s"
                        logger.warning(
                            f"[isolated_benchmark] Run {idx+1}/{num_runs} timed out after {timeout_seconds}s – will skip and continue"
                        )
                        # Try terminate first (SIGTERM) for cleaner shutdown
                        proc.terminate()
                        proc.join(timeout=0.5)  # Wait up to 0.5s for clean exit
                        if proc.is_alive():
                            # If still alive, force kill (SIGKILL)
                            proc.kill()
                            proc.join()

                        # Add a small delay to allow system cleanup after killing the process
                        # This helps prevent performance degradation in subsequent runs
                        time.sleep(0.1)  # 100ms delay

                        # Record timeout and continue; we will still succeed as long as
                        # *at least one* run finishes successfully.  This avoids rare
                        # false-positive timeouts due to sporadic kernel stalls or
                        # first-touch page-fault storms.
                        run_results.append({
                            "warmup_ns": None,
                            "timed_ns": None,
                            "timeout": True,
                        })
                        continue  # attempt next run

                    if not ret.get("success", False):
                        run_error = ret.get('error')
                        if not run_error:
                            # Provide more detailed unknown error information
                            ret_keys = list(ret.keys()) if ret else []
                            run_error = f"Process failed without error message. Return dict keys: {ret_keys}. Process may have crashed or timed out."
                        # Extract clean error message with code context
                        clean_error = _extract_clean_error_with_context(run_error)
                        logger.warning(
                            f"[isolated_benchmark] Run {idx+1}/{num_runs} failed: {clean_error}"
                        )
                        
                        # Record execution error and continue.  We still treat the
                        # whole benchmark as successful if another run finishes.
                        run_results.append({
                            "warmup_ns": None,
                            "timed_ns": None,
                            "error": clean_error,
                        })
                        continue  # try next run

                    warmup_ns = ret.get("warmup_ns")
                    timed_ns = ret.get("timed_ns")
                    if warmup_ns is None or timed_ns is None:
                        timing_error = "No timing information returned from worker process"
                        logger.warning(
                            f"[isolated_benchmark] Run {idx+1}/{num_runs} did not return timing info – skipping"
                        )
                        run_results.append({
                            "warmup_ns": None,
                            "timed_ns": None,
                            "error": timing_error,
                        })
                        continue

                    # Capture stdout if available
                    captured_stdout = ret.get("stdout", "")
                    
                    run_results.append({
                        "warmup_ns": int(warmup_ns),
                        "timed_ns": int(timed_ns),
                        "warmup_ms": warmup_ns / 1e6,
                        "timed_ms": timed_ns / 1e6,
                        "stdout": captured_stdout,
                    })
                    try:
                        last_result = pickle.loads(ret.get("out_pickle", b""))
                    except Exception:
                        last_result = None
                    
                    # Increment manager usage counter
                    manager_usage += 1

        finally:
            # Clean up Manager if it exists
            if mgr is not None:
                logger.debug(f"[isolated_benchmark] Cleaning up Manager after {manager_usage} total uses")
                try:
                    mgr.shutdown()
                except Exception as e:
                    logger.warning(f"[isolated_benchmark] Error shutting down Manager during cleanup: {e}")

        return {"run_results": run_results, "last_result": last_result}

    # Use retry wrapper for manager context
    manager_result = _run_with_manager_retry(_run_with_manager, task_name=task_name)
    
    if not manager_result.get("success", True):
        return manager_result
    
    run_results = manager_result["run_results"] 
    last_result = manager_result["last_result"]

    # Filter out entries that have real timing info
    successful_runs = [r for r in run_results if r.get("warmup_ns") is not None]

    if not successful_runs:
        # Check if all failures were timeouts or if there were actual errors
        timeout_runs = [r for r in run_results if r.get("timeout", False)]
        error_runs = [r for r in run_results if r.get("error") is not None]
        
        if error_runs:
            # If there were any errors, report the first one with context
            first_error = error_runs[0].get("error", "Unknown error")
            # Categorise the error type based on the message so that callers can
            # react appropriately (e.g. halt evaluation on OOM conditions).
            _lower_err = first_error.lower()
            if any(_kw in _lower_err for _kw in ("unable to allocate", "memoryerror", "out of memory", "arraymemoryerror", "cannot allocate memory", "killed")):
                detected_error_type = "memory_error"
            elif "importerror" in _lower_err:
                detected_error_type = "import_error"
            else:
                detected_error_type = "execution_error"
            result = {
                "success": False,
                "error": first_error,
                "timeout_occurred": len(timeout_runs) > 0,
                "error_type": detected_error_type,
                "runs": num_runs,
                "num_errors": len(error_runs),
                "num_timeouts": len(timeout_runs),
            }
        else:
            # All failures were timeouts
            result = {
                "success": False,
                "error": "All runs timed out",
                "timeout_occurred": True,
                "error_type": "timeout",
                "runs": num_runs,
            }
        logger.warning(f"[isolated_benchmark] Returning failure result: {result}")
        return result

    # Extract timing data
    warmup_times_ns = [r["warmup_ns"] for r in successful_runs]
    timed_times_ns = [r["timed_ns"] for r in successful_runs]

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    logger.debug(f"[isolated_benchmark] Aggregating results: collected {len(successful_runs)} timing measurements")
    
    min_timed_ns = min(timed_times_ns)
    mean_timed_ns = statistics.mean(timed_times_ns)
    mean_warmup_ns = statistics.mean(warmup_times_ns)

    # Collect stdout from all runs (they should be the same, so just use the last one)
    stdout_content = ""
    if successful_runs:
        # Get stdout from the last successful run
        stdout_content = successful_runs[-1].get("stdout", "")
    
    result = {
        "success": True,
        "individual_results": run_results,
        "warmup_times_ns": warmup_times_ns,
        "timed_times_ns": timed_times_ns,
        "num_runs_executed": len(successful_runs),
        "min_ns": min_timed_ns,
        "mean_ns": mean_timed_ns,
        "min_time_ms": min_timed_ns / 1e6,
        "mean_time_ms": mean_timed_ns / 1e6,
        "elapsed_ms": min_timed_ns / 1e6,
        "mean_warmup_ms": mean_warmup_ns / 1e6,
        "timeout_occurred": len(successful_runs) < num_runs,
        "num_timeouts": sum(1 for r in run_results if r.get("timeout")),
        "num_errors": sum(1 for r in run_results if r.get("error") is not None),
        "stdout": stdout_content,
        "stderr": "",  # We don't capture stderr for eval_input
        "result": last_result  # Add the last result for validation
    }
    
    logger.debug(f"[isolated_benchmark] Summary: mean_warmup={mean_warmup_ns / 1e6:.3f}ms, mean_timed={mean_timed_ns / 1e6:.3f}ms, runs={len(successful_runs)}")
    return result


def run_isolated_benchmark_with_fetch(
    *,
    task_name: str,
    code_dir: str,
    warmup_fetch_info: Dict[str, Any],
    timed_fetch_info: Dict[str, Any],
    num_runs: int = 5,
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    """Memory-efficient version of run_isolated_benchmark that loads problems inside workers.
    
    This avoids keeping large problem instances in the parent process memory.
    
    Args:
        task_name: Name of the task
        code_dir: Directory containing solver code
        warmup_fetch_info: Info for fetching warmup problem (type + parameters)
        timed_fetch_info: Info for fetching timed problem (type + parameters)
        num_runs: Number of timing runs
        timeout_seconds: Timeout per subprocess
        
    Returns:
        Same format as run_isolated_benchmark
    """
    logger = logging.getLogger(__name__)
    
    # Always use memory-efficient path to avoid pickling issues with mmap objects
    logger.debug(f"[isolated_benchmark_fetch] Using memory-efficient streaming approach for fetch types: "
                 f"warmup={warmup_fetch_info.get('type')}, timed={timed_fetch_info.get('type')}")
    
    # Similar setup to run_isolated_benchmark
    if mp.current_process().daemon:
        logger.warning(
            f"run_isolated_benchmark_with_fetch called from a daemonic process – attempting workaround. "
            f"Process: {mp.current_process().name}, PID: {os.getpid()}"
        )
        # For memory-efficient streaming, we need to use a different approach in daemon processes
        # Fall back to loading the problems and using regular isolated benchmark
        
        # Load problems from fetch info
        if warmup_fetch_info["type"] == "jsonl_streaming":
            import orjson
            with open(warmup_fetch_info["path"], 'r') as f:
                for i, line in enumerate(f):
                    if i == warmup_fetch_info["index"]:
                        warmup_data = orjson.loads(line.strip()).get("problem")
                        break
        else:
            warmup_data = warmup_fetch_info["data"]
            
        if timed_fetch_info["type"] == "jsonl_streaming":
            import orjson
            with open(timed_fetch_info["path"], 'r') as f:
                for i, line in enumerate(f):
                    if i == timed_fetch_info["index"]:
                        timed_data = orjson.loads(line.strip()).get("problem")
                        break
        else:
            timed_data = timed_fetch_info["data"]
            
        # Use regular isolated benchmark with loaded data
        return run_isolated_benchmark(
            task_name=task_name,
            code_dir=code_dir,
            warmup_problem=warmup_data,
            timed_problem=timed_data,
            num_runs=num_runs,
            timeout_seconds=timeout_seconds
        )
    
    def _run_with_manager_fetch(ctx):
        """Inner function that uses the manager context for fetch-based benchmark."""
        run_results: List[Dict[str, float]] = []
        last_result = None
        
        # Load manager refresh interval from config
        try:
            from AlgoTuner.config.config import load_config
            config = load_config()
            MANAGER_REFRESH_INTERVAL = config.get("benchmark", {}).get("manager_refresh_interval", 50)
            logger.debug(f"[isolated_benchmark_fetch] Using manager_refresh_interval={MANAGER_REFRESH_INTERVAL} from config")
        except Exception as e:
            MANAGER_REFRESH_INTERVAL = 50
            logger.debug(f"[isolated_benchmark_fetch] Failed to load manager_refresh_interval from config: {e}. Using default: {MANAGER_REFRESH_INTERVAL}")
        
        # Track Manager usage
        manager_usage = 0
        mgr = None
        
        try:
            for idx in range(num_runs):
                # Create or refresh Manager if needed
                if mgr is None or manager_usage >= MANAGER_REFRESH_INTERVAL:
                    if mgr is not None:
                        logger.debug(f"[isolated_benchmark_fetch] Refreshing Manager after {manager_usage} uses")
                        try:
                            mgr.shutdown()
                        except Exception as e:
                            logger.warning(f"[isolated_benchmark_fetch] Error shutting down Manager: {e}")
                        del mgr
                        gc.collect()  # Force cleanup of Manager resources
                    
                    logger.debug(f"[isolated_benchmark_fetch] Creating new Manager for run {idx+1}/{num_runs}")
                    mgr = ctx.Manager()
                    manager_usage = 0
                with tempfile.TemporaryDirectory() as tmp_dir:
                    ret = mgr.dict()
                    proc = ctx.Process(
                        target=_fork_run_worker_with_fetch,
                        args=(task_name, code_dir, tmp_dir, warmup_fetch_info, timed_fetch_info, ret),
                        daemon=False,
                    )
                    proc.start()
                    proc.join(timeout_seconds)

                    if proc.is_alive():
                        timeout_error = f"Process timed out after {timeout_seconds}s"
                        logger.warning(f"[isolated_benchmark_fetch] Run {idx+1}/{num_runs} timed out")
                        proc.kill()
                        proc.join()
                        return {
                            "success": False,
                            "error": timeout_error,
                            "timeout_occurred": True,
                            "error_type": "timeout",
                            "runs": num_runs,
                            "num_runs_executed": idx,
                        }

                    if not ret.get("success", False):
                        run_error = ret.get('error', 'Unknown error')
                        clean_error = _extract_clean_error_with_context(run_error)
                        logger.warning(f"[isolated_benchmark_fetch] Run {idx+1}/{num_runs} failed: {clean_error}")
                        return {
                            "success": False,
                            "error": clean_error,
                            "timeout_occurred": False,
                            "error_type": "execution_error",
                            "runs": num_runs,
                            "num_runs_executed": idx,
                        }

                    warmup_ns = ret.get("warmup_ns")
                    timed_ns = ret.get("timed_ns")
                    if warmup_ns is None or timed_ns is None:
                        timing_error = f"No timing information returned from worker process"
                        logger.warning(f"[isolated_benchmark_fetch] Run {idx+1}/{num_runs} no timing info")
                        return {
                            "success": False,
                            "error": timing_error,
                            "timeout_occurred": False,
                            "error_type": "timing_error",
                            "runs": num_runs,
                            "num_runs_executed": idx,
                        }

                    run_results.append({
                        "warmup_ns": int(warmup_ns),
                        "timed_ns": int(timed_ns),
                        "warmup_ms": warmup_ns / 1e6,
                        "timed_ms": timed_ns / 1e6
                    })
                    
                    # Clear the manager dict to free memory
                    ret.clear()
                    
                    # Increment manager usage counter
                    manager_usage += 1
                    
                    # Force GC every few runs to prevent memory buildup
                    if (idx + 1) % 3 == 0:
                        gc.collect()
        
        finally:
            # Clean up Manager if it exists
            if mgr is not None:
                logger.debug(f"[isolated_benchmark_fetch] Cleaning up Manager after {manager_usage} total uses")
                try:
                    mgr.shutdown()
                except Exception as e:
                    logger.warning(f"[isolated_benchmark_fetch] Error shutting down Manager during cleanup: {e}")
        
        return {"run_results": run_results}

    # Use retry wrapper for manager context
    manager_result = _run_with_manager_retry(_run_with_manager_fetch, task_name=f"{task_name}_fetch")
    
    if not manager_result.get("success", True):
        return manager_result
    
    run_results = manager_result["run_results"]

    if not run_results:
        return {
            "success": False,
            "error": "No successful runs completed",
            "timeout_occurred": False,
            "error_type": "unknown",
            "runs": num_runs,
        }

    # Calculate statistics
    timed_times_ns = [r["timed_ns"] for r in run_results]
    warmup_times_ns = [r["warmup_ns"] for r in run_results]
    min_timed_ns = min(timed_times_ns)
    mean_timed_ns = statistics.mean(timed_times_ns)
    mean_warmup_ns = statistics.mean(warmup_times_ns)

    result = {
        "success": True,
        "individual_results": run_results,
        "warmup_times_ns": warmup_times_ns,
        "timed_times_ns": timed_times_ns,
        "num_runs_executed": len(run_results),
        "min_ns": min_timed_ns,
        "mean_ns": mean_timed_ns,
        "min_time_ms": min_timed_ns / 1e6,
        "mean_time_ms": mean_timed_ns / 1e6,
        "elapsed_ms": min_timed_ns / 1e6,
        "mean_warmup_ms": mean_warmup_ns / 1e6,
        "timeout_occurred": False,
        "num_timeouts": sum(1 for r in run_results if r.get("timeout")),
    }
    
    logger.debug(f"[isolated_benchmark_fetch] Summary: mean_timed={mean_timed_ns / 1e6:.3f}ms, runs={len(run_results)}")
    return result


def _fork_run_worker_with_fetch(
    task_name: str,
    code_dir: str,
    tmp_dir: str,
    warmup_fetch_info: Dict[str, Any],
    timed_fetch_info: Dict[str, Any],
    ret_dict,
):
    """Worker that fetches problems inside the subprocess to minimize parent memory usage."""
    import traceback
    import os
    os.environ["ALGOTUNER_SOLVER_WORKER"] = "1"  # mark as solver worker
    import sys
    
    worker_logger = logging.getLogger("isolated_worker_fetch")

    # ------------------------------------------------------------------
    # Memory safety identical to _fork_run_worker – cap RLIMIT_AS to 14 GB.
    # ------------------------------------------------------------------
    # Check if RLIMIT_AS should be disabled from config
    disable_rlimit_as = False
    try:
        from AlgoTuner.config.loader import load_config
        config = load_config()
        disable_rlimit_as = config.get("benchmark", {}).get("validation_pool", {}).get("disable_rlimit_as", False)
        if disable_rlimit_as:
            # Set environment variable to skip RLIMIT_AS in ProcessMemoryMonitor
            os.environ['SKIP_RLIMIT_AS'] = '1'
            worker_logger.debug(
                "RLIMIT_AS disabled by configuration in isolated benchmark fetch-worker (%d)",
                os.getpid(),
            )
    except Exception as config_err:  # noqa: BLE001
        worker_logger.warning(
            "Could not load config to check disable_rlimit_as in fetch-worker (%d): %s",
            os.getpid(),
            config_err,
        )
    
    if not disable_rlimit_as:
        try:
            from AlgoTuner.utils.process_monitor import init_worker_memory_monitor

            _mem_mon = init_worker_memory_monitor(14.0)  # noqa: WPS437
            worker_logger.debug(
                "ProcessMemoryMonitor initialised – RLIMIT_AS capped at 14 GB in isolated benchmark fetch-worker (%d)",
                os.getpid(),
            )
        except Exception as _mm_err:  # noqa: BLE001
            worker_logger.warning(
                "Could not initialise ProcessMemoryMonitor in isolated benchmark fetch-worker (%d): %s",
                os.getpid(),
                _mm_err,
            )

    # Apply PySAT fixes
    try:
        if importlib.util.find_spec("pysat") is not None:
            # PySAT present – apply lightweight patches.
            from AlgoTuner.utils.pysat_fix import apply_pysat_fixes
            apply_pysat_fixes()
            worker_logger.debug("PYSAT_FIX: patches applied in fetch worker")
        else:
            worker_logger.debug("PYSAT_FIX: PySAT not found – skipping patch application")
    except Exception as exc:
        worker_logger.warning(f"PYSAT_FIX: skipped due to import error – {exc}")

    try:
        # Setup environment
        os.environ.setdefault("CODE_DIR", code_dir)
        os.environ["CURRENT_TASK_NAME"] = task_name
        os.environ["PYTHONPYCACHEPREFIX"] = tmp_dir
        for _var in ("NUMBA_DEBUG", "NUMBA_DUMP_IR", "NUMBA_DUMP_CFG", "NUMBA_DUMP_OPT_STATS"):
            os.environ.pop(_var, None)
        os.chdir(tmp_dir)

        # Load solver (same logic as original worker)
        code_dir_path = Path(code_dir)
        from AlgoTuner.utils.solver_loader import load_solver_module, get_fresh_solve_callable

        alt_filename = f"{task_name}.py"
        alt_file_path = code_dir_path / alt_filename

        if alt_file_path.is_file():
            solver_module = load_solver_module(code_dir_path, solver_filename=alt_filename)
        else:
            solver_file = code_dir_path / "solver.py"
            if solver_file.is_file():
                solver_module = load_solver_module(code_dir_path)
            else:
                # Auto-detection logic (same as original)
                env_task_name = os.environ.get('CURRENT_TASK_NAME', task_name)
                if env_task_name != task_name:
                    task_name = env_task_name
                    alt_filename = f"{task_name}.py"
                
                # First check if code_dir already points to the task directory
                # (e.g., code_dir is already /app/AlgoTuneTasks/sha256_hashing)
                if code_dir_path.name == task_name and (code_dir_path / alt_filename).is_file():
                    # We're already in the task directory
                    solver_module = load_solver_module(code_dir_path, solver_filename=alt_filename)
                else:
                    # Try subdirectories
                    possible_task_dirs = [
                        code_dir_path / "AlgoTuneTasks" / task_name,
                        Path("/app/AlgoTuneTasks") / task_name,
                        Path("/app") / "AlgoTuneTasks" / task_name,
                    ]
                    
                    for possible_dir in possible_task_dirs:
                        possible_task_file = possible_dir / f"{task_name}.py"
                        if possible_task_file.is_file():
                            solver_module = load_solver_module(possible_dir, f"{task_name}.py")
                            break
                    else:
                        raise FileNotFoundError(
                            f"Neither '{alt_filename}' nor 'solver.py' found in {code_dir_path} or auto-detected paths"
                        )

        # Get solver (same wrapper logic as original)
        if hasattr(solver_module, "Solver"):
            solve = get_fresh_solve_callable(solver_module)
        elif hasattr(solver_module, "solve"):
            class _AutoSolver:
                def solve(self, problem):
                    return solver_module.solve(problem)
            solve = _AutoSolver().solve
        else:
            # Try task-based solver
            # First, collect classes whose names end with 'Task' *and* are defined in this module
            task_classes = []
            from AlgoTuneTasks.base import Task as _BaseTask
            for _name, _obj in vars(solver_module).items():
                if not isinstance(_obj, type):
                    continue
                if not _name.endswith("Task"):
                    continue
                # Ensure class is defined in this module, not an import of the abstract base
                if getattr(_obj, "__module__", None) != solver_module.__name__:
                    continue
                # Skip the abstract base Task itself
                if _obj is _BaseTask or getattr(_obj, "solve", None) is getattr(_BaseTask, "solve", None):
                    continue
                task_classes.append(_obj)

            # Fallback: if still empty, look for any concrete subclass of Task defined in this module
            if not task_classes:
                task_classes = [obj for obj in vars(solver_module).values()
                                if isinstance(obj, type) and issubclass(obj, _BaseTask)
                                and obj is not _BaseTask
                                and getattr(obj, "__module__", None) == solver_module.__name__]

            if task_classes:
                task_cls = task_classes[0]
                class _TaskWrapperSolver:
                    def __init__(self):
                        self.task_instance = task_cls()
                    def solve(self, problem):
                        return self.task_instance.solve(problem)
                solve = _TaskWrapperSolver().solve
            else:
                raise AttributeError("No solve method or Solver class found")

        # Fetch problems inside the worker (memory-efficient!)
        def _fetch_problem(fetch_info):
            if fetch_info["type"] == "direct":
                return fetch_info["data"]
            elif fetch_info["type"] in ("jsonl_streaming", "jsonl_seek"):
                import orjson
                import functools
                from AlgoTuner.utils.serialization import dataset_decoder
                import os
                
                jsonl_path = fetch_info["path"]
                # Use index if streaming, otherwise use byte offset for seek.
                use_seek = fetch_info["type"] == "jsonl_seek"
                target_index = fetch_info.get("index")  # may be None
                target_offset = fetch_info.get("offset")  # may be None
                
                # Efficiently read only the target line without keeping previous records in memory
                actual_base_dir = os.path.dirname(jsonl_path)
                object_hook_for_load = functools.partial(dataset_decoder, base_dir=actual_base_dir)
                
                if use_seek and target_offset is not None:
                    with open(jsonl_path, 'rb') as f:
                        f.seek(target_offset)
                        line = f.readline()
                        try:
                            raw_record = orjson.loads(line)
                            processed_record = object_hook_for_load(raw_record)
                            return processed_record.get("problem", processed_record)
                        except orjson.JSONDecodeError as e:
                            raise RuntimeError(f"JSON Decode Error (seek) in {jsonl_path} at offset {target_offset}: {e}")
                else:
                    with open(jsonl_path, 'r') as f:
                        current_index = 0
                        for line in f:
                            if current_index == target_index:
                                line = line.strip()
                                if not line:
                                    raise RuntimeError("Empty line while streaming JSONL")
                                try:
                                    raw_record = orjson.loads(line)
                                    processed_record = object_hook_for_load(raw_record)
                                    return processed_record.get("problem", processed_record)
                                except orjson.JSONDecodeError as e:
                                    raise RuntimeError(f"JSON Decode Error in {jsonl_path}, line {current_index}: {e}")
                            current_index += 1
                    raise IndexError("JSONL index not found")
            else:
                raise ValueError(f"Unknown fetch type: {fetch_info['type']}")

        # Load problems one at a time
        warmup_problem = _fetch_problem(warmup_fetch_info)
        timed_problem = _fetch_problem(timed_fetch_info)

        # Run timing (same as original worker)
        import time
        from AlgoTuner.utils.timing_config import WARMUPS

        # Warmup phase - standard: 1 warmup
        t_w0 = time.perf_counter_ns()
        warmup_result = solve(warmup_problem)
        warmup_result = deep_materialize_fast(warmup_result)  # Force materialization
        if warmup_result is None:
            raise ValueError("Solver returned None during warmup instead of a valid result dictionary")
        warmup_ns = time.perf_counter_ns() - t_w0

        # Timed phase
        t0 = time.perf_counter_ns()
        out = solve(timed_problem)
        out = deep_materialize_fast(out)  # Force materialization
        if out is None:
            raise ValueError("Solver returned None instead of a valid result dictionary")
        timed_ns = time.perf_counter_ns() - t0

        # Return results
        ret_dict["success"] = True
        ret_dict["warmup_ns"] = warmup_ns
        ret_dict["timed_ns"] = timed_ns
        # Skip pickling the output for baseline evaluation - we only need timing
        # This avoids expensive pickling of large outputs (e.g., 189MB ciphertext)
        ret_dict["out_pickle"] = b""

    except Exception as exc:
        tb_str = traceback.format_exc()
        ret_dict["success"] = False
        ret_dict["error"] = tb_str
        worker_logger.error(f"Worker failed: {exc}", exc_info=True)


def _is_manager_error(exception: Exception) -> bool:
    """
    Determine if an exception is a retryable manager-related error.
    
    Args:
        exception: The exception to classify
        
    Returns:
        True if the error is likely a transient manager issue that should be retried
    """
    error_msg = str(exception).lower()
    error_type = type(exception).__name__
    
    # Common manager/multiprocessing connection issues
    retryable_patterns = [
        "manager",
        "connection",
        "broken pipe",
        "connection refused",
        "resource temporarily unavailable",
        "semlock",
        "shared memory",
        "ipc",
        "pipe",
        "socket",
        "no child process",  # handle ECHILD errors
        "no child processes"  # common wording on some platforms
    ]
    
    # Exception types that are typically retryable
    retryable_types = [
        "ConnectionError",
        "BrokenPipeError", 
        "OSError",
        "IOError",
        "RemoteError",
        "ChildProcessError"  # treat child process errors as retryable
    ]
    
    # Check if error message contains retryable patterns
    message_match = any(pattern in error_msg for pattern in retryable_patterns)
    
    # Check if exception type is retryable
    type_match = error_type in retryable_types
    
    return message_match or type_match


def _run_with_manager_retry(
    manager_func,
    max_retries: int = 3,
    base_delay: float = 0.5,
    task_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Execute a function that uses multiprocessing.Manager with retry logic.
    
    Args:
        manager_func: Function that takes a multiprocessing context and returns a result
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (with exponential backoff)
        task_name: Task name for logging
        
    Returns:
        Result from manager_func or error dict if all retries failed
    """
    ctx = mp.get_context("forkserver")
    
    for attempt in range(max_retries):
        try:
            logging.debug(f"[isolated_benchmark] Attempt {attempt + 1}/{max_retries} for task '{task_name}'")
            
            # Force garbage collection before each attempt to free resources
            if attempt > 0:
                gc.collect()
            
            result = manager_func(ctx)
            
            if attempt > 0:
                logging.info(f"[isolated_benchmark] Task '{task_name}' succeeded on attempt {attempt + 1}")
            
            return result
            
        except Exception as e:
            is_retryable = _is_manager_error(e)
            
            if attempt == max_retries - 1:
                # Final attempt failed
                logging.error(f"[isolated_benchmark] Task '{task_name}' failed after {max_retries} attempts. Final error: {e}")
                return {
                    "success": False,
                    "error": f"Manager context failed after {max_retries} attempts: {e}",
                    "timeout_occurred": False,
                    "error_type": "manager_retry_exhausted",
                    "runs": 0,
                    "num_runs_executed": 0,
                }
            elif is_retryable:
                # Retryable error - wait and try again
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)  # Exponential backoff with jitter
                logging.warning(
                    f"[isolated_benchmark] Task '{task_name}' attempt {attempt + 1} failed with retryable error: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

                # If the error relates to missing child processes, switch to 'spawn' start-method for next attempt
                try:
                    child_err = isinstance(e, ChildProcessError) or "no child process" in str(e).lower()
                except Exception:
                    child_err = False

                if child_err:
                    try:
                        mp.set_start_method("spawn", force=True)
                        ctx = mp.get_context("spawn")
                        logging.warning(
                            f"[isolated_benchmark] Switching multiprocessing start_method to 'spawn' for task '{task_name}' after ChildProcessError"
                        )
                    except Exception as _sm_err:
                        logging.debug(f"[isolated_benchmark] Failed to switch start method: {_sm_err}")
            else:
                # Non-retryable error - fail immediately
                logging.error(f"[isolated_benchmark] Task '{task_name}' failed with non-retryable error: {e}")
                return {
                    "success": False,
                    "error": f"Non-retryable error: {e}",
                    "timeout_occurred": False,
                    "error_type": "non_retryable_error",
                    "runs": 0,
                    "num_runs_executed": 0,
                }
    
    # Should never reach here, but just in case
    return {
        "success": False,
        "error": "Unexpected retry loop exit",
        "timeout_occurred": False,
        "error_type": "retry_logic_error",
        "runs": 0,
        "num_runs_executed": 0,
    }