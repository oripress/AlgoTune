"""
AlgoTuner Solver Loading Module
==============================

This module provides functionality to dynamically load and execute solver code.
"""

import sys
try:
    import AlgoTuneTasks
    import AlgoTuneTasks.base as _algotunetas_base_module
    if 'AlgoTune' not in sys.modules:
        class AlgoTuneModule:
            def __getattr__(self, name):
                if name == 'tasks':
                    class AlgoTuneTasksNestedModule:
                        def __getattr__(self, nested_name):
                            if nested_name == 'base':
                                return _algotunetas_base_module
                            return getattr(AlgoTuneTasks, nested_name)
                    return AlgoTuneTasksNestedModule()
                return getattr(AlgoTuneTasks, name)
        sys.modules['AlgoTune'] = AlgoTuneModule()
        sys.modules['AlgoTune.tasks'] = sys.modules['AlgoTune'].tasks
        sys.modules['AlgoTune.tasks.base'] = _algotunetas_base_module
        
except ImportError:
    pass

from pathlib import Path
import importlib.util
from types import ModuleType
import os
import logging
import sys
import inspect
from contextlib import contextmanager
import gc
import signal
import time

from AlgoTuner.utils.error_utils import SolverFileNotFoundError, SOLVER_NOT_FOUND_GENERIC_MSG
from AlgoTuner.utils.dace_config import configure_dace_cache


def _filesystem_operation_with_timeout(operation_func, timeout_seconds=10, operation_name="filesystem operation", error_container=None):
    """
    Execute a filesystem operation with a timeout to prevent indefinite hanging.
    
    Args:
        operation_func: Function to execute (should take no arguments)
        timeout_seconds: Maximum time to wait for operation
        operation_name: Description for logging
        
    Returns:
        True if operation succeeded, False if timed out or failed
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"{operation_name} timed out after {timeout_seconds} seconds")
    
    try:
        # For module execution, execute directly with enhanced error handling
        if operation_name == "module execution":
            import errno
            try:
                operation_func()
                return True
            except (PermissionError, OSError) as e:
                # Import error_utils to get proper context extraction
                from AlgoTuner.utils.error_utils import extract_error_context
                
                # Use standardized error message for all file operation errors
                error_msg = "Error: Reading and writing files is not allowed."
                
                # Get the code context using the standard error_utils function
                import traceback
                tb_str = traceback.format_exc()
                context_result = extract_error_context(tb_str, error_msg)
                context = context_result.get("code_context_snippet")
                
                logging.error(error_msg)
                if context:
                    logging.error(f"Code Context:\n{context}")
                
                if error_container is not None:
                    error_container['error'] = error_msg
                    if context:
                        error_container['context'] = context
                return False
            except Exception as e:
                # Import error_utils to get proper context extraction
                from AlgoTuner.utils.error_utils import extract_error_context
                import traceback
                
                tb_str = traceback.format_exc()
                error_msg = str(e)
                
                # Get the code context using the standard error_utils function
                context_result = extract_error_context(tb_str, error_msg)
                context = context_result.get("code_context_snippet")
                
                logging.warning(f"load_solver_module: {operation_name} failed: {error_msg}")
                logging.debug(f"load_solver_module: Full traceback:\n{tb_str}")
                
                if context:
                    logging.error(f"Code Context:\n{context}")
                
                if error_container is not None:
                    error_container['error'] = error_msg
                    if context:
                        error_container['context'] = context
                return False
        else:
            # Use signal-based timeout for other operations
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            # Execute the operation
            operation_func()
            
            # Clear the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
            return True
        
    except TimeoutError as e:
        logging.warning(f"load_solver_module: {e}")
        return False
    except Exception as e:
        logging.warning(f"load_solver_module: {operation_name} failed: {e}")
        return False
    finally:
        # Ensure alarm is cleared even if something goes wrong
        if operation_name != "module execution":
            try:
                signal.alarm(0)
                if 'old_handler' in locals():
                    signal.signal(signal.SIGALRM, old_handler)
            except:
                pass


def _purge_modules_from_dir(target_dir: Path) -> None:
    """Remove every entry in ``sys.modules`` whose ``__file__`` resides within
    *target_dir* (or one of its sub-directories).

    Parameters
    ----------
    target_dir : Path
        The directory whose imported modules should be evicted from the module
        cache.
    """
    try:
        target_dir = target_dir.resolve()
    except Exception:
        target_dir = Path(os.path.abspath(target_dir))

    removed = 0
    for mod_name, mod in list(sys.modules.items()):
        try:
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                continue

            mod_path = Path(mod_file).resolve()
            if str(mod_path).startswith(str(target_dir)):
                del sys.modules[mod_name]
                removed += 1
        except Exception:
            continue

    if removed:
        logging.debug(
            f"load_solver_module: Purged {removed} modules originating from '{target_dir}'."
        )



@contextmanager
def with_working_dir(target_dir):
    """Temporarily change the working directory to target_dir for the duration of the context."""
    prev_dir = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(prev_dir)

def load_solver_module(code_dir: Path, solver_filename: str = "solver.py") -> ModuleType:
    """
    Dynamically load the given solver file from the specified directory.
    Returns the imported module object.
    Raises FileNotFoundError or ImportError on failure.
    """
    logging.debug(f"load_solver_module: Starting to load {solver_filename} from {code_dir}")
    code_dir = Path(code_dir)
    solver_file = code_dir / solver_filename
    if not solver_file.is_file():
        logging.error(f"load_solver_module: Solver file not found at {solver_file}")
        raise SolverFileNotFoundError("Error: solver.py not found.")

    logging.debug(f"load_solver_module: Purging modules from {code_dir} to ensure fresh code is loaded")
    _purge_modules_from_dir(code_dir)

    import tempfile, uuid
    temp_cache_dir = Path(tempfile.gettempdir()) / f"dace_cache_{uuid.uuid4().hex[:8]}"
    
    # Create temp directory with timeout to prevent filesystem hangs
    def create_temp_dir():
        temp_cache_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir_created = _filesystem_operation_with_timeout(
        create_temp_dir, 
        timeout_seconds=10, 
        operation_name="temp cache directory creation"
    )
    
    if not temp_dir_created:
        # Fallback to system temp dir if creation failed/timed out
        temp_cache_dir = Path(tempfile.gettempdir())
        logging.warning(f"load_solver_module: Falling back to system temp dir: {temp_cache_dir}")

    logging.debug(f"load_solver_module: Configuring DaCe cache directory to temporary path {temp_cache_dir}")
    
    # Configure DaCe cache with timeout protection
    def configure_cache():
        configure_dace_cache(temp_cache_dir)
    
    cache_configured = _filesystem_operation_with_timeout(
        configure_cache,
        timeout_seconds=10,
        operation_name="DaCe cache configuration"
    )
    
    if cache_configured:
        logging.debug("load_solver_module: Completed DaCe cache configuration")
    else:
        logging.warning("load_solver_module: DaCe cache configuration failed/timed out, continuing without custom cache")

    logging.debug(f"load_solver_module: About to change working directory to {code_dir}")
    with with_working_dir(code_dir):
        logging.debug(f"load_solver_module: Changed to working directory {code_dir}")
        
        logging.debug(f"load_solver_module: About to create module spec for {solver_file}")
        spec = importlib.util.spec_from_file_location(solver_file.stem, str(solver_file))
        if spec is None or spec.loader is None:
            logging.error(f"load_solver_module: Could not create module spec for {solver_file}")
            raise ImportError(f"Could not create module spec for solver at {solver_file}")
        
        logging.debug(f"load_solver_module: Created module spec for {spec.name}")
        
        if spec and spec.name in sys.modules:
            try:
                del sys.modules[spec.name]
                logging.debug(f"load_solver_module: Removed cached module '{spec.name}' to force fresh load.")
            except Exception as mod_del_err:
                logging.warning(f"load_solver_module: Failed to delete cached module '{spec.name}': {mod_del_err}")
        
        logging.debug(f"load_solver_module: About to create module from spec")
        module = importlib.util.module_from_spec(spec)
        logging.debug(f"load_solver_module: Created module, setting __path__")
        module.__path__ = [str(code_dir)]
        
        logging.debug(f"load_solver_module: About to execute module")
        
        # Execute module with timeout to prevent hanging on import
        def execute_module():
            spec.loader.exec_module(module)
        
        error_details = {}
        module_executed = _filesystem_operation_with_timeout(
            execute_module,
            timeout_seconds=30,  # Longer timeout for module execution
            operation_name="module execution",
            error_container=error_details
        )
        
        if not module_executed:
            # Build detailed error message with context
            error_msg = error_details.get('error', f"Module execution failed for {solver_file}")
            context = error_details.get('context', '')
            
            if context:
                full_error = f"{error_msg}\n\nCode Context:\n{context}"
            else:
                full_error = error_msg
            
            raise ImportError(full_error)
        
        logging.debug(f"load_solver_module: Successfully executed module")
        
        sys.modules[spec.name] = module
        

        logging.debug(f"load_solver_module: Successfully loaded solver module from {solver_file}")
        return module

def _detect_expensive_initialization(solver_module: ModuleType) -> bool:
    """
    Detect if a solver module contains expensive initialization patterns
    that are likely to timeout during instantiation.
    
    Returns True if expensive patterns are detected.
    """
    try:
        import inspect
        import ast
        
        SolverClass = getattr(solver_module, "Solver", None)
        if SolverClass is None:
            return False
            
        try:
            source = inspect.getsource(SolverClass)
        except (OSError, TypeError):
            return False
            
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
            
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                init_source = ast.get_source_segment(source, node) or ""
                
                expensive_patterns = [
                    "mp.zetazero",
                    "mp.siegelz",
                    "mp.findroot",
                    "range(1, N_MAX",
                    "for k in range(",
                ]
                
                for pattern in expensive_patterns:
                    if pattern in init_source:
                        logging.warning(f"Detected expensive initialization pattern: '{pattern}' in Solver.__init__")
                        return True
                        
        return False
        
    except Exception as e:
        logging.debug(f"Error detecting expensive initialization: {e}")
        return False

def get_solve_callable(solver_module: ModuleType):
    """
    Given an imported solver module, find the `Solver` class, instantiate it,
    and return its `solve` method.
    
    Creates a new Solver instance each time it's called to prevent caching issues
    when evaluating multiple tasks.
    
    Raises AttributeError if no valid solver entrypoint is found.
    """
    logging.info(f"get_solve_callable: Looking for Solver class in module")
    # Find the Solver class
    SolverClass = getattr(solver_module, "Solver", None)
    if SolverClass is None:
        logging.error(f"get_solve_callable: Class 'Solver' not found in solver module")
        raise AttributeError("Class 'Solver' not found in solver module")

    logging.info(f"get_solve_callable: Found Solver class: {SolverClass}")

    # Check for expensive initialization patterns
    if _detect_expensive_initialization(solver_module):
        logging.error(f"get_solve_callable: Detected expensive initialization patterns that are likely to timeout")
        logging.error(f"get_solve_callable: Hint - Use simple approaches like mp.nzeros() instead of complex mathematical formulas in __init__")
        
        # Create error with clean message and code context using error_utils
        from AlgoTuner.utils.error_utils import create_standard_error_result
        import traceback
        
        timeout_error = TimeoutError("Solver contains expensive initialization patterns that would likely timeout. Use simpler approaches like mp.nzeros() in your implementation.")
        error_result = create_standard_error_result(
            exception=timeout_error,
            traceback_str=traceback.format_exc(),
            error_type_override="expensive_initialization_timeout",
            default_error_msg="Expensive initialization patterns detected"
        )
        
        # Log the enhanced error for debugging
        logging.error(f"Enhanced error: {error_result.get('error')}")
        if error_result.get('code_context'):
            logging.error(f"Code Context:\n{error_result['code_context']}")
        
        # Still raise the original timeout error for backward compatibility
        raise timeout_error

    # Create solver instance once and return its solve method
    # This matches the baseline behavior where task_obj.solve is pre-instantiated
    logging.info(f"get_solve_callable: Creating Solver instance with 120s timeout")
    
    # Add timeout for solver instantiation in case it takes a long time
    from AlgoTuner.utils.precise_timing import time_limit, TimeoutError
    try:
        with time_limit(120.0):  # 120 second timeout for initialization
            solver_instance = SolverClass()
    except TimeoutError as timeout_exc:
        logging.error(f"get_solve_callable: Solver instantiation timed out after 120 seconds")
        logging.error(f"get_solve_callable: Hint - Move expensive computations from __init__ to solve() method")
        
        # Create error with clean message and code context using error_utils
        from AlgoTuner.utils.error_utils import create_standard_error_result
        import traceback
        
        error_result = create_standard_error_result(
            exception=timeout_exc,
            traceback_str=traceback.format_exc(),
            error_type_override="initialization_timeout",
            default_error_msg="Solver instantiation timed out - move expensive computations to solve() method"
        )
        
        # Log the enhanced error for debugging
        logging.error(f"Enhanced error: {error_result.get('error')}")
        if error_result.get('code_context'):
            logging.error(f"Code Context:\n{error_result['code_context']}")
        
        # Re-raise with enhanced message for better user feedback
        enhanced_timeout_error = TimeoutError(error_result.get('error', str(timeout_exc)))
        # Preserve the code context in the exception for potential later use
        enhanced_timeout_error.code_context = error_result.get('code_context')
        raise enhanced_timeout_error
    
    logging.info(f"get_solve_callable: Looking for solve method on instance")
    solve_method = getattr(solver_instance, "solve", None)
    if not callable(solve_method):
        logging.error(f"get_solve_callable: Method 'solve' not found or not callable on Solver instance")
        raise AttributeError("Method 'solve' not found or not callable on Solver instance")
    
    logging.info(f"get_solve_callable: Returning pre-instantiated solve method")
    return solve_method




def get_fresh_solve_callable(solver_module: ModuleType, solver_attr: str = "Solver"):
    """
    Returns a factory function that creates fresh Solver instances for each call.
    This prevents instance-level and class-level caching between timing runs.
    
    Args:
        solver_module: Module containing the solver class
        solver_attr: Name of the solver class attribute (default: "Solver")
    
    Returns:
        Callable that creates a new Solver() and calls solve() on each invocation
    """
    logging.debug(f"get_fresh_solve_callable: Creating factory for fresh Solver instances (attr: {solver_attr})")
    
    existing_solver = getattr(solver_module, solver_attr, None)
    is_pysat_solver = (solver_attr == "Solver" and existing_solver is not None and 
                      getattr(existing_solver, "__module__", "").startswith("pysat"))
    
    if is_pysat_solver:
        logging.debug(f"get_fresh_solve_callable: Detected PySAT Solver, looking for Task subclass")
        task_class = None
        for name, obj in vars(solver_module).items():
            if not isinstance(obj, type):
                continue
            if getattr(obj, "__module__", None) != solver_module.__name__:
                continue
            if hasattr(obj, "solve") and callable(getattr(obj, "solve", None)):
                from AlgoTuneTasks.base import Task
                if obj != Task and issubclass(obj, Task):
                    task_class = obj
                    logging.debug(f"get_fresh_solve_callable: Found Task subclass: {name}")
                    break
        
        if task_class:
            def fresh_solve_wrapper(problem):
                task_instance = task_class()
                result = task_instance.solve(problem)
                del task_instance
                if not hasattr(fresh_solve_wrapper, "_call_count"):
                    fresh_solve_wrapper._call_count = 0
                fresh_solve_wrapper._call_count += 1
                if fresh_solve_wrapper._call_count % 5 == 0:
                    gc.collect()
                return result
            logging.debug(f"get_fresh_solve_callable: Returning Task-based wrapper")
            return fresh_solve_wrapper
    
    def fresh_solve_wrapper(problem):
        """Factory function that creates fresh Solver instance per call."""

        SolverClass = getattr(solver_module, solver_attr, None)
        if SolverClass is None:
            raise AttributeError(f"Class '{solver_attr}' not found in solver module after reload")


        solver_instance = SolverClass()

        if not hasattr(fresh_solve_wrapper, "_call_count"):
            fresh_solve_wrapper._call_count = 0

        result = solver_instance.solve(problem)

        fresh_solve_wrapper._call_count += 1
        if fresh_solve_wrapper._call_count % 5 == 0:
            gc.collect()

        return result
    
    logging.debug(f"get_fresh_solve_callable: Returning fresh instance factory")
    return fresh_solve_wrapper


def get_fresh_solve_callable_with_module_reload(solver_module: ModuleType):
    """
    Returns a factory function that reloads the module and creates fresh Solver instances.
    This provides maximum isolation by clearing ALL state including class-level caches.
    
    Use this for agent mode where complete cache isolation is required.
    
    Returns:
        Callable that reloads module and creates a new Solver() on each invocation
    """
    import importlib
    
    logging.info(f"get_fresh_solve_callable_with_module_reload: Creating module-reloading factory")
    
    module_name = solver_module.__name__
    module_file = solver_module.__file__
    
    SolverClass = getattr(solver_module, "Solver", None)
    if SolverClass is None:
        logging.error(f"get_fresh_solve_callable_with_module_reload: Class 'Solver' not found in solver module")
        raise AttributeError("Class 'Solver' not found in solver module")

    def fresh_solve_wrapper_with_reload(problem):
        """Factory function that reloads module and creates fresh Solver instance per call"""
        try:
            reloaded_module = importlib.reload(solver_module)
            
            ReloadedSolverClass = getattr(reloaded_module, "Solver", None)
            if ReloadedSolverClass is None:
                raise AttributeError("Class 'Solver' not found in reloaded solver module")
            

            solver_instance = ReloadedSolverClass()
            result = None
            try:
                result = solver_instance.solve(problem)
                return result
            finally:
                del solver_instance
                del result
                gc.collect()
                
        except Exception as reload_error:
            logging.warning(f"Module reload failed: {reload_error}. Falling back to regular fresh instance.")
            solver_instance = SolverClass()
            result = None
            try:
                result = solver_instance.solve(problem)
                return result
            finally:
                del solver_instance
                del result
                gc.collect()
    
    logging.info(f"get_fresh_solve_callable_with_module_reload: Returning module-reloading factory")
    return fresh_solve_wrapper_with_reload

def locate_solver_file(task_name, code_dir=None):
    """
    Pick the solver file:
      - AGENT_MODE=1 → CODE_DIR/solver.py (error if missing)
      - AGENT_MODE=0 → baseline in task folder (<task_name>.py)
    
    Parameters:
    - task_name: Name of the task (required)
    - code_dir: Directory containing code (when provided, used for agent mode)
    """
    logging.info(f"locate_solver_file: Starting to locate solver file for task '{task_name}'")
    
    if not task_name:
        logging.error(f"locate_solver_file: task_name is required")
        raise AttributeError(f"task_name is required")
    
    logging.info(f"locate_solver_file: Using task name: {task_name}")
    
    agent_mode = os.getenv("AGENT_MODE", "0")
    logging.info(f"locate_solver_file: AGENT_MODE={agent_mode}")
    
    if agent_mode == "1":
        code_dir_str = code_dir or os.getenv("CODE_DIR", "llm_src")
        logging.info(f"locate_solver_file: CODE_DIR={code_dir_str}")
        logging.info(f"locate_solver_file: Provided code_dir parameter: {code_dir}")
        logging.info(f"locate_solver_file: Environment CODE_DIR: {os.getenv('CODE_DIR')}")
        if not code_dir_str:
            logging.error(f"locate_solver_file: CODE_DIR environment variable must be set in agent mode")
            raise EnvironmentError("CODE_DIR environment variable must be set in agent mode.")
        llm_solver = Path(code_dir_str) / "solver.py"
        logging.info(f"locate_solver_file: LLM solver path: {llm_solver}")
        logging.info(f"locate_solver_file: Current working directory: {os.getcwd()}")
        if not llm_solver.is_file():
            try:
                parent_dir = llm_solver.parent
                if parent_dir.exists():
                    files_in_dir = list(parent_dir.glob("*.py"))
                    logging.error(f"locate_solver_file: Directory {parent_dir} exists but solver.py not found")
                    logging.error(f"locate_solver_file: Python files in directory: {files_in_dir}")
                else:
                    logging.error(f"locate_solver_file: Directory {parent_dir} does not exist")
            except Exception as debug_e:
                logging.error(f"locate_solver_file: Debug error: {debug_e}")
            
            logging.error(f"locate_solver_file: LLM solver file not found at {llm_solver}")
            raise SolverFileNotFoundError("Error: solver.py not found.")
        logging.info(f"locate_solver_file: Returning LLM solver path: {llm_solver}")
        return llm_solver
    
    logging.info(f"locate_solver_file: Baseline mode - need to construct task path for {task_name}")
    
    logging.warning(f"locate_solver_file: Baseline mode with new signature not yet fully implemented")
    baseline_filename = f"{task_name}.py"
    baseline = Path(".") / baseline_filename
    logging.info(f"locate_solver_file: Returning baseline path: {baseline}")
    return baseline 