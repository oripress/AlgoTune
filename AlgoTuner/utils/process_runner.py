import os
import sys
import logging
import time
import atexit
import io
import traceback
import multiprocessing
import pickle
import importlib
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor

# Import the new error utility
from AlgoTuner.utils.error_utils import create_standard_error_result

# Track which modules we've already reloaded in this process
_reloaded_modules = set()
_process_initialized = False

def _initialize_process():
    """Initialize the process with common imports and setup."""
    global _process_initialized
    if _process_initialized:
        return
        
    try:
        # Import common modules that might be needed
        import numpy
        import math
        import gc
        import sys
        import os
        
        # Initialize numpy if available
        try:
            numpy.random.seed()  # Initialize RNG state
        except Exception as e:
            logging.debug(f"Non-critical error initializing numpy: {e}")
        
        # Force garbage collection to start clean
        gc.collect()
        
        _process_initialized = True
        logging.info(f"Process {os.getpid()} initialized successfully")
    except ImportError as e:
        logging.warning(f"Non-critical import error during process initialization: {e}")
        _process_initialized = True  # Still mark as initialized
    except Exception as e:
        logging.error(f"Error initializing process {os.getpid()}: {e}")
        logging.error(traceback.format_exc())
        raise  # Re-raise to ensure the error is caught by the process pool

class EvaluatorProcessPoolExecutor(ProcessPoolExecutor):
    """Custom process pool that initializes processes on creation."""
    def __init__(self, *args, **kwargs):
        # Use forkserver context for better isolation
        ctx = multiprocessing.get_context('forkserver')
        kwargs['initializer'] = _initialize_process
        kwargs['mp_context'] = ctx
        super().__init__(*args, **kwargs)

def _ensure_module_loaded(func):
    """Ensure module is loaded and reloaded once per process."""
    if not hasattr(func, "__module__") or func.__module__ == "__main__":
        return func
        
    module_name = func.__module__
    if module_name not in _reloaded_modules:
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)
            _reloaded_modules.add(module_name)
            if hasattr(module, func.__name__):
                func = getattr(module, func.__name__)
        except Exception as e:
            logging.error(f"Error reloading module {module_name}: {str(e)}")
    return func

# Ensure the file doesn't end abruptly if _run_wrapper was the last thing
# (Add a newline or keep existing code if any follows)
