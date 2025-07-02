"""
Process pool management for the evaluator module.
"""

import logging
import time
import traceback
import multiprocessing
import atexit
import concurrent.futures
from concurrent.futures import TimeoutError as FuturesTimeoutError


# Simple wrapper for execution without timing, suitable for warmup
def _execute_simple(func, args, kwargs):
    """Executes a function and returns success status and result/error."""
    try:
        result = func(*args, **kwargs)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        # Keep error simple for warmup checks
        return {"success": False, "result": None, "error": str(e)}


class ProcessPoolManager:
    """
    Manages a process pool for evaluation tasks.
    
    This class provides a singleton pattern for accessing a shared process pool,
    ensuring resources are properly managed and cleaned up.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProcessPoolManager, cls).__new__(cls)
            cls._instance._pool = None
            cls._instance._configured_pool_size = None # Initialize config variable
            atexit.register(cls._instance.cleanup)
        return cls._instance
    
    def configure(self, pool_size: int):
        """Configure the pool size before the pool is created."""
        if self._pool is not None:
             logging.warning("ProcessPoolManager: Pool already created. configure() call ignored.")
             return
        if pool_size is not None and pool_size > 0:
             self._configured_pool_size = pool_size
             logging.info(f"ProcessPoolManager configured with pool_size={pool_size}")
        else:
             logging.warning(f"ProcessPoolManager: Invalid pool_size {pool_size} passed to configure(). Using default.")
             self._configured_pool_size = None # Reset to use default
    
    def get_pool(self):
        """
        Get or create the process pool.
        
        Returns:
            ProcessPoolExecutor instance
        """
        if self._pool is None:
            from AlgoTuner.utils.process_runner import EvaluatorProcessPoolExecutor
            # Determine the number of workers based on configuration or default
            max_workers = self._configured_pool_size if self._configured_pool_size else multiprocessing.cpu_count()
            logging.info(f"ProcessPoolManager: Creating pool with max_workers={max_workers}")
            self._pool = EvaluatorProcessPoolExecutor(max_workers=max_workers)
        return self._pool
    
    def cleanup(self):
        """Clean up the process pool."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
            
            # Clean up any remaining shared memory resources from loky/joblib
            try:
                import multiprocessing.resource_tracker
                import warnings
                # Suppress any resource tracker cleanup warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Force cleanup of shared memory resources
                    import gc
                    gc.collect()
            except Exception as e:
                logging.debug(f"ProcessPoolManager cleanup: Could not clean shared memory resources: {e}")
    
    def reset_pool(self):
        """
        Reset the process pool if it's in a broken state.
        
        Returns:
            Fresh ProcessPoolExecutor instance
        """
        self.cleanup()
        return self.get_pool()


# Define dummy_solve outside of warmup_evaluator to make it picklable
def _dummy_solve(x):
    """
    Dummy function for warming up the process pool.
    Does some actual computation to ensure JIT is triggered.
    
    Args:
        x: Input value, will be returned unchanged
        
    Returns:
        The input unchanged
    """
    try:
        # Do some actual computation to warm up the JIT
        result = 0
        for i in range(1000):
            result += i
        
        # Also do some numpy operations to warm up numpy
        try:
            import numpy as np
            arr = np.ones((100, 100))
            result += float(np.sum(arr))  # Convert to float to avoid numpy type issues
        except ImportError:
            pass  # numpy is optional
        except Exception as e:
            logging.debug(f"Non-critical numpy error in warmup: {e}")
            
        # Return the input unchanged to verify correct execution
        return x
    except Exception as e:
        logging.error(f"Error in _dummy_solve: {str(e)}")
        raise  # Re-raise to ensure the error is caught and logged by the wrapper


def warmup_evaluator():
    """
    Warm up the evaluation system to avoid first-evaluation overhead.
    This:
    1. Creates the process pool
    2. Runs multiple dummy evaluations to trigger module imports
    3. Warms up the Python JIT
    4. Ensures all processes in the pool are initialized
    5. Performs additional JIT warmup to ensure consistent timing
    """
    logging.info("Warming up evaluator...")
    
    # Initialize process pool
    pool_manager = ProcessPoolManager()
    pool = pool_manager.get_pool()
    
    # Get the number of processes in the pool
    num_processes = pool._max_workers
    logging.info(f"Warming up {num_processes} processes in the pool")
    
    # Track successful warmups
    successful_warmups = 0
    required_warmups = num_processes * 2  # We want multiple successful warmups per process for better JIT optimization
    max_attempts = required_warmups * 3  # Allow for some failures
    
    # Run multiple dummy evaluations to trigger imports and JIT
    try:
        attempt = 0
        while successful_warmups < required_warmups and attempt < max_attempts:
            try:
                # Submit the simple execution wrapper
                future = pool.submit(
                    _execute_simple, # Use the new simple executor
                    _dummy_solve,
                    ([attempt],),  # Different input for each call
                    {}
                    # No need for capture_output flag
                )
                
                # Wait for the result with timeout
                result = future.result(timeout=5.0)  # Increased timeout for slower systems
                
                # Verify the result is correct based on _execute_simple's return format
                if result.get("success") and result.get("result") == attempt:
                    successful_warmups += 1
                    logging.info(f"Successful warmup {successful_warmups}/{required_warmups}")
                else:
                    logging.warning(f"Warmup returned unexpected result or error: {result}")
            except Exception as e:
                logging.error(f"Warmup attempt {attempt + 1} failed: {str(e)}")
            
            attempt += 1
        
        if successful_warmups >= required_warmups:
            logging.info(f"Evaluator warmup complete with {successful_warmups} successful warmups")
        else:
            logging.warning(f"Evaluator warmup partially complete with only {successful_warmups}/{required_warmups} successful warmups")
            
    except Exception as e:
        logging.error(f"Warmup failed: {str(e)}")
        logging.error(traceback.format_exc())
        # Don't raise the error - allow the system to continue with partial warmup 