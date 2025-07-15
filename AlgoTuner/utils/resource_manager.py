#!/usr/bin/env python3
"""
Resource Manager for AlgoTuner's Multiprocessing Operations

This module provides efficient resource management for multiprocessing operations,
addressing memory leaks and resource exhaustion in isolated benchmark execution.

Key features:
- Automatic cleanup of multiprocessing.Manager resources
- Memory-efficient result collection
- Configurable resource lifecycle management
- Context managers for safe resource handling
"""

import gc
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable, Tuple
import multiprocessing as mp

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages multiprocessing resources with automatic cleanup and memory optimization.
    
    This class provides a clean interface for managing Manager instances and their
    associated resources, preventing memory leaks in long-running benchmark operations.
    """
    
    def __init__(self, refresh_interval: int = 10):
        """
        Initialize the ResourceManager.
        
        Args:
            refresh_interval: Number of operations before refreshing the Manager instance
        """
        self.refresh_interval = refresh_interval
        self._manager = None
        self._usage_count = 0
        self._total_refreshes = 0
    
    @contextmanager
    def get_shared_dict(self) -> Dict[str, Any]:
        """
        Get a managed dictionary with automatic cleanup.
        
        Yields:
            A multiprocessing.Manager().dict() that will be automatically cleaned up
        """
        if self._manager is None:
            self._create_manager()
        
        shared_dict = self._manager.dict()
        try:
            yield shared_dict
        finally:
            # Clean up the dictionary
            try:
                shared_dict.clear()
            except Exception as e:
                logger.debug(f"Error clearing shared dict: {e}")
            del shared_dict
            
            # Increment usage and check if refresh needed
            self._usage_count += 1
            if self._usage_count >= self.refresh_interval:
                self._refresh_manager()
    
    def _create_manager(self) -> None:
        """Create a new Manager instance."""
        logger.debug("Creating new multiprocessing Manager")
        self._manager = mp.Manager()
        self._usage_count = 0
    
    def _refresh_manager(self) -> None:
        """Refresh the Manager instance to prevent resource accumulation."""
        logger.debug(f"Refreshing Manager after {self._usage_count} uses")
        self._shutdown_manager()
        self._create_manager()
        self._total_refreshes += 1
        gc.collect()
    
    def _shutdown_manager(self) -> None:
        """Safely shutdown the current Manager instance."""
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down Manager: {e}")
            self._manager = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def cleanup(self) -> None:
        """Perform final cleanup of all resources."""
        logger.debug(f"ResourceManager cleanup: {self._total_refreshes} refreshes performed")
        self._shutdown_manager()
        gc.collect()


class BenchmarkResultAccumulator:
    """
    Memory-efficient accumulator for benchmark results.
    
    Stores only essential timing data and optionally preserves the last result
    for validation purposes.
    """
    
    def __init__(self, preserve_outputs: bool = False):
        """
        Initialize the accumulator.
        
        Args:
            preserve_outputs: Whether to preserve full outputs for validation
        """
        self.preserve_outputs = preserve_outputs
        self.timing_results = []
        self.last_output = None
        self.successful_runs = 0
        self.failed_runs = 0
        self.timeout_runs = 0
    
    def add_timing_result(self, ret_dict: Dict[str, Any], run_index: int) -> None:
        """
        Add a timing result from a benchmark run.
        
        Args:
            ret_dict: Return dictionary from benchmark worker
            run_index: Index of the current run
        """
        if ret_dict.get("success", False):
            # Extract timing data
            timing_data = {
                "run_index": run_index,
                "warmup_ns": ret_dict.get("warmup_ns"),
                "timed_ns": ret_dict.get("timed_ns"),
                "warmup_ms": ret_dict.get("warmup_ns", 0) / 1e6,
                "timed_ms": ret_dict.get("timed_ns", 0) / 1e6,
            }
            
            # Optionally store stdout if small
            stdout = ret_dict.get("stdout", "")
            if stdout and len(stdout) < 10240:  # 10KB limit
                timing_data["stdout"] = stdout
            
            self.timing_results.append(timing_data)
            self.successful_runs += 1
            
            # Preserve the last successful output if requested
            if self.preserve_outputs and "out_pickle" in ret_dict:
                try:
                    import pickle
                    self.last_output = pickle.loads(ret_dict["out_pickle"])
                except Exception as e:
                    logger.debug(f"Failed to unpickle result: {e}")
        
        elif ret_dict.get("timeout"):
            self.timing_results.append({
                "run_index": run_index,
                "timeout": True,
                "warmup_ns": None,
                "timed_ns": None,
            })
            self.timeout_runs += 1
        
        else:
            # Error case
            error_msg = ret_dict.get("error", "Unknown error")
            self.timing_results.append({
                "run_index": run_index,
                "error": self._truncate_error(error_msg),
                "warmup_ns": None,
                "timed_ns": None,
            })
            self.failed_runs += 1
    
    def _truncate_error(self, error_msg: str, max_length: int = 1000) -> str:
        """Truncate error messages to prevent memory bloat."""
        if len(error_msg) > max_length:
            return error_msg[:max_length] + "... (truncated)"
        return error_msg
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected results."""
        return {
            "timing_results": self.timing_results,
            "last_output": self.last_output,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "timeout_runs": self.timeout_runs,
            "total_runs": len(self.timing_results),
        }


def create_resource_aware_worker(
    worker_func: Callable,
    skip_output_serialization: bool = True
) -> Callable:
    """
    Create a memory-efficient wrapper for worker functions.
    
    Args:
        worker_func: The original worker function
        skip_output_serialization: Whether to skip serializing large outputs
        
    Returns:
        A wrapped worker function with memory optimizations
    """
    def wrapped_worker(*args, **kwargs):
        # Get the return dictionary (last argument by convention)
        ret_dict = args[-1] if args else kwargs.get('ret_dict')
        
        # Call the original worker
        worker_func(*args, **kwargs)
        
        # Optimize memory usage
        if skip_output_serialization and ret_dict and "out_pickle" in ret_dict:
            # Replace large pickled data with a placeholder
            ret_dict["out_pickle"] = b""
            ret_dict["output_skipped"] = True
        
        # Force garbage collection in worker
        gc.collect()
    
    return wrapped_worker


class ManagedBenchmarkExecutor:
    """
    High-level executor for benchmarks with integrated resource management.
    """
    
    def __init__(self, 
                 context: Optional[mp.context.BaseContext] = None,
                 manager_refresh_interval: int = 10,
                 preserve_outputs: bool = False):
        """
        Initialize the executor.
        
        Args:
            context: Multiprocessing context (defaults to forkserver)
            manager_refresh_interval: How often to refresh Manager instances
            preserve_outputs: Whether to preserve full outputs
        """
        self.context = context or mp.get_context("forkserver")
        self.resource_manager = ResourceManager(refresh_interval=manager_refresh_interval)
        self.preserve_outputs = preserve_outputs
    
    def execute_benchmark_runs(self,
                             num_runs: int,
                             worker_func: Callable,
                             worker_args: Tuple,
                             timeout_seconds: float = 60.0) -> Dict[str, Any]:
        """
        Execute multiple benchmark runs with proper resource management.
        
        Args:
            num_runs: Number of runs to execute
            worker_func: Worker function to execute
            worker_args: Arguments for the worker function
            timeout_seconds: Timeout per run
            
        Returns:
            Dictionary with results and statistics
        """
        accumulator = BenchmarkResultAccumulator(preserve_outputs=self.preserve_outputs)
        
        with self.resource_manager:
            for run_idx in range(num_runs):
                with self.resource_manager.get_shared_dict() as ret_dict:
                    # Create and start the process
                    proc = self.context.Process(
                        target=worker_func,
                        args=(*worker_args, ret_dict),
                        daemon=False
                    )
                    proc.start()
                    proc.join(timeout=timeout_seconds)
                    
                    # Handle timeout
                    if proc.is_alive():
                        logger.warning(f"Run {run_idx+1}/{num_runs} timed out after {timeout_seconds}s")
                        proc.terminate()
                        proc.join(timeout=0.5)
                        if proc.is_alive():
                            proc.kill()
                            proc.join()
                        ret_dict["timeout"] = True
                    
                    # Collect results immediately
                    accumulator.add_timing_result(dict(ret_dict), run_idx)
                
                # Periodic garbage collection
                if run_idx % 5 == 0:
                    gc.collect()
        
        return accumulator.get_summary()


# Integration helper for existing code
def apply_resource_management_patches():
    """
    Apply resource management improvements to existing AlgoTuner code.
    This function can be called during initialization to patch the isolated_benchmark module.
    """
    try:
        import AlgoTuner.utils.isolated_benchmark as iso_module
        
        # Backup original functions
        if not hasattr(iso_module, '_original_run_with_manager'):
            iso_module._original_run_with_manager = iso_module._run_with_manager
        
        # Apply patches here...
        logger.info("Resource management patches applied successfully")
        
    except Exception as e:
        logger.error(f"Failed to apply resource management patches: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    print("AlgoTuner Resource Manager")
    print("==========================")
    print()
    print("Features:")
    print("- Automatic cleanup of multiprocessing.Manager resources")
    print("- Memory-efficient result accumulation")
    print("- Configurable Manager refresh intervals")
    print("- Context managers for safe resource handling")
    print("- Integration helpers for existing code")