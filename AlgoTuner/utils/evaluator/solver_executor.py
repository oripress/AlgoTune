"""
Solver executor that runs solver code in isolation.
This component is responsible ONLY for execution, not validation.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

from AlgoTuner.utils.evaluator.evaluation_types import (
    ExecutionResult, 
    TimingMetrics, 
    ErrorType,
    RunnerConfig
)
from AlgoTuner.utils.evaluator.runner import (
    run_solver_evaluation,
    _calculate_timeout_seconds
)
# categorize_error is not available, will implement inline


class SolverExecutor:
    """Executes solver code in isolation without validation."""
    
    def __init__(self, config: RunnerConfig):
        """
        Initialize the solver executor.
        
        Args:
            config: Configuration for execution behavior
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def execute(
        self,
        solver_func: Any,
        problem: Any,
        baseline_time_ms: Optional[float] = None,
        problem_metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute solver on a single problem.
        
        Args:
            solver_func: The solver function/method to execute
            problem: The problem instance to solve
            baseline_time_ms: Baseline time for timeout calculation
            problem_metadata: Optional metadata about the problem
            
        Returns:
            ExecutionResult with output, timing, and error info
        """
        start_time = time.perf_counter()
        
        try:
            # Use isolated execution when not in daemon process
            import multiprocessing as mp
            is_daemon = mp.current_process().daemon
            self.logger.info(f"SolverExecutor.execute: is_daemon={is_daemon}, use_isolated={self.config.use_isolated_execution}")
            
            if not is_daemon and self.config.use_isolated_execution:
                result = self._execute_isolated(
                    solver_func, problem, baseline_time_ms, problem_metadata
                )
            else:
                # Fallback to in-process execution
                self.logger.info("Using in-process execution (daemon or isolated disabled)")
                result = self._execute_in_process(
                    solver_func, problem, baseline_time_ms
                )
            
            elapsed_s = time.perf_counter() - start_time
            self.logger.info(f"Solver execution completed in {elapsed_s:.2f}s, success={result.success}, has_timing={result.timing is not None}")
            
            # If no timing info, add elapsed time
            if result.success and not result.timing:
                self.logger.warning(f"No timing info from execution, adding elapsed time: {elapsed_s*1000:.2f}ms")
                result = result._replace(
                    timing=TimingMetrics(
                        mean_ms=elapsed_s * 1000,
                        min_ms=elapsed_s * 1000,
                        max_ms=elapsed_s * 1000,
                        stddev_ms=0.0,
                        values_ms=[elapsed_s * 1000],
                        warmup_ms=0.0,
                        num_runs=1
                    )
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error in solver execution: {e}")
            return self._create_error_result(e, time.perf_counter() - start_time)
    
    def _execute_isolated(
        self,
        solver_func: Any,
        problem: Any,
        baseline_time_ms: Optional[float],
        problem_metadata: Optional[Dict[str, Any]]
    ) -> ExecutionResult:
        """Execute solver in isolated subprocess."""
        import os
        
        # Check if we're in AGENT_MODE - isolated execution only works with solver loaded from disk
        agent_mode = os.environ.get("AGENT_MODE", "0")
        self.logger.info(f"SolverExecutor: AGENT_MODE={agent_mode}, use_isolated={self.config.use_isolated_execution}")
        if agent_mode == "0":
            # In baseline mode, we can't use isolated execution since solver is a method
            self.logger.info("Baseline mode detected, using in-process execution")
            return self._execute_in_process(solver_func, problem, baseline_time_ms)
        
        from AlgoTuner.utils.isolated_benchmark import run_isolated_benchmark
        
        # Get task name from metadata or environment
        task_name = "unknown_task"
        if problem_metadata and "task_name" in problem_metadata:
            task_name = problem_metadata["task_name"]
        elif os.environ.get("CURRENT_TASK_NAME"):
            task_name = os.environ["CURRENT_TASK_NAME"]
        
        # Get code directory
        code_dir = os.environ.get("CODE_DIR", "llm_src")
        
        # Calculate timeout
        timeout_seconds = 60.0  # Default
        if baseline_time_ms:
            per_run_s = baseline_time_ms / 1000.0
            timeout_seconds = (1 + self.config.warmup_runs) * per_run_s * 10.0  # warmup + timed + buffer
            timeout_seconds = min(timeout_seconds, 300.0)  # Cap at 5 minutes
            # Ensure we never set an unrealistically low timeout (e.g. sub-second) which can
            # be hit simply by solver startup costs or validation overhead.  Use a sensible
            # lower bound so that very fast baseline times do not convert into premature
            # time-outs that hide the real error.
            timeout_seconds = max(timeout_seconds, 10.0)  # At least 10 seconds overall
        
        try:
            # Use isolated benchmark with forkserver
            benchmark_result = run_isolated_benchmark(
                task_name=task_name,
                code_dir=code_dir,
                warmup_problem=problem,
                timed_problem=problem,
                num_runs=self.config.num_runs,
                timeout_seconds=timeout_seconds,
            )
            
            # Log benchmark result for debugging
            self.logger.info(f"Isolated benchmark result keys: {list(benchmark_result.keys())}")
            if benchmark_result.get("success"):
                timing_fields = ["mean", "min", "max", "mean_ms", "min_ms", "max_ms", "elapsed_ms"]
                timing_info = {k: benchmark_result.get(k) for k in timing_fields if k in benchmark_result}
                self.logger.info(f"Timing fields in result: {timing_info}")
            
            # Check if we already have the result from the benchmark
            if benchmark_result.get("success") and benchmark_result.get("result") is None:
                # This should not happen with the updated isolated benchmark
                self.logger.warning("Benchmark succeeded but no result was returned - validation will be skipped")
            
            return self._convert_benchmark_result(benchmark_result)
            
        except Exception as e:
            self.logger.error(f"Error in isolated execution: {e}")
            # Fall back to in-process execution
            return self._execute_in_process(solver_func, problem, baseline_time_ms)
    
    def _execute_in_process(
        self,
        solver_func: Any,
        problem: Any,
        baseline_time_ms: Optional[float]
    ) -> ExecutionResult:
        """Execute solver in current process (for testing/debugging)."""
        import numpy as np
        from AlgoTuner.utils.precise_timing import time_function_precise
        
        try:
            # Warmup runs
            for _ in range(self.config.warmup_runs):
                _ = solver_func(problem)
            
            # Timed runs
            times_ms = []
            outputs = []
            
            for _ in range(self.config.num_runs):
                start_ns = time.perf_counter_ns()
                output = solver_func(problem)
                elapsed_ns = time.perf_counter_ns() - start_ns
                
                times_ms.append(elapsed_ns / 1e6)
                outputs.append(output)
            
            # Use the last output
            final_output = outputs[-1]
            
            # Calculate timing metrics
            timing = TimingMetrics(
                mean_ms=float(np.mean(times_ms)),
                min_ms=float(np.min(times_ms)),
                max_ms=float(np.max(times_ms)),
                stddev_ms=float(np.std(times_ms)),
                values_ms=times_ms,
                warmup_ms=0.0,  # Not tracked in simple mode
                num_runs=self.config.num_runs
            )
            
            return ExecutionResult(
                success=True,
                output=final_output,
                timing=timing,
                stdout="",
                stderr="",
                error=None,
                error_type=ErrorType.NONE,
                traceback=None,
                timeout_occurred=False
            )
            
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            error_type = self._categorize_error(e, tb_str)
            
            return ExecutionResult(
                success=False,
                output=None,
                timing=None,
                stdout="",
                stderr="",
                error=str(e),
                error_type=error_type,
                traceback=tb_str,
                timeout_occurred=False
            )
    
    def _convert_benchmark_result(self, benchmark_result: Dict[str, Any]) -> ExecutionResult:
        """Convert legacy benchmark result to ExecutionResult."""
        success = benchmark_result.get("success", False)
        
        # Extract timing if successful
        timing = None
        if success:
            # Try both seconds and milliseconds fields
            if "mean" in benchmark_result or "mean_ms" in benchmark_result:
                # Handle both seconds and milliseconds fields
                mean_ms = benchmark_result.get("mean_ms") or (benchmark_result.get("mean", 0.0) * 1000)
                min_ms = benchmark_result.get("min_ms") or (benchmark_result.get("min", 0.0) * 1000)
                max_ms = benchmark_result.get("max_ms") or (benchmark_result.get("max", 0.0) * 1000)
                stddev_ms = benchmark_result.get("stddev_ms") or (benchmark_result.get("stddev", 0.0) * 1000)
                
                # Get values in milliseconds
                values_ms = benchmark_result.get("values_ms", [])
                if not values_ms and "values" in benchmark_result:
                    # Convert seconds to milliseconds
                    values_ms = [v * 1000 for v in benchmark_result.get("values", [])]
                
                timing = TimingMetrics(
                    mean_ms=mean_ms,
                    min_ms=min_ms,
                    max_ms=max_ms,
                    stddev_ms=stddev_ms,
                    values_ms=values_ms,
                    warmup_ms=benchmark_result.get("first_warmup_result", {}).get("elapsed_ms", 0.0),
                    num_runs=benchmark_result.get("num_runs_executed", self.config.num_runs)
                )
            elif "elapsed_ms" in benchmark_result:
                # Single run result
                elapsed_ms = benchmark_result.get("elapsed_ms", 0.0)
                timing = TimingMetrics(
                    mean_ms=elapsed_ms,
                    min_ms=elapsed_ms,
                    max_ms=elapsed_ms,
                    stddev_ms=0.0,
                    values_ms=[elapsed_ms],
                    warmup_ms=0.0,
                    num_runs=1
                )
        
        # Determine error type
        error_type = ErrorType.NONE
        if not success:
            if benchmark_result.get("timeout_occurred", False):
                # Treat timeouts as non-critical errors so that the orchestrator does not abort
                # the entire evaluation. We keep success=False but set the error_type to NONE so
                # that it will be counted as an invalid/timeout sample instead of triggering the
                # critical-error early-exit logic.
                error_type = ErrorType.NONE
            elif benchmark_result.get("error_type") == "import_error":
                error_type = ErrorType.IMPORT_ERROR
            elif benchmark_result.get("error_type") == "memory_error":
                error_type = ErrorType.MEMORY_ERROR
            else:
                error_type = ErrorType.EXECUTION_ERROR
        
        return ExecutionResult(
            success=success,
            output=benchmark_result.get("result"),
            timing=timing,
            stdout=benchmark_result.get("stdout", ""),
            stderr=benchmark_result.get("stderr", ""),
            error=benchmark_result.get("error"),
            error_type=error_type,
            traceback=benchmark_result.get("traceback"),
            timeout_occurred=benchmark_result.get("timeout_occurred", False)
        )
    
    def _create_error_result(self, exception: Exception, elapsed_s: float) -> ExecutionResult:
        """Create an ExecutionResult for an unexpected error."""
        import traceback
        tb_str = traceback.format_exc()
        
        return ExecutionResult(
            success=False,
            output=None,
            timing=None,
            stdout="",
            stderr="",
            error=str(exception),
            error_type=self._categorize_error(exception, tb_str),
            traceback=tb_str,
            timeout_occurred=False
        )
    
    def _categorize_error(self, exception: Exception, traceback_str: str) -> ErrorType:
        """Categorize an exception into an ErrorType."""
        # Simple error categorization based on exception type
        if isinstance(exception, TimeoutError):
            return ErrorType.TIMEOUT
        elif isinstance(exception, ImportError):
            return ErrorType.IMPORT_ERROR
        elif isinstance(exception, MemoryError):
            return ErrorType.MEMORY_ERROR
        elif isinstance(exception, TypeError):
            return ErrorType.TYPE_ERROR
        elif "validation" in str(exception).lower():
            return ErrorType.VALIDATION_ERROR
        else:
            return ErrorType.EXECUTION_ERROR