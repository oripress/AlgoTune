"""
Main orchestrator that coordinates all evaluation components.
This is the single entry point for the clean architecture.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from AlgoTuner.utils.evaluator.evaluation_types import (
    DatasetResults,
    ProblemResult,
    RunnerConfig,
    ExecutionResult,
    ErrorType
)
from AlgoTuner.utils.evaluator.solver_executor import SolverExecutor
from AlgoTuner.utils.evaluator.validation_pipeline import ValidationPipeline
from AlgoTuner.utils.evaluator.memory_optimizer import MemoryOptimizer
from AlgoTuner.utils.evaluator.result_aggregator import ResultAggregator


class EvaluationOrchestrator:
    """
    Orchestrates the entire evaluation pipeline.
    
    This is the main entry point that coordinates:
    1. Solver execution
    2. Solution validation
    3. Memory optimization
    4. Result aggregation
    """
    
    def __init__(self, config: Optional[RunnerConfig] = None):
        """
        Initialize the orchestrator with all components.
        
        Args:
            config: Configuration for the evaluation pipeline
        """
        self.config = config or RunnerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.executor = SolverExecutor(self.config)
        self.validator = ValidationPipeline()
        self.memory_optimizer = MemoryOptimizer(
            size_threshold_bytes=self.config.max_solution_size_bytes
        )
        self.aggregator = ResultAggregator()
        
        self.logger.info("Initialized evaluation orchestrator")
    
    def evaluate_dataset(
        self,
        task_instance: Any,
        dataset: List[Dict[str, Any]],
        solver_func: Callable,
        task_name: Optional[str] = None,
        baseline_func: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> DatasetResults:
        """
        Evaluate a solver on an entire dataset.
        
        This is the main entry point for dataset evaluation.
        
        Args:
            task_instance: Task instance with is_solution method
            dataset: List of problems to evaluate
            solver_func: The solver function to evaluate
            task_name: Optional name of the task
            baseline_func: Optional baseline function for speedup calculation
            progress_callback: Optional callback for progress updates
            
        Returns:
            DatasetResults with all results and metrics
        """
        start_time = time.perf_counter()
        task_name = task_name or getattr(task_instance, '__class__.__name__', 'Unknown')
        
        self.logger.info(f"Starting evaluation of {len(dataset)} problems for task {task_name}")
        
        results = []
        
        for i, problem_data in enumerate(dataset):
            # Extract problem and metadata
            problem, metadata = self._extract_problem_data(problem_data)
            problem_id = metadata.get("id", f"problem_{i+1}")
            
            # Log individual problem start
            self.logger.info(f"Evaluating problem {i+1}/{len(dataset)}: {problem_id}")
            
            # Ensure task_name is in metadata for isolated execution
            metadata["task_name"] = task_name
            
            # Evaluate single problem
            result = self.evaluate_single(
                task_instance=task_instance,
                problem=problem,
                solver_func=solver_func,
                problem_id=problem_id,
                problem_index=i,
                baseline_time_ms=metadata.get("baseline_time_ms"),
                problem_metadata=metadata
            )
            
            results.append(result)
            
            # Log individual problem completion with detailed metrics
            status = "valid" if result.is_valid else "invalid"
            speedup_str = f"speedup={result.speedup:.2f}x" if result.speedup else "speedup=N/A"
            time_str = f"time={result.execution.timing.min_ms:.2f}ms" if result.execution.timing else "time=N/A"
            error_str = f", error={result.execution.error}" if result.execution.error else ""
            
            self.logger.info(
                f"Completed problem {i+1}/{len(dataset)}: {problem_id} - {status}, "
                f"{speedup_str}, {time_str}{error_str}"
            )
            
            # Check for critical errors that should stop evaluation
            if result.execution.error_type in [
                ErrorType.MEMORY_ERROR,
                ErrorType.IMPORT_ERROR,
                ErrorType.EXECUTION_ERROR,
            ]:
                self.logger.error(f"Critical error encountered, stopping evaluation: {result.execution.error_type.value}")
                # Create a special result that indicates early exit
                return DatasetResults(
                    task_name=task_name,
                    results=results,  # Include results up to this point
                    metrics=self.aggregator._compute_metrics(results),
                    invalid_contexts=[],  # Don't collect invalid contexts on early exit
                    evaluation_time_s=time.perf_counter() - start_time
                )
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(dataset), result)
            
            # Log progress
            if (i + 1) % max(1, len(dataset) // 10) == 0:
                self.logger.info(f"Progress: {i+1}/{len(dataset)} problems evaluated")
        
        # Clear validation context cache after dataset
        self.validator.clear_context_cache()
        
        # Aggregate results
        evaluation_time_s = time.perf_counter() - start_time
        dataset_results = self.aggregator.aggregate(
            task_name=task_name,
            results=results,
            evaluation_time_s=evaluation_time_s
        )
        
        # Log summary
        self.logger.info(
            f"Completed evaluation in {evaluation_time_s:.2f}s. "
            f"Valid: {dataset_results.metrics.num_valid}/{dataset_results.metrics.num_evaluated}"
        )
        
        return dataset_results
    
    def evaluate_single(
        self,
        task_instance: Any,
        problem: Any,
        solver_func: Callable,
        problem_id: str = "problem",
        problem_index: int = 0,
        baseline_time_ms: Optional[float] = None,
        problem_metadata: Optional[Dict[str, Any]] = None
    ) -> ProblemResult:
        """
        Evaluate a solver on a single problem.
        
        This method coordinates the entire evaluation pipeline for one problem:
        1. Execute solver
        2. Validate solution (if execution succeeded)
        3. Strip solution (if needed)
        4. Calculate metrics
        
        Args:
            task_instance: Task instance with is_solution method
            problem: The problem to solve
            solver_func: The solver function
            problem_id: Identifier for this problem
            problem_index: Index of this problem in dataset
            baseline_time_ms: Baseline time for speedup calculation
            problem_metadata: Optional metadata about the problem
            
        Returns:
            ProblemResult with execution, validation, and metrics
        """
        self.logger.debug(f"Evaluating problem {problem_id}")
        
        # Step 1: Execute solver
        execution_result = self.executor.execute(
            solver_func=solver_func,
            problem=problem,
            baseline_time_ms=baseline_time_ms,
            problem_metadata=problem_metadata
        )
        
        # Step 2: Validate if execution succeeded
        validation_result = None
        if execution_result.success and execution_result.output is not None:
            self.logger.info(f"Validating solution for {problem_id}: output type={type(execution_result.output).__name__}")
            validation_result = self.validator.validate(
                task_instance=task_instance,
                problem=problem,
                solution=execution_result.output,
                capture_context=True
            )
            self.logger.info(f"Validation complete for {problem_id}: is_valid={validation_result.is_valid}")
        else:
            self.logger.info(f"Skipping validation for {problem_id}: success={execution_result.success}, has_output={execution_result.output is not None}")
        
        # Step 3: Calculate metrics
        speedup = None
        solver_time_ms = None
        
        if execution_result.timing:
            solver_time_ms = execution_result.timing.min_ms
            
            if baseline_time_ms and baseline_time_ms > 0:
                speedup = baseline_time_ms / solver_time_ms
            else:
                self.logger.info(f"No baseline time for problem {problem_id}: baseline_time_ms={baseline_time_ms}")
        
        # Step 4: Create initial result
        result = ProblemResult(
            problem_id=problem_id,
            problem_index=problem_index,
            execution=execution_result,
            validation=validation_result,
            speedup=speedup,
            baseline_time_ms=baseline_time_ms,
            solver_time_ms=solver_time_ms
        )
        
        # Step 5: Optimize memory by stripping large solutions
        if self.config.strip_solutions:
            result = self.memory_optimizer.optimize_result(result)
        
        return result
    
    def _extract_problem_data(
        self,
        problem_data: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Extract problem and metadata from dataset entry.
        
        Args:
            problem_data: Entry from dataset (could be dict or raw problem)
            
        Returns:
            Tuple of (problem, metadata)
        """
        if isinstance(problem_data, dict):
            # Extract problem and metadata from dict
            problem = problem_data.get("problem", problem_data)
            
            # Build metadata
            metadata = {
                "id": problem_data.get("id", problem_data.get("k", None)),
                "baseline_time_ms": problem_data.get("baseline_time_ms"),
                "baseline_time_us": problem_data.get("baseline_time_us"),
            }
            
            # Include any other keys as metadata
            for key, value in problem_data.items():
                if key not in ["problem", "id", "k"]:
                    metadata[key] = value
            
            return problem, metadata
        else:
            # Raw problem, no metadata
            return problem_data, {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        return {
            "memory_optimizer": self.memory_optimizer.get_statistics(),
            "config": {
                "num_runs": self.config.num_runs,
                "warmup_runs": self.config.warmup_runs,
                "strip_solutions": self.config.strip_solutions,
                "use_isolated_execution": self.config.use_isolated_execution,
            }
        }