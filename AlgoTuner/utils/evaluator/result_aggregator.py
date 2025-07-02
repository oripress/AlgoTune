"""
Result aggregator for computing metrics and formatting output.
This component handles all metric calculations in one place.
"""

import logging
import statistics
from typing import List, Optional, Tuple

from AlgoTuner.utils.evaluator.evaluation_types import (
    ProblemResult,
    AggregateMetrics,
    DatasetResults,
    ErrorType
)


class ResultAggregator:
    """Aggregates evaluation results and computes metrics."""
    
    def __init__(self):
        """Initialize the result aggregator."""
        self.logger = logging.getLogger(__name__)
    
    def aggregate(
        self,
        task_name: str,
        results: List[ProblemResult],
        evaluation_time_s: float = 0.0
    ) -> DatasetResults:
        """
        Aggregate problem results into dataset results with metrics.
        
        Args:
            task_name: Name of the task being evaluated
            results: List of individual problem results
            evaluation_time_s: Total evaluation time in seconds
            
        Returns:
            DatasetResults with computed metrics and formatted contexts
        """
        if not results:
            # Empty results
            return DatasetResults(
                task_name=task_name,
                results=[],
                metrics=self._empty_metrics(),
                invalid_contexts=[],
                evaluation_time_s=evaluation_time_s
            )
        
        # Compute aggregate metrics
        metrics = self._compute_metrics(results)
        
        # Extract invalid contexts
        invalid_contexts = self._extract_invalid_contexts(results)
        
        # Create dataset results
        dataset_results = DatasetResults(
            task_name=task_name,
            results=results,
            metrics=metrics,
            invalid_contexts=invalid_contexts,
            evaluation_time_s=evaluation_time_s
        )
        
        self.logger.info(
            f"Aggregated {len(results)} results: "
            f"{metrics.num_valid} valid, "
            f"{metrics.num_invalid} invalid, "
            f"{metrics.num_errors} errors, "
            f"{metrics.num_timeouts} timeouts"
        )
        
        return dataset_results
    
    def _compute_metrics(self, results: List[ProblemResult]) -> AggregateMetrics:
        """Compute aggregate metrics from problem results."""
        num_evaluated = len(results)
        
        # Count different result types
        num_valid = sum(1 for r in results if r.is_valid)
        num_errors = sum(1 for r in results if not r.is_success)
        num_timeouts = sum(1 for r in results if r.execution.timeout_occurred)
        num_invalid = sum(
            1 for r in results 
            if r.is_success and not r.is_valid
        )
        
        # Calculate rates
        success_rate = (num_evaluated - num_errors) / num_evaluated if num_evaluated > 0 else 0.0
        validity_rate = num_valid / num_evaluated if num_evaluated > 0 else 0.0
        
        # Calculate speedups
        speedups = [r.speedup for r in results if r.speedup is not None]
        finite_speedups = [s for s in speedups if s != float('inf')]
        num_inf_speedup = len([s for s in speedups if s == float('inf')])
        
        mean_speedup = None
        median_speedup = None
        
        if finite_speedups:
            mean_speedup = statistics.mean(finite_speedups)
            median_speedup = statistics.median(finite_speedups)
        elif speedups and all(s == float('inf') for s in speedups):
            # All speedups are infinite
            mean_speedup = float('inf')
            median_speedup = float('inf')
        
        # Calculate average times
        solver_times = [
            r.solver_time_ms for r in results 
            if r.solver_time_ms is not None
        ]
        baseline_times = [
            r.baseline_time_ms for r in results 
            if r.baseline_time_ms is not None
        ]
        
        avg_solver_time = statistics.mean(solver_times) if solver_times else None
        avg_baseline_time = statistics.mean(baseline_times) if baseline_times else None
        
        return AggregateMetrics(
            num_evaluated=num_evaluated,
            num_valid=num_valid,
            num_invalid=num_invalid,
            num_errors=num_errors,
            num_timeouts=num_timeouts,
            success_rate=success_rate,
            validity_rate=validity_rate,
            mean_speedup=mean_speedup,
            median_speedup=median_speedup,
            num_inf_speedup=num_inf_speedup,
            avg_solver_time_ms=avg_solver_time,
            avg_baseline_time_ms=avg_baseline_time
        )
    
    def _extract_invalid_contexts(
        self,
        results: List[ProblemResult],
        max_contexts: int = 3
    ) -> List[str]:
        """Extract formatted contexts for invalid solutions."""
        contexts = []
        
        for result in results:
            # Only collect contexts for invalid solutions
            if result.is_success and not result.is_valid:
                context_str = None
                
                # Case 1: Validation was performed and has context
                if (result.validation and result.validation.context):
                    context_str = result.validation.context.format_for_display()
                    if context_str and context_str != "No context available":
                        contexts.append(context_str)
                
                # Case 2: Validation was skipped due to missing output
                elif result.validation is None and result.execution.output is None:
                    context_str = (
                        f"Problem: {result.problem_id}\n"
                        f"Issue: Solver returned None (no output)\n"
                        f"This usually means:\n"
                        f"  - The solver function didn't return anything\n"
                        f"  - There's a missing 'return' statement\n"
                        f"  - The solver encountered an unhandled error"
                    )
                    contexts.append(context_str)
                
                # Case 3: Other invalid cases (validation failed for other reasons)
                elif result.validation and not result.validation.is_valid:
                    # Use error message if available
                    error_msg = result.validation.error_message or "Solution validation failed"
                    context_str = (
                        f"Problem: {result.problem_id}\n"
                        f"Issue: {error_msg}"
                    )
                    contexts.append(context_str)
                    
                if len(contexts) >= max_contexts:
                    break
        
        return contexts
    
    def _empty_metrics(self) -> AggregateMetrics:
        """Create empty metrics for no results."""
        return AggregateMetrics(
            num_evaluated=0,
            num_valid=0,
            num_invalid=0,
            num_errors=0,
            num_timeouts=0,
            success_rate=0.0,
            validity_rate=0.0,
            mean_speedup=None,
            median_speedup=None,
            num_inf_speedup=0,
            avg_solver_time_ms=None,
            avg_baseline_time_ms=None
        )
    
    def format_summary(self, dataset_results: DatasetResults) -> str:
        """
        Format dataset results as a human-readable summary.
        
        Args:
            dataset_results: The results to format
            
        Returns:
            Formatted string summary
        """
        metrics = dataset_results.metrics
        
        lines = []
        
        # Header
        lines.append(f"Task: {dataset_results.task_name}")
        lines.append(f"Evaluation Time: {dataset_results.evaluation_time_s:.2f}s")
        lines.append("")
        
        # Results summary
        lines.append(f"Results: {metrics.num_evaluated} problems evaluated")
        lines.append(f"  Valid: {metrics.num_valid} ({metrics.validity_rate*100:.1f}%)")
        lines.append(f"  Invalid: {metrics.num_invalid}")
        lines.append(f"  Errors: {metrics.num_errors}")
        lines.append(f"  Timeouts: {metrics.num_timeouts}")
        lines.append("")
        
        # Performance summary
        if metrics.mean_speedup is not None:
            if metrics.mean_speedup == float('inf'):
                lines.append("Speedup: ∞ (baseline time ~0)")
            else:
                lines.append(f"Speedup: {metrics.mean_speedup:.2f}x mean, "
                           f"{metrics.median_speedup:.2f}x median")
                if metrics.num_inf_speedup > 0:
                    lines.append(f"  ({metrics.num_inf_speedup} problems with ∞ speedup)")
        else:
            lines.append("Speedup: N/A")
        
        # Timing summary
        if metrics.avg_solver_time_ms is not None:
            lines.append(f"Average solver time: {metrics.avg_solver_time_ms:.2f}ms")
        if metrics.avg_baseline_time_ms is not None:
            lines.append(f"Average baseline time: {metrics.avg_baseline_time_ms:.2f}ms")
        
        # Invalid examples
        if dataset_results.invalid_contexts:
            lines.append("")
            lines.append("Invalid Solution Examples:")
            for i, context in enumerate(dataset_results.get_invalid_examples(), 1):
                lines.append("")
                lines.append(context)
        
        return "\n".join(lines)