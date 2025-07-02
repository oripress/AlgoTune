"""
Immutable data classes for the clean evaluation architecture.
These types ensure data flows through the pipeline without side effects.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ErrorType(Enum):
    """Standard error types in the evaluation pipeline."""
    NONE = "none"
    EXECUTION_ERROR = "execution_error"
    VALIDATION_ERROR = "validation_error"
    INVALID_SOLUTION = "invalid_solution"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    BENCHMARK_POOL_ERROR = "benchmark_pool_error"


@dataclass(frozen=True)
class TimingMetrics:
    """Immutable timing metrics for a single execution."""
    mean_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    values_ms: List[float] = field(default_factory=list)
    warmup_ms: float = 0.0
    num_runs: int = 1
    
    @property
    def total_time_ms(self) -> float:
        """Total time including warmup."""
        return self.warmup_ms + sum(self.values_ms)


@dataclass(frozen=True)
class ExecutionResult:
    """Result from executing a solver on a single problem."""
    success: bool
    output: Any  # The solver's output (will be stripped later if needed)
    timing: Optional[TimingMetrics] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    error_type: ErrorType = ErrorType.NONE
    traceback: Optional[str] = None
    timeout_occurred: bool = False
    
    @property
    def has_output(self) -> bool:
        """Check if execution produced output."""
        return self.success and self.output is not None


@dataclass(frozen=True)
class ValidationContext:
    """Context about why validation failed."""
    failure_line: Optional[int] = None
    failure_reason: Optional[str] = None
    code_snippet: Optional[str] = None
    full_context: Optional[str] = None
    
    def format_for_display(self) -> str:
        """Format context for user display."""
        if self.full_context:
            return self.full_context
        
        parts = []
        if self.failure_reason:
            parts.append(f"Reason: {self.failure_reason}")
        if self.failure_line:
            parts.append(f"Failed at line: {self.failure_line}")
        if self.code_snippet:
            parts.append(f"Code:\n{self.code_snippet}")
        
        return "\n".join(parts) if parts else "No context available"


@dataclass(frozen=True)
class ValidationResult:
    """Result from validating a solution."""
    is_valid: bool
    error_type: ErrorType = ErrorType.NONE
    error_message: Optional[str] = None
    context: Optional[ValidationContext] = None
    validation_time_ms: float = 0.0
    
    @property
    def has_context(self) -> bool:
        """Check if validation has failure context."""
        return self.context is not None and self.context.full_context is not None


@dataclass(frozen=True)
class StrippedSolution:
    """Placeholder for a stripped solution."""
    original_type: str
    original_size_bytes: Optional[int] = None
    validation_completed: bool = True
    is_valid: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "__stripped__": True,
            "type": self.original_type,
            "size_bytes": self.original_size_bytes,
            "validation_completed": self.validation_completed,
            "is_valid": self.is_valid
        }


@dataclass(frozen=True)
class ProblemResult:
    """Complete result for evaluating a single problem."""
    problem_id: str
    problem_index: int
    execution: ExecutionResult
    validation: Optional[ValidationResult] = None
    speedup: Optional[float] = None
    baseline_time_ms: Optional[float] = None
    solver_time_ms: Optional[float] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if the solution is valid."""
        return (self.execution.success and 
                self.validation is not None and 
                self.validation.is_valid)
    
    @property
    def is_success(self) -> bool:
        """Check if execution succeeded (regardless of validation)."""
        return self.execution.success
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility."""
        result = {
            "problem_id": self.problem_id,
            "success": self.execution.success,
            "is_valid": self.is_valid,
            "timeout_occurred": self.execution.timeout_occurred,
        }
        
        # Add execution data
        if self.execution.timing:
            result.update({
                "min_time_ms": self.execution.timing.min_ms,
                "mean_ms": self.execution.timing.mean_ms,
                "values_ms": self.execution.timing.values_ms,
                "elapsed_ms": self.execution.timing.total_time_ms,
            })
        
        # Add validation data
        if self.validation:
            result["validation_result"] = {
                "success": self.validation.is_valid,
                "error_type": self.validation.error_type.value,
                "error": self.validation.error_message,
            }
            if self.validation.context:
                result["code_context"] = self.validation.context.format_for_display()
        
        # Add error data
        if not self.execution.success:
            result.update({
                "error": self.execution.error,
                "error_type": self.execution.error_type.value,
                "traceback": self.execution.traceback,
            })
        
        # Add performance data
        if self.speedup is not None:
            result["speedup"] = self.speedup
        if self.baseline_time_ms is not None:
            result["baseline_time_ms"] = self.baseline_time_ms
        if self.solver_time_ms is not None:
            result["solver_min_time_ms"] = self.solver_time_ms
            
        return result


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate metrics for a dataset evaluation."""
    num_evaluated: int
    num_valid: int
    num_invalid: int
    num_errors: int
    num_timeouts: int
    success_rate: float
    validity_rate: float
    mean_speedup: Optional[float] = None
    median_speedup: Optional[float] = None
    num_inf_speedup: int = 0
    avg_solver_time_ms: Optional[float] = None
    avg_baseline_time_ms: Optional[float] = None
    
    @property
    def overall_valid(self) -> bool:
        """Check if all solutions are valid."""
        return self.num_evaluated > 0 and self.num_valid == self.num_evaluated


@dataclass(frozen=True)
class DatasetResults:
    """Results for evaluating a solver on an entire dataset."""
    task_name: str
    results: List[ProblemResult]
    metrics: AggregateMetrics
    invalid_contexts: List[str] = field(default_factory=list)
    evaluation_time_s: float = 0.0
    
    @property
    def success(self) -> bool:
        """Check if evaluation succeeded overall."""
        return self.metrics.overall_valid
    
    def get_invalid_examples(self, max_examples: int = 3) -> List[str]:
        """Get formatted invalid solution examples."""
        examples = []
        for i, context in enumerate(self.invalid_contexts[:max_examples]):
            examples.append(f"Invalid Example #{i+1}:\n{context}")
        return examples
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy AttributedList format."""
        # Convert results to legacy format
        legacy_results = [r.to_legacy_format() for r in self.results]
        
        # Create the legacy response structure
        return {
            "results": legacy_results,
            "aggregate_metrics": {
                "num_evaluated": self.metrics.num_evaluated,
                "num_valid": self.metrics.num_valid,
                "num_invalid": self.metrics.num_invalid,
                "num_errors": self.metrics.num_errors,
                "num_timeouts": self.metrics.num_timeouts,
                "overall_valid": self.metrics.overall_valid,
                "success_rate": self.metrics.success_rate,
                "mean_speedup": self.metrics.mean_speedup,
                "median_speedup": self.metrics.median_speedup,
                "num_inf_speedup": self.metrics.num_inf_speedup,
                "avg_solver_time_ms": self.metrics.avg_solver_time_ms,
                "avg_oracle_time_ms": self.metrics.avg_baseline_time_ms,
            },
            "invalid_solution_analysis": self.invalid_contexts,
            "success": self.success,
            "evaluation_type": "dataset"
        }


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for the evaluation runner."""
    num_runs: int = 10
    warmup_runs: int = 1
    timeout_multiplier: float = 10.0
    capture_output: bool = False
    validate_in_process: bool = True
    strip_solutions: bool = True
    max_solution_size_bytes: int = 10 * 1024 * 1024  # 10MB
    use_isolated_execution: bool = True
    feature_flags: Dict[str, bool] = field(default_factory=dict)