"""
Memory optimizer for managing solution stripping.
This ensures solutions are stripped consistently in one place.
"""

import logging
import sys
from typing import Any, Dict, Optional, Tuple

from AlgoTuner.utils.evaluator.evaluation_types import (
    ExecutionResult,
    ProblemResult,
    StrippedSolution,
    ValidationResult
)


class MemoryOptimizer:
    """Manages memory by stripping large solutions after validation."""
    
    # Standard marker for stripped solutions
    STRIPPED_MARKER = "__stripped__"
    
    def __init__(self, size_threshold_bytes: int = 10 * 1024 * 1024):
        """
        Initialize the memory optimizer.
        
        Args:
            size_threshold_bytes: Solutions larger than this are stripped (default 10MB)
        """
        self.size_threshold_bytes = size_threshold_bytes
        self.logger = logging.getLogger(__name__)
        self._strip_count = 0
        self._total_bytes_saved = 0
    
    def optimize_result(self, result: ProblemResult) -> ProblemResult:
        """
        Optimize a problem result by stripping large solutions.
        
        Args:
            result: The problem result to optimize
            
        Returns:
            New ProblemResult with stripped solution if needed
        """
        # Only strip if execution succeeded and we have output
        if not result.execution.success or result.execution.output is None:
            return result
        
        # Check if solution should be stripped
        solution = result.execution.output
        if not self.should_strip(solution):
            return result
        
        # Strip the solution
        stripped = self.strip_solution(
            solution,
            is_valid=result.is_valid if result.validation else None
        )
        
        # Create new execution result with stripped solution
        new_execution = ExecutionResult(
            success=result.execution.success,
            output=stripped.to_dict(),
            timing=result.execution.timing,
            stdout=result.execution.stdout,
            stderr=result.execution.stderr,
            error=result.execution.error,
            error_type=result.execution.error_type,
            traceback=result.execution.traceback,
            timeout_occurred=result.execution.timeout_occurred
        )
        
        # Return new result with stripped solution
        return ProblemResult(
            problem_id=result.problem_id,
            problem_index=result.problem_index,
            execution=new_execution,
            validation=result.validation,
            speedup=result.speedup,
            baseline_time_ms=result.baseline_time_ms,
            solver_time_ms=result.solver_time_ms
        )
    
    def should_strip(self, solution: Any) -> bool:
        """
        Determine if a solution should be stripped.
        
        Args:
            solution: The solution to check
            
        Returns:
            True if solution should be stripped
        """
        # Already stripped
        if self._is_stripped(solution):
            return False
        
        # Estimate size
        size_bytes = self._estimate_size(solution)
        
        # Strip if over threshold
        should_strip = size_bytes > self.size_threshold_bytes
        
        if should_strip:
            self.logger.debug(
                f"Solution size {size_bytes:,} bytes exceeds threshold "
                f"{self.size_threshold_bytes:,} bytes, will strip"
            )
        
        return should_strip
    
    def strip_solution(
        self,
        solution: Any,
        is_valid: Optional[bool] = None
    ) -> StrippedSolution:
        """
        Strip a solution to minimal representation.
        
        Args:
            solution: The solution to strip
            is_valid: Optional validation status
            
        Returns:
            StrippedSolution object
        """
        # Get size before stripping
        size_bytes = self._estimate_size(solution)
        
        # Create stripped representation
        stripped = StrippedSolution(
            original_type=type(solution).__name__,
            original_size_bytes=size_bytes,
            validation_completed=is_valid is not None,
            is_valid=is_valid
        )
        
        # Update statistics
        self._strip_count += 1
        self._total_bytes_saved += size_bytes
        
        self.logger.debug(
            f"Stripped {type(solution).__name__} solution of size {size_bytes:,} bytes. "
            f"Total stripped: {self._strip_count}, saved: {self._total_bytes_saved:,} bytes"
        )
        
        return stripped
    
    def _is_stripped(self, solution: Any) -> bool:
        """Check if a solution is already stripped."""
        if isinstance(solution, dict):
            return (self.STRIPPED_MARKER in solution or
                    solution.get("stripped_after_validation", False))
        return False
    
    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate the memory size of an object in bytes.
        
        This is an approximation that handles common data types.
        """
        try:
            # For simple types, use sys.getsizeof
            if isinstance(obj, (int, float, str, bytes, bool, type(None))):
                return sys.getsizeof(obj)
            
            # For numpy arrays, use nbytes
            if hasattr(obj, 'nbytes'):
                return obj.nbytes
            
            # For lists/tuples, sum the sizes
            if isinstance(obj, (list, tuple)):
                size = sys.getsizeof(obj)
                for item in obj:
                    size += self._estimate_size(item)
                return size
            
            # For dicts, sum keys and values
            if isinstance(obj, dict):
                size = sys.getsizeof(obj)
                for key, value in obj.items():
                    size += self._estimate_size(key)
                    size += self._estimate_size(value)
                return size
            
            # For sets
            if isinstance(obj, set):
                size = sys.getsizeof(obj)
                for item in obj:
                    size += self._estimate_size(item)
                return size
            
            # For custom objects, try to estimate based on __dict__
            if hasattr(obj, '__dict__'):
                return self._estimate_size(obj.__dict__)
            
            # Default fallback
            return sys.getsizeof(obj)
            
        except Exception as e:
            self.logger.warning(f"Error estimating size: {e}")
            # Return a default size
            return 1024  # 1KB default
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        return {
            "solutions_stripped": self._strip_count,
            "total_bytes_saved": self._total_bytes_saved,
            "threshold_bytes": self.size_threshold_bytes,
            "average_bytes_per_strip": (
                self._total_bytes_saved / self._strip_count 
                if self._strip_count > 0 else 0
            )
        }
    
    def reset_statistics(self):
        """Reset the statistics counters."""
        self._strip_count = 0
        self._total_bytes_saved = 0
        self.logger.debug("Reset memory optimization statistics")