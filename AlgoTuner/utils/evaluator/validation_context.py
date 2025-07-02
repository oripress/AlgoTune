"""
Centralized validation context management to reduce code duplication.
"""

import logging
from typing import Any, Dict, Optional, Tuple

class ValidationContextManager:
    """Manages validation failure context in a centralized way."""
    
    def __init__(self):
        self._contexts: Dict[Tuple[Any, Any], str] = {}
    
    def capture_context(self, task_instance: Any, problem: Any, solution: Any) -> Optional[str]:
        """
        Capture validation failure context for a given task/problem/solution.
        Returns the captured context or None if already captured.
        """
        # Check if solution is stripped
        if self._is_stripped(solution):
            # Return existing context if available
            context = getattr(task_instance, '_last_is_solution_failure_context', None)
            if context:
                return context
            return "Solution was stripped before context could be captured."
        
        # Check if we already have context for this task instance
        if hasattr(task_instance, '_last_is_solution_failure_context'):
            return task_instance._last_is_solution_failure_context
        
        # Capture new context
        from AlgoTuner.utils.evaluator.failure_analyzer import trace_is_solution_failure
        trace_is_solution_failure(task_instance, problem, solution)
        return getattr(task_instance, '_last_is_solution_failure_context', None)
    
    def _is_stripped(self, solution: Any) -> bool:
        """Check if a solution has been stripped."""
        if solution is None:
            return True
        if isinstance(solution, dict):
            return solution.get("__stripped__", False) or solution.get("stripped_after_validation", False)
        return False
    
    def store_context(self, key: Tuple[Any, Any], context: str) -> None:
        """Store context for later retrieval."""
        self._contexts[key] = context
    
    def get_context(self, key: Tuple[Any, Any]) -> Optional[str]:
        """Retrieve stored context."""
        return self._contexts.get(key)
    
    def clear(self) -> None:
        """Clear all stored contexts."""
        self._contexts.clear()