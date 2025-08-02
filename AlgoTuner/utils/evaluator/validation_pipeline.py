"""
Single point of validation for all solutions.
This ensures validation happens exactly once and context is properly captured.
"""

import logging
import time
from typing import Any, Optional

from AlgoTuner.utils.evaluator.evaluation_types import (
    ValidationResult,
    ValidationContext,
    ErrorType
)
from AlgoTuner.utils.evaluator.failure_analyzer import trace_is_solution_failure


class ValidationPipeline:
    """Handles all solution validation in a single, consistent way."""
    
    def __init__(self):
        """Initialize the validation pipeline."""
        self.logger = logging.getLogger(__name__)
        self._context_cache = {}  # Cache contexts by task instance id
    
    def validate(
        self,
        task_instance: Any,
        problem: Any,
        solution: Any,
        capture_context: bool = True
    ) -> ValidationResult:
        """
        Validate a solution against a problem.
        
        This is the ONLY place where validation should happen in the new architecture.
        
        Args:
            task_instance: The task instance with is_solution method
            problem: The problem to validate against
            solution: The solution to validate
            capture_context: Whether to capture failure context
            
        Returns:
            ValidationResult with validation status and optional context
        """
        start_time = time.perf_counter()
        
        # Check if solution is already stripped
        if self._is_stripped_solution(solution):
            self.logger.warning("Attempting to validate a stripped solution")
            return ValidationResult(
                is_valid=False,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message="Cannot validate stripped solution",
                context=None,
                validation_time_ms=0.0
            )
        
        try:
            # Call is_solution method
            self.logger.info(f"Calling task.is_solution for problem type {type(problem).__name__}, solution type {type(solution).__name__}")
            
            # Check if solution is None
            if solution is None:
                self.logger.error("Solution is None! This should not happen!")
                return ValidationResult(
                    is_valid=False,
                    error_type=ErrorType.VALIDATION_ERROR,
                    error_message="Solution is None",
                    context=None,
                    validation_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            # Log solution structure
            if isinstance(solution, tuple) and len(solution) == 2:
                self.logger.info(f"Solution structure: tuple with {len(solution[0])} eigenvalues and {len(solution[1])} eigenvectors")
            else:
                self.logger.info(f"Solution structure: {type(solution)}, length: {len(solution) if hasattr(solution, '__len__') else 'N/A'}")
            
            is_valid = task_instance.is_solution(problem, solution)
            self.logger.info(f"Validation result: {is_valid}")
            
            # Convert to boolean explicitly
            is_valid = bool(is_valid)
            
            validation_time_ms = (time.perf_counter() - start_time) * 1000
            
            if is_valid:
                self.logger.debug("Solution is valid")
                return ValidationResult(
                    is_valid=True,
                    error_type=ErrorType.NONE,
                    error_message=None,
                    context=None,
                    validation_time_ms=validation_time_ms
                )
            else:
                self.logger.debug("Solution is invalid, capturing context")
                # Solution is invalid - capture context immediately before any stripping
                context = None
                if capture_context:
                    # Capture failure context immediately while solution is still intact
                    try:
                        self.logger.info("Capturing is_solution failure context immediately")
                        trace_is_solution_failure(task_instance, problem, solution)
                    except Exception as e:
                        self.logger.warning(f"Failed to capture is_solution failure context: {e}")
                    
                    context = self._capture_failure_context(
                        task_instance, problem, solution
                    )
                    self.logger.info(f"ValidationPipeline: captured context is None: {context is None}")
                    if context:
                        self.logger.info(f"ValidationPipeline: context has full_context: {context.full_context is not None}")
                        if context.full_context:
                            self.logger.info(f"ValidationPipeline: full_context length: {len(context.full_context)}")
                    else:
                        self.logger.warning("ValidationPipeline: No context captured during validation failure")
                
                return ValidationResult(
                    is_valid=False,
                    error_type=ErrorType.INVALID_SOLUTION,
                    error_message="Solution failed validation",
                    context=context,
                    validation_time_ms=validation_time_ms
                )
                
        except Exception as e:
            # Validation threw an exception
            import traceback
            tb_str = traceback.format_exc()
            
            self.logger.error(f"Exception in is_solution: {e}")
            
            validation_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Try to capture context even on exception
            context = None
            if capture_context:
                try:
                    context = self._capture_exception_context(e, tb_str)
                except Exception as ctx_error:
                    self.logger.warning(f"Failed to capture exception context: {ctx_error}")
            
            return ValidationResult(
                is_valid=False,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message=f"Validation Error: {str(e)}",
                context=context,
                validation_time_ms=validation_time_ms
            )
    
    def _is_stripped_solution(self, solution: Any) -> bool:
        """Check if a solution has been stripped."""
        if solution is None:
            return False
        
        if isinstance(solution, dict):
            return (solution.get("__stripped__", False) or 
                    solution.get("stripped_after_validation", False))
        
        return False
    
    def _capture_failure_context(
        self,
        task_instance: Any,
        problem: Any,
        solution: Any
    ) -> Optional[ValidationContext]:
        """Capture context about why validation failed."""
        try:
            # Check if we already have cached context
            task_id = id(task_instance)
            if task_id in self._context_cache:
                cached_context = self._context_cache[task_id]
                self.logger.debug("Using cached validation context")
                return cached_context
            
            # First check if we already have context stored from a previous call
            raw_context = getattr(task_instance, "_last_is_solution_failure_context", None)
            
            if raw_context:
                self.logger.debug(f"Using existing context of length {len(raw_context)}")
                # Parse the context to extract details
                context = self._parse_failure_context(raw_context)
                
                # Cache for potential reuse
                self._context_cache[task_id] = context
                
                return context
            
            # Use the failure analyzer to trace the failure
            self.logger.debug("Tracing is_solution failure")
            trace_is_solution_failure(task_instance, problem, solution)
            
            # Extract the context from the task instance
            raw_context = getattr(task_instance, "_last_is_solution_failure_context", None)
            
            if raw_context:
                self.logger.debug(f"Captured context of length {len(raw_context)}")
                
                # Parse the context to extract details
                context = self._parse_failure_context(raw_context)
                
                # Cache for potential reuse
                self._context_cache[task_id] = context
                
                return context
            else:
                self.logger.warning("No failure context captured")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing failure context: {e}")
            return ValidationContext(
                failure_reason=f"Failed to capture context: {str(e)}",
                full_context=None
            )
    
    def _parse_failure_context(self, raw_context: str) -> ValidationContext:
        """Parse raw failure context into structured format."""
        # Look for the failing line marker
        failure_line = None
        lines = raw_context.split('\n')
        
        for i, line in enumerate(lines):
            if line.startswith('>'):
                # This is the failing line
                try:
                    parts = line.split(':', 1)
                    if len(parts) >= 1:
                        line_num_str = parts[0].strip('> ')
                        failure_line = int(line_num_str)
                except (ValueError, IndexError):
                    pass
                break
        
        # Extract the reason if it's in the context
        failure_reason = None
        if "Solution is not a list" in raw_context:
            failure_reason = "Solution is not a list of the expected length"
        elif "not normalized" in raw_context:
            failure_reason = "Eigenvector is not properly normalized"
        elif "relative error" in raw_context:
            failure_reason = "Solution has excessive relative error"
        
        return ValidationContext(
            failure_line=failure_line,
            failure_reason=failure_reason,
            code_snippet=None,  # Could extract relevant code lines
            full_context=raw_context
        )
    
    def _capture_exception_context(
        self,
        exception: Exception,
        traceback_str: str
    ) -> ValidationContext:
        """Create context from a validation exception."""
        from AlgoTuner.utils.error_utils import extract_error_context
        
        try:
            # Use existing error context extraction utility
            context_info = extract_error_context(traceback_str, str(exception))
            enhanced_error_msg = context_info.get("enhanced_error_message", str(exception))
            code_context = context_info.get("code_context_snippet")
            
            # Extract line number from enhanced error message
            failure_line = None
            if " at line " in enhanced_error_msg:
                try:
                    line_part = enhanced_error_msg.split(" at line ")[1].split()[0]
                    failure_line = int(line_part)
                except (ValueError, IndexError):
                    pass
            
            return ValidationContext(
                failure_line=failure_line,
                failure_reason=f"Exception in validation: {type(exception).__name__}",
                code_snippet=code_context,
                full_context=None  # Use structured context instead of raw traceback
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to extract structured context: {e}")
            # Fallback to basic context
            return ValidationContext(
                failure_line=None,
                failure_reason=f"{type(exception).__name__}: {str(exception)}",
                code_snippet=None,
                full_context=None
            )
    
    def clear_context_cache(self):
        """Clear the context cache to free memory."""
        self._context_cache.clear()
        self.logger.debug("Cleared validation context cache")