"""
Analyzes failed is_solution calls after evaluation completes.
"""

import sys
import inspect
import logging
import traceback
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from AlgoTuner.utils.error_utils import create_standard_error_result


MAX_FAILURES_TO_ANALYZE_PER_TASK = 3

class IsSolutionTracer:
    """
    A tracer specifically designed to find the line number where
    a target function (is_solution) returns False.
    """
    def __init__(self, target_code_object, start_line_no):
        self.target_code_object = target_code_object
        self.start_line_no = start_line_no # For reference, not strictly needed by trace
        self.last_line_in_target = None
        self.failing_line_number = None
        self.call_depth = 0  # Track nested call depth within is_solution
        self.watched_calls = {}  # Track potentially relevant function calls
        logging.debug(f"Tracer initialized for code: {target_code_object.co_filename}:{start_line_no}")

    def trace(self, frame, event, arg):
        """Main trace function that's registered with sys.settrace()."""
        # Simplifying the tracer: focus on returns with False value in is_solution
        
        # Check if we're in our target function
        if frame.f_code == self.target_code_object:
            if event == 'line':
                # Keep track of the current line in the target function
                self.last_line_in_target = frame.f_lineno
                # Capture conditional expressions that might lead to returns
                if 'return ' in frame.f_code.co_consts and frame.f_lineno in frame.f_lasti:
                    logging.debug(f"Potential return at line {frame.f_lineno} detected")
                logging.debug(f"Line event at line {frame.f_lineno}")
                
            elif event == 'return':
                # Check if we're returning False (falsey values count as well)
                # This is the key point: identify when is_solution returns a falsey value
                if not arg:  # This catches False, None, 0, empty sequences, etc.
                    logging.debug(f"Found return with falsey value {arg} at line {self.last_line_in_target}")
                    self.failing_line_number = self.last_line_in_target
                else:
                    logging.debug(f"Return with truthy value {arg} ignored")
                    
            # Capture all calls made from is_solution for additional context
            elif event == 'call':
                called_func_name = frame.f_code.co_name
                self.watched_calls[called_func_name] = self.watched_calls.get(called_func_name, 0) + 1
                logging.debug(f"Captured call to {called_func_name} from is_solution")
                
            # Continue tracing inside target function
            return self.trace
            
        # Ignore events outside our target
        return None

def trace_is_solution_failure(task_instance: Any, problem: Any, solution: Any) -> List[str]:
    """
    Traces a single call to task_instance.is_solution(problem, solution)
    to find the line number returning False and returns context lines.
    Returns an empty list if analysis fails or no failure is found.
    """
    # ------------------------------------------------------------------
    # Skip tracing when the solution object has been stripped to reduce
    # memory (it is represented by a small sentinel dict produced in
    # runner.run_solver_evaluation).  Calling is_solution with such a
    # stub – or with ``None`` – would raise and pollute the logs.
    # ------------------------------------------------------------------

    if solution is None or (isinstance(solution, dict) and (solution.get("__stripped__", False) or solution.get("stripped_after_validation", False))):
        # Return the already-stored context if it exists (it was captured before stripping)
        if hasattr(task_instance, '_last_is_solution_failure_context') and task_instance._last_is_solution_failure_context:
            return [task_instance._last_is_solution_failure_context]
        # Only set the "stripped" message if we don't already have failure context
        else:
            task_instance._last_is_solution_failure_context = (
                "Solution object was stripped after initial validation –\n"
                "skipping second is_solution call to avoid spurious errors."
            )
            return []

    is_solution_func = task_instance.is_solution
    try:
        # Trace the is_solution call
        func_code = is_solution_func.__code__
        source_lines, start_lineno = inspect.getsourcelines(func_code)
        tracer = IsSolutionTracer(func_code, start_lineno)
        sys.settrace(tracer.trace)
        try:
            # This is where an exception inside is_solution would occur
            _ = is_solution_func(problem, solution)
        except Exception as e:
            # It's a failure in their code, not ours. Report it as such.
            tb_str = traceback.format_exc()
            error_info = create_standard_error_result(e, tb_str)
            # Format a user-friendly message with the error and context
            context_msg = error_info.get('code_context', 'No code context available.')
            error_msg = f"Error in 'is_solution': {error_info.get('error', str(e))}\n{context_msg}"
            task_instance._last_is_solution_failure_context = error_msg
            return [] # Return empty list as analysis "succeeded" in finding the error
        finally:
            sys.settrace(None)
        
        fl = tracer.failing_line_number
        if fl is None:
            task_instance._last_is_solution_failure_context = "No failure line found in is_solution"
            return []

        # Build context: 15 lines above the failing line, none below
        end_lineno = start_lineno + len(source_lines) - 1
        above_lines = 15  # number of lines to show above failure
        cs = max(start_lineno, fl - above_lines)
        ce = fl
        lines = []
        for ln in range(cs, ce + 1):
            text = source_lines[ln - start_lineno].rstrip('\n')
            prefix = '>' if ln == fl else ' '
            lines.append(f"{prefix} {ln}: {text}")
        snippet = "\n".join(lines)
        task_instance._last_is_solution_failure_context = snippet
        return [snippet]
    except Exception as e:
        # This is now for genuine analysis/tracing failures
        analysis_error_msg = f"Error during failure analysis trace: {e}"
        task_instance._last_is_solution_failure_context = analysis_error_msg
        return []

def analyze_is_solution_failures(failed_instances: Dict[str, List[Tuple[Any, Any, Any]]]) -> List[str]:
    """
    Main entry point to analyze stored is_solution failures.
    Iterates through stored failures, triggers tracing, and returns collected analysis strings.
    """
    all_analysis_logs = []
    if not failed_instances:
        logging.info("No is_solution failures recorded for analysis.")
        return all_analysis_logs

    logging.info(f"--- Starting Post-Evaluation Analysis of is_solution Failures ({MAX_FAILURES_TO_ANALYZE_PER_TASK} max per task) ---")
    total_analyzed = 0
    for task_name, failures in failed_instances.items():
        # Removed task-specific log header here, as requested
        task_analysis_lines = [] # Store lines for this specific task temporarily
        processed_failures_for_task = 0
        for i, (task_instance, problem, solution) in enumerate(failures):
            # Add "Failure X:" prefix before the analysis for each instance
            task_analysis_lines.append(f"Failure {i + 1}:")
            analysis_lines = trace_is_solution_failure(task_instance, problem, solution)
            if analysis_lines:
                 task_analysis_lines.extend(analysis_lines)
                 task_analysis_lines.append("") # Add a blank line for separation between failure contexts
                 processed_failures_for_task += 1
            total_analyzed += 1
            
        # Only add the header and the collected lines if we actually processed failures for this task
        if processed_failures_for_task > 0:
            all_analysis_logs.extend(task_analysis_lines)
        # Removed else block for logging no failures found for task

    logging.info(f"--- Finished Post-Evaluation Analysis ({total_analyzed} total failures analyzed) ---")
    return all_analysis_logs 