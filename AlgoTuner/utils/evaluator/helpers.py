"""
Helper functions for the evaluator module.
"""

import logging
import numpy as np
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from numpy.typing import NDArray


def describe_type(value: Any) -> str:
    """
    Helper function to get a concise type description.
    
    Args:
        value: Any Python value
        
    Returns:
        A string describing the type of the value
    """
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{type(value).__name__}[]"
        sample = value[0]
        if isinstance(sample, (list, tuple)):
            inner_type = type(sample[0]).__name__ if sample else "any"
            return f"{type(value).__name__}[{type(sample).__name__}[{inner_type},...]]"
        return f"{type(value).__name__}[{type(sample).__name__}]"
    return type(value).__name__


def truncate_value(value_str: str, max_length: int = 100) -> str:
    """
    Truncate a string representation of a value.
    
    Args:
        value_str: String to truncate
        max_length: Maximum length
        
    Returns:
        Truncated string
    """
    if len(value_str) <= max_length:
        return value_str
    return value_str[:max_length] + "..."


def are_types_compatible(result: Any, expected: Any) -> bool:
    """
    Check if two types are compatible for evaluation purposes.
    
    This is more flexible than strict type equality and allows for common
    type conversions that would work in practice.
    
    Args:
        result: The result value to check
        expected: The expected value to compare against
        
    Returns:
        True if the types are compatible, False otherwise
    """
    # Always consider numpy arrays compatible with lists/tuples and vice versa
    if isinstance(result, np.ndarray) and isinstance(expected, (list, tuple)):
        return True
    
    if isinstance(expected, np.ndarray) and isinstance(result, (list, tuple)):
        return True
    
    # If types are exactly the same, they're compatible
    if type(result) == type(expected):
        return True
    
    # Special case for list and tuple - they're structurally compatible
    if isinstance(result, (list, tuple)) and isinstance(expected, (list, tuple)):
        # Empty sequences are compatible
        if not result or not expected:
            return True
        
        # Check if the first elements are compatible
        if len(result) > 0 and len(expected) > 0:
            # If first elements are also sequences, check their types
            if isinstance(result[0], (list, tuple)) and isinstance(expected[0], (list, tuple)):
                return True
            # Otherwise, check if the first elements have the same type
            return type(result[0]) == type(expected[0])
        
        return True
    
    # Special case for numeric types
    if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
        return True
    
    # Special case for string-like types
    if isinstance(result, str) and isinstance(expected, str):
        return True
    
    # Default to True - assume compatibility
    return True


def format_type_mismatch(result: Any, expected_solution: Any) -> str:
    """
    Format a clear type mismatch error message.
    
    Args:
        result: The actual result
        expected_solution: The expected result
        
    Returns:
        Formatted error message
    """
    result_type = describe_type(result)
    expected_type = describe_type(expected_solution)
    
    # If the type descriptions are identical or nearly identical, this is likely a validation
    # issue rather than an actual type mismatch - check the actual types
    if result_type == expected_type or (
        # Special case for list[list[int,...]] vs list[list[int,...]],
        # which can happen due to different representations of the same type
        'list[list[' in result_type and 'list[list[' in expected_type and 
        result_type.endswith(',...]]') and expected_type.endswith(',...]]')
    ):
        # For structurally equivalent types, check if the data validates correctly
        if are_types_compatible(result, expected_solution):
            # If there's a mismatch but types are compatible, suggest checking the task's is_solution method
            return (
                f"Your solution has the correct type structure but doesn't validate according to the task's requirements.\n"
                f"This could be due to differences in how the data is arranged or constraints not being met.\n"
                f"Your output: {truncate_value(str(result))}\n"
                f"Example format: {truncate_value(str(expected_solution))}\n\n"
                f"Please check the task requirements and ensure your solution meets all constraints."
            )
    
    # Add a note about list/tuple compatibility if that's the issue
    note = ""
    if (isinstance(result, tuple) and isinstance(expected_solution, list)) or \
       (isinstance(result, list) and isinstance(expected_solution, tuple)):
        note = "\nNote: You can convert between tuple and list using list() or tuple() functions."
    
    # Special diagnostics for None result
    if result is None:
        import logging
        logging.debug("DEBUG: format_type_mismatch received None as result")
        
        # Try to capture the most recent traceback
        import traceback
        recent_tb = traceback.format_exc()
        if recent_tb and recent_tb.strip() != "NoneType: None":
            logging.error(f"Recent traceback when None was returned: {recent_tb}")
    
    # Truncate the output for better readability
    truncated_result = truncate_value(str(result))
    truncated_expected = truncate_value(str(expected_solution))
    
    # Create a more helpful error message with concrete examples
    return (
        f"Invalid solution format: Type mismatch: Your solution returned {result_type} but we expect {expected_type}.{note}\n"
        f"Your output: {truncated_result}\n"
        f"Expected format: {truncated_expected}\n\n"
        f"The solver function must return the correct type to be considered valid."
    )


def make_error_response(
    error_msg: Optional[str] = None, 
    traceback_str: Optional[str] = None, 
    stdout: str = "", 
    stderr: str = "", 
    elapsed_ms: float = 0
) -> Tuple[None, Dict[str, Any], float, str]:
    """
    Create a standardized error response.
    
    Args:
        error_msg: Error message
        traceback_str: Traceback string
        stdout: Captured standard output
        stderr: Captured standard error
        elapsed_ms: Elapsed time in milliseconds
        
    Returns:
        Tuple of (None, metadata, elapsed_ms, "error")
    """
    return (
        None,
        {
            "stdout": stdout,
            "stderr": stderr,
            "error": error_msg,
            "traceback": traceback_str
        },
        elapsed_ms,
        "error"
    )


def prepare_problem_for_solver(problem: Any, task_instance: Any) -> Any:
    """
    Prepare the problem input to be in the correct format for the solver.
    This is a generic function that handles different input types and structures.
    
    Args:
        problem: The problem input in any format
        task_instance: The task instance to determine expected format
        
    Returns:
        The prepared problem in the correct format for the solver
    """
    import numpy as np
    import inspect
    from numpy.typing import NDArray
    
    # If problem is None, return as is
    if problem is None:
        logging.warning("None problem passed to prepare_problem_for_solver")
        return problem
        
    # Always ensure the task_instance itself is available for validation
    if not task_instance:
        logging.warning("No task_instance provided to prepare_problem_for_solver")
        return problem
    
    # Log the initial problem type for debugging
    logging.debug(f"Initial problem type: {type(problem)}")
    
    # Step 1: Handle lists and convert to numpy arrays when appropriate
    if isinstance(problem, list):
        logging.debug(f"Problem is a list with {len(problem)} elements")
        
        # Check if the problem is empty
        if not problem:
            logging.warning("Empty list problem, returning as is")
            return problem
            
        # If all elements are lists of the same length, it's likely a 2D structure
        # that should be converted to a 2D numpy array
        all_lists = all(isinstance(x, list) for x in problem)
        if all_lists and problem:
            first_len = len(problem[0])
            all_same_len = all(len(x) == first_len for x in problem)
            
            if all_same_len:
                try:
                    np_problem = np.array(problem)
                    logging.info(f"Converted 2D list structure to numpy array with shape {np_problem.shape}")
                    problem = np_problem
                except Exception as e:
                    logging.warning(f"Failed to convert 2D list to numpy array: {e}")
        
        # Even if it's not a 2D structure, try to convert simple lists to numpy arrays
        # This is especially important for problems loaded from data files
        if isinstance(problem, list):
            try:
                np_problem = np.array(problem)
                logging.info(f"Converted list to numpy array with shape {np_problem.shape}")
                problem = np_problem
            except Exception as e:
                logging.warning(f"Failed to convert list to numpy array: {e}")
    
    # Step 2: Check if is_solution method requires numpy arrays
    if hasattr(task_instance, 'is_solution'):
        try:
            # Inspect the is_solution method to see if it expects numpy arrays
            source = inspect.getsource(task_instance.is_solution)
            
            # If the is_solution method uses shape or other numpy attributes, 
            # it likely expects a numpy array
            needs_numpy = any(attr in source for attr in ['.shape', '.ndim', '.dtype', 
                                                         'np.array', 'numpy.array'])
            
            if needs_numpy and not isinstance(problem, np.ndarray):
                logging.info("is_solution needs numpy arrays, converting problem")
                try:
                    np_problem = np.array(problem)
                    logging.info(f"Converted problem to numpy array with shape {np_problem.shape}")
                    problem = np_problem
                except Exception as e:
                    logging.warning(f"Failed to convert problem to numpy array for is_solution: {e}")
        except Exception as e:
            logging.warning(f"Error inspecting is_solution: {e}")
    
    # Step 3: Check solve method's parameter annotations
    try:
        solve_signature = inspect.signature(task_instance.solve)
        param_annotations = None
        
        # Try to get the parameter annotation for problem or coefficients
        if 'problem' in solve_signature.parameters:
            param_annotations = solve_signature.parameters.get('problem')
        elif 'coefficients' in solve_signature.parameters:
            param_annotations = solve_signature.parameters.get('coefficients')
        else:
            # Just take the first parameter, whatever it is
            first_param = next(iter(solve_signature.parameters.values()), None)
            if first_param:
                param_annotations = first_param
                
        # Check the parameter annotation if we found one
        if param_annotations and param_annotations.annotation:
            annotation = param_annotations.annotation
            annotation_str = str(annotation)
            
            # More comprehensive check for NDArray annotations
            is_ndarray = (
                annotation == NDArray or 
                (hasattr(annotation, '__origin__') and annotation.__origin__ == NDArray) or
                'ndarray' in annotation_str.lower() or
                'numpy' in annotation_str.lower()
            )
            
            if is_ndarray and not isinstance(problem, np.ndarray):
                logging.info(f"Solve method expects NDArray, current type is {type(problem)}")
                
                # Try to convert the problem to numpy array
                if isinstance(problem, (list, tuple)):
                    try:
                        np_problem = np.array(problem)
                        logging.info(f"Converted problem to NumPy array with shape {np_problem.shape} based on type annotation")
                        return np_problem
                    except Exception as e:
                        logging.warning(f"Failed to convert to NumPy array from annotation: {e}")
                
                # If problem is a list of one NDArray, extract it
                if isinstance(problem, list) and len(problem) == 1 and isinstance(problem[0], np.ndarray):
                    logging.info("Extracted single NumPy array from list")
                    return problem[0]
                    
                # Special case for nested lists of arrays
                if isinstance(problem, list) and all(isinstance(item, np.ndarray) for item in problem):
                    # Try to stack arrays appropriately
                    try:
                        stacked = np.vstack(problem) if len(problem) > 1 else problem[0]
                        logging.info(f"Stacked multiple arrays into shape {stacked.shape}")
                        return stacked
                    except Exception as e1:
                        try:
                            stacked = np.hstack(problem)
                            logging.info(f"Horizontally stacked arrays into shape {stacked.shape}")
                            return stacked
                        except Exception as e2:
                            logging.warning(f"Failed to stack arrays: {e1}, {e2}")
    except Exception as e:
        logging.warning(f"Error checking task signature: {e}")
    
    # Step 4: Generate example problem to determine expected format
    try:
        example = task_instance.generate_problem(n=1, random_seed=42)
        logging.debug(f"Generated example problem of type {type(example)}")
        
        # If the example is a NumPy array and our problem isn't, convert
        if isinstance(example, np.ndarray) and not isinstance(problem, np.ndarray):
            try:
                np_problem = np.array(problem)
                logging.info(f"Based on example, converted problem to NumPy array with shape {np_problem.shape}")
                return np_problem
            except Exception as e:
                logging.warning(f"Failed to convert problem to match example: {e}")
                
        # Special handling for lists of arrays
        if isinstance(example, np.ndarray) and isinstance(problem, list):
            if len(problem) == 1 and isinstance(problem[0], np.ndarray):
                logging.info("Based on example, extracting single NumPy array from list")
                return problem[0]
            elif all(isinstance(item, np.ndarray) for item in problem):
                for item in problem:
                    if item.shape == example.shape:
                        logging.info(f"Found array with matching shape {item.shape}")
                        return item
                        
                # Try combining arrays if none match exactly
                try:
                    if len(problem) > 1:
                        stacked = np.vstack(problem)
                        logging.info(f"Vertically stacked multiple arrays to shape {stacked.shape}")
                        return stacked
                except Exception:
                    try:
                        stacked = np.hstack(problem)
                        logging.info(f"Horizontally stacked multiple arrays to shape {stacked.shape}")
                        return stacked
                    except Exception as e:
                        logging.warning(f"Failed to combine arrays: {e}")
    except Exception as e:
        logging.warning(f"Error generating example problem: {e}")
    
    # If all the special handling above failed, return the problem as is
    logging.debug(f"Returning problem as-is with type {type(problem)}")
    return problem 


# --- format_shape --- 
def format_shape(obj: Any) -> str:
    """Format the shape of an object in a human-readable way"""
    # Import from utils to avoid duplication
    from AlgoTuner.utils.utils import format_object_shape
    return format_object_shape(obj) 


def clean_traceback(tb_str: Optional[str]) -> str:
    """Clean a traceback by removing sensitive information"""
    # Import from utils to avoid duplication
    from AlgoTuner.utils.utils import clean_traceback as clean_tb_utils
    return clean_tb_utils(tb_str) 