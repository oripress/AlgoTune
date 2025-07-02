import os
from typing import Any, Optional, Tuple
import logging
import sys
import importlib
import traceback
from pathlib import Path

from AlgoTuner.utils.error_utils import create_standard_error_result, SolverFileNotFoundError, SOLVER_NOT_FOUND_GENERIC_MSG
from AlgoTuner.utils.solver_loader import load_solver_module, get_solve_callable, with_working_dir

def validate_solver_setup(code_dir: Path, command_source: Optional[str] = None) -> Tuple[bool, Optional[dict]]:
    """Validate solver.py setup and imports."""
    solver_file = code_dir / "solver.py"
    logging.info(f"Looking for solver at: {solver_file}")
    
    if not solver_file.is_file():
        error_msg = SOLVER_NOT_FOUND_GENERIC_MSG
        logging.error(f"{error_msg} (Path checked: {solver_file})")
        return False, {
            "success": False,
            "error": error_msg,
            "error_type": "solver_not_found_error",
            "output_logs": "",
            "command_source": command_source,
        }

    # Dynamically load and validate the solver entrypoint using the central loader
    try:
        with with_working_dir(code_dir):
            solver_module = load_solver_module(code_dir)
    except SolverFileNotFoundError as e:
        error_msg = SOLVER_NOT_FOUND_GENERIC_MSG
        logging.error(f"{error_msg} - Details: {e}")
        return False, {
            "success": False,
            "error": error_msg,
            "error_type": "solver_not_found_error",
            "output_logs": "",
            "command_source": command_source,
        }
    except ImportError as e:
        # The ImportError already contains our formatted error message, don't wrap it
        error_msg = str(e)
        logging.error(f"Failed to import solver.py:\n{error_msg}\n{traceback.format_exc()}")
        return False, {
            "success": False,
            "error": error_msg,
            "error_type": "import_error",
            "output_logs": "",
            "command_source": command_source,
        }
    # --- NEW: Explicitly catch OSError (e.g., forbidden file I/O) and return standardized result ---
    except OSError as e:
        # Use standardized error utility to classify and format
        import traceback as _tb
        from AlgoTuner.utils.error_utils import create_standard_error_result

        logging.error(f"OSError encountered while loading solver.py: {e}\n{_tb.format_exc()}")

        error_dict = create_standard_error_result(
            exception=e,
            traceback_str=_tb.format_exc(),
        )

        # Ensure expected keys for downstream processing
        error_dict.setdefault("output_logs", "")
        error_dict["command_source"] = command_source

        return False, error_dict
    try:
        # Check for Solver class and solve method
        SolverClass = getattr(solver_module, "Solver", None)
        if SolverClass is None:
            raise AttributeError("Class 'Solver' not found in solver.py")
        if not callable(getattr(SolverClass, "solve", None)):
            raise AttributeError("Method 'solve' not found or not callable in Solver class")
        logging.info("Found Solver class with solve method in solver.py")
        return True, None
    except AttributeError as e: 
        # Custom error messages based on what was missing
        if "Class 'Solver' not found" in str(e):
            error_msg = "Solver class not found in solver.py. Please define a class named 'Solver' with a 'solve' method."
            error_type = "missing_solver_class"
        elif "Method 'solve' not found" in str(e):
            error_msg = "Class 'Solver' found but no callable 'solve' method. Please add a method named 'solve' to your Solver class."
            error_type = "missing_solver_method"
        else: # Other potential AttributeErrors
            error_msg = f"Error accessing Solver/solve in solver.py: {e}"
            error_type = "attribute_error"
        logging.error(error_msg)
        return False, {
            "success": False,
            "error": error_msg,
            "error_type": error_type,
            "output_logs": "",
            "command_source": command_source,
        }
    except Exception as e:
        error_msg = f"Error validating solver.py: {e}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return False, {
            "success": False,
            "error": error_msg,
            "error_type": "validation_error",
            "output_logs": "",
            "command_source": command_source,
        }
    # Should not reach here
    return True, None

def validate_dataset(data: Any, data_subset: str, command_source: Optional[str] = None) -> Tuple[bool, Optional[dict], Any]:
    """Validate dataset and handle dataset subsets."""
    if not data:
        return False, {
            "success": False,
            "error": "No data available for evaluation",
            "output_logs": "",
            "command_source": command_source,
        }, None

    # Handle dataset subsets
    if isinstance(data, tuple):
        train_data, test_data = data
        data = train_data if data_subset == "train" else test_data

    if not data:
        return False, {
            "success": False,
            "error": f"No data available for {data_subset} subset",
            "output_logs": "",
            "command_source": command_source,
        }, None
    
    return True, None, data

def validate_structure(example: Any, input_data: Any) -> bool:
    """Recursively validate the structure of input_data against example."""
    if type(example) != type(input_data):
        # Special case: allow bool values in lists where example has numeric types
        if isinstance(example, (list, tuple)) and isinstance(input_data, (list, tuple)):
            return len(example) == len(input_data)
        return False
    
    if isinstance(example, (list, tuple)):
        if len(example) == 0 and len(input_data) == 0:
            return True
        if len(example) == 0 or len(input_data) == 0:
            return False
        # For lists/tuples, also validate length if example is non-empty
        if len(example) > 0 and len(example) != len(input_data):
            return False
        return all(validate_structure(e, i) for e, i in zip(example, input_data))
    
    return True

def validate_optimal_time(
    optimal_time: Optional[float],
    idx: int,
    last_output_logs: str,
    command_source: Optional[str] = None
) -> Tuple[bool, Optional[dict]]:
    """Validate optimal time exists and is valid."""
    if optimal_time is None:
        return False, {
            "success": False,
            "error": f"Problem {idx + 1} is missing its optimal solve time. Please regenerate the dataset.",
            "output_logs": last_output_logs,
            "command_source": command_source,
        }
    return True, None 