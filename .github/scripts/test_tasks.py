#!/usr/bin/env python3
"""
Task Signature and Description Testing Script

This script validates Task implementations for:
1. Correct method signatures and type hints for core methods.
2. Presence, non-emptiness, and required headers in description.txt.
"""

import os
import sys
import inspect
import importlib
import ast
import textwrap
import traceback
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, repo_root)
# -----------------------------------------------------------

# Silence third‑party logs (Keep minimal logging setup)
logging.basicConfig(level=logging.CRITICAL)
for name in ["ortools", "numpy", "pulp", "pyomo", "gurobipy", "cplex", "scipy"]:
    lg = logging.getLogger(name)
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False
root = logging.getLogger()
root.handlers = []
root.addHandler(logging.NullHandler())

# Pre-import typing for checks if needed
try:
    from typing import Union, get_args, get_type_hints, Any
except ImportError:
    Union = None
    get_args = lambda x: []
    get_type_hints = lambda x: {}
    Any = object()
    print("Warning: 'typing' module not fully available. Some type hint checks may be limited.", file=sys.stderr)


##############################
#         CHECKS             #
##############################

def check_explicit_type_hints(method_name: str, method_obj, class_name: str):
    """Ensure method parameters and return value have explicit type hints."""
    # Skip for base class or methods known not to require hints (like __init__ if simplified)
    if class_name == "Task" or method_name.startswith("__") and method_name != "__init__":
        # Allow __init__ check if desired, but skip others like __repr__
        return
    try:
        sig = inspect.signature(method_obj)
    except (ValueError, TypeError):
        # Cannot get signature (e.g., built-in function/method)
        # Consider raising an error if we expect all Task methods to be inspectable
        # print(f" [WARNING: Could not inspect signature for {class_name}.{method_name}]", end='')
        return # Skip check if signature is not available

    # Check parameters for type hints
    missing_param_hints = []
    for param_name, param in sig.parameters.items():
        # Ignore 'self'/'cls' and **kwargs parameters for hinting requirement
        if param_name in ('self', 'cls') or param.kind == param.VAR_KEYWORD:
            continue
        # Check if annotation is missing or is inspect._empty
        if param.annotation is inspect.Parameter.empty:
            missing_param_hints.append(param_name)

    if missing_param_hints:
        raise AssertionError(
            f"In {class_name}.{method_name}, parameters {missing_param_hints} are missing type hints."
        )

    # Check return value for type hint
    # Allow missing return hint for __init__
    if method_name != "__init__" and sig.return_annotation is inspect.Signature.empty:
            raise AssertionError(
            f"In {class_name}.{method_name}, the return value is missing a type hint (use '-> None' if applicable)."
        )

    # Basic check for Any usage (optional, can be strict)
    try:
        hints = get_type_hints(method_obj)
        for param_name, hint in hints.items():
            if param_name == 'return' and method_name == '__init__': continue # Skip return for init
            # Check if the hint itself is Any or contains Any (e.g., Union[Any, int])
            if hint is Any:
                print(f" [INFO: {class_name}.{method_name} uses 'Any' for parameter '{param_name}']", end='') # Use INFO or WARNING
            elif Union and hasattr(hint, '__args__') and Any in get_args(hint):
                print(f" [INFO: {class_name}.{method_name} uses 'Any' within Union/Optional for '{param_name}']", end='')
        # Check return hint
        if 'return' in hints and hints['return'] is Any and method_name != '__init__':
            print(f" [INFO: {class_name}.{method_name} uses 'Any' for return type]", end='')
        elif 'return' in hints and Union and hasattr(hints['return'], '__args__') and Any in get_args(hints['return']) and method_name != '__init__':
            print(f" [INFO: {class_name}.{method_name} uses 'Any' within Union/Optional for return type]", end='')
    except Exception:
        # Ignore errors during get_type_hints (e.g., NameErrors if types aren't imported)
        # print(f" [WARNING: Could not resolve type hints for {class_name}.{method_name}]", end='')
        pass


def check_description_file(task_dir):
    """Check for description.txt, non-emptiness, and required headers."""
    desc_path = os.path.join(task_dir, 'description.txt')
    if not os.path.isfile(desc_path):
        raise AssertionError(f"Missing required file: description.txt in {task_dir}")

    with open(desc_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        raise AssertionError(f"File description.txt in {task_dir} is empty.")

    # Check for category specification
    import re
    category_pattern = r'^Category:\s*(.+)$'
    category_found = False

    for line in content.split('\n'):
        match = re.match(category_pattern, line.strip())
        if match:
            category_found = True
            category_value = match.group(1).strip()
            if not category_value:
                raise AssertionError(f"Category specification in {task_dir}/description.txt is empty.")
            break

    if not category_found:
        raise AssertionError(f"Missing required 'Category:' specification in {task_dir}/description.txt")


# --- MODIFIED test_task_class with granular timeouts ---
def test_task_class(task_class):
    """Run signature and description checks on a Task subclass."""
    class_name = task_class.__name__
    module_name = task_class.__module__
    print(f"\n--- Testing Signatures & Description: {class_name} ({module_name}) ---")
    task_failed = False

    # Define required method signatures (parameter names)
    # Focusing on presence and core parameters now.
    required_methods = {
        # Method: (required_params)
        'generate_problem': ['n'],
        'solve': ['problem'],
        'is_solution': ['problem', 'solution'],
    }

    # Check Task directory for description.txt
    try:
        print("  - Checking description.txt...", end='', flush=True)
        # Find task directory relative to module path
        module_path = sys.modules[module_name].__file__
        task_dir = os.path.dirname(module_path)
        check_description_file(task_dir)
        print(" [PASS]")
    except (AssertionError, OSError, KeyError) as e:
        print(f" [FAIL]\n    ERROR: {e}")
        task_failed = True # Mark failure but continue

    print("  - Checking method signatures and type hints:")
    all_methods_passed = True
    methods_to_check = ['__init__', 'generate_problem', 'solve', 'is_solution']
    for method_name in methods_to_check:
        method_obj = getattr(task_class, method_name, None)

        print(f"    - Checking '{method_name}'...", end='', flush=True)
        if not method_obj or not callable(method_obj):
            if method_name != '__init__': # Allow missing __init__ if base is sufficient
                print(f" [FAIL]\n      ERROR: Missing or non-callable required method '{method_name}'.")
                all_methods_passed = False
            else:
                print(" [SKIP] (Optional)") # Skip optional __init__
            continue

        try:
            # A. Check Signature Parameters (basic check)
            if method_name in required_methods:
                required_params = required_methods[method_name]
                try:
                    sig = inspect.signature(method_obj)
                    params = list(sig.parameters.keys())
                except (ValueError, TypeError):
                    raise AssertionError("Cannot inspect signature.")
                # Remove self/cls if present
                if params and params[0] in ('self', 'cls'):
                    params.pop(0)
                missing_required = set(required_params) - set(params)
                if missing_required:
                    raise AssertionError(f"Missing required parameters: {missing_required}.")

            # B. Check Type Hints (for all checked methods)
            check_explicit_type_hints(method_name, method_obj, class_name)

            print(" [PASS]")

        except AssertionError as e:
            all_methods_passed = False
            print(f" [FAIL]\n      ERROR: {e}")
        except Exception as e:
            all_methods_passed = False
            print(f" [FAIL]\n      ERROR: Unexpected error during check: {e}")
            traceback.print_exc(limit=1)

    if not all_methods_passed:
        print("  - [FAIL] Method checks failed.")
        task_failed = True

    # --- Final Result ---
    if task_failed:
        print(f"--- Task {class_name} FAILED Signatures/Description Checks ---")
        # Keep the AssertionError to make the script exit with non-zero status
        raise AssertionError(f"Checks failed for {class_name}")
    else:
        print(f"--- Task {class_name} PASSED Signatures/Description Checks ---")
        return True


def main():
    start_time_main = time.time() # Use a different name
    tasks_dir = 'AlgoTuneTasks'
    excluded_dirs = {'__pycache__', 'base', 'template', '.git', '.ipynb_checkpoints'}
    # Optional: Exclude specific known-problematic tasks if needed
    # excluded_dirs.add('some_task_dir')
    if not os.path.isdir(tasks_dir):
        print(f"ERROR: Tasks directory '{tasks_dir}' not found in {os.getcwd()}.", file=sys.stderr)
        sys.exit(1)

    specified_tasks = []
    if len(sys.argv) > 1:
        specified_tasks = sys.argv[1:]
        print(f"Testing only specified tasks: {', '.join(specified_tasks)}")

    all_subdirs = sorted([
        d for d in os.listdir(tasks_dir)
        if os.path.isdir(os.path.join(tasks_dir, d)) and d not in excluded_dirs and not d.startswith('.')
    ])

    subdirs_to_test = []
    if specified_tasks:
        specified_tasks_set = set(specified_tasks)
        for d in all_subdirs:
            if d in specified_tasks_set:
                subdirs_to_test.append(d)
        tested_set = set(subdirs_to_test)
        missing_tasks = specified_tasks_set - tested_set
        if missing_tasks:
            print(f"WARNING: Specified tasks not found or excluded: {', '.join(missing_tasks)}", file=sys.stderr)
    else:
        subdirs_to_test = all_subdirs

    if not subdirs_to_test:
        print("No tasks found or specified to test.")
        sys.exit(0)

    print(f"Found {len(subdirs_to_test)} task directories to check: {', '.join(subdirs_to_test)}")

    failed_tasks = []
    tested_task_classes = set() # Track tested classes to avoid double testing if registered multiple times
    # Dynamically import Task classes from subdirectories
    # Use the registration mechanism if available, otherwise fallback to discovery
    try:
        # Try importing the central registration point
        from AlgoTuneTasks import TASK_REGISTRY # Assumes AlgoTuneTasks/__init__.py populates this
        print(f"Using central TASK_REGISTRY with {len(TASK_REGISTRY)} entries.")
        task_classes_to_test = {}
        for task_name, task_class in TASK_REGISTRY.items():
            # Derive expected directory from task_name
            if task_name in subdirs_to_test:
                # Check if already added (could happen with multiple files registering same name)
                if task_class not in tested_task_classes:
                    task_classes_to_test[task_name] = task_class
                else:
                    print(f"  Skipping duplicate registration for task class: {task_class.__name__} (name: {task_name})")
            else:
                # print(f"  Skipping registered task '{task_name}' as its directory is not selected for testing.")
                pass # Silently skip if not in subdirs_to_test
        # Identify selected subdirs for which no registered task was found
        registered_dirs = set(task_classes_to_test.keys())
        missing_registrations = set(subdirs_to_test) - registered_dirs
        if missing_registrations:
            # Treat missing registrations as errors
            error_msg = f"ERROR: No tasks found in TASK_REGISTRY for selected directories: {', '.join(missing_registrations)}. Ensure task files exist and are imported in AlgoTuneTasks/__init__.py."
            print(error_msg, file=sys.stderr)
            for missing_task_dir in missing_registrations:
                 failed_tasks.append(f"{missing_task_dir} (Not Registered)")
            # Do not proceed to fallback if registry was found but incomplete for selected tasks
    except (ImportError, AttributeError):
        # Fallback: Manually find and import Task subclasses if registry fails
        print("WARNING: TASK_REGISTRY not found or accessible. Falling back to manual import...")
        task_classes_to_test = {}
        for subdir in subdirs_to_test:
            module_path_base = os.path.join(tasks_dir, subdir)
            print(f"  Searching in: {module_path_base}")
            found_in_subdir = False
            for filename in os.listdir(module_path_base):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = f"AlgoTuneTasks.{subdir}.{filename[:-3]}"
                    try:
                        module = importlib.import_module(module_name)
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            # Check inheritance from 'Task' (needs careful import)
                            try:
                                from AlgoTuneTasks.base import Task # Assuming base class location
                                if issubclass(obj, Task) and obj is not Task and obj.__module__ == module_name:
                                    # Basic check: does class name match module/file pattern? Often yes.
                                    # Using subdir name as the assumed task name for mapping
                                    if obj not in tested_task_classes:
                                        print(f"    Found Task subclass: {obj.__name__} in {module_name}")
                                        task_classes_to_test[subdir] = obj # Map dir name to class
                                        found_in_subdir = True
                                        break # Take first Task subclass found in the file
                            except ImportError:
                                print("ERROR: Could not import AlgoTuneTasks.base.Task to verify inheritance.", file=sys.stderr)
                                sys.exit(1)
                    except ImportError as e:
                        print(f"    ERROR importing module {module_name}: {e}")
                        failed_tasks.append(f"{subdir} (Import Error: {e})")
                    except Exception as e:
                        print(f"    ERROR processing module {module_name}: {e}")
                        failed_tasks.append(f"{subdir} (Processing Error: {e})")
                        traceback.print_exc(limit=1)
            if not found_in_subdir:
                # Treat missing Task class in fallback mode as an error
                error_msg = f"ERROR: No Task subclass found directly in modules within {subdir}. Ensure the task's Python file exists and contains a Task subclass."
                print(error_msg, file=sys.stderr)
                failed_tasks.append(f"{subdir} (No Task Class Found)")
    # --- Run Tests ---
    print(f"\nRunning checks on {len(task_classes_to_test)} task classes...")
    for task_name, task_class in task_classes_to_test.items():
        if task_class in tested_task_classes:
            continue # Avoid re-testing the same class object
        try:
            test_task_class(task_class) # Pass class object directly
            tested_task_classes.add(task_class)
        except AssertionError:
            # Failure message printed inside test_task_class
            failed_tasks.append(f"{task_class.__name__} ({task_name})")
            tested_task_classes.add(task_class) # Add even if failed to avoid re-testing
        except Exception as e:
            # Catch unexpected errors during the test function itself
            print(f"\n--- Task {task_class.__name__} ({task_name}) CRASHED --- ")
            print(f"    Unexpected Error: {type(e).__name__}: {e}")
            traceback.print_exc() # Print full traceback for crashes
            failed_tasks.append(f"{task_class.__name__} ({task_name} - CRASHED)")
            tested_task_classes.add(task_class)
    # --- Final Summary ---
    end_time_main = time.time() # Use a different name
    print("\n" + "="*40)
    print(f"Testing Finished in {end_time_main - start_time_main:.2f} seconds.")
    if failed_tasks:
        print(f"\nSummary: {len(failed_tasks)} Task(s) FAILED signature/description checks:")
        for task in failed_tasks:
            print(f"  - {task}")
        sys.exit(1)
    else:
        print(f"\nSummary: All {len(tested_task_classes)} tested tasks PASSED signature/description checks.")
        sys.exit(0)


# Guard execution
if __name__ == "__main__":
    # Need to import time here if not already imported globally
    import time
    main()
