import sys
import logging
import importlib
from pathlib import Path
import traceback
import io # Added for log capturing

# Assume utils.py is in the same directory or accessible via sys.path
# If utils.py is in .github/scripts/, this should work.
from utils import find_task_class

logging.basicConfig(level=logging.INFO)

project_root = Path(__file__).resolve().parent.parent.parent
tasks_dir = project_root / "AlgoTuneTasks"

def find_task_dirs(base_dir: Path) -> list[Path]:
    """Finds all valid task directories."""
    assert base_dir.is_dir()
    return sorted([
        item for item in base_dir.iterdir()
        if item.is_dir() and not item.name.startswith(('.', '_')) and (item / f"{item.name}.py").exists()
    ])

def main():
    logging.info("Starting is_solution return type check...")
    task_dirs = find_task_dirs(tasks_dir)
    failed_tasks = []
    passed_tasks = 0

    assert len(task_dirs) > 0, f"No task directories found in {tasks_dir}"

    # Dynamically import all task modules to ensure they are registered
    for task_dir in task_dirs:
        task_name = task_dir.name
        module_name = f"AlgoTuneTasks.{task_name}.{task_name}"
        importlib.import_module(module_name)

    for task_dir in task_dirs:
        task_name = task_dir.name
        logging.info(f"--- Checking Task: {task_name} ---")

        try:
            task_instance = find_task_class(task_name)()

            # Generate a minimal problem instance (try n=2, seed=1)
            # Some tasks might require n >= 2
            n_test = 2
            seed_test = 1
            problem_input = task_instance.generate_problem(n=n_test, random_seed=seed_test)

            log_capture_string = io.StringIO()
            log_handler = logging.StreamHandler(log_capture_string)
            log_handler.setLevel(logging.ERROR) # Capture ERROR level and above
            logging.getLogger().addHandler(log_handler)

            task_failed_due_to_log = False

            # Solve the problem
            solution = task_instance.solve(problem_input)
            log_contents = log_capture_string.getvalue()
            if log_contents:
                logging.error(f"  FAILURE: Error logged during solve:\n---\n{log_contents.strip()}\n---")
                task_failed_due_to_log = True

            # Reset buffer for the next check
            log_capture_string.seek(0)
            log_capture_string.truncate(0)

            # Check the is_solution method returns a boolean True
            is_valid_result = task_instance.is_solution(problem_input, solution)
            log_contents = log_capture_string.getvalue()
            if log_contents:
                 logging.error(f"  FAILURE: Error logged during is_solution:\n---\n{log_contents.strip()}\n---")
                 task_failed_due_to_log = True

            # Remove the handler AFTER checking logs for both methods
            logging.getLogger().removeHandler(log_handler)

            if task_failed_due_to_log:
                 failed_tasks.append(task_name)
            # Can't do `== True` since eg numpy arrays override __eq__
            elif type(is_valid_result) is bool and is_valid_result:
                logging.info("  SUCCESS: is_solution returned type bool and is_solution returned True.")
                passed_tasks += 1
            else:
                logging.error(f"  FAILURE: is_solution returned type {type(is_valid_result)}, expected bool. is_valid_result={is_valid_result}, expected True.")
                failed_tasks.append(task_name)

        except Exception as e:
            # Ensure handler is removed even if exception occurs before removal
            if 'log_handler' in locals() and log_handler in logging.getLogger().handlers:
                 logging.getLogger().removeHandler(log_handler)
            logging.error(f"  ERROR: Unexpected error processing task {task_name}: {e}\n{traceback.format_exc()}")
            failed_tasks.append(task_name)

    logging.info("\n--- Check Summary ---")
    logging.info(f"Total tasks checked: {len(task_dirs)}")
    logging.info(f"Passed: {passed_tasks}")
    logging.info(f"Failed: {len(failed_tasks)}")

    if failed_tasks:
        logging.error(f"Failed tasks: {', '.join(sorted(list(set(failed_tasks))))}")
        sys.exit(1)
    else:
        logging.info("All checked tasks returned the correct bool type from is_solution.")
        sys.exit(0)

if __name__ == "__main__":
    main()
