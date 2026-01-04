import subprocess
import json
import subprocess
import json
import sys
import logging
import argparse
from pathlib import Path

# --- Configuration ---
THRESHOLD_SECONDS = 0.05
TIMEOUT_SECONDS = 30
MAX_N_POWER = 25 # Stop searching if n reaches 2**25
# --- End Configuration ---

logging.basicConfig(level=logging.DEBUG)

project_root = Path(__file__).resolve().parent.parent.parent

def find_tasks(tasks_dir: Path) -> list[str]:
    """
    Scans the tasks directory and returns a list of task names (directory names).
    """
    assert tasks_dir.is_dir()
    return sorted([item.name for item in tasks_dir.iterdir() if item.is_dir() and not item.name.startswith(('.', '_'))])

def run_single_task_instance(task_name: str, n: int, run_task_script_path: Path, seed: int | None = None) -> tuple[float | None, bool | None, str, str | None, str | None]:
    """
    Runs subprocess for a given task, n, and optional seed.
    Returns duration, validity status, status string, stdout, and stderr.
    """
    stdout, stderr = None, None
    try:
        process_result = subprocess.run(
            [
                sys.executable,
                str(run_task_script_path),
                task_name,
                str(n), # Pass n as a string argument
                f"--seed={seed if seed is not None else hash(n)}"
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=TIMEOUT_SECONDS
        )
        stdout = process_result.stdout
        stderr = process_result.stderr
        logging.info(f"Task {task_name} (n={n})")
    except subprocess.TimeoutExpired:
        logging.warning(f"Task {task_name} (n={n})")
        return None, None, "timeout", stdout, stderr

    if process_result.returncode != 0:
        logging.debug(f"exit code: {process_result.returncode}")
        logging.debug(f"Stdout:\n{stdout}")
        logging.debug(f"Stderr:\n{stderr}")
        # Return the captured output even on failure
        return None, None, f"subprocess_error (code {process_result.returncode})", stdout, stderr

    try:
        # Take the last line of stdout to work around https://github.com/cvxpy/cvxpy/pull/2798.
        output_data = json.loads(stdout.strip().splitlines()[-1])
        duration = output_data["solve_duration_seconds"]
        is_valid = output_data["is_valid"]
        assert isinstance(duration, float)
        assert isinstance(is_valid, bool)
        return duration, is_valid, "ok", stdout, stderr
    except (json.JSONDecodeError, KeyError, IndexError, AssertionError) as e:
        logging.error(f"Error parsing output for task {task_name} (n={n}): {e}")
        logging.debug(f"Stdout:\n{stdout}")
        logging.debug(f"Stderr:\n{stderr}")
        return None, None, "output_parse_error", stdout, stderr


def binary_search_for_threshold(task_name: str, n_low: int, n_high: int, run_task_script_path: Path) -> tuple[int | None, float | None, bool | None, str]:
    """
    Performs binary search between n_low and n_high to find the first n
    that exceeds THRESHOLD_SECONDS without timing out or validation error.
    Returns: (found_n, found_duration, found_validity, status_string)
    """
    logging.info(f"Timeout at n={n_high}. Starting binary search in range [{n_low + 1}, {n_high - 1}]...")
    low = n_low + 1
    high = n_high - 1

    if low > high:
        logging.warning(f"  Binary search range is empty ({low} > {high}). Cannot refine timeout at n={n_high}.")
        return None, None, None, f"Timeout at n={n_high}, empty binary search range"

    while low <= high:
        mid = low + (high - low) // 2
        if mid == n_low: # Avoid re-testing n_low
            low = mid + 1
            continue
        if mid == n_high: # Avoid re-testing n_high
            high = mid - 1
            continue

        logging.info(f"  Binary search testing n = {mid}...")
        # Use the same seed logic (n) as the main loop for consistency during search
        # We don't need stdout/stderr from binary search runs for now
        duration, is_valid, status, _, _ = run_single_task_instance(task_name, mid, run_task_script_path, seed=mid)
        # Allow for errors during binary search as well
        # assert status in {"ok", "timeout"} # Relax this assertion

        if status == "ok":
            logging.info(f"    -> Solve time: {duration:.6f} seconds. Valid: {is_valid}")
            if not is_valid:
                # Treat invalid solution during search as an error for this n
                logging.error(f"    -> Invalid solution found at n={mid}. Stopping binary search for {task_name}.")
                return None, None, None, f"Invalid solution during binary search at n={mid}"
            elif duration > THRESHOLD_SECONDS:
                # Found a valid n > threshold, return immediately
                logging.info(f"  Binary search complete. Found valid n={mid} > threshold.")
                return mid, duration, is_valid, "Binary search found threshold"
            else:
                # Valid solution below threshold, need to search higher values
                low = mid + 1
        else:
            # Timeout occurred at mid, try lower values
            logging.warning(f"    -> Timeout at n={mid}. Reducing search range.")
            high = mid - 1 # Timeout means we must try lower values

    # If the loop finishes without returning, no suitable n was found
    logging.error(f"  Binary search complete. No n in range [{n_low + 1}, {high}] found meeting threshold > {THRESHOLD_SECONDS}s and < {TIMEOUT_SECONDS}s timeout.")
    # Return a more descriptive status message for the summary
    return None, None, None, f"No suitable n found in range [{n_low + 1}, {high}] (Timeout at n={n_high}, threshold={THRESHOLD_SECONDS}s, timeout={TIMEOUT_SECONDS}s)"


def main():
    # tasks_dir is relative to the project root
    tasks_dir = project_root / "AlgoTuneTasks"
    # run_task.py is in the same directory as this script
    run_task_script_path = Path(__file__).resolve().parent / "run_task.py"
    assert run_task_script_path.exists()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run timing tests on HardProblemBench tasks to find the 'n' value exceeding a threshold.")
    parser.add_argument(
        "task_list_str",
        nargs='?', # Optional positional argument
        type=str,
        default=None, # Default to None if not provided
        help="Optional comma-separated list of specific task names to run (e.g., 'task1,task2'). If not provided, runs all found tasks."
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # Split the comma-separated string into a list of task names
    task_names = [name.strip() for name in args.task_list_str.split(',') if name.strip()] if args.task_list_str else find_tasks(tasks_dir)
    assert len(task_names) > 0, "no tasks found"

    results = {} # {task_name: result_string}

    for task_name in task_names:
        logging.info(f"\n--- Processing Task: {task_name} ---")
        # Initialize variables for the search and validation for this task
        threshold_n = None
        threshold_duration = None
        threshold_valid = None # Validity at the point threshold was met
        final_status_msg = "Threshold not met" # Default status
        found_threshold_or_error = False # Flag if we found n > threshold OR hit an error/timeout
        validation_passed = True # Assume validation passes unless proven otherwise

        # --- Exponential Search ---
        n = 1
        n_power = 0
        last_successful_n = 0
        # last_successful_valid = None # Not strictly needed here, checked inline

        while n_power <= MAX_N_POWER:
            current_n = n
            logging.info(f"Testing {task_name} with n = {current_n} (2^{n_power})...")
            # Use current_n as seed during search for consistency
            # Capture stdout/stderr here in case of immediate failure
            duration, is_valid, status, search_stdout, search_stderr = run_single_task_instance(task_name, current_n, run_task_script_path, seed=current_n)

            if status == "ok":
                logging.info(f"  -> Solve time: {duration:.6f} seconds. Valid: {is_valid}")
                last_successful_n = current_n
                # last_successful_valid = is_valid # Store if needed later

                if not is_valid:
                     # Treat invalid solution as an error for this n
                     final_status_msg = f"Error at n={current_n} (Invalid Solution)"
                     logging.error(f"    -> Invalid solution found during initial search at n={current_n}.")
                     # Log the output from the failed run
                     logging.debug(f"--- Subprocess Output (n={current_n}, seed={current_n}) ---")
                     logging.debug(f"Stdout:\n{search_stdout}")
                     logging.debug(f"Stderr:\n{search_stderr}")
                     logging.debug(f"--- End Subprocess Output ---")
                     found_threshold_or_error = True
                     break # Stop search

                if duration > THRESHOLD_SECONDS:
                    # Found potential threshold, store details and stop search
                    threshold_n = current_n
                    threshold_duration = duration
                    threshold_valid = is_valid # Should be True here
                    final_status_msg = f"n = {threshold_n} ({threshold_duration:.6f}s)"
                    found_threshold_or_error = True
                    break # Stop exponential search

                # If duration is below threshold and valid, continue exponential search
                n *= 2
                n_power += 1

            else:
                # Error occurred (timeout, subprocess error, parse error, etc.)
                logging.debug(f"DEBUG: Error condition entered. status='{status}', last_successful_n={last_successful_n}")
                if "timeout" in status.lower() and last_successful_n > 0: # Check if we had a prior success
                    # Perform binary search if timeout occurred after a successful run
                    # Note: We don't strictly need last_successful_duration <= THRESHOLD_SECONDS check here,
                    # because if it was > threshold, the main loop would have broken already.
                    bs_n, bs_duration, bs_valid, bs_status = binary_search_for_threshold(
                        task_name, last_successful_n, current_n, run_task_script_path
                    )
                    if bs_n is not None:
                        # Found threshold via binary search
                        threshold_n = bs_n
                        threshold_duration = bs_duration
                        threshold_valid = bs_valid # Should be True if bs_n is not None
                        final_status_msg = f"n = {threshold_n} ({threshold_duration:.6f}s) [BS]"
                    else:
                        # Binary search didn't find a value, report original timeout/error range
                        final_status_msg = f"Error between n={last_successful_n} and n={current_n} ({bs_status})"
                else:
                    # Error on first attempt or non-timeout error
                    final_status_msg = f"Error at n={current_n} ({status})"

                found_threshold_or_error = True # Treat error/timeout as finding a limit
                break # Stop searching for this task

        if not found_threshold_or_error:
            # Loop finished without exceeding threshold or erroring
            final_status_msg = f"Threshold not met (max n={last_successful_n})"

        # --- Validation Step ---
        if threshold_n is not None and threshold_valid is True:
            logging.info(f"--- Running Validation for {task_name} at n={threshold_n} ---")
            # validation_passed flag is already initialized before the search loop
            for i in range(3):
                # Generate a unique seed for each validation run, different from search seeds
                validation_seed = threshold_n + 1000 + i
                logging.info(f"  Validation run {i+1}/3 with seed={validation_seed}...")

                # Run the task instance for validation, capture output
                val_duration, val_is_valid, val_status, val_stdout, val_stderr = run_single_task_instance(
                    task_name, threshold_n, run_task_script_path, seed=validation_seed
                )

                # Check results of the validation run
                if val_status != "ok":
                    # Error occurred during the validation run (Timeout, Subprocess Error, Parse Error)
                    logging.error(f"    -> Validation run {i+1} FAILED (Error: {val_status})")
                    # Log the output from the failed validation run
                    logging.debug(f"--- Subprocess Output (Validation run {i+1}, seed={validation_seed}) ---")
                    logging.debug(f"Stdout:\n{val_stdout}")
                    logging.debug(f"Stderr:\n{val_stderr}")
                    logging.debug(f"--- End Subprocess Output ---")
                    final_status_msg += f" | Validation Failed [seed={validation_seed}] ({val_status})"
                    validation_passed = False
                    break # Stop validation on first error
                elif not val_is_valid:
                    # Solution was explicitly invalid according to run_task.py
                    logging.error(f"    -> Validation run {i+1} FAILED (Invalid Solution)")
                    # Log the output from the invalid validation run
                    logging.debug(f"--- Subprocess Output (Validation run {i+1}, seed={validation_seed}) ---")
                    logging.debug(f"Stdout:\n{val_stdout}")
                    logging.debug(f"Stderr:\n{val_stderr}")
                    logging.debug(f"--- End Subprocess Output ---")
                    final_status_msg += f" | Validation Failed [seed={validation_seed}] (Invalid Solution)"
                    validation_passed = False
                    break # Stop validation on first invalid solution
                else:
                    # Validation run passed
                    logging.info(f"    -> Validation run {i+1} PASSED (Duration: {val_duration:.6f}s)")

            # Append overall validation status if all runs passed
            if validation_passed:
                 final_status_msg += " | Validation Passed"
        elif threshold_n is not None and threshold_valid is False:
             # This case implies the run that met the threshold was already invalid
             final_status_msg += " | Initial run was invalid"
        # else: final_status_msg already contains Error or Threshold not met message

        results[task_name] = final_status_msg # Store the final message including validation status

    # Print summary
    print("\n--- Threshold Search Summary ---")
    print(f"Threshold: {THRESHOLD_SECONDS} seconds")
    print(f"Timeout per run: {TIMEOUT_SECONDS} seconds")
    print("-" * 30)
    max_len = max(len(name) for name in results.keys()) if results else 10
    failed_tasks = [] # List to collect names of failed tasks
    for task, result in sorted(results.items()):
        print(f"{task:<{max_len}} : {result}")
        # Check if the result indicates a failure (Error, Timeout, Threshold not met, Validation Failed, Load Error, Invalid Solution)
        is_failure = (
            "Error" in result or
            "Threshold not met" in result or
            "Timeout" in result or
            "Validation Failed" in result or
            "Load Error" in result or
            "Invalid Solution" in result
        )
        if is_failure:
             failed_tasks.append(task)
    print("--- End Summary ---")

    if failed_tasks:
        print(f"\nFailed Tasks: {','.join(failed_tasks)}")
        print("\nFAILED")
        sys.exit(1)
    else:
        print("\nSUCCESS")
        sys.exit(0)


if __name__ == "__main__":
    main()
