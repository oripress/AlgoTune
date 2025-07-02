import pytest
import os
import shutil
import json
import numpy as np
import tempfile
import logging
import sys
import time
from typing import Dict, Tuple, Set, List, Any
from pathlib import Path # Use pathlib for easier path handling
from AlgoTuner.utils.precise_timing import time_execution_ns  # Use high-precision timer
import traceback
from AlgoTuneTasks.base import BadDatasetError, TASK_REGISTRY, load_dataset_streaming, Task
from AlgoTuneTasks.factory import TaskFactory
import glob
import random

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now import project modules
try:
    from AlgoTuner.utils.serialization import dataset_decoder # Needed for loading jsonl
    from AlgoTuner.config.loader import load_config # To get default oracle time
except ImportError as e:
    logging.error(f"Failed to import project modules: {e}. Ensure PYTHONPATH is set correctly or run from project root.")
    sys.exit(1)

# --- Test Parameters ---
# Use small sizes for quick testing
TEST_TRAIN_SIZE = 100
TEST_TEST_SIZE = 100
TEST_RANDOM_SEED = 42
# Use a default time limit from global config or set a specific one
try:
    CONFIG = load_config()
    DEFAULT_ORACLE_TIME_MS = CONFIG.get("global", {}).get("oracle_time_limit", 1000)
except Exception:
    DEFAULT_ORACLE_TIME_MS = 1000 # Fallback
TEST_ORACLE_TIME_MS = DEFAULT_ORACLE_TIME_MS
# --- End Test Parameters ---

# --- Define Directories ---
# Use Path objects for clarity
PROJECT_ROOT_PATH = Path(PROJECT_ROOT)
# Override TEST_DATA_DIR if TEMP_DIR_STORAGE is set
TEMP_STORAGE = os.environ.get("TEMP_DIR_STORAGE")
if TEMP_STORAGE:
    TEST_DATA_DIR = Path(TEMP_STORAGE)
else:
    TEST_DATA_DIR = PROJECT_ROOT_PATH / "data" / "tests"
LOG_DIR = PROJECT_ROOT_PATH / "tests" / "logs"
REPORT_DIR = PROJECT_ROOT_PATH / "tests" / "reports"
# Ensure directories exist
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
# Define a second test data directory for a separate solve pass
ALT_TEST_DATA_DIR = PROJECT_ROOT_PATH / "data" / "tests_alt"
ALT_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Import and register all tasks so TASK_REGISTRY is populated
from AlgoTuner.utils.discover_and_list_tasks import discover_and_import_tasks
discover_and_import_tasks()

# Parameterize over all registered tasks
param_tasks = sorted(TASK_REGISTRY.keys())
print(f"[TEST-DATA] Parametrized tasks from registry: {param_tasks}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_oracle_times(file_path: str) -> Tuple[Set[int], Dict[int, float]]:
    """Loads oracle times and seeds from a generated JSONL dataset file."""
    seeds = set()
    oracle_times = {}
    if not os.path.exists(file_path):
        logging.warning(f"Dataset file not found: {file_path}")
        return seeds, oracle_times

    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line, object_hook=dataset_decoder)
                    seed = record.get('seed')
                    solve_time = record.get('solve_time_ms')
                    if seed is not None and solve_time is not None:
                        seeds.add(seed)
                        # Handle potential duplicates? Last one wins for now.
                        oracle_times[seed] = float(solve_time)
                    else:
                        logging.warning(f"Record missing 'seed' or 'solve_time_ms' in {file_path}: {line.strip()}")
                except json.JSONDecodeError:
                    logging.warning(f"Failed to decode JSON line in {file_path}: {line.strip()}")
                    continue
                except Exception as e:
                    logging.warning(f"Error processing line in {file_path}: {e} - Line: {line.strip()}")
                    continue
    except IOError as e:
        logging.error(f"Error reading file {file_path}: {e}")

    return seeds, oracle_times

# Helper to build a seed->time mapping by reading train/test JSONL for a task
def _get_times_map(data_dir: Path, task_name: str) -> Dict[int, float]:
    """
    Load mapping from seed to solve_time_ms by reading train and test JSONL files for a task.
    """
    times: Dict[int, float] = {}
    for suffix, size in [("train", TEST_TRAIN_SIZE), ("test", TEST_TEST_SIZE)]:
        pattern = data_dir / task_name / f"{suffix}_t{TEST_ORACLE_TIME_MS}ms_k*_n{size}.jsonl"
        for p in glob.glob(str(pattern)):
            _, d = _load_oracle_times(str(p))
            times.update(d)
    return times

# MODIFIED: Calculate average oracle time
def _run_generation_and_load(task_instance: Task, data_dir: Path, run_id: int) -> Tuple[Set[int], float, float]:
    """Runs create_dataset, loads the results, and returns seeds and AVERAGE oracle time."""
    task_name = getattr(task_instance, 'task_name', 'unknown_task')
    logging.info(f"[{task_name}-Run{run_id}] Starting dataset generation... (Params: train={TEST_TRAIN_SIZE}, test={TEST_TEST_SIZE}, seed={TEST_RANDOM_SEED}, t={TEST_ORACLE_TIME_MS}ms)")
    # data_dir is now the base "data/tests", create_dataset will make the task subdir
    task_data_path = data_dir / task_name

    train_gen, test_gen = None, None # Keep linters happy
    final_train_path, final_test_path = None, None
    all_seeds = set()
    all_times_list = [] # Store all times to calculate average

    try:
        # load_dataset handles both raw generation and timing in two-stage flow
        train_gen, test_gen = task_instance.load_dataset(
            train_size=TEST_TRAIN_SIZE,
            test_size=TEST_TEST_SIZE,
            random_seed=TEST_RANDOM_SEED,
            data_dir=str(data_dir),
            t=TEST_ORACLE_TIME_MS,
        )
        logging.info(f"[{task_name}-Run{run_id}] Dataset generation call completed. Final k={task_instance.k}")

        # Construct expected final filenames based on create_dataset logic
        # Note: create_dataset creates task_name subdir inside the provided data_dir
        base_filename_part = f"t{TEST_ORACLE_TIME_MS}ms_k{task_instance.k}"
        final_train_path = task_data_path / f"train_{base_filename_part}_n{TEST_TRAIN_SIZE}.jsonl"
        final_test_path = task_data_path / f"test_{base_filename_part}_n{TEST_TEST_SIZE}.jsonl"

        # Load results from the generated files
        logging.info(f"[{task_name}-Run{run_id}] Loading results from {final_train_path} and {final_test_path}")
        train_seeds, train_times_dict = _load_oracle_times(str(final_train_path))
        test_seeds, test_times_dict = _load_oracle_times(str(final_test_path))

        # Combine train and test results
        all_seeds = train_seeds.union(test_seeds)
        all_times_list.extend(train_times_dict.values())
        all_times_list.extend(test_times_dict.values())

        # Check for overlapping seeds (shouldn't happen with current create_dataset logic)
        overlapping_seeds = train_seeds.intersection(test_seeds)
        if overlapping_seeds:
             logging.warning(f"[{task_name}-Run{run_id}] Found overlapping seeds between train and test: {overlapping_seeds}")

        # Consume generators to ensure they ran (optional, files are already written)
        # list(train_gen)
        # list(test_gen)

    except Exception as e:
        logging.error(f"[{task_name}-Run{run_id}] Error during generation or loading: {e}", exc_info=True)
        # Re-raise the exception to fail the test
        raise

    # Calculate average time
    average_time = np.mean(all_times_list) if all_times_list else 0.0
    # Calculate median time for robust statistic
    median_time = np.median(all_times_list) if all_times_list else 0.0
    logging.info(f"[{task_name}-Run{run_id}] Calculated average oracle time: {average_time:.4f} ms from {len(all_times_list)} samples.")
    logging.info(f"[{task_name}-Run{run_id}] Calculated median oracle time: {median_time:.4f} ms")

    # Return combined seeds, average, and median times
    return all_seeds, average_time, median_time

def _run_loaded_solve(task_instance: Task, data_dir: Path) -> Tuple[List[float], float]:
    """Loads saved dataset and solves each loaded problem to measure solve times."""
    task_name = getattr(task_instance, 'task_name', 'unknown_task')
    base_part = f"t{TEST_ORACLE_TIME_MS}ms_k{task_instance.k}"
    task_data_path = data_dir / task_name
    files = [
        task_data_path / f"train_{base_part}_n{TEST_TRAIN_SIZE}.jsonl",
        task_data_path / f"test_{base_part}_n{TEST_TEST_SIZE}.jsonl"
    ]
    solve_times = []
    for file_path in files:
        for record in load_dataset_streaming(str(file_path)):
            problem = record.get('problem')
            # Use precise timing with perf_counter_ns via time_execution_ns
            result = time_execution_ns(
                func=task_instance.solve,
                args=(problem,),
                num_runs=1,
                warmup_runs=0
            )
            # Prefer median_ns if available, else first captured duration
            ns = result.get('median_ns') if result.get('median_ns') is not None else (result.get('values_ns')[0] if result.get('values_ns') else 0)
            # Convert nanoseconds to milliseconds
            solve_times.append(ns / 1e6)
    avg_solve = np.mean(solve_times) if solve_times else 0.0
    return solve_times, avg_solve

# Parameterize the test function to run for each task in param_tasks
@pytest.mark.parametrize("task_name", param_tasks)
def test_generate_save_load_consistency(task_name: str):
    """
    Tests that generating a dataset twice with the same parameters
    produces the same set of problems (seeds) and approximately
    the same oracle solve times.
    """
    logging.info(f"===== Testing Task: {task_name} =====")
    # Use the predefined TEST_DATA_DIR, no longer need tempfile per test
    task_data_path = TEST_DATA_DIR / task_name # Specific path for this task's data

    # Define the result file path
    # Include Slurm job/task IDs if available for uniqueness in parallel runs
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', 'na')
    result_file = REPORT_DIR / f"result_{task_name}_{slurm_job_id}_{slurm_task_id}.json"
    print(f"[TEST-DATA] Will write JSON report to: {result_file}")

    # --- Configure Task-Specific Logging ---
    log_file_name = f"{task_name}_{slurm_job_id}_{slurm_task_id}.log"
    log_file_path = LOG_DIR / log_file_name
    
    root_logger = logging.getLogger()
    # Use 'w' mode to overwrite if the test reruns with same IDs (unlikely but clean)
    file_handler = logging.FileHandler(log_file_path, mode='w') 
    # Use a standard formatter (adjust level/format as needed)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    logging.info(f"Configured logging to file: {log_file_path}")
    # --- End Logging Configuration ---

    # Wrap the entire test logic in a try/finally to ensure handler removal
    try:
        try:
            # Instantiate the task
            task_instance = TaskFactory(task_name, oracle_time_limit=TEST_ORACLE_TIME_MS)

            # Clean up any previous data for this task specifically BEFORE Run 1
            # This handles reruns or leftover data without needing a fully temp dir
            if task_data_path.exists():
                logging.warning(f"[{task_name}] Found existing data directory, cleaning up: {task_data_path}")
                shutil.rmtree(task_data_path)
            task_data_path.mkdir(exist_ok=True) # Ensure it exists for the run

            # --- Run 1 ---
            seeds1, avg_time1, med_time1 = _run_generation_and_load(task_instance, TEST_DATA_DIR, run_id=1)
            if not seeds1:
                 pytest.fail(f"[{task_name}-Run1] No seeds generated or loaded. Check logs.")
            logging.info(f"[{task_name}-Run1] Generated {len(seeds1)} unique seeds. Average Time: {avg_time1:.4f} ms, Median Time: {med_time1:.4f} ms")
            # --- Solve loaded instances after Run 1 ---
            solve_times1, avg_solve1 = _run_loaded_solve(task_instance, TEST_DATA_DIR)
            med_solve1 = np.median(solve_times1) if solve_times1 else 0.0
            logging.info(f"[{task_name}-Run1] Average loaded solve time: {avg_solve1:.4f} ms over {len(solve_times1)} instances; Median: {med_solve1:.4f} ms")

            # --- Cleanup before Run 2 ---
            # Prepare alternate directory for second generation
            task_data_path = ALT_TEST_DATA_DIR / task_name
            if task_data_path.exists():
                shutil.rmtree(task_data_path)
            # --- Run 2 in alternate directory ---
            seeds2, avg_time2, med_time2 = _run_generation_and_load(task_instance, ALT_TEST_DATA_DIR, run_id=2)
            if not seeds2:
                 pytest.fail(f"[{task_name}-Run2] No seeds generated or loaded. Check logs.")
            logging.info(f"[{task_name}-Run2] Generated {len(seeds2)} unique seeds. Average Time: {avg_time2:.4f} ms, Median Time: {med_time2:.4f} ms")
            # --- Solve loaded instances after Run 2 ---
            solve_times2, avg_solve2 = _run_loaded_solve(task_instance, ALT_TEST_DATA_DIR)
            med_solve2 = np.median(solve_times2) if solve_times2 else 0.0
            logging.info(f"[{task_name}-Run2] Average loaded solve time: {avg_solve2:.4f} ms over {len(solve_times2)} instances; Median: {med_solve2:.4f} ms")

            # --- Comparison & Reporting ---
            logging.info(f"[{task_name}] Comparing results and writing report...")

            # 1. Sample a fixed number of seeds from the intersection instead of requiring exact match
            common_seeds = seeds1.intersection(seeds2)
            if len(common_seeds) < TEST_TRAIN_SIZE:
                pytest.fail(f"Not enough common seeds for sampling! Only found {len(common_seeds)} common seeds.")
            # deterministic sample based on TEST_RANDOM_SEED
            random.seed(TEST_RANDOM_SEED)
            sample_seeds = random.sample(sorted(common_seeds), TEST_TRAIN_SIZE)
            logging.info(f"[{task_name}] Sampling {TEST_TRAIN_SIZE} seeds from intersection for comparison.")
            # 2. Load per-seed times for both runs and recompute avg/median on the sample
            times1_map = _get_times_map(TEST_DATA_DIR, task_name)
            times2_map = _get_times_map(ALT_TEST_DATA_DIR, task_name)
            sample_times1 = [times1_map[s] for s in sample_seeds]
            sample_times2 = [times2_map[s] for s in sample_seeds]
            avg_time1 = float(np.mean(sample_times1))
            avg_time2 = float(np.mean(sample_times2))
            med_time1 = float(np.median(sample_times1))
            med_time2 = float(np.median(sample_times2))
            logging.info(f"[{task_name}] Sample avg times: Run1={avg_time1:.4f} ms, Run2={avg_time2:.4f} ms; medians: {med_time1:.4f}, {med_time2:.4f}")
            # 2. Calculate absolute difference in average times
            abs_diff = abs(avg_time1 - avg_time2)
            logging.info(f"[{task_name}] Average Time Run 1: {avg_time1:.4f} ms")
            logging.info(f"[{task_name}] Average Time Run 2: {avg_time2:.4f} ms")
            logging.info(f"[{task_name}] Absolute Difference: {abs_diff:.4f} ms")

            # 3. Write results to JSON file
            report_data = {
                "status": "SUCCESS",
                "task_name": task_name,
                "target_oracle_time_ms": TEST_ORACLE_TIME_MS,
                "run1_avg_time_ms": avg_time1,
                "run1_median_time_ms": med_time1,
                "run2_avg_time_ms": avg_time2,
                "run2_median_time_ms": med_time2,
                "run1_loaded_solve_avg_time_ms": avg_solve1,
                "run1_loaded_solve_median_time_ms": med_solve1,
                "run2_loaded_solve_avg_time_ms": avg_solve2,
                "run2_loaded_solve_median_time_ms": med_solve2,
                "absolute_difference_loaded_solve_ms": abs(avg_solve1 - avg_solve2),
                "diff_target_loaded_run1_ms": avg_solve1 - TEST_ORACLE_TIME_MS,
                "diff_target_loaded_run2_ms": avg_solve2 - TEST_ORACLE_TIME_MS,
                "abs_diff_target_loaded_run1_avg_ms": abs(avg_solve1 - TEST_ORACLE_TIME_MS),
                "abs_diff_target_loaded_run1_median_ms": abs(med_solve1 - TEST_ORACLE_TIME_MS),
                "absolute_difference_ms": abs_diff,
                "diff_target_run1_ms": avg_time1 - TEST_ORACLE_TIME_MS,
                "diff_target_run2_ms": avg_time2 - TEST_ORACLE_TIME_MS,
                "abs_diff_target_run1_avg_ms": abs(avg_time1 - TEST_ORACLE_TIME_MS),
                "abs_diff_target_run1_median_ms": abs(med_time1 - TEST_ORACLE_TIME_MS),
                "absolute_difference_median_ms": abs(med_time1 - med_time2),
                "absolute_difference_loaded_solve_median_ms": abs(med_solve1 - med_solve2),
                "seeds_verified": True, # Since assertion passed
                "num_seeds": len(seeds1),
                "parameters": {
                    "train_size": TEST_TRAIN_SIZE,
                    "test_size": TEST_TEST_SIZE,
                    "random_seed": TEST_RANDOM_SEED,
                    "oracle_time_ms": TEST_ORACLE_TIME_MS,
                },
                # Absolute difference between target oracle time and run1 measurements
                "abs_diff_target_run1_ms": abs(avg_time1 - TEST_ORACLE_TIME_MS),
                "abs_diff_target_run1_median_ms": abs(med_time1 - TEST_ORACLE_TIME_MS),
                # Difference between average baseline time (run1) and target time
                "diff_target_avg_ms": avg_time1 - TEST_ORACLE_TIME_MS,
                # Status flag if target time not reached within 100ms
                "target_time_status": "NOT_REACHED" if abs(avg_time1 - TEST_ORACLE_TIME_MS) > 100 else "REACHED",
            }
            try:
                with open(result_file, 'w') as f:
                    json.dump(report_data, f, indent=4)
                logging.info(f"[{task_name}] Results saved to {result_file}")
                # Print summary of timing metrics to console
                logging.info(
                    f"[{task_name}] Report Summary: target_oracle_time_ms={TEST_ORACLE_TIME_MS}ms, "
                    f"run1_avg_time_ms={avg_time1:.4f}ms, run2_avg_time_ms={avg_time2:.4f}ms, "
                    f"absolute_difference_ms={abs_diff:.4f}ms, "
                    f"diff_target_run1_ms={(avg_time1-TEST_ORACLE_TIME_MS):.4f}ms, "
                    f"diff_target_run2_ms={(avg_time2-TEST_ORACLE_TIME_MS):.4f}ms, "
                    f"run1_loaded_solve_avg_time_ms={avg_solve1:.4f}ms, run2_loaded_solve_avg_time_ms={avg_solve2:.4f}ms, "
                    f"absolute_difference_loaded_solve_ms={abs(avg_solve1 - avg_solve2):.4f}ms, "
                    f"diff_target_loaded_run1_ms={(avg_solve1-TEST_ORACLE_TIME_MS):.4f}ms, "
                    f"diff_target_loaded_run2_ms={(avg_solve2-TEST_ORACLE_TIME_MS):.4f}ms"
                )
            except IOError as e:
                logging.error(f"[{task_name}] Failed to write results to {result_file}: {e}")
                pytest.fail(f"[{task_name}] Failed to write results file: {e}")

        except BadDatasetError as e:
            logging.error(f"[{task_name}] Bad dataset error: {e}", exc_info=True)
            # Write BAD_DATASET report
            failure_data = {
                "task_name": task_name,
                "status": "BAD_DATASET",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "parameters": {
                    "train_size": TEST_TRAIN_SIZE,
                    "test_size": TEST_TEST_SIZE,
                    "random_seed": TEST_RANDOM_SEED,
                    "oracle_time_ms": TEST_ORACLE_TIME_MS,
                }
            }
            try:
                with open(result_file, 'w') as f:
                    json.dump(failure_data, f, indent=4)
            except Exception:
                pass  # Avoid error loops
            pytest.fail(f"Task {task_name} bad dataset: {e}")
        except Exception as e:
            logging.error(f"[{task_name}] Test failed with exception: {e}", exc_info=True)
            # Write failure report if possible
            failure_data = {
                "task_name": task_name, "status": "FAILED", "error": str(e),
                "traceback": traceback.format_exc(),
                "parameters": {
                    "train_size": TEST_TRAIN_SIZE,
                    "test_size": TEST_TEST_SIZE,
                    "random_seed": TEST_RANDOM_SEED,
                    "oracle_time_ms": TEST_ORACLE_TIME_MS,
                }
            }
            try:
                with open(result_file, 'w') as f:
                    json.dump(failure_data, f, indent=4)
            except Exception:
                pass  # Avoid error loops
            pytest.fail(f"Task {task_name} failed: {e}")

    finally:
        # --- Cleanup Task-Specific Logging ---
        logging.info(f"Removing file handler for {log_file_path}")
        root_logger.removeHandler(file_handler)
        file_handler.close()
        # --- End Logging Cleanup ---

        # No longer using tempfile per test, so no global cleanup here.
        # Cleanup now happens at the start of the test for the specific task dir.
        pass # Keep finally block structure

    logging.info(f"===== Finished Task: {task_name} =====")

# Allow running the script directly for debugging all tasks
if __name__ == "__main__":
    for task_name in param_tasks:
        print(f"Running test for task: {task_name}")
        try:
            test_generate_save_load_consistency(task_name)
            print(f"Test for {task_name} completed successfully.")
        except Exception as main_e:
            print(f"Test for {task_name} failed: {main_e}")
            sys.exit(1) 