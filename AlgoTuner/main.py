import argparse
import json
import logging
import math
import multiprocessing
import os
import random
import sys
import time
import warnings
from pathlib import Path

import litellm

from AlgoTuner.config.loader import load_config
from AlgoTuner.config.model_config import GenericAPIModelConfig, GlobalConfig
from AlgoTuner.interfaces.llm_interface import LLMInterface
from AlgoTuner.utils.initialize_solver import initialize_solver_from_task
from AlgoTuner.utils.logger import setup_logging
from AlgoTuneTasks.factory import TaskFactory


# Ensure proper multiprocessing initialization before any imports that might use it
if __name__ == "__main__":
    # The AlgoTuner system uses forkserver for process isolation
    try:
        multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError:
        # Already set, which is fine
        pass

    # Also set NUMBA threading layer for fork safety
    if "NUMBA_THREADING_LAYER" not in os.environ:
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

warnings.filterwarnings("ignore", ".*resource_tracker.*")
warnings.filterwarnings("ignore", ".*loky.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*resource_tracker.*")


def acquire_lock(lock_file_path, timeout=60):
    """Attempts to acquire an exclusive lock using a lock file."""
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        try:
            fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            logging.info(f"Acquired lock: {lock_file_path}")
            return True
        except FileExistsError:
            sleep_time = random.uniform(0.1, 0.5)
            logging.debug(f"Lock exists, sleeping {sleep_time:.2f}s before retry...")
            time.sleep(sleep_time)
        except Exception as e:
            logging.error(f"Error acquiring lock {lock_file_path}: {e}")
            return False
    logging.error(f"Timeout acquiring lock after {timeout}s: {lock_file_path}")
    return False


def release_lock(lock_file_path):
    """Releases the lock by removing the lock file."""
    try:
        os.remove(lock_file_path)
        logging.info(f"Released lock: {lock_file_path}")
    except FileNotFoundError:
        logging.warning(f"Attempted to release lock, but file not found: {lock_file_path}")
    except Exception as e:
        logging.error(f"Error releasing lock {lock_file_path}: {e}")


def normalize_summary_model_name(model_name: str) -> str:
    if not model_name:
        return model_name
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    return model_name


def update_summary_json(
    summary_file_path_str: str,
    task_name: str,
    model_name: str,
    speedup: float | None,
    error: str | None = None,
):
    """Atomically updates the summary JSON file with the final speedup or error status."""
    summary_file_path = Path(summary_file_path_str)
    lock_file_path = summary_file_path.with_suffix(".json.lock")
    logging.info(f"Attempting to update summary file: {summary_file_path}")

    if not acquire_lock(str(lock_file_path)):
        logging.error("Failed to acquire lock, cannot update summary file.")
        return

    try:
        summary_data = {}
        if summary_file_path.exists():
            try:
                with open(summary_file_path) as f:
                    summary_data = json.load(f)
                    if not isinstance(summary_data, dict):
                        logging.warning(
                            f"Summary file {summary_file_path} did not contain a JSON object, resetting."
                        )
                        summary_data = {}
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from {summary_file_path}, resetting.")
                summary_data = {}
            except Exception as e:
                logging.error(
                    f"Error reading summary file {summary_file_path}: {e}. Proceeding with empty data."
                )
                summary_data = {}

        # Determine the speedup string based on error status
        if error is not None:
            speedup_str = "Error"
        elif speedup is None or not math.isfinite(speedup):
            speedup_str = "N/A"
        else:
            speedup_str = f"{speedup:.5f}"

        normalized_model = normalize_summary_model_name(model_name)
        task_entry = summary_data.setdefault(task_name, {})
        task_entry[normalized_model] = {"final_speedup": speedup_str}
        logging.info(
            f"Updating summary for Task: {task_name}, Model: {normalized_model} with Speedup: {speedup_str}"
        )

        try:
            with open(summary_file_path, "w") as f:
                json.dump(summary_data, f, indent=4)
            logging.info(f"Successfully updated summary file: {summary_file_path}")
        except Exception as e:
            logging.error(f"Error writing updated summary file {summary_file_path}: {e}")

    finally:
        release_lock(str(lock_file_path))


def update_failure_json(
    failure_file_path_str: str,
    task_name: str,
    model_name: str,
    reason: str,
    details: str | None = None,
):
    """Atomically records a failed run in a dedicated failure file."""
    failure_file_path = Path(failure_file_path_str)
    lock_file_path = failure_file_path.with_suffix(".json.lock")
    logging.info(f"Attempting to update failure file: {failure_file_path}")

    if not acquire_lock(str(lock_file_path)):
        logging.error("Failed to acquire lock, cannot update failure file.")
        return

    try:
        failure_data = {}
        if failure_file_path.exists():
            try:
                with open(failure_file_path) as f:
                    failure_data = json.load(f)
                    if not isinstance(failure_data, dict):
                        failure_data = {}
            except Exception:
                failure_data = {}

        normalized_model = normalize_summary_model_name(model_name)
        task_entry = failure_data.setdefault(task_name, {})
        task_entry[normalized_model] = {
            "reason": reason,
            "details": details or "",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        with open(failure_file_path, "w") as f:
            json.dump(failure_data, f, indent=4)
        logging.info(f"Successfully updated failure file: {failure_file_path}")
    finally:
        release_lock(str(lock_file_path))


def main():
    parser = argparse.ArgumentParser(description="LLM Interface Script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use (e.g., 'gpt-4o', 'human') as defined in config.yaml",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name to run (e.g., 'tsp', 'tsp_fuel') as defined in config.yaml",
    )

    args = parser.parse_args()

    task_name = args.task
    desired_model_name = args.model

    # Initialize memory monitoring for parent process using same config as workers
    memory_monitor = None
    try:
        from AlgoTuner.utils.process_monitor import init_worker_memory_monitor

        # Load memory limit from config - use evaluation_pool settings
        config = load_config()
        memory_limit_gb = (
            config.get("benchmark", {}).get("evaluation_pool", {}).get("memory_limit_per_worker")
        )

        if memory_limit_gb is not None:
            # Initialize process memory monitor (sets RLIMIT_AS)
            memory_monitor = init_worker_memory_monitor(memory_limit_gb)
            logging.info(
                f"Initialized parent process memory monitor with {memory_limit_gb}GB limit"
            )
        else:
            logging.info(
                "No memory limit configured in benchmark.evaluation_pool.memory_limit_per_worker"
            )
    except Exception as e:
        logging.warning(f"Could not initialize parent process memory monitor: {e}")

    summary_file_env = os.environ.get("SUMMARY_FILE")
    if not summary_file_env:
        logging.warning("SUMMARY_FILE environment variable not set. Cannot update summary JSON.")

    logger = setup_logging(task=task_name, model=desired_model_name)

    # Configure per-job isolated Python cache to avoid network filesystem stress
    import uuid

    cache_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity
    cache_dir = f"/tmp/pycache_{os.getpid()}_{cache_id}"
    os.environ["PYTHONPYCACHEPREFIX"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Set PYTHONPYCACHEPREFIX to {cache_dir}")

    llm_model_name = desired_model_name

    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    global_config_data = config.get("global", {})
    global_config = GlobalConfig(**global_config_data)

    if desired_model_name == "human":
        task_instance = TaskFactory(task_name, oracle_time_limit=global_config.oracle_time_limit)
        llm_interface = LLMInterface(
            model_config=None,
            global_config=None,
            model_name="human",
            task_instance=task_instance,
        )
        llm_interface.run_human_mode()
        return

    model_info = config["models"].get(desired_model_name)
    if not model_info:
        logger.critical(f"Model '{desired_model_name}' is not defined in the configuration.")
        sys.exit(1)

    budget = model_info.get("spend_limit", global_config.spend_limit)
    model_info["spend_limit"] = budget

    logger.info(f"Configuration loaded successfully. Budget: ${budget:.4f}")
    logger.info(f"Config loaded: {global_config}")

    api_key_env = model_info.get("api_key_env")
    if not api_key_env:
        logger.critical(f"Missing 'api_key_env' for model '{desired_model_name}' in config.")
        sys.exit(1)
    api_key = os.getenv(api_key_env)
    if not api_key:
        logger.critical(f"API key not found. Set the {api_key_env} environment variable.")
        sys.exit(1)

    litellm.drop_params = True
    try:
        model_info_from_litellm = litellm.get_model_info(llm_model_name)
    except Exception as e:
        logger.warning(f"Could not get model info from litellm for {llm_model_name}: {e}")
        model_info_from_litellm = {}
    # Re-enable parameters for normal completion calls
    litellm.drop_params = False

    raw_max_input_tokens = model_info_from_litellm.get("max_input_tokens")
    if isinstance(raw_max_input_tokens, int) and raw_max_input_tokens > 0:
        max_input_tokens = raw_max_input_tokens
    else:
        max_input_tokens = 65536
        logger.info(
            f"No valid max_input_tokens from litellm for model '{desired_model_name}'; defaulting context_length to {max_input_tokens}."
        )
    max_output_tokens = model_info_from_litellm.get("max_output_tokens", 4096)

    # Check for max_completion_tokens first (for models like gpt-5, gpt-5-mini)
    configured_max_completion_tokens = model_info.get("max_completion_tokens", None)
    configured_max_tokens = model_info.get("max_tokens", None)
    configured_context_length = model_info.get("context_length", None)

    config_params = {
        "name": llm_model_name,
        "api_key": api_key,
        "spend_limit": model_info.get("spend_limit", 0.0),
        "api_key_env": api_key_env,
    }

    # Use context_length for history truncation if available.
    if configured_context_length:
        config_params["context_length"] = configured_context_length
        logger.info(
            f"Using context_length from config for model '{desired_model_name}': {configured_context_length}."
        )
    elif isinstance(max_input_tokens, int) and max_input_tokens > 0:
        config_params["context_length"] = max_input_tokens
        logger.info(
            f"Using context_length from provider defaults for model '{desired_model_name}': {max_input_tokens}."
        )

    # Add the appropriate token limit parameter (output cap)
    if configured_max_completion_tokens:
        config_params["max_completion_tokens"] = configured_max_completion_tokens
        logger.info(
            f"Using max_completion_tokens from config for model '{desired_model_name}': {configured_max_completion_tokens}."
        )
    elif configured_max_tokens:
        config_params["max_tokens"] = configured_max_tokens
        logger.info(
            f"Using max_tokens from config for model '{desired_model_name}': {configured_max_tokens}."
        )
    else:
        logger.info(
            f"No max_tokens override for model '{desired_model_name}'; using provider default."
        )

    # Only set temperature if explicitly provided in config
    if "temperature" in model_info:
        config_params["temperature"] = model_info["temperature"]

    # Only set top_p if explicitly provided in config
    if "top_p" in model_info:
        config_params["top_p"] = model_info["top_p"]

    model_config = GenericAPIModelConfig(**config_params)

    default_params = model_info.get("default_params", {})
    if default_params:
        logger.info(f"Found default_params for model: {default_params}")

    logger.info(f"Passing model-specific config to LLMInterface: {model_info}")

    task_config = config.get("tasks", {}).get(task_name, {})

    oracle_time_limit = global_config.oracle_time_limit
    evaluator_time_limit = oracle_time_limit

    logger.info(
        f"Oracle time limit: {oracle_time_limit} ms, Evaluator time limit: {evaluator_time_limit} ms"
    )

    task_n = os.environ.get("TASK_N")
    task_dataset_size = os.environ.get("TASK_DATASET_SIZE")
    task_target_time_ms = os.environ.get("TASK_TARGET_TIME_MS")

    task_params_for_factory = {}
    try:
        if task_n is not None:
            task_params_for_factory["n"] = int(task_n)
        if task_dataset_size is not None:
            task_params_for_factory["dataset_size"] = int(task_dataset_size)
        if task_target_time_ms is not None:
            task_params_for_factory["target_time_ms"] = int(task_target_time_ms)
        logger.info(f"Read dataset params from env: {task_params_for_factory}")
    except ValueError as e:
        logger.error(
            f"Error converting dataset env vars to int: {e}. Check submit_agent.sh export."
        )
        task_params_for_factory = {}

    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        logger.warning(
            "DATA_DIR environment variable not set. Dataset loading might fail or use default path."
        )
    else:
        logger.info(f"Using DATA_DIR from environment: {data_dir}")

    task_instance = TaskFactory(
        task_name, oracle_time_limit=oracle_time_limit, data_dir=data_dir, **task_params_for_factory
    )

    code_dir = Path(os.environ.get("CODE_DIR", "llm_src"))
    code_dir.mkdir(exist_ok=True)
    initialize_solver_from_task(task_instance, str(code_dir))

    llm_interface = LLMInterface(
        model_config=model_config,
        global_config=global_config,
        model_name=desired_model_name,
        task_instance=task_instance,
        model_specific_config=model_info,
    )

    try:
        logger.info(
            "Starting LLM interface run_task (with final snapshot restore and test evaluation)..."
        )
        llm_interface.run_task()
        logger.info("LLM interface run_task completed successfully.")

        if summary_file_env:
            test_speedup = None
            failure_file_env = str(Path(summary_file_env).with_name("agent_failures.json"))

            if hasattr(llm_interface, "_final_eval_metrics") and llm_interface._final_eval_metrics:
                test_speedup = llm_interface._final_eval_metrics.get("mean_speedup")
                logger.info(f"Using test dataset speedup for summary: {test_speedup}")

            if test_speedup is not None and math.isfinite(test_speedup):
                logger.info(f"Recording test speedup ({test_speedup}) to summary file.")
                update_summary_json(summary_file_env, task_name, desired_model_name, test_speedup)
            else:
                final_eval_success = getattr(llm_interface, "_final_eval_success", None)
                final_eval_error = getattr(llm_interface, "_final_eval_error", None)
                reason = (
                    final_eval_error
                    or ("final_evaluation_failed" if final_eval_success is False else "missing_metrics")
                )
                logger.warning(
                    "No valid final speedup available; writing N/A to summary and recording failure."
                )
                update_summary_json(summary_file_env, task_name, desired_model_name, None)
                update_failure_json(
                    failure_file_env,
                    task_name,
                    desired_model_name,
                    reason=str(reason),
                    details=f"final_eval_success={final_eval_success}",
                )
        else:
            logger.warning("Skipping summary file update because SUMMARY_FILE env var was not set.")

    except MemoryError:
        # Handle memory limit exceeded with proper context
        logger.error(
            f"Memory limit exceeded during evaluation of task '{task_name}' with model '{desired_model_name}'"
        )

        # Try to save error information if summary file was specified
        if summary_file_env:
            try:
                # Get memory limit info for error message
                if memory_monitor and hasattr(memory_monitor, "memory_limit_gb"):
                    memory_info = f"Memory limit ({memory_monitor.memory_limit_gb}GB) exceeded"
                else:
                    memory_info = "Memory limit exceeded"

                failure_file_env = str(Path(summary_file_env).with_name("agent_failures.json"))
                update_failure_json(
                    failure_file_env,
                    task_name,
                    desired_model_name,
                    reason=memory_info,
                    details="memory_error",
                )
                logger.info("Saved memory error to failure file")
            except Exception as save_error:
                logger.error(f"Could not save error to summary: {save_error}")

        # Exit with error code
        sys.exit(137)  # Standard exit code for OOM
    except Exception as e:
        logger.exception(f"An error occurred during LLMInterface run: {e}")

        # Try to save error information if summary file was specified
        if summary_file_env:
            try:
                error_type = type(e).__name__
                error_msg = str(e)

                # Categorize the error for better reporting
                if "payment" in error_msg.lower() or "credit" in error_msg.lower() or "402" in error_msg:
                    error_info = "Payment/Credit Error"
                elif "APIError" in error_type or "RateLimitError" in error_type:
                    error_info = f"API Error ({error_type})"
                else:
                    error_info = f"Error ({error_type})"

                failure_file_env = str(Path(summary_file_env).with_name("agent_failures.json"))
                update_failure_json(
                    failure_file_env,
                    task_name,
                    desired_model_name,
                    reason=error_info,
                    details=error_msg,
                )
                logger.info(f"Saved error status to failure file: {error_info}")
            except Exception as save_error:
                logger.error(f"Could not save error to summary: {save_error}")

        sys.exit(1)
    finally:
        pass

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
