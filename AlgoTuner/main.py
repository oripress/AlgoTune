import argparse
import os
import litellm
import sys
import logging
import warnings
from AlgoTuner.config.loader import load_config
from AlgoTuner.config.model_config import GlobalConfig, GenericAPIModelConfig
from AlgoTuner.interfaces.llm_interface import LLMInterface
from AlgoTuner.utils.logger import setup_logging
from AlgoTuneTasks.factory import TaskFactory
from AlgoTuner.utils.initialize_solver import initialize_solver_from_task
import json
import time
import random
import fcntl
import math
from pathlib import Path
from typing import Optional

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

def update_summary_json(summary_file_path_str: str, task_name: str, model_name: str, speedup: Optional[float]):
    """Atomically updates the summary JSON file with the final speedup."""
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
                with open(summary_file_path, 'r') as f:
                    summary_data = json.load(f)
                    if not isinstance(summary_data, dict):
                         logging.warning(f"Summary file {summary_file_path} did not contain a JSON object, resetting.")
                         summary_data = {}
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from {summary_file_path}, resetting.")
                summary_data = {}
            except Exception as e:
                 logging.error(f"Error reading summary file {summary_file_path}: {e}. Proceeding with empty data.")
                 summary_data = {}

        if speedup is None or not math.isfinite(speedup):
            speedup_str = "N/A"
        else:
            speedup_str = f"{speedup:.4f}"

        task_entry = summary_data.setdefault(task_name, {})
        task_entry[model_name] = {"final_speedup": speedup_str}
        logging.info(f"Updating summary for Task: {task_name}, Model: {model_name} with Speedup: {speedup_str}")

        try:
            with open(summary_file_path, 'w') as f:
                json.dump(summary_data, f, indent=4)
            logging.info(f"Successfully updated summary file: {summary_file_path}")
        except Exception as e:
            logging.error(f"Error writing updated summary file {summary_file_path}: {e}")

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

    summary_file_env = os.environ.get("SUMMARY_FILE")
    if not summary_file_env:
         logging.warning("SUMMARY_FILE environment variable not set. Cannot update summary JSON.")

    logger = setup_logging(task=task_name, model=desired_model_name)


    llm_model_name = desired_model_name

    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    global_config_data = config.get("global", {})
    global_config = GlobalConfig(**global_config_data)

    if desired_model_name == "human":
        task_instance = TaskFactory(
            task_name, oracle_time_limit=global_config.oracle_time_limit
        )
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
        logger.critical(
            f"Model '{desired_model_name}' is not defined in the configuration."
        )
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
        logger.critical(
            f"API key not found. Set the {api_key_env} environment variable."
        )
        sys.exit(1)

    litellm.drop_params = True
    try:
        model_info_from_litellm = litellm.get_model_info(llm_model_name)
    except Exception as e:
        logger.warning(f"Could not get model info from litellm for {llm_model_name}: {e}")
        model_info_from_litellm = {}
    # Re-enable parameters for normal completion calls
    litellm.drop_params = False

    max_input_tokens = model_info_from_litellm.get("max_input_tokens", 4096)
    max_output_tokens = model_info_from_litellm.get("max_output_tokens", 4096)

    # Use max_tokens from config if specified, otherwise use the model's max output tokens
    configured_max_tokens = model_info.get("max_tokens", None)
    if configured_max_tokens:
        max_tokens = configured_max_tokens
        logger.info(f"Using max_tokens from config for model '{desired_model_name}': {max_tokens}.")
    else:
        max_tokens = max_output_tokens
        logger.info(f"Using default max_output_tokens for model '{desired_model_name}': {max_tokens}.")

    model_config = GenericAPIModelConfig(
        name=llm_model_name,
        api_key=api_key,
        temperature=model_info.get("temperature", 0.0),
        top_p=model_info.get("top_p", 0.95),
        max_tokens=max_tokens,
        spend_limit=model_info.get("spend_limit", 0.0),
        api_key_env=api_key_env,
    )

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

    task_n = os.environ.get('TASK_N')
    task_dataset_size = os.environ.get('TASK_DATASET_SIZE')
    task_target_time_ms = os.environ.get('TASK_TARGET_TIME_MS')
    
    task_params_for_factory = {}
    try:
        if task_n is not None: task_params_for_factory['n'] = int(task_n)
        if task_dataset_size is not None: task_params_for_factory['dataset_size'] = int(task_dataset_size)
        if task_target_time_ms is not None: task_params_for_factory['target_time_ms'] = int(task_target_time_ms)
        logger.info(f"Read dataset params from env: {task_params_for_factory}")
    except ValueError as e:
        logger.error(f"Error converting dataset env vars to int: {e}. Check submit_agent.sh export.")
        task_params_for_factory = {}

    data_dir = os.environ.get('DATA_DIR')
    if not data_dir:
        logger.warning("DATA_DIR environment variable not set. Dataset loading might fail or use default path.")
    else:
        logger.info(f"Using DATA_DIR from environment: {data_dir}")

    task_instance = TaskFactory(
        task_name,
        oracle_time_limit=oracle_time_limit,
        data_dir=data_dir,
        **task_params_for_factory
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
        logger.info("Starting LLM interface run_task (with final snapshot restore and test evaluation)...")
        llm_interface.run_task()
        logger.info("LLM interface run_task completed successfully.")

        if summary_file_env:
             test_speedup = None
             
             if hasattr(llm_interface, '_final_eval_metrics') and llm_interface._final_eval_metrics:
                 test_speedup = llm_interface._final_eval_metrics.get('mean_speedup')
                 logger.info(f"Using test dataset speedup for summary: {test_speedup}")
             else:
                 logger.info("No test evaluation metrics available, will use N/A in summary")
             
             logger.info(f"Recording test speedup ({test_speedup}) to summary file.")
             update_summary_json(summary_file_env, task_name, desired_model_name, test_speedup)
        else:
             logger.warning("Skipping summary file update because SUMMARY_FILE env var was not set.")

    except Exception as e:
        logger.exception(f"An error occurred during LLMInterface run: {e}")
        sys.exit(1)
    finally:
        pass

    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main()
