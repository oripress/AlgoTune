"""
Module reloading and task loading for the evaluator.
"""

import os
import sys
import logging
import importlib
import networkx as nx
import pkgutil
import time
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from multiprocessing import Manager
from functools import partial
import traceback
from AlgoTuneTasks.base import TASK_REGISTRY

# --- Import and run task discovery ---
from AlgoTuner.utils.discover_and_list_tasks import discover_and_import_tasks
try:
    discover_and_import_tasks()
    logging.info("utils.evaluator.loader: Successfully discovered and imported tasks.")
except Exception as e:
    logging.warning(f"utils.evaluator.loader: Task auto-discovery failed: {e}")
# --- End task discovery ---


# Directory where LLM-generated code is stored
CODE_DIR = os.environ.get("CODE_DIR", "llm_src")

# Ensure CODE_DIR is in sys.path
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def load_task(task_name, data_dir: Optional[str] = None):
    """
    Load a task instance dynamically using TaskFactory.
    This will import and register the task if needed.
    """
    from AlgoTuneTasks.factory import TaskFactory
    try:
        # TaskFactory handles import, registration, and instantiation
        instance = TaskFactory(task_name, data_dir=data_dir)
        return instance
    except Exception as e:
        # Raise a consistent ValueError if import or registration fails
        raise ValueError(f"Task '{task_name}' is not registered.") from e


def reload_all_llm_src(code_dir: str) -> None:
    """Reloads all modules in the llm_src directory and the solver module.
    
    Args:
        code_dir: The path to the code directory containing solver.py.
    """
    logging.info("Attempting to reload all LLM source modules...")
    
    try:
        # Temporarily add code_dir to sys.path so imports resolve
        sys.path.insert(0, code_dir)
        code_dir_path = os.path.abspath(code_dir)
        reloaded_count = 0
        # Reload any modules whose file lives under CODE_DIR
        for mod_name, mod in list(sys.modules.items()):
            mod_file = getattr(mod, "__file__", None)
            if mod_file and os.path.abspath(mod_file).startswith(code_dir_path):
                try:
                    importlib.reload(mod)
                    logging.debug(f"Reloaded LLM module: {mod_name}")
                    reloaded_count += 1
                except Exception as e:
                    logging.warning(f"Could not reload LLM module {mod_name}: {e}")
        logging.info(f"Reloaded {reloaded_count} modules from CODE_DIR '{code_dir_path}'")
        # Finally import or reload the solver entrypoint itself
        try:
            if 'solver' in sys.modules:
                importlib.reload(sys.modules['solver'])
                logging.info("Reloaded solver module 'solver'")
            else:
                importlib.import_module('solver')
                logging.info("Imported solver module 'solver'")
        except Exception as e:
            logging.warning(f"Failed to import/reload solver module: {e}")
    finally:
        # Ensure the path is removed even if errors occur
        if code_dir in sys.path:
            sys.path.remove(code_dir)


class EvaluationTimeoutError(Exception):
    """Custom exception for evaluation timeouts."""
    pass 