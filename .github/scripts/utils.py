"""Utility functions, esp. for converting task names to the corresponding Python
class."""

import importlib
import inspect
import logging
import os
import sys

# Ensure AlgoTuneTasks.base can be found by adding repo root to path
# This assumes the utils script is in .github/scripts/
current_utils_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_utils_dir, '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
tasks_dir = os.path.join(repo_root, 'AlgoTuneTasks')

from AlgoTuneTasks.base import Task

def find_task_class(task_name: str) -> Task:
    """
    Finds the Task subclass corresponding to the task_name.
    Imports based on directory structure (AlgoTuneTasks/<task_name>/<task_name>.py) with
    importlib and searches for the corresponding class.
    Args:
        task_name (str): The name of the task (should match directory name).
    Returns:
        The Task subclass if found, otherwise raises an Exception.
    """
    logging.debug(f"Searching for task '{task_name}' in directory: {tasks_dir}")

    # Construct the full module name expected by importlib
    mod_name = f"AlgoTuneTasks.{task_name}.{task_name}"
    module = importlib.import_module(mod_name)
    found_classes = [obj for _, obj in inspect.getmembers(module, inspect.isclass) if issubclass(obj, Task) and obj is not Task and obj.__module__ == mod_name]
    assert len(found_classes) == 1, "Found more than one Task-ish looking class"
    return found_classes[0]
