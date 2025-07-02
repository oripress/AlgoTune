import importlib
import pkgutil
import logging
import AlgoTuneTasks
import os
from typing import Optional, Type

from AlgoTuneTasks.base import TASK_REGISTRY


def TaskFactory(task_name, **kwargs):
    """
    Lazy-import factory function to create Task instances based on a registered name.
    It searches for the task in subdirectories of the tasks package.

    :param task_name: Name of the task to instantiate.
    :param kwargs: Keyword arguments to pass to the Task subclass constructor.
                  Should include oracle_time_limit.
    :return: An instance of the requested Task subclass.
    """

    # Try to find the task in the registry (by raw name)
    task_cls = TASK_REGISTRY.get(task_name)
    # if task_cls is None:

    # If not found, try to dynamically import the module
    if task_cls is None:
        # Use raw task_name for module path construction
        module_name = f"AlgoTuneTasks.{task_name}.{task_name}"
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logging.error(f"Could not import module {module_name}: {e}")
            raise ValueError(f"Task '{task_name}' could not be imported (tried {module_name})")
        # Try again to get the class after import, using raw name only
        task_cls = TASK_REGISTRY.get(task_name)
        if task_cls is None:
            # Pass original task_name and the module_name attempted
            raise ValueError(f"Task '{task_name}' is not registered after import (tried {module_name})")

    # Ensure target_time_ms is passed if oracle_time_limit is present in kwargs
    if 'oracle_time_limit' in kwargs and 'target_time_ms' not in kwargs:
        kwargs['target_time_ms'] = kwargs.pop('oracle_time_limit')
        logging.info(f"TaskFactory: Mapped oracle_time_limit to target_time_ms: {kwargs['target_time_ms']}")

    # Create task instance
    task_instance = task_cls(**kwargs)
    setattr(task_instance, 'task_name', task_name)
    logging.info(f"TaskFactory: Set task_instance.task_name to raw '{task_name}'")
    return task_instance