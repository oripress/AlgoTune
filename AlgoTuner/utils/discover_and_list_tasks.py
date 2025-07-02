#!/usr/bin/env python3

"""
Helper script to discover, import, and list all registered tasks.
It imports every tasks.task_<X>.<task_name> module so that
@register_task decorators run, then prints the sorted registry keys.
"""

import os
import sys
import importlib
import logging

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import registry
from AlgoTuneTasks.base import TASK_REGISTRY

def discover_and_import_tasks():
    """Dynamically imports all task modules under AlgoTuneTasks to populate TASK_REGISTRY."""
    # --- Memory-safety guard ---------------------------------------------------
    # When running inside the lightweight evaluation agent (e.g. in the
    # AlgoTuner test harness where `AGENT_MODE` environment variable is set to
    # "1"), importing **all** task modules may pull in heavy dependencies such
    # as JAX, PyTorch, CVXPY, etc.  On memory-constrained runners this can lead
    # to the process being OOM-killed before the actual task under evaluation
    # (e.g. `sha256_hashing`) even executes.  In that context we only need the
    # specific task requested by the harness, which will be imported lazily via
    # `TaskFactory`.  Therefore, if we detect that we are running in such an
    # environment we safely *skip* the exhaustive discovery below.

    if os.environ.get("AGENT_MODE") == "1":
        logging.info(
            "discover_and_import_tasks: AGENT_MODE==1 → skipping full task discovery to save RAM; tasks will be lazily imported on demand."
        )
        return

    logging.info("discover_and_import_tasks: Starting file-based task discovery.")
    tasks_root = os.path.join(PROJECT_ROOT, "AlgoTuneTasks")
    if not os.path.isdir(tasks_root):
        logging.warning(f"discover_and_import_tasks: tasks_root not found: {tasks_root}")
        return
    for dirpath, dirnames, filenames in os.walk(tasks_root):
        # Skip any cache directories
        if '__pycache__' in dirpath:
            continue
        for fname in filenames:
            if not fname.endswith('.py') or fname in ('__init__.py', 'base.py', 'registry.py', 'factory.py', 'base_old.py'):
                continue
            file_path = os.path.join(dirpath, fname)
            # Compute module name relative to PROJECT_ROOT
            relpath = os.path.relpath(file_path, PROJECT_ROOT)
            # Convert path to module name: AlgoTuneTasks.<subdir>.<module>
            module_name = os.path.splitext(relpath)[0].replace(os.sep, '.')
            logging.debug(f"discover_and_import_tasks: Importing module {module_name} from {file_path}")
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except (ImportError, ModuleNotFoundError, TypeError, SyntaxError) as e:
                # Skip modules with missing dependencies or Python version syntax issues
                logging.debug(f"discover_and_import_tasks: Skipping module {module_name}: {e}")
            except Exception as e:
                logging.error(f"discover_and_import_tasks: Error importing module {module_name}: {e}", exc_info=True)
    logging.info(f"discover_and_import_tasks: Finished discovery. TASK_REGISTRY keys: {list(TASK_REGISTRY.keys())}")

if __name__ == "__main__":
    discover_and_import_tasks()
    # Print space-separated list of registered task names
    print(" ".join(sorted(TASK_REGISTRY.keys())), flush=True) 