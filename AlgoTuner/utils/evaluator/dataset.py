"""
Dataset manipulation utilities for the evaluator.
"""

import os
import logging
import traceback
import numpy as np
import inspect
from typing import Optional, List, Dict, Any
from numpy.typing import NDArray

from AlgoTuneTasks.factory import TaskFactory
from AlgoTuneTasks.base import TASK_REGISTRY
from AlgoTuner.utils.serialization import dataset_decoder
from AlgoTuner.utils.streaming_json import stream_jsonl


def validate_datasets(task_name: Optional[str] = None, force: bool = False, data_dir: str = "./data") -> int:
    """
    Validate datasets for the specified task or all tasks.
    Will raise an error if a bad dataset is found rather than attempting to repair.
    
    Args:
        task_name: Name of the task to check, or None for all tasks
        force: Ignored (kept for backward compatibility)
        data_dir: Directory where the datasets are stored
        
    Returns:
        Number of datasets validated
    """
    # If task_name is None, get all registered tasks
    if task_name is None:
        task_names = list(TASK_REGISTRY.keys())
    else:
        task_names = [task_name]
    
    # Track statistics
    total_validated = 0
    
    for name in task_names:
        logging.info(f"Checking datasets for task '{name}'")
        task_data_dir = os.path.join(data_dir, name)
        
        # Skip if task data directory doesn't exist
        if not os.path.exists(task_data_dir):
            logging.info(f"Task data directory for '{name}' doesn't exist, skipping")
            continue
            
        # List all dataset files for this task
        dataset_files = []
        for filename in os.listdir(task_data_dir):
            if filename.endswith('.json') or filename.endswith('.pkl'):
                if 'train_target_' in filename:
                    dataset_files.append(filename)
        
        if not dataset_files:
            logging.info(f"No dataset files found for task '{name}', skipping")
            continue
            
        # Process each dataset file
        for filename in dataset_files:
            filepath = os.path.join(task_data_dir, filename)
            
            # Try to load the dataset to validate it
            logging.info(f"Validating dataset from {filepath}")
            
            try:
                # Stream and decode each record to validate the JSONL file
                valid_count = 0
                for idx, raw_record in enumerate(stream_jsonl(filepath)):
                    # Reconstruct typed objects
                    if isinstance(raw_record, dict):
                        _ = dataset_decoder(raw_record)
                    valid_count += 1
                if valid_count == 0:
                    raise RuntimeError(f"Empty dataset in {filepath}")
                # Basic validation passed
                total_validated += 1
                logging.info(f"Dataset {filepath} passed basic validation ({valid_count} records)")
            except Exception as e:
                error_msg = f"Dataset validation failed for {filepath}: {e}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
    
    logging.info(f"Dataset validation completed: {total_validated} datasets validated")
    return total_validated


# Compatibility stub for backward compatibility
def repair_datasets(task_name: Optional[str] = None, force: bool = False, data_dir: str = "./data") -> int:
    """
    Repair datasets for the specified task or all tasks.
    
    Args:
        task_name: Name of the task to repair, or None for all tasks
        force: Whether to force repair even if dataset seems valid
        data_dir: Directory where the datasets are stored
        
    Returns:
        Number of datasets repaired
    """
    logging.warning("repair_datasets is deprecated. Use validate_datasets instead.")
    return validate_datasets(task_name, force, data_dir) 