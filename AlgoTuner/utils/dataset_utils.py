"""Utilities for dataset file discovery and loading."""

import os
from typing import List, Optional

from AlgoTuner.utils.dataset_manager import DatasetManager

# Global instance for backward compatibility
_default_manager = None

def _get_manager(data_dir: str) -> DatasetManager:
    """Get or create default manager."""
    global _default_manager
    if _default_manager is None or _default_manager.data_dir != data_dir:
        _default_manager = DatasetManager(data_dir)
    return _default_manager

def find_dataset_files(
    task_name: str, 
    data_dir: str,
    target_time_ms: Optional[int] = None,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    subset: str = "train"
) -> List[str]:
    """
    Find dataset files for a task with consistent logic.
    
    This is a compatibility wrapper around DatasetManager.
    
    Args:
        task_name: Name of the task
        data_dir: Base data directory
        target_time_ms: Target time in milliseconds (optional, wildcards if None)
        train_size: Training set size (optional, wildcards if None)
        test_size: Test set size (optional, wildcards if None) 
        subset: "train" or "test"
        
    Returns:
        List of matching file paths, sorted by n value (descending)
    """
    manager = _get_manager(data_dir)
    
    # Find all matching datasets
    matches = []
    
    # Try with exact parameters first
    dataset_info = manager.find_dataset(task_name, target_time_ms, subset, train_size, test_size)
    if dataset_info:
        matches.append(dataset_info.path)
    
    # If no exact match and target_time_ms was specified, try without it
    if not matches and target_time_ms:
        dataset_info = manager.find_dataset(task_name, None, subset, train_size, test_size)
        if dataset_info:
            matches.append(dataset_info.path)
    
    return matches

def load_dataset_streaming(file_path: str):
    """Load dataset from JSONL file as a generator."""
    from AlgoTuner.utils.streaming_json import stream_jsonl
    return stream_jsonl(file_path)