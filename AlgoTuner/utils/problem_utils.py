"""Utilities for handling problems and warmup problem selection."""

import random
from typing import Any, List, Optional


def get_warmup_problem(dataset: List[Any], current_index: Optional[int] = None) -> Any:
    """
    Get a warmup problem from the dataset.
    
    Args:
        dataset: List of problems to select from
        current_index: If provided, selects (current_index - 1) % len(dataset).
                      If None, selects a random problem from the dataset.
    
    Returns:
        A problem instance from the dataset to use for warmup
        
    Raises:
        ValueError: If dataset is empty or None
    """
    if not dataset:
        raise ValueError("Dataset required for warmup problem selection - cannot proceed without proper dataset context")
    
    if current_index is not None:
        return dataset[(current_index - 1) % len(dataset)]
    else:
        return dataset[random.randint(0, len(dataset) - 1)]