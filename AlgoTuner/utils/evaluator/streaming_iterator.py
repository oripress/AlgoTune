"""
Streaming dataset iterator for memory-efficient evaluation.
"""

import gc
from typing import Iterator, Tuple, Any, Optional, Dict


class StreamingDatasetIterator:
    """
    Iterator that provides (warmup_problem, timed_problem, index) tuples
    while maintaining only a 2-problem window in memory.
    
    Ensures strict isolation between problems and prevents cache contamination.
    """
    
    def __init__(self, dataset_iter: Iterator[Any], max_samples: Optional[int] = None):
        """
        Args:
            dataset_iter: Iterator over dataset problems
            max_samples: Maximum number of samples to process (None for all)
        """
        self.dataset_iter = dataset_iter
        self.max_samples = max_samples
        self.prev_problem = None
        self.count = 0
    
    def __iter__(self) -> Iterator[Tuple[Any, Any, int]]:
        """
        Yields (warmup_problem, timed_problem, index) tuples.
        
        For the first problem (i=0), both warmup and timed are the same problem.
        For subsequent problems, warmup is problem i-1 and timed is problem i.
        """
        for i, problem in enumerate(self.dataset_iter):
            if self.max_samples and i >= self.max_samples:
                break
            
            # For first problem, use itself as warmup
            # For others, use previous problem as warmup
            warmup_problem = self.prev_problem if i > 0 else problem
            
            # Yield the tuple
            yield (warmup_problem, problem, i)
            
            # Update state for next iteration
            self.prev_problem = problem
            self.count = i + 1
            
            # Force garbage collection to free memory
            # This ensures we don't accumulate references
            gc.collect()
    
    def get_count(self) -> int:
        """Get the number of problems processed so far."""
        return self.count


class StreamingDatasetWrapper:
    """
    Wrapper that handles problem data extraction and metadata.
    Converts raw dataset entries to (problem, metadata) tuples.
    """
    
    def __init__(self, dataset_iter: Iterator[Any]):
        self.dataset_iter = dataset_iter
    
    def __iter__(self) -> Iterator[Tuple[Any, Dict[str, Any]]]:
        """
        Yields (problem, metadata) tuples from dataset entries.
        """
        for entry in self.dataset_iter:
            if isinstance(entry, dict):
                # Extract problem and metadata from dict
                problem = entry.get("problem", entry)
                
                # Build metadata
                metadata = {
                    "id": entry.get("id", entry.get("k", None)),
                    "baseline_time_ms": entry.get("baseline_time_ms"),
                    "baseline_time_us": entry.get("baseline_time_us"),
                }
                
                # Include any other keys as metadata
                for key, value in entry.items():
                    if key not in ["problem", "id", "k"]:
                        metadata[key] = value
                
                yield (problem, metadata)
            else:
                # Raw problem, no metadata
                yield (entry, {})
            
            # Force GC after each yield
            gc.collect()