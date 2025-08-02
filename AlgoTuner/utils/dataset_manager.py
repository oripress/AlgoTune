"""Centralized dataset access and management."""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import orjson
from AlgoTuner.utils.serialization import dataset_decoder
from AlgoTuner.utils.streaming_json import stream_jsonl


@dataclass
class DatasetInfo:
    """Metadata about a dataset file."""
    path: str
    task_name: str
    target_time_ms: int
    n_value: int
    size: int  # Number of problems in this subset
    subset: str  # "train" or "test"
    
    @classmethod
    def from_path(cls, path: str) -> Optional['DatasetInfo']:
        """Parse dataset info from filename."""
        filename = os.path.basename(path)
        pattern = r"^(.+?)_T(\d+)ms_n(\d+)_size(\d+)_(train|test)\.jsonl$"
        match = re.match(pattern, filename)
        
        if not match:
            return None
            
        return cls(
            path=path,
            task_name=match.group(1),
            target_time_ms=int(match.group(2)),
            n_value=int(match.group(3)),
            size=int(match.group(4)),
            subset=match.group(5)
        )


class DatasetManager:
    """Centralized dataset access and management."""
    
    def __init__(self, data_dir: str):
        """Initialize with base data directory."""
        self.data_dir = Path(data_dir)
        self._metadata_cache: Dict[str, DatasetInfo] = {}
        self._size_cache: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
    
    def find_dataset(
        self, 
        task_name: str, 
        target_time_ms: Optional[int] = None,
        subset: str = "train",
        train_size: Optional[int] = None,
        test_size: Optional[int] = None
    ) -> Optional[DatasetInfo]:
        """
        Find best matching dataset with clear precedence rules.
        
        Search order:
        1. Base directory with exact match
        2. Task subdirectory with exact match
        3. Task subdirectory with any time (if target_time_ms not specified)
        
        Returns None if no dataset found.
        """
        size_filter = train_size if subset == "train" else test_size
        candidates = []
        
        # Pattern components
        time_part = f"T{target_time_ms}ms" if target_time_ms else "T*ms"
        size_part = f"size{size_filter}" if size_filter else "size*"
        pattern = f"{task_name}_{time_part}_n*_{size_part}_{subset}.jsonl"
        
        # Search locations in order of preference
        search_paths = [
            self.data_dir / pattern,
            self.data_dir / task_name / pattern,
        ]
        
        # If exact time match not required, also search with wildcard
        if target_time_ms:
            wildcard_pattern = f"{task_name}_T*ms_n*_{size_part}_{subset}.jsonl"
            search_paths.append(self.data_dir / task_name / wildcard_pattern)
        
        # Find all matching files
        import glob
        for search_pattern in search_paths:
            matches = glob.glob(str(search_pattern))
            for match in matches:
                info = DatasetInfo.from_path(match)
                if info:
                    candidates.append(info)
            
            # If we found exact matches with target time, stop searching
            if target_time_ms and candidates and all(c.target_time_ms == target_time_ms for c in candidates):
                break
        
        if not candidates:
            return None
        
        # Sort by preference: exact time match first, then by n value (descending)
        if target_time_ms:
            candidates.sort(key=lambda x: (x.target_time_ms != target_time_ms, -x.n_value))
        else:
            candidates.sort(key=lambda x: -x.n_value)
        
        selected = candidates[0]
        self._metadata_cache[selected.path] = selected
        
        self.logger.info(f"Selected dataset: {selected.path} (n={selected.n_value}, T={selected.target_time_ms}ms)")
        return selected
    
    def get_dataset_info(self, dataset_path: str) -> Optional[DatasetInfo]:
        """Get cached metadata about dataset."""
        if dataset_path in self._metadata_cache:
            return self._metadata_cache[dataset_path]
        
        info = DatasetInfo.from_path(dataset_path)
        if info:
            self._metadata_cache[dataset_path] = info
        return info
    
    def count_dataset_size(self, dataset_path: str) -> int:
        """Count records in dataset, with caching."""
        if dataset_path in self._size_cache:
            return self._size_cache[dataset_path]
        
        count = 0
        with open(dataset_path, 'r') as f:
            for _ in f:
                count += 1
        
        self._size_cache[dataset_path] = count
        return count
    
    def load_problem(self, dataset_path: str, index: int) -> Any:
        """Load a single problem by index."""
        # Get base directory for resolving external references
        base_dir = os.path.dirname(dataset_path)
        
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    # Parse JSON
                    raw_data = orjson.loads(line)
                    # Apply dataset_decoder to resolve external references
                    data = dataset_decoder(raw_data, base_dir=base_dir)
                    return data.get('problem', data)
        
        raise IndexError(f"Index {index} out of range for dataset {dataset_path}")
    
    def load_problem_with_metadata(self, dataset_path: str, index: int) -> Dict[str, Any]:
        """Load full problem record including metadata."""
        # Get base directory for resolving external references
        base_dir = os.path.dirname(dataset_path)
        
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    # Parse JSON
                    raw_data = orjson.loads(line)
                    # Apply dataset_decoder to resolve external references
                    return dataset_decoder(raw_data, base_dir=base_dir)
        
        raise IndexError(f"Index {index} out of range for dataset {dataset_path}")
    
    def stream_dataset(self, dataset_path: str) -> Generator[Dict[str, Any], None, None]:
        """Stream entire dataset efficiently."""
        return stream_jsonl(dataset_path)
    
    def get_warmup_index(self, current_index: int, dataset_size: int) -> int:
        """
        Standard warmup index calculation.
        Uses (current - 1) % size to ensure different problem.
        """
        return (current_index - 1) % dataset_size
    
    def get_warmup_problem(
        self, 
        task_name: str,
        current_problem: Optional[Any] = None,
        current_index: Optional[int] = None
    ) -> Tuple[Any, str]:
        """
        Get a warmup problem for evaluation.
        
        Returns:
            Tuple of (warmup_problem, dataset_path_used)
        """
        dataset_info = self.find_dataset(task_name)
        if not dataset_info:
            raise ValueError(f"No dataset found for task {task_name}")
        
        # Get actual size from file if not in metadata
        if dataset_info.size == 0 or not hasattr(dataset_info, '_counted_size'):
            actual_size = self.count_dataset_size(dataset_info.path)
            dataset_info._counted_size = actual_size
        else:
            actual_size = getattr(dataset_info, '_counted_size', dataset_info.size)
        
        if actual_size == 0:
            raise ValueError(f"Dataset {dataset_info.path} is empty")
        
        # Determine warmup index
        if current_index is not None:
            warmup_idx = self.get_warmup_index(current_index, actual_size)
        else:
            # For single problem eval, pick a random one
            import random
            warmup_idx = random.randint(0, actual_size - 1)
        
        warmup_problem = self.load_problem(dataset_info.path, warmup_idx)
        
        self.logger.debug(f"Selected warmup problem at index {warmup_idx} from {dataset_info.path}")
        return warmup_problem, dataset_info.path
    
    def clear_cache(self):
        """Clear all caches."""
        self._metadata_cache.clear()
        self._size_cache.clear()