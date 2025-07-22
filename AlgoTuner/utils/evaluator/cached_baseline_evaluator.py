"""
Enhanced baseline evaluator with integrated array caching.

This module provides a cached version of evaluate_baseline_dataset that
significantly reduces I/O overhead during evaluation.
"""

import os
import json
import logging
import tempfile
import time
from typing import Any, Dict, Optional, Iterable

from AlgoTuner.utils.caching import ArrayCache, CachedDatasetLoader
from AlgoTuner.utils.streaming_json import stream_jsonl
from AlgoTuner.utils.timing_config import DATASET_RUNS, DATASET_WARMUPS, EVAL_RUNS
from AlgoTuner.config.loader import load_config


def evaluate_baseline_dataset_cached(
    task_obj: Any,
    dataset_iterable: Iterable[Dict[str, Any]],
    num_runs: Optional[int] = None,
    warmup_runs: Optional[int] = None,
    output_file: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    cache: Optional[ArrayCache] = None,
    **kwargs
) -> str:
    """
    Cached version of evaluate_baseline_dataset.
    
    This function wraps the baseline evaluation with array caching to reduce
    repeated disk I/O. It's a drop-in replacement that adds caching.
    
    Args:
        task_obj: Task instance
        dataset_iterable: Dataset iterator (will be replaced if jsonl_path provided)
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        output_file: Output file path
        jsonl_path: Path to JSONL file (enables caching)
        cache: ArrayCache instance (creates new if None)
        **kwargs: Additional arguments passed to original function
        
    Returns:
        Path to output JSON file
    """
    logger = logging.getLogger(__name__)
    
    # Import the original function
    from AlgoTuner.utils.evaluator.main import evaluate_baseline_dataset
    
    # Determine number of runs from config
    if num_runs is None:
        config = load_config()
        # Check if we're in evaluation mode (3 array jobs with EVAL_RUNS each)
        if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
            num_runs = EVAL_RUNS
            logger.info(f"Using EVAL_RUNS={num_runs} for array job evaluation")
        else:
            num_runs = DATASET_RUNS
            logger.info(f"Using DATASET_RUNS={num_runs} for standard evaluation")
    
    # If we have a JSONL path and can use caching
    if jsonl_path and os.path.exists(jsonl_path):
        logger.info(f"Enabling array caching for evaluation of {jsonl_path}")
        
        # Create or use provided cache
        if cache is None:
            # Configure cache based on environment
            cache_config = load_config().get('benchmark', {}).get('cache', {})
            cache = ArrayCache(
                max_entries=cache_config.get('max_entries', 100),
                max_memory_mb=cache_config.get('max_memory_mb', 2048),  # 2GB default
                ttl_seconds=cache_config.get('ttl_seconds', 900)  # 15 min default
            )
        
        # Create cached loader
        cached_loader = CachedDatasetLoader(cache)
        
        # Override dataset_iterable with cached version
        dataset_iterable = cached_loader.stream_jsonl(jsonl_path)
        
        # Log cache stats periodically
        start_time = time.time()
        problems_processed = 0
        
        # Run original evaluation with cached dataset
        try:
            result_path = evaluate_baseline_dataset(
                task_obj=task_obj,
                dataset_iterable=dataset_iterable,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                output_file=output_file,
                jsonl_path=jsonl_path,
                **kwargs
            )
            
            # Log final cache statistics
            elapsed_time = time.time() - start_time
            cache_stats = cache.get_stats()
            logger.info(
                f"Evaluation completed in {elapsed_time:.1f}s with cache stats: "
                f"hit_rate={cache_stats['hit_rate']:.2%}, "
                f"hits={cache_stats['hits']}, "
                f"misses={cache_stats['misses']}, "
                f"memory_mb={cache_stats['memory_mb']:.1f}"
            )
            
            return result_path
            
        finally:
            # Clear cache to free memory
            if cache:
                cache.clear()
                logger.info("Cleared array cache after evaluation")
    
    else:
        # No JSONL path or caching not possible, use original function
        logger.info("Running evaluation without array caching")
        return evaluate_baseline_dataset(
            task_obj=task_obj,
            dataset_iterable=dataset_iterable,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            output_file=output_file,
            jsonl_path=jsonl_path,
            **kwargs
        )