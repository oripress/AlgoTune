"""
Cached dataset loader that integrates with the existing streaming JSON system.

This module provides a drop-in replacement for dataset loading that uses
the ArrayCache to avoid repeated disk I/O.
"""

import os
import logging
import functools
from typing import Iterator, Dict, Any, Optional
import orjson

from AlgoTuner.utils.serialization import dataset_decoder
from AlgoTuner.utils.caching.array_cache import ArrayCache


class CachedDatasetLoader:
    """
    Dataset loader with integrated array caching.
    
    This class wraps the existing dataset loading logic and adds
    transparent caching for memory-mapped arrays.
    """
    
    def __init__(self, cache: Optional[ArrayCache] = None):
        """
        Initialize the cached loader.
        
        Args:
            cache: ArrayCache instance to use (creates new if None)
        """
        self.cache = cache or ArrayCache()
        self.logger = logging.getLogger(__name__)
    
    def stream_jsonl(self, file_path: str, decoder_base_dir: Optional[str] = None) -> Iterator[Dict]:
        """
        Stream JSONL file with cached array loading.
        
        This is a drop-in replacement for AlgoTuner.utils.streaming_json.stream_jsonl
        that adds caching for ndarray_ref entries.
        
        Args:
            file_path: Path to JSONL file
            decoder_base_dir: Base directory for resolving references
            
        Yields:
            Decoded problem dictionaries with cached arrays
        """
        actual_base_dir = decoder_base_dir if decoder_base_dir is not None else os.path.dirname(file_path)
        
        self.logger.info(f"Streaming {file_path} with array caching (base_dir: {actual_base_dir})")
        
        try:
            with open(file_path, 'r') as f:
                # Create cached decoder
                cached_decoder = functools.partial(
                    self._cached_dataset_decoder, 
                    base_dir=actual_base_dir
                )
                
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON
                        raw_record = orjson.loads(line)
                        # Apply cached decoder
                        processed_record = cached_decoder(raw_record)
                        yield processed_record
                        
                    except orjson.JSONDecodeError as e:
                        self.logger.error(f"JSON Decode Error in {file_path}, line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
                        continue
                        
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return iter([])
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return iter([])
    
    def _cached_dataset_decoder(self, obj: Any, base_dir: Optional[str] = None) -> Any:
        """
        Custom decoder that uses cache for ndarray_ref entries.
        
        This wraps the standard dataset_decoder but intercepts ndarray_ref
        entries to use the cache.
        """
        # Handle ndarray_ref with cache
        if isinstance(obj, dict) and obj.get("__type__") == "ndarray_ref":
            if base_dir is None:
                self.logger.error("base_dir not provided for ndarray_ref")
                return obj
            
            npy_path = os.path.join(base_dir, obj["npy_path"])
            
            # Try to get from cache
            array = self.cache.get(npy_path)
            if array is not None:
                return array
            else:
                # Cache miss or load failure, fallback to original decoder
                return dataset_decoder(obj, base_dir)
        
        # For all other types, use standard decoder recursively
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Special handling for known keys that might contain ndarray_refs
                if k in ['problem', 'y2', 'data', 'matrix'] and isinstance(v, dict):
                    result[k] = self._cached_dataset_decoder(v, base_dir)
                else:
                    result[k] = self._cached_dataset_decoder(v, base_dir) if isinstance(v, (dict, list)) else v
            
            # Apply standard decoder for typed objects
            if "__type__" in obj and obj["__type__"] != "ndarray_ref":
                return dataset_decoder(result, base_dir)
            return result
            
        elif isinstance(obj, list):
            return [self._cached_dataset_decoder(item, base_dir) for item in obj]
        else:
            return obj
    
    def clear_cache(self):
        """Clear the array cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()