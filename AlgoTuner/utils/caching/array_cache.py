"""
Thread-safe LRU cache for memory-mapped numpy arrays.

This cache maintains references to memory-mapped arrays to avoid repeated
disk metadata lookups while respecting memory constraints.
"""

import os
import threading
import time
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple
import numpy as np
import weakref


class ArrayCache:
    """
    LRU cache for memory-mapped numpy arrays with memory tracking.
    
    Features:
    - Thread-safe operations
    - LRU eviction policy
    - Memory usage tracking
    - Hit/miss statistics
    - Weak references to allow garbage collection
    """
    
    def __init__(self, 
                 max_entries: int = 100,
                 max_memory_mb: Optional[float] = None,
                 ttl_seconds: Optional[float] = None):
        """
        Initialize the array cache.
        
        Args:
            max_entries: Maximum number of entries to cache
            max_memory_mb: Maximum memory usage in MB (None = no limit)
            ttl_seconds: Time-to-live for cache entries (None = no expiry)
        """
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.ttl_seconds = ttl_seconds
        
        # Main cache storage
        self._cache: OrderedDict[str, Tuple[weakref.ref, float, int]] = OrderedDict()
        # Strong references to prevent premature garbage collection
        self._strong_refs: OrderedDict[str, np.ndarray] = OrderedDict()
        
        # Threading
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_bytes = 0
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, path: str) -> Optional[np.ndarray]:
        """
        Retrieve an array from cache or load it.
        
        Args:
            path: Path to the .npy file
            
        Returns:
            Memory-mapped numpy array or None if loading fails
        """
        with self._lock:
            # Check if in cache
            if path in self._cache:
                weak_ref, timestamp, size = self._cache[path]
                
                # Check if TTL expired
                if self.ttl_seconds and (time.time() - timestamp) > self.ttl_seconds:
                    self._evict(path)
                    return self._load_and_cache(path)
                
                # Try to get strong reference
                array = weak_ref()
                if array is not None:
                    # Move to end (most recently used)
                    self._cache.move_to_end(path)
                    self._strong_refs.move_to_end(path)
                    self._hits += 1
                    return array
                else:
                    # Weak reference died, reload
                    self._evict(path)
                    return self._load_and_cache(path)
            else:
                self._misses += 1
                return self._load_and_cache(path)
    
    def _load_and_cache(self, path: str) -> Optional[np.ndarray]:
        """Load array and add to cache."""
        if not os.path.exists(path):
            self.logger.error(f"Array file not found: {path}")
            return None
        
        try:
            # Load with memory mapping
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            
            # Calculate size (for memory tracking)
            # Note: For mmap, this is virtual memory, not physical
            size = array.nbytes
            
            # Check memory limit
            if self.max_memory_mb:
                if (self._total_bytes + size) / (1024 * 1024) > self.max_memory_mb:
                    self._evict_until_space(size)
            
            # Check entry limit
            while len(self._cache) >= self.max_entries:
                self._evict_oldest()
            
            # Add to cache
            self._cache[path] = (weakref.ref(array), time.time(), size)
            self._strong_refs[path] = array
            self._total_bytes += size
            
            self.logger.debug(f"Cached array from {path}: shape={array.shape}, dtype={array.dtype}")
            return array
            
        except Exception as e:
            self.logger.error(f"Failed to load array from {path}: {e}")
            return None
    
    def _evict(self, path: str):
        """Remove entry from cache."""
        if path in self._cache:
            _, _, size = self._cache[path]
            del self._cache[path]
            self._strong_refs.pop(path, None)
            self._total_bytes -= size
            self._evictions += 1
    
    def _evict_oldest(self):
        """Evict the least recently used entry."""
        if self._cache:
            oldest_path = next(iter(self._cache))
            self._evict(oldest_path)
    
    def _evict_until_space(self, needed_bytes: int):
        """Evict entries until there's enough space."""
        target_bytes = self.max_memory_mb * 1024 * 1024 if self.max_memory_mb else float('inf')
        
        while self._total_bytes + needed_bytes > target_bytes and self._cache:
            self._evict_oldest()
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._strong_refs.clear()
            self._total_bytes = 0
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'entries': len(self._cache),
                'memory_mb': self._total_bytes / (1024 * 1024),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'total_requests': total_requests
            }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.clear()