"""
Thread lifecycle management system for worker processes.

Provides centralized tracking, cleanup, and monitoring of threads to prevent
thread accumulation and resource exhaustion in long-lived workers.
"""

import threading
import time
import logging
import os
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - thread manager will use limited functionality")


class WorkerThreadManager:
    """Centralized thread lifecycle management for worker processes."""
    
    def __init__(self):
        self._threads: Dict[str, threading.Thread] = {}
        self._thread_metadata: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self.pid = os.getpid()
        self._creation_count = 0
        self._cleanup_count = 0
        
        logging.info(f"WorkerThreadManager (PID: {self.pid}): Initialized")
    
    def register_thread(self, name: str, thread: threading.Thread, metadata: Optional[dict] = None):
        """
        Register a thread for cleanup tracking.
        
        Args:
            name: Unique name for the thread
            thread: Thread object to track
            metadata: Optional metadata about the thread
        """
        with self._lock:
            if name in self._threads:
                logging.warning(f"WorkerThreadManager (PID: {self.pid}): Thread '{name}' already registered, replacing")
                # Clean up the old thread first
                self._cleanup_thread_unsafe(name, timeout=1.0)
            
            self._threads[name] = thread
            self._thread_metadata[name] = metadata or {}
            self._thread_metadata[name].update({
                'created_at': time.time(),
                'daemon': thread.daemon,
                'alive': thread.is_alive()
            })
            self._creation_count += 1
            
            logging.debug(f"WorkerThreadManager (PID: {self.pid}): Registered thread '{name}' (total: {len(self._threads)})")
    
    def cleanup_thread(self, name: str, timeout: float = 5.0) -> bool:
        """
        Clean up a specific thread with timeout.
        
        Args:
            name: Name of thread to clean up
            timeout: Maximum time to wait for thread to finish
            
        Returns:
            True if thread was cleaned up successfully
        """
        with self._lock:
            return self._cleanup_thread_unsafe(name, timeout)
    
    def _cleanup_thread_unsafe(self, name: str, timeout: float) -> bool:
        """Internal thread cleanup without lock."""
        if name not in self._threads:
            return True
            
        thread = self._threads[name]
        metadata = self._thread_metadata.get(name, {})
        
        try:
            if thread.is_alive():
                logging.debug(f"WorkerThreadManager (PID: {self.pid}): Stopping thread '{name}'")
                
                # If it's a daemon thread, we can't join it reliably
                if thread.daemon:
                    logging.debug(f"WorkerThreadManager (PID: {self.pid}): Thread '{name}' is daemon, marking for cleanup")
                else:
                    # Try to join non-daemon threads
                    thread.join(timeout=timeout)
                    if thread.is_alive():
                        logging.warning(f"WorkerThreadManager (PID: {self.pid}): Thread '{name}' did not stop within {timeout}s")
                        return False
            
            # Remove from tracking
            del self._threads[name]
            del self._thread_metadata[name]
            self._cleanup_count += 1
            
            logging.debug(f"WorkerThreadManager (PID: {self.pid}): Cleaned up thread '{name}' (remaining: {len(self._threads)})")
            return True
            
        except Exception as e:
            logging.error(f"WorkerThreadManager (PID: {self.pid}): Error cleaning up thread '{name}': {e}")
            # Still remove from tracking even if cleanup failed
            self._threads.pop(name, None)
            self._thread_metadata.pop(name, None)
            return False
    
    def cleanup_all_threads(self, timeout: float = 10.0) -> Tuple[int, int]:
        """
        Clean up all registered threads.
        
        Args:
            timeout: Total timeout for all cleanup operations
            
        Returns:
            (successful_cleanups, failed_cleanups)
        """
        with self._lock:
            thread_names = list(self._threads.keys())
            
        if not thread_names:
            return (0, 0)
            
        logging.info(f"WorkerThreadManager (PID: {self.pid}): Cleaning up {len(thread_names)} threads")
        
        successful = 0
        failed = 0
        per_thread_timeout = timeout / len(thread_names) if thread_names else timeout
        
        for name in thread_names:
            if self.cleanup_thread(name, timeout=per_thread_timeout):
                successful += 1
            else:
                failed += 1
        
        logging.info(f"WorkerThreadManager (PID: {self.pid}): Thread cleanup complete - success: {successful}, failed: {failed}")
        return (successful, failed)
    
    def get_thread_count(self) -> int:
        """Get current registered thread count."""
        with self._lock:
            return len(self._threads)
    
    def get_system_thread_count(self) -> int:
        """Get total system thread count for this process."""
        if not PSUTIL_AVAILABLE:
            logging.debug(f"WorkerThreadManager (PID: {self.pid}): psutil not available, returning threading.active_count()")
            return threading.active_count()
            
        try:
            process = psutil.Process(self.pid)
            return process.num_threads()
        except Exception as e:
            logging.warning(f"WorkerThreadManager (PID: {self.pid}): Error getting system thread count: {e}")
            return threading.active_count()  # Fallback to threading module
    
    def should_recycle_worker(self, max_threads: int = 20) -> Tuple[bool, str]:
        """
        Check if worker should be recycled due to thread count.
        
        Args:
            max_threads: Maximum allowed threads before recycling
            
        Returns:
            (should_recycle, reason)
        """
        registered_count = self.get_thread_count()
        system_count = self.get_system_thread_count()
        
        # Check registered threads
        if registered_count > max_threads:
            return (True, f"Too many registered threads: {registered_count} > {max_threads}")
        
        # Check system threads (more aggressive threshold)
        system_threshold = max_threads * 2  # Allow some untracked threads
        if system_count > system_threshold:
            return (True, f"Too many system threads: {system_count} > {system_threshold}")
        
        # Check for thread leaks (system threads much higher than registered)
        if system_count > 0 and registered_count > 0:
            leak_ratio = system_count / max(registered_count, 1)
            if leak_ratio > 5:  # System threads 5x more than registered
                return (True, f"Potential thread leak: {system_count} system vs {registered_count} registered")
        
        return (False, "Thread count OK")
    
    def get_thread_stats(self) -> dict:
        """Get comprehensive thread statistics."""
        with self._lock:
            registered_threads = []
            for name, thread in self._threads.items():
                metadata = self._thread_metadata.get(name, {})
                registered_threads.append({
                    'name': name,
                    'alive': thread.is_alive(),
                    'daemon': thread.daemon,
                    'created_at': metadata.get('created_at'),
                    'age_seconds': time.time() - metadata.get('created_at', time.time())
                })
            
            stats = {
                'pid': self.pid,
                'registered_count': len(self._threads),
                'system_count': self.get_system_thread_count(),
                'creation_count': self._creation_count,
                'cleanup_count': self._cleanup_count,
                'registered_threads': registered_threads
            }
        
        # Add system threading info
        stats['active_count'] = threading.active_count()
        stats['main_thread_alive'] = threading.main_thread().is_alive()
        
        return stats
    
    def diagnose_threads(self) -> str:
        """Generate a diagnostic report of current thread state."""
        stats = self.get_thread_stats()
        
        report = [
            f"WorkerThreadManager Diagnostic Report (PID: {self.pid})",
            f"=" * 50,
            f"Registered threads: {stats['registered_count']}",
            f"System threads: {stats['system_count']}",
            f"Active threads (threading): {stats['active_count']}",
            f"Created total: {stats['creation_count']}",
            f"Cleaned total: {stats['cleanup_count']}",
            f"Main thread alive: {stats['main_thread_alive']}",
            "",
            "Registered threads:"
        ]
        
        for thread_info in stats['registered_threads']:
            report.append(f"  - {thread_info['name']}: alive={thread_info['alive']}, "
                         f"daemon={thread_info['daemon']}, age={thread_info['age_seconds']:.1f}s")
        
        return "\n".join(report)


# Global instance for worker processes
_worker_thread_manager: Optional[WorkerThreadManager] = None


def get_worker_thread_manager() -> WorkerThreadManager:
    """Get or create the global worker thread manager."""
    global _worker_thread_manager
    if _worker_thread_manager is None:
        _worker_thread_manager = WorkerThreadManager()
    return _worker_thread_manager


def register_worker_thread(name: str, thread: threading.Thread, metadata: Optional[dict] = None):
    """Convenience function to register a thread."""
    manager = get_worker_thread_manager()
    manager.register_thread(name, thread, metadata)


def cleanup_worker_threads(timeout: float = 10.0) -> Tuple[int, int]:
    """Convenience function to clean up all worker threads."""
    manager = get_worker_thread_manager()
    return manager.cleanup_all_threads(timeout)


def should_recycle_worker_for_threads(max_threads: int = 20) -> Tuple[bool, str]:
    """Convenience function to check if worker should be recycled."""
    manager = get_worker_thread_manager()
    return manager.should_recycle_worker(max_threads)


def get_worker_thread_stats() -> dict:
    """Convenience function to get thread statistics."""
    manager = get_worker_thread_manager()
    return manager.get_thread_stats()


def diagnose_worker_threads() -> str:
    """Convenience function to diagnose thread state."""
    manager = get_worker_thread_manager()
    return manager.diagnose_threads()