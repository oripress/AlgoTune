"""
Worker health monitoring system.

Tracks worker resource usage, task completion patterns, and determines
when workers should be recycled to maintain system stability.
"""

import time
import logging
import os
from typing import Tuple, List, Dict, Any, Optional
from collections import deque

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - worker health monitoring will use limited functionality")

from .thread_manager import get_worker_thread_manager
from .process_monitor import get_worker_memory_monitor


class WorkerHealthMonitor:
    """Monitor worker health and determine when recycling is needed."""
    
    def __init__(self, max_history: int = 50):
        """
        Initialize worker health monitor.
        
        Args:
            max_history: Maximum number of task records to keep
        """
        self.pid = os.getpid()
        self.max_history = max_history
        
        # Task tracking
        self.task_count = 0
        self.start_time = time.time()
        
        # Resource usage history
        self.thread_count_history = deque(maxlen=max_history)
        self.memory_usage_history = deque(maxlen=max_history)
        self.task_duration_history = deque(maxlen=max_history)
        
        # Health metrics
        self.recycling_events = 0
        self.last_health_check = time.time()
        
        # Thresholds (configurable)
        self.max_tasks_per_worker = 25  # Reduced from default 100
        self.max_threads_per_worker = 20
        self.max_memory_growth_mb = 1000  # 1GB growth before concern
        self.max_task_duration_growth = 2.0  # 2x slowdown before concern
        
        logging.info(f"WorkerHealthMonitor (PID: {self.pid}): Initialized with max_history={max_history}")
    
    def record_task_start(self) -> dict:
        """Record the start of a new task and return baseline metrics."""
        self.task_count += 1
        current_time = time.time()
        
        # Get current resource usage
        thread_stats = get_worker_thread_manager().get_thread_stats()
        memory_monitor = get_worker_memory_monitor()
        memory_stats = memory_monitor.get_memory_stats() if memory_monitor else {}
        
        baseline = {
            'task_number': self.task_count,
            'start_time': current_time,
            'threads_registered': thread_stats.get('registered_count', 0),
            'threads_system': thread_stats.get('system_count', 0),
            'memory_rss_mb': memory_stats.get('rss_mb', 0),
            'memory_vms_mb': memory_stats.get('vms_mb', 0)
        }
        
        logging.debug(f"WorkerHealthMonitor (PID: {self.pid}): Task {self.task_count} started - "
                     f"threads: {baseline['threads_system']}, memory: {baseline['memory_rss_mb']:.1f}MB")
        
        return baseline
    
    def record_task_completion(self, baseline: dict, success: bool = True):
        """
        Record task completion and current resource usage.
        
        Args:
            baseline: Baseline metrics from record_task_start()
            success: Whether the task completed successfully
        """
        current_time = time.time()
        task_duration = current_time - baseline['start_time']
        
        # Get current resource usage
        thread_stats = get_worker_thread_manager().get_thread_stats()
        memory_monitor = get_worker_memory_monitor()
        memory_stats = memory_monitor.get_memory_stats() if memory_monitor else {}
        
        # Record metrics
        self.thread_count_history.append({
            'timestamp': current_time,
            'registered': thread_stats.get('registered_count', 0),
            'system': thread_stats.get('system_count', 0),
            'active': thread_stats.get('active_count', 0)
        })
        
        self.memory_usage_history.append({
            'timestamp': current_time,
            'rss_mb': memory_stats.get('rss_mb', 0),
            'vms_mb': memory_stats.get('vms_mb', 0),
            'rss_ratio': memory_stats.get('rss_ratio', 0)
        })
        
        self.task_duration_history.append({
            'timestamp': current_time,
            'duration': task_duration,
            'success': success,
            'task_number': self.task_count
        })
        
        self.last_health_check = current_time
        
        logging.debug(f"WorkerHealthMonitor (PID: {self.pid}): Task {self.task_count} completed in {task_duration:.2f}s - "
                     f"threads: {thread_stats.get('system_count', 0)}, memory: {memory_stats.get('rss_mb', 0):.1f}MB")
    
    def should_recycle_worker(self) -> Tuple[bool, str]:
        """
        Determine if worker should be recycled and why.
        
        Returns:
            (should_recycle, reason)
        """
        reasons = []
        
        # Check task count limit
        if self.task_count >= self.max_tasks_per_worker:
            reasons.append(f"Task limit reached: {self.task_count} >= {self.max_tasks_per_worker}")
        
        # Check thread count via thread manager
        thread_manager = get_worker_thread_manager()
        should_recycle_threads, thread_reason = thread_manager.should_recycle_worker(self.max_threads_per_worker)
        if should_recycle_threads:
            reasons.append(f"Thread issue: {thread_reason}")
        
        # Check memory growth
        memory_reason = self._check_memory_growth()
        if memory_reason:
            reasons.append(f"Memory issue: {memory_reason}")
        
        # Check performance degradation
        performance_reason = self._check_performance_degradation()
        if performance_reason:
            reasons.append(f"Performance issue: {performance_reason}")
        
        # Check worker age
        worker_age = time.time() - self.start_time
        if worker_age > 3600:  # 1 hour
            reasons.append(f"Worker age: {worker_age/60:.1f} minutes")
        
        if reasons:
            return (True, "; ".join(reasons))
        
        return (False, "Worker health OK")
    
    def _check_memory_growth(self) -> Optional[str]:
        """Check for concerning memory growth patterns."""
        if len(self.memory_usage_history) < 10:
            return None
            
        # Get recent memory usage
        recent = list(self.memory_usage_history)[-10:]
        first = recent[0]
        last = recent[-1]
        
        # Check absolute growth
        rss_growth = last['rss_mb'] - first['rss_mb']
        if rss_growth > self.max_memory_growth_mb:
            return f"RSS growth {rss_growth:.1f}MB > {self.max_memory_growth_mb}MB"
        
        # Check if memory ratio is consistently high
        high_ratio_count = sum(1 for m in recent if m['rss_ratio'] > 0.8)
        if high_ratio_count >= 8:  # 8 out of 10 tasks
            return f"High memory ratio in {high_ratio_count}/10 recent tasks"
        
        return None
    
    def _check_performance_degradation(self) -> Optional[str]:
        """Check for performance degradation over time."""
        if len(self.task_duration_history) < 20:
            return None
            
        durations = [t['duration'] for t in self.task_duration_history if t['success']]
        if len(durations) < 10:
            return None
            
        # Compare recent vs early performance
        early_durations = durations[:5]
        recent_durations = durations[-5:]
        
        early_avg = sum(early_durations) / len(early_durations)
        recent_avg = sum(recent_durations) / len(recent_durations)
        
        if early_avg > 0 and recent_avg / early_avg > self.max_task_duration_growth:
            return f"Performance degraded {recent_avg/early_avg:.1f}x (from {early_avg:.2f}s to {recent_avg:.2f}s)"
        
        return None
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        current_time = time.time()
        worker_age = current_time - self.start_time
        
        # Get current resource stats
        thread_stats = get_worker_thread_manager().get_thread_stats()
        memory_monitor = get_worker_memory_monitor()
        memory_stats = memory_monitor.get_memory_stats() if memory_monitor else {}
        
        # Calculate averages from history
        avg_threads = 0
        avg_memory = 0
        avg_duration = 0
        
        if self.thread_count_history:
            avg_threads = sum(t['system'] for t in self.thread_count_history) / len(self.thread_count_history)
        
        if self.memory_usage_history:
            avg_memory = sum(m['rss_mb'] for m in self.memory_usage_history) / len(self.memory_usage_history)
        
        successful_durations = [t['duration'] for t in self.task_duration_history if t['success']]
        if successful_durations:
            avg_duration = sum(successful_durations) / len(successful_durations)
        
        should_recycle, recycle_reason = self.should_recycle_worker()
        
        return {
            'pid': self.pid,
            'worker_age_minutes': worker_age / 60,
            'tasks_completed': self.task_count,
            'recycling_events': self.recycling_events,
            
            # Current state
            'current_threads': thread_stats.get('system_count', 0),
            'current_memory_mb': memory_stats.get('rss_mb', 0),
            'current_memory_ratio': memory_stats.get('rss_ratio', 0),
            
            # Averages
            'avg_threads': avg_threads,
            'avg_memory_mb': avg_memory,
            'avg_task_duration': avg_duration,
            
            # Health status
            'should_recycle': should_recycle,
            'recycle_reason': recycle_reason,
            
            # Limits
            'max_tasks': self.max_tasks_per_worker,
            'max_threads': self.max_threads_per_worker,
            'max_memory_growth_mb': self.max_memory_growth_mb,
            
            # History sizes
            'thread_history_size': len(self.thread_count_history),
            'memory_history_size': len(self.memory_usage_history),
            'duration_history_size': len(self.task_duration_history)
        }
    
    def reset_for_new_worker(self):
        """Reset counters for a new worker process."""
        old_count = self.task_count
        
        self.task_count = 0
        self.start_time = time.time()
        self.thread_count_history.clear()
        self.memory_usage_history.clear()
        self.task_duration_history.clear()
        self.recycling_events += 1
        
        logging.info(f"WorkerHealthMonitor (PID: {self.pid}): Reset for new worker (previous completed {old_count} tasks)")


# Global instance for worker processes
_worker_health_monitor: Optional[WorkerHealthMonitor] = None


def get_worker_health_monitor() -> WorkerHealthMonitor:
    """Get or create the global worker health monitor."""
    global _worker_health_monitor
    if _worker_health_monitor is None:
        _worker_health_monitor = WorkerHealthMonitor()
    return _worker_health_monitor


def record_worker_task_start() -> dict:
    """Convenience function to record task start."""
    monitor = get_worker_health_monitor()
    return monitor.record_task_start()


def record_worker_task_completion(baseline: dict, success: bool = True):
    """Convenience function to record task completion."""
    monitor = get_worker_health_monitor()
    monitor.record_task_completion(baseline, success)


def should_recycle_worker() -> Tuple[bool, str]:
    """Convenience function to check if worker should be recycled."""
    monitor = get_worker_health_monitor()
    return monitor.should_recycle_worker()


def get_worker_health_summary() -> Dict[str, Any]:
    """Convenience function to get health summary."""
    monitor = get_worker_health_monitor()
    return monitor.get_health_summary()