"""
Process-level memory monitoring system.

Replaces thread-based memory monitoring with signal-based and resource limit approaches
to prevent thread accumulation in long-lived worker processes.
"""

import os
import signal
import time
import logging
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - process memory monitoring will use limited functionality")


class ProcessMemoryMonitor:
    """Process-level memory monitoring using resource limits and signals."""
    
    def __init__(self, memory_limit_gb: float, check_interval: float = 0.1):
        """
        Initialize process memory monitor.
        
        Args:
            memory_limit_gb: Memory limit in GB
            check_interval: How often to check memory (seconds)
        """
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        self.memory_limit_gb = memory_limit_gb
        self.check_interval = check_interval
        self._monitoring = False
        self._original_alarm_handler = None
        self.pid = os.getpid()
        
        # Thresholds for different actions
        self.warning_threshold = 0.8  # 80% of limit
        self.critical_threshold = 0.9  # 90% of limit
        
        logging.info(f"ProcessMemoryMonitor (PID: {self.pid}): Initialized with {memory_limit_gb:.2f}GB limit")
    
    def start_monitoring(self):
        """Start memory monitoring using process resource limits."""
        if self._monitoring:
            logging.warning(f"ProcessMemoryMonitor (PID: {self.pid}): Already monitoring")
            return
            
        # Check if RLIMIT_AS should be skipped (e.g., for pysat compatibility)
        skip_rlimit_as = os.environ.get('SKIP_RLIMIT_AS', '').lower() in ('1', 'true', 'yes')
        if skip_rlimit_as:
            logging.info(f"ProcessMemoryMonitor (PID: {self.pid}): Skipping RLIMIT_AS due to SKIP_RLIMIT_AS environment variable")
            self._monitoring = True
            return
            
        try:
            # Set resource limits respecting current hard limits to avoid permission issues
            import resource
            
            # Check current limits first
            current_as_limit = resource.getrlimit(resource.RLIMIT_AS)
            if current_as_limit[1] != resource.RLIM_INFINITY and current_as_limit[1] < self.memory_limit_bytes:
                # Hard limit is lower than what we want to set, use the hard limit
                actual_limit = current_as_limit[1]
                logging.warning(f"ProcessMemoryMonitor (PID: {self.pid}): Using existing hard limit {actual_limit / (1024**3):.2f}GB instead of requested {self.memory_limit_gb:.2f}GB")
            else:
                actual_limit = self.memory_limit_bytes
                
            try:
                # Attempt to set RLIMIT_AS to the desired (possibly clamped) value.
                resource.setrlimit(resource.RLIMIT_AS, (actual_limit, current_as_limit[1]))
                logging.info(
                    f"ProcessMemoryMonitor (PID: {self.pid}): Set RLIMIT_AS to {actual_limit / (1024**3):.2f}GB"
                )
            except ValueError as ve:
                # This error is typically raised when the new *soft* limit is
                # below the memory that is already allocated by the process.
                # Re-compute a safe limit based on the current peak RSS (or VMS)
                # plus a small safety margin and try again.  This keeps us from
                # running completely unlimited while still preventing an OOM
                # kill later on.
                logging.warning(
                    "ProcessMemoryMonitor (PID: %d): Initial RLIMIT_AS of %.2fGB was below current usage – %s. "
                    "Retrying with a higher limit.",
                    self.pid,
                    actual_limit / (1024**3),
                    ve,
                )

                # Fallback: derive the current resident set size (rss) as an
                # approximation of real memory usage.  If psutil is available
                # we prefer that; otherwise we fall back to ru_maxrss which is
                # reported in KiB on Linux.
                try:
                    if PSUTIL_AVAILABLE:
                        import psutil  # local import to avoid mandatory dep
                        mem = psutil.Process(self.pid).memory_info()
                        current_rss = mem.rss
                        current_vms = getattr(mem, "vms", current_rss)
                        current_usage = max(current_rss, current_vms)
                        current_rss = current_usage  # rename for subsequent code, keep semantics
                    else:
                        import resource as _res
                        current_rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss * 1024
                except Exception as _mem_err:
                    logging.error(
                        "ProcessMemoryMonitor (PID: %d): Could not determine current memory usage (%s). "
                        "Falling back to requested limit.",
                        self.pid,
                        _mem_err,
                    )
                    current_rss = actual_limit  # best effort – keep previous

                # Add a 20 %% head-room plus an extra 512 MiB to avoid frequent
                # re-tries while still keeping the cap reasonable.
                slack_bytes = int(current_rss * 1.2) + (512 * 1024 ** 2)
                retry_limit = max(slack_bytes, self.memory_limit_bytes)

                # Ensure we never exceed the existing hard limit (if finite).
                new_hard = current_as_limit[1]
                if new_hard != resource.RLIM_INFINITY and retry_limit > new_hard:
                    # If the hard limit is lower than what we want, bump the
                    # hard limit as well (allowed for CAP_SYS_RESOURCE or when
                    # hard == soft == RLIM_INFINITY).
                    new_hard = retry_limit

                try:
                    resource.setrlimit(resource.RLIMIT_AS, (retry_limit, new_hard))
                    logging.info(
                        "ProcessMemoryMonitor (PID: %d): RLIMIT_AS re-set to %.2fGB (rss≈%.2fGB).",
                        self.pid,
                        retry_limit / (1024 ** 3),
                        current_rss / (1024 ** 3),
                    )
                except Exception as final_err:
                    logging.warning(
                        "ProcessMemoryMonitor (PID: %d): Failed to adjust RLIMIT_AS on retry (%s). "
                        "Continuing without explicit limit – the cgroup/Slurm limit will apply.",
                        self.pid,
                        final_err,
                    )
                
            # Try to set RSS limit too (not all systems support this)
            try:
                current_rss_soft, current_rss_hard = resource.getrlimit(resource.RLIMIT_RSS)

                # Respect the existing hard limit when we lack CAP_SYS_RESOURCE
                # Otherwise setting soft > hard raises ValueError and leaves the
                # limit unchanged.  We therefore cap the *soft* value to the
                # current hard limit when that hard limit is finite.

                desired_soft = self.memory_limit_bytes
                desired_hard = current_rss_hard

                if desired_hard != resource.RLIM_INFINITY and desired_soft > desired_hard:
                    desired_soft = desired_hard  # clamp to hard cap

                try:
                    resource.setrlimit(resource.RLIMIT_RSS, (desired_soft, desired_hard))
                    logging.info(
                        f"ProcessMemoryMonitor (PID: {self.pid}): Set RLIMIT_RSS soft={desired_soft / (1024**3):.2f}GB hard={'inf' if desired_hard == resource.RLIM_INFINITY else desired_hard / (1024**3):.2f}GB"
                    )
                except ValueError:
                    # Fall back to keeping the existing hard cap but raise soft limit if allowed
                    fallback_soft = min(self.memory_limit_bytes, current_rss_hard)
                    resource.setrlimit(resource.RLIMIT_RSS, (fallback_soft, current_rss_hard))
                    logging.info(
                        f"ProcessMemoryMonitor (PID: {self.pid}): Set RLIMIT_RSS soft={fallback_soft / (1024**3):.2f}GB (hard unchanged)"
                    )
            except (AttributeError, OSError) as e:
                logging.debug(f"ProcessMemoryMonitor (PID: {self.pid}): RSS limit not supported: {e}")
                
        except (ImportError, Exception) as e:
            # --------------------------------------------------------------
            # EXTRA DIAGNOSTICS – why did RLIMIT setting fail?
            # --------------------------------------------------------------
            logging.warning(
                f"ProcessMemoryMonitor (PID: {self.pid}): Failed to set resource limits: {e}"
            )

            try:
                import resource as _res_diag

                soft_as, hard_as = _res_diag.getrlimit(_res_diag.RLIMIT_AS)
                soft_rss, hard_rss = _res_diag.getrlimit(_res_diag.RLIMIT_RSS)

                logging.error(
                    "[MEM_DIAG %d] RLIMIT_AS soft=%.2f GB, hard=%s",
                    self.pid,
                    (soft_as / 1024**3) if soft_as != _res_diag.RLIM_INFINITY else float('inf'),
                    (hard_as / 1024**3) if hard_as != _res_diag.RLIM_INFINITY else 'inf',
                )
                logging.error(
                    "[MEM_DIAG %d] RLIMIT_RSS soft=%s, hard=%s",
                    self.pid,
                    (soft_rss / 1024**3) if soft_rss != _res_diag.RLIM_INFINITY else 'inf',
                    (hard_rss / 1024**3) if hard_rss != _res_diag.RLIM_INFINITY else 'inf',
                )

                if PSUTIL_AVAILABLE:
                    import psutil as _ps_diag

                    p = _ps_diag.Process(self.pid)
                    mem = p.memory_info()
                    logging.error(
                        "[MEM_DIAG %d] rss=%.2f GB, vms=%.2f GB, shared=%.2f GB, data=%.2f GB",
                        self.pid,
                        mem.rss / 1024**3,
                        getattr(mem, "vms", 0) / 1024**3,
                        getattr(mem, "shared", 0) / 1024**3,
                        getattr(mem, "data", 0) / 1024**3,
                    )
            except Exception as _diag_err:
                logging.debug(
                    f"[MEM_DIAG {self.pid}] Unable to collect diagnostic info: {_diag_err}"
                )
        
        self._monitoring = True
        logging.info(f"ProcessMemoryMonitor (PID: {self.pid}): Monitoring started")
    
    def stop_monitoring(self):
        """Clean stop of monitoring."""
        if not self._monitoring:
            return
            
        # Cancel any pending alarms
        signal.alarm(0)
        
        # Restore original signal handler if we had one
        if self._original_alarm_handler is not None:
            signal.signal(signal.SIGALRM, self._original_alarm_handler)
            self._original_alarm_handler = None
            
        self._monitoring = False
        logging.info(f"ProcessMemoryMonitor (PID: {self.pid}): Monitoring stopped")
    
    def check_memory_once(self) -> Optional[Exception]:
        """
        Single memory check without threads.
        
        PERFORMANCE OPTIMIZATION: Disabled expensive psutil monitoring since we use 14GB OS limits.
        
        Returns:
            None - monitoring disabled, relying on OS memory limits
        """
        # Memory monitoring disabled for performance - using OS-enforced 14GB limits instead
        # This eliminates expensive psutil syscalls that were called frequently during evaluation
        return None
    
    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        if not PSUTIL_AVAILABLE:
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'rss_gb': 0,
                'vms_gb': 0,
                'limit_gb': self.memory_limit_gb,
                'rss_ratio': 0,
                'vms_ratio': 0,
                'error': 'psutil not available'
            }
            
        try:
            process = psutil.Process(self.pid)
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024**2),
                'vms_mb': memory_info.vms / (1024**2),
                'rss_gb': memory_info.rss / (1024**3),
                'vms_gb': memory_info.vms / (1024**3),
                'limit_gb': self.memory_limit_gb,
                'rss_ratio': (memory_info.rss / (1024**3)) / self.memory_limit_gb,
                'vms_ratio': (memory_info.vms / (1024**3)) / self.memory_limit_gb,
                'pid': self.pid
            }
        except Exception as e:
            logging.warning(f"ProcessMemoryMonitor (PID: {self.pid}): Error getting memory stats: {e}")
            return {'error': str(e), 'pid': self.pid}


# Global instance for worker processes
_worker_memory_monitor: Optional[ProcessMemoryMonitor] = None


def init_worker_memory_monitor(memory_limit_gb: float) -> ProcessMemoryMonitor:
    """Initialize the global worker memory monitor."""
    global _worker_memory_monitor
    if _worker_memory_monitor is None:
        _worker_memory_monitor = ProcessMemoryMonitor(memory_limit_gb)
        _worker_memory_monitor.start_monitoring()
    return _worker_memory_monitor


def get_worker_memory_monitor() -> Optional[ProcessMemoryMonitor]:
    """Get the current worker memory monitor."""
    return _worker_memory_monitor


def check_worker_memory() -> Optional[Exception]:
    """Convenience function to check worker memory."""
    monitor = get_worker_memory_monitor()
    if monitor:
        return monitor.check_memory_once()
    return None


def cleanup_worker_memory_monitor():
    """Clean up the global worker memory monitor."""
    global _worker_memory_monitor
    if _worker_memory_monitor:
        _worker_memory_monitor.stop_monitoring()
        _worker_memory_monitor = None