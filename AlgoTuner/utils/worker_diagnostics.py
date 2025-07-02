"""
Worker diagnostics and debugging utilities.

Provides comprehensive diagnostic tools for debugging worker health,
thread management, and memory usage issues.
"""

import logging
import time
import os
from typing import Dict, Any, Optional


def diagnose_worker_health() -> Dict[str, Any]:
    """
    Generate comprehensive worker health diagnostic report.
    
    Returns:
        Detailed diagnostic information about worker state
    """
    diagnostics = {
        'timestamp': time.time(),
        'pid': os.getpid(),
        'diagnostic_errors': []
    }
    
    # Thread diagnostics
    try:
        from AlgoTuner.utils.thread_manager import get_worker_thread_manager, diagnose_worker_threads
        
        thread_manager = get_worker_thread_manager()
        diagnostics['thread_stats'] = thread_manager.get_thread_stats()
        diagnostics['thread_diagnosis'] = diagnose_worker_threads()
        
        should_recycle, reason = thread_manager.should_recycle_worker()
        diagnostics['thread_recycle_recommendation'] = {
            'should_recycle': should_recycle,
            'reason': reason
        }
        
    except Exception as e:
        diagnostics['diagnostic_errors'].append(f"Thread diagnostics failed: {e}")
        diagnostics['thread_stats'] = {'error': str(e)}
    
    # Memory diagnostics
    try:
        from AlgoTuner.utils.process_monitor import get_worker_memory_monitor
        
        memory_monitor = get_worker_memory_monitor()
        if memory_monitor:
            diagnostics['memory_stats'] = memory_monitor.get_memory_stats()
            memory_error = memory_monitor.check_memory_once()
            diagnostics['memory_check'] = {
                'status': 'OK' if memory_error is None else 'ERROR',
                'error': str(memory_error) if memory_error else None
            }
        else:
            diagnostics['memory_stats'] = {'error': 'No memory monitor initialized'}
            
    except Exception as e:
        diagnostics['diagnostic_errors'].append(f"Memory diagnostics failed: {e}")
        diagnostics['memory_stats'] = {'error': str(e)}
    
    # Worker health diagnostics
    try:
        from AlgoTuner.utils.worker_health import get_worker_health_monitor
        
        health_monitor = get_worker_health_monitor()
        diagnostics['health_summary'] = health_monitor.get_health_summary()
        
        should_recycle, reason = health_monitor.should_recycle_worker()
        diagnostics['health_recycle_recommendation'] = {
            'should_recycle': should_recycle,
            'reason': reason
        }
        
    except Exception as e:
        diagnostics['diagnostic_errors'].append(f"Health diagnostics failed: {e}")
        diagnostics['health_summary'] = {'error': str(e)}
    
    # System diagnostics
    try:
        try:
            import psutil
            psutil_available = True
        except ImportError:
            psutil_available = False
            
        import threading
        
        if not psutil_available:
            diagnostics['system_stats'] = {'error': 'psutil not available'}
            diagnostics['system_memory'] = {'error': 'psutil not available'}
        else:
            process = psutil.Process(os.getpid())
            diagnostics['system_stats'] = {
                'memory_info': process.memory_info()._asdict(),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'cpu_percent': process.cpu_percent(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections()),
                'threading_active_count': threading.active_count(),
                'threading_enumerate': [t.name for t in threading.enumerate()]
            }
            
            # System memory info
            vm = psutil.virtual_memory()
            diagnostics['system_memory'] = {
                'total_gb': vm.total / (1024**3),
                'available_gb': vm.available / (1024**3),
                'percent_used': vm.percent,
                'free_gb': vm.free / (1024**3)
            }
        
    except Exception as e:
        diagnostics['diagnostic_errors'].append(f"System diagnostics failed: {e}")
        diagnostics['system_stats'] = {'error': str(e)}
    
    # Overall assessment
    overall_issues = []
    
    if diagnostics.get('thread_recycle_recommendation', {}).get('should_recycle'):
        overall_issues.append(f"Thread issues: {diagnostics['thread_recycle_recommendation']['reason']}")
    
    if diagnostics.get('health_recycle_recommendation', {}).get('should_recycle'):
        overall_issues.append(f"Health issues: {diagnostics['health_recycle_recommendation']['reason']}")
    
    if diagnostics.get('memory_check', {}).get('status') == 'ERROR':
        overall_issues.append(f"Memory issues: {diagnostics['memory_check']['error']}")
    
    diagnostics['overall_assessment'] = {
        'status': 'HEALTHY' if not overall_issues else 'ISSUES_DETECTED',
        'issues': overall_issues,
        'recommendation': 'Worker should be recycled' if overall_issues else 'Worker is healthy'
    }
    
    return diagnostics


def log_worker_diagnostics(level: int = logging.INFO):
    """Log comprehensive worker diagnostics."""
    try:
        diagnostics = diagnose_worker_health()
        
        logging.log(level, "=== WORKER DIAGNOSTICS ===")
        logging.log(level, f"PID: {diagnostics['pid']}")
        logging.log(level, f"Overall Status: {diagnostics['overall_assessment']['status']}")
        
        if diagnostics['overall_assessment']['issues']:
            logging.log(level, f"Issues: {'; '.join(diagnostics['overall_assessment']['issues'])}")
            logging.log(level, f"Recommendation: {diagnostics['overall_assessment']['recommendation']}")
        
        # Thread info
        thread_stats = diagnostics.get('thread_stats', {})
        if 'error' not in thread_stats:
            logging.log(level, f"Threads: {thread_stats.get('registered_count', 0)} registered, "
                              f"{thread_stats.get('system_count', 0)} system, "
                              f"{thread_stats.get('active_count', 0)} active")
        
        # Memory info
        memory_stats = diagnostics.get('memory_stats', {})
        if 'error' not in memory_stats:
            logging.log(level, f"Memory: {memory_stats.get('rss_gb', 0):.2f}GB RSS "
                              f"({memory_stats.get('rss_ratio', 0):.1%} of limit)")
        
        # Health info
        health_summary = diagnostics.get('health_summary', {})
        if 'error' not in health_summary:
            logging.log(level, f"Health: {health_summary.get('tasks_completed', 0)} tasks completed, "
                              f"worker age {health_summary.get('worker_age_minutes', 0):.1f}min")
        
        if diagnostics['diagnostic_errors']:
            logging.log(level, f"Diagnostic errors: {'; '.join(diagnostics['diagnostic_errors'])}")
        
        logging.log(level, "=== END DIAGNOSTICS ===")
        
    except Exception as e:
        logging.error(f"Failed to log worker diagnostics: {e}")


def format_worker_report(diagnostics: Optional[Dict[str, Any]] = None) -> str:
    """
    Format a human-readable worker diagnostic report.
    
    Args:
        diagnostics: Optional diagnostics dict, if None will generate new one
        
    Returns:
        Formatted report string
    """
    if diagnostics is None:
        diagnostics = diagnose_worker_health()
    
    lines = [
        "Worker Diagnostic Report",
        "=" * 50,
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(diagnostics['timestamp']))}",
        f"PID: {diagnostics['pid']}",
        "",
        f"Overall Status: {diagnostics['overall_assessment']['status']}",
        f"Recommendation: {diagnostics['overall_assessment']['recommendation']}",
    ]
    
    if diagnostics['overall_assessment']['issues']:
        lines.extend([
            "",
            "Issues Detected:",
            *[f"  - {issue}" for issue in diagnostics['overall_assessment']['issues']]
        ])
    
    # Thread section
    lines.extend([
        "",
        "Thread Information:",
        "-" * 20
    ])
    
    thread_stats = diagnostics.get('thread_stats', {})
    if 'error' in thread_stats:
        lines.append(f"  Error: {thread_stats['error']}")
    else:
        lines.extend([
            f"  Registered threads: {thread_stats.get('registered_count', 0)}",
            f"  System threads: {thread_stats.get('system_count', 0)}",
            f"  Active threads: {thread_stats.get('active_count', 0)}",
            f"  Total created: {thread_stats.get('creation_count', 0)}",
            f"  Total cleaned: {thread_stats.get('cleanup_count', 0)}"
        ])
        
        if thread_stats.get('registered_threads'):
            lines.append("  Registered thread details:")
            for thread in thread_stats['registered_threads']:
                lines.append(f"    - {thread['name']}: alive={thread['alive']}, "
                           f"daemon={thread['daemon']}, age={thread['age_seconds']:.1f}s")
    
    # Memory section
    lines.extend([
        "",
        "Memory Information:",
        "-" * 20
    ])
    
    memory_stats = diagnostics.get('memory_stats', {})
    if 'error' in memory_stats:
        lines.append(f"  Error: {memory_stats['error']}")
    else:
        lines.extend([
            f"  RSS: {memory_stats.get('rss_gb', 0):.2f}GB ({memory_stats.get('rss_ratio', 0):.1%} of limit)",
            f"  VMS: {memory_stats.get('vms_gb', 0):.2f}GB ({memory_stats.get('vms_ratio', 0):.1%} of limit)",
            f"  Limit: {memory_stats.get('limit_gb', 0):.2f}GB"
        ])
    
    memory_check = diagnostics.get('memory_check', {})
    if memory_check:
        lines.append(f"  Status: {memory_check['status']}")
        if memory_check.get('error'):
            lines.append(f"  Error: {memory_check['error']}")
    
    # Health section
    lines.extend([
        "",
        "Health Information:",
        "-" * 20
    ])
    
    health_summary = diagnostics.get('health_summary', {})
    if 'error' in health_summary:
        lines.append(f"  Error: {health_summary['error']}")
    else:
        lines.extend([
            f"  Tasks completed: {health_summary.get('tasks_completed', 0)}",
            f"  Worker age: {health_summary.get('worker_age_minutes', 0):.1f} minutes",
            f"  Average threads: {health_summary.get('avg_threads', 0):.1f}",
            f"  Average memory: {health_summary.get('avg_memory_mb', 0):.1f}MB",
            f"  Average task duration: {health_summary.get('avg_task_duration', 0):.2f}s"
        ])
    
    # System section
    lines.extend([
        "",
        "System Information:",
        "-" * 20
    ])
    
    system_stats = diagnostics.get('system_stats', {})
    if 'error' in system_stats:
        lines.append(f"  Error: {system_stats['error']}")
    else:
        memory_info = system_stats.get('memory_info', {})
        lines.extend([
            f"  Process memory: {memory_info.get('rss', 0) / (1024**2):.1f}MB RSS, "
            f"{memory_info.get('vms', 0) / (1024**2):.1f}MB VMS",
            f"  Process threads: {system_stats.get('num_threads', 0)}",
            f"  Threading active: {system_stats.get('threading_active_count', 0)}",
            f"  CPU percent: {system_stats.get('cpu_percent', 0):.1f}%",
            f"  Open files: {system_stats.get('open_files', 0)}",
            f"  Network connections: {system_stats.get('connections', 0)}"
        ])
    
    system_memory = diagnostics.get('system_memory', {})
    if system_memory:
        lines.extend([
            f"  System memory: {system_memory.get('percent_used', 0):.1f}% used "
            f"({system_memory.get('available_gb', 0):.2f}GB available of "
            f"{system_memory.get('total_gb', 0):.2f}GB total)"
        ])
    
    if diagnostics['diagnostic_errors']:
        lines.extend([
            "",
            "Diagnostic Errors:",
            "-" * 20,
            *[f"  - {error}" for error in diagnostics['diagnostic_errors']]
        ])
    
    return "\n".join(lines)


def emergency_worker_cleanup():
    """
    Emergency cleanup function for worker processes experiencing issues.
    
    Attempts to clean up threads, reset monitoring, and provide diagnostics.
    """
    logging.critical("EMERGENCY_WORKER_CLEANUP: Starting emergency cleanup")
    
    cleanup_results = {
        'thread_cleanup': False,
        'memory_cleanup': False,
        'health_reset': False,
        'errors': []
    }
    
    # Thread cleanup
    try:
        from AlgoTuner.utils.thread_manager import cleanup_worker_threads
        successful, failed = cleanup_worker_threads(timeout=2.0)  # Short timeout for emergency
        cleanup_results['thread_cleanup'] = failed == 0
        logging.critical(f"EMERGENCY_WORKER_CLEANUP: Thread cleanup - success: {successful}, failed: {failed}")
    except Exception as e:
        cleanup_results['errors'].append(f"Thread cleanup failed: {e}")
        logging.critical(f"EMERGENCY_WORKER_CLEANUP: Thread cleanup failed: {e}")
    
    # Memory monitor cleanup
    try:
        from AlgoTuner.utils.process_monitor import cleanup_worker_memory_monitor
        cleanup_worker_memory_monitor()
        cleanup_results['memory_cleanup'] = True
        logging.critical("EMERGENCY_WORKER_CLEANUP: Memory monitor cleanup completed")
    except Exception as e:
        cleanup_results['errors'].append(f"Memory cleanup failed: {e}")
        logging.critical(f"EMERGENCY_WORKER_CLEANUP: Memory cleanup failed: {e}")
    
    # Health monitor reset
    try:
        from AlgoTuner.utils.worker_health import get_worker_health_monitor
        health_monitor = get_worker_health_monitor()
        health_monitor.reset_for_new_worker()
        cleanup_results['health_reset'] = True
        logging.critical("EMERGENCY_WORKER_CLEANUP: Health monitor reset completed")
    except Exception as e:
        cleanup_results['errors'].append(f"Health reset failed: {e}")
        logging.critical(f"EMERGENCY_WORKER_CLEANUP: Health reset failed: {e}")
    
    # Log final diagnostics
    try:
        log_worker_diagnostics(level=logging.CRITICAL)
    except Exception as e:
        logging.critical(f"EMERGENCY_WORKER_CLEANUP: Failed to log final diagnostics: {e}")
    
    success = all([
        cleanup_results['thread_cleanup'],
        cleanup_results['memory_cleanup'], 
        cleanup_results['health_reset']
    ])
    
    logging.critical(f"EMERGENCY_WORKER_CLEANUP: Completed with success={success}, errors={len(cleanup_results['errors'])}")
    return cleanup_results