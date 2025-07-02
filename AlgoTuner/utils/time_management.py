"""
Time management utilities.
This module re-exports timing utilities from precise_timing.py for backward compatibility.
"""

from AlgoTuner.utils.precise_timing import time_limit, TimeoutError

def calculate_timeout(optimal_time: float, multiplier: float = 10.0) -> float:
    """Calculate timeout based on optimal time."""
    return optimal_time * multiplier

def format_time_ms(ms: float) -> str:
    """Format milliseconds into a human-readable string with 3 decimal places."""
    if ms < 1:
        # For values under 1ms, show microseconds
        return f"{ms*1000:.3f}μs"
    elif ms < 1000:
        # For values under 1 second, show milliseconds
        return f"{ms:.3f}ms"
    elif ms < 60000:
        # For values under 1 minute, show seconds
        seconds = ms / 1000
        return f"{seconds:.3f}s"
    else:
        # For values over 1 minute (and under 1 hour)
        minutes = int(ms / 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:.3f}s"

__all__ = ['time_limit', 'TimeoutError', 'calculate_timeout', 'format_time_ms'] 