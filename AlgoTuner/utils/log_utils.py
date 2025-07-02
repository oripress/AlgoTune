"""Utility helpers for logging.

Currently only provides a low-overhead timestamp based on time.perf_counter
so that debug messages across the code base can avoid the ambiguities of
`time.time()` (which jumps if the system clock changes).

It intentionally returns **seconds** as a float, matching the semantics that
existing log statements expect for the `:.3f` format specifier.
"""

from time import perf_counter as _perf_counter

__all__ = ["ts"]

def ts() -> float:  # noqa: D401
    """Return a monotonically increasing timestamp in seconds."""
    return _perf_counter() 