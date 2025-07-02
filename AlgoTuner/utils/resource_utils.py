import logging
from typing import Optional

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:  # pragma: no cover – non-POSIX systems
    RESOURCE_AVAILABLE = False
    resource = None  # type: ignore

__all__ = ["raise_rlimit_as"]

def _to_readable(bytes_val: int) -> str:
    """Helper to format bytes as GiB with two decimals."""
    return f"{bytes_val / (1024 ** 3):.2f}GB"

def raise_rlimit_as(min_bytes: int, logger: Optional[logging.Logger] = None) -> None:
    """Ensure RLIMIT_AS soft/hard limits are *at least* ``min_bytes``.

    If the current limits are already higher, nothing is changed.  If the
    process lacks the privilege to raise the hard limit, we still attempt to
    raise the soft limit up to the existing hard cap.

    Parameters
    ----------
    min_bytes : int
        Desired minimum allowed virtual memory in **bytes**.
    logger : Optional[logging.Logger]
        Logger to use for diagnostic messages (defaults to ``logging`` root).
    """
    if not RESOURCE_AVAILABLE or min_bytes <= 0:
        return

    log = logger or logging.getLogger(__name__)

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    
    # Enhanced logging for debugging
    log.error(f"raise_rlimit_as: ENTRY - Current limits: soft={soft if soft != resource.RLIM_INFINITY else 'INFINITY'}, hard={hard if hard != resource.RLIM_INFINITY else 'INFINITY'}")
    log.error(f"raise_rlimit_as: ENTRY - Requested min_bytes: {min_bytes} ({min_bytes / (1024**3):.2f}GB)")
    if soft != resource.RLIM_INFINITY:
        log.error(f"raise_rlimit_as: ENTRY - Current soft limit: {soft / (1024**3):.2f}GB")
    if hard != resource.RLIM_INFINITY:
        log.error(f"raise_rlimit_as: ENTRY - Current hard limit: {hard / (1024**3):.2f}GB")

    # Determine targets – never lower existing limits
    target_soft = max(soft, min_bytes)
    target_hard = max(hard, min_bytes)

    # Enhanced logging for targets
    log.error(f"raise_rlimit_as: TARGETS - target_soft={target_soft if target_soft != resource.RLIM_INFINITY else 'INFINITY'}, target_hard={target_hard if target_hard != resource.RLIM_INFINITY else 'INFINITY'}")
    if target_soft != resource.RLIM_INFINITY:
        log.error(f"raise_rlimit_as: TARGETS - target_soft: {target_soft / (1024**3):.2f}GB")
    if target_hard != resource.RLIM_INFINITY:
        log.error(f"raise_rlimit_as: TARGETS - target_hard: {target_hard / (1024**3):.2f}GB")

    # Fast-path: nothing to do
    if (target_soft, target_hard) == (soft, hard):
        log.error("raise_rlimit_as: FAST-PATH - No changes needed, limits already sufficient")
        return

    try:
        # First try to raise both limits
        log.error(f"raise_rlimit_as: ATTEMPT - Trying to set both limits to soft={target_soft}, hard={target_hard}")
        resource.setrlimit(resource.RLIMIT_AS, (target_soft, target_hard))
        log.error(
            "raise_rlimit_as: SUCCESS - RLIMIT_AS raised to soft=%s, hard=%s",
            _to_readable(target_soft),
            _to_readable(target_hard),
        )
    except (ValueError, PermissionError, OSError) as e:
        # Could not change hard limit – try raising just the soft limit up to the current hard cap
        log.error(f"raise_rlimit_as: FAILED - Could not set both limits: {e}")
        try:
            capped_soft = min(target_soft, hard)
            log.error(f"raise_rlimit_as: FALLBACK - Trying to set only soft limit to {capped_soft} (capped by hard limit {hard})")
            if capped_soft > soft:
                resource.setrlimit(resource.RLIMIT_AS, (capped_soft, hard))
                log.error(
                    "raise_rlimit_as: FALLBACK SUCCESS - Soft limit raised to %s (hard=%s remains)",
                    _to_readable(capped_soft),
                    _to_readable(hard),
                )
            else:
                log.error(f"raise_rlimit_as: FALLBACK SKIP - Capped soft limit {capped_soft} not higher than current {soft}")
        except Exception as e:
            log.error("raise_rlimit_as: FALLBACK FAILED - Could not raise RLIMIT_AS – %s", e) 