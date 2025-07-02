"""
blas_utils.py – central place to keep the BLAS thread-count configuration consistent
across baseline measurements and solver benchmarks.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

# Vars understood by the usual BLAS / OpenMP libraries
_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _desired_thread_count(override: Optional[int] = None) -> int:
    """Pick the thread count that should be used for *all* timings."""
    if override is not None and override > 0:
        return int(override)

    # 1) explicit env override wins
    env_val = os.environ.get("ALGOTUNE_BLAS_THREADS")
    if env_val and env_val.isdigit():
        return int(env_val)

    # 2) honour current CPU-affinity / cpuset if available
    try:
        import os
        affinity_cnt = len(os.sched_getaffinity(0))
        if affinity_cnt > 0:
            return affinity_cnt
    except Exception:
        pass  # not available on this platform

    # 3) fall back to total logical cores
    try:
        import psutil
        return psutil.cpu_count(logical=True) or 1
    except Exception:
        return os.cpu_count() or 1


def set_blas_threads(n_threads: Optional[int] = None) -> int:
    """Make *this* process use exactly *n_threads* BLAS / OpenMP threads.

    The value is also exported via environment so that forked/spawned children
    inherit the setting.
    Returns the thread count in effect so callers can log it.
    """
    n_threads = _desired_thread_count(n_threads)

    # 1) Export to env so that future subprocesses inherit it
    for var in _ENV_VARS:
        os.environ[var] = str(n_threads)

    # 2) Change runtime threadpools where possible (after NumPy import)
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(n_threads)
    except Exception:
        # threadpoolctl not installed or failed – silently ignore
        pass

    # NumPy 2.0 exposes set_num_threads; older versions do not – ignore errors
    try:
        import numpy as _np

        if hasattr(_np, "set_num_threads"):
            _np.set_num_threads(n_threads)
    except Exception:
        pass

    logging.info(
        f"BLAS thread configuration set to {n_threads} threads (exported via {_ENV_VARS})."
    )
    return n_threads


def log_current_blas_threads(msg_prefix: str = "") -> None:
    """Emit a log entry with the current threadpoolctl information."""
    try:
        import threadpoolctl

        pools = threadpoolctl.threadpool_info()
        logging.info(
            f"{msg_prefix}Current BLAS pools: " + ", ".join(
                f"{p['internal_api']}({p['prefix']}):{p['num_threads']}" for p in pools
            )
        )
    except Exception as e:
        logging.debug(f"Could not obtain BLAS pool info: {e}")


def log_cpu_affinity(prefix: str = "") -> None:
    """Log the number of CPUs the current process is allowed to run on."""
    try:
        import os
        cpus = os.sched_getaffinity(0)
        logging.info(f"{prefix}CPU affinity: {len(cpus)} cores ({sorted(cpus)[:8]}{'...' if len(cpus)>8 else ''})")
    except AttributeError:
        logging.debug("sched_getaffinity not available on this platform")


def log_thread_env(prefix: str = "") -> None:
    """Log current thread-related environment variables (OMP, MKL, etc.)."""
    vals = {var: os.environ.get(var, "<unset>") for var in _ENV_VARS}
    logging.info(
        f"{prefix}Thread env vars: " + ", ".join(f"{k}={v}" for k, v in vals.items())
    ) 