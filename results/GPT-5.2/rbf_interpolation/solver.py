from __future__ import annotations

from typing import Any

import os

import numpy as np

try:
    from scipy.interpolate import RBFInterpolator
except Exception:  # pragma: no cover
    RBFInterpolator = None  # type: ignore[assignment]

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None  # type: ignore[assignment]

class Solver:
    """
    RBF interpolation matching the reference (SciPy global RBFInterpolator),
    with lightweight performance tweaks.

    - Avoid Python list conversion (return NumPy array directly).
    - Optionally control BLAS thread counts. Instead of using a context manager
      (which has overhead every call), we set thread limits persistently and only
      change them when the heuristic target changes.
    """

    def __init__(self) -> None:
        max_threads = int(os.cpu_count() or 1)
        self._max_threads = 8 if max_threads > 8 else max_threads

        # Threadpool controller state (returned by threadpool_limits when used as a function).
        self._tp_controller: Any | None = None
        self._tp_n: int | None = None

    def _set_threads(self, n_threads: int) -> None:
        if threadpool_limits is None:
            return
        if self._tp_n == n_threads:
            return

        # Restore original limits if we previously changed them.
        if self._tp_controller is not None:
            try:
                self._tp_controller.restore_original_limits()
            except Exception:
                pass
            self._tp_controller = None
            self._tp_n = None

        # Set new limits persistently (until restored/changed again).
        try:
            self._tp_controller = threadpool_limits(limits=n_threads)
            self._tp_n = n_threads
        except Exception:
            self._tp_controller = None
            self._tp_n = None

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        rbf_cls = RBFInterpolator
        if rbf_cls is None:  # pragma: no cover
            raise RuntimeError("SciPy is required for this solver.")

        x_train = np.asarray(problem["x_train"], dtype=np.float64)
        y_train = np.asarray(problem["y_train"], dtype=np.float64).ravel()
        x_test = np.asarray(problem["x_test"], dtype=np.float64)

        rbf_config = problem.get("rbf_config") or {}
        n = int(x_train.shape[0])

        # Heuristic: avoid threading overhead for small/medium n; scale threads with n.
        # (_set_threads is a no-op if threadpoolctl is unavailable.)
        if n <= 400:
            nt = 1
        elif n <= 900:
            nt = 2
        elif n <= 1600:
            nt = 4
        else:
            nt = self._max_threads
        self._set_threads(nt)

        rbf = rbf_cls(
            x_train,
            y_train,
            kernel=rbf_config.get("kernel"),
            epsilon=rbf_config.get("epsilon"),
            smoothing=rbf_config.get("smoothing"),
        )
        return {"y_pred": rbf(x_test)}