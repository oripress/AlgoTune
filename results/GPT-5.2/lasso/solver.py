from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    """
    Optimized wrapper around sklearn.linear_model.Lasso:

    - cache the estimator (avoid per-call construction)
    - set copy_X=False (reference uses copy_X=True)
    - build X directly as Fortran-ordered when input is a Python list, to avoid
      sklearn's internal extra copy to Fortran order.
    - use cyclic CD (same as reference default) for determinism
    """

    def __init__(self, alpha: float = 0.1, max_iter: int = 1000, tol: float = 1e-4) -> None:
        from sklearn.linear_model import Lasso  # type: ignore  # init not timed

        self._clf = Lasso(
            alpha=float(alpha),
            fit_intercept=False,
            copy_X=False,
            max_iter=int(max_iter),
            tol=float(tol),
            selection="cyclic",
            warm_start=False,
        )

    @staticmethod
    def _as_fortran_float64(X_in: Any) -> np.ndarray:
        # If already an ndarray, request Fortran order without unnecessary copy.
        if isinstance(X_in, np.ndarray):
            return np.asarray(X_in, dtype=np.float64, order="F")
        # For Python lists, construct directly in Fortran order (single allocation/copy).
        return np.array(X_in, dtype=np.float64, order="F")

    @staticmethod
    def _as_float64_1d(y_in: Any) -> np.ndarray:
        if isinstance(y_in, np.ndarray):
            y = np.asarray(y_in, dtype=np.float64)
        else:
            y = np.array(y_in, dtype=np.float64)
        if y.ndim != 1:
            y = y.reshape(-1)
        return y

    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        clf = self._clf

        # Keep overrides (if any) cheap.
        a = kwargs.get("alpha", None)
        if a is not None:
            a = float(a)
            if a != clf.alpha:
                clf.alpha = a
        mi = kwargs.get("max_iter", None)
        if mi is not None:
            mi = int(mi)
            if mi != clf.max_iter:
                clf.max_iter = mi
        t = kwargs.get("tol", None)
        if t is not None:
            t = float(t)
            if t != clf.tol:
                clf.tol = t

        X = self._as_fortran_float64(problem["X"])
        y = self._as_float64_1d(problem["y"])

        clf.fit(X, y)
        return clf.coef_.tolist()