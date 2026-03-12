from __future__ import annotations

from typing import Any

import numpy as np

try:
    import scipy.linalg as _sla  # type: ignore
except Exception:  # pragma: no cover
    _sla = None


class Solver:
    def __init__(self) -> None:
        pass

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        k = int(problem["n_components"])
        X_in = problem["X"]

        # Avoid mutating the input if it's already an ndarray.
        X = np.asarray(X_in, dtype=np.float64)
        if isinstance(X_in, np.ndarray):
            X = X.copy()

        # Center in-place.
        X -= X.mean(axis=0)

        # SciPy wins on larger matrices when the LAPACK call can avoid checks/copies.
        if _sla is not None and X.size >= 50_000:
            if not X.flags.f_contiguous:
                X = np.asfortranarray(X)
            _u, _s, vt = _sla.svd(
                X,
                full_matrices=False,
                overwrite_a=True,
                check_finite=False,
                lapack_driver="gesdd",
            )
            return vt[:k]

        _u, _s, vt = np.linalg.svd(X, full_matrices=False)
        return vt[:k]
