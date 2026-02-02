from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

# Small global caches to avoid repeated allocations for common shapes.
_ARANGE_CACHE: dict[int, np.ndarray] = {}
_BOUNDS_CACHE: dict[
    tuple[int, int, int], list[tuple[float | None, float | None]]
] = {}
_AEQ_STRUCT_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

class Solver:
    def __init__(self) -> None:
        # Initialization not counted towards runtime.
        pass

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solve quantile regression via linear programming (HiGHS through SciPy),
        matching sklearn.linear_model.QuantileRegressor with alpha=0.

        Notes:
        - sklearn returns coef_ as shape (n_features,), so we return a 1D list.
        - intercept is returned as list of length 1 to match the reference.
        """
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        q = float(problem["quantile"])
        fit_intercept = bool(problem["fit_intercept"])

        n, p = X.shape

        # Variable order:
        #   beta_pos (p, >=0),
        #   beta_neg (p, >=0),
        #   intercept (1, free) if fit_intercept,
        #   u (n, >=0),
        #   v (n, >=0)
        ib = 1 if fit_intercept else 0
        k_dense = 2 * p + ib
        m = k_dense + 2 * n

        # Build A_eq in CSC with cached (indices, indptr) and fast data fill.
        ar = _ARANGE_CACHE.get(n)
        if ar is None:
            ar = np.arange(n, dtype=np.int32)
            _ARANGE_CACHE[n] = ar

        skey = (n, p, ib)
        struct = _AEQ_STRUCT_CACHE.get(skey)
        if struct is None:
            # indptr
            indptr = np.empty(m + 1, dtype=np.int64)
            indptr[: k_dense + 1] = (np.arange(k_dense + 1, dtype=np.int64) * n)
            start = k_dense * n
            indptr[k_dense + 1 : k_dense + n + 1] = start + np.arange(
                1, n + 1, dtype=np.int64
            )
            indptr[k_dense + n + 1 : m + 1] = start + n + np.arange(
                1, n + 1, dtype=np.int64
            )

            # indices
            nnz = start + 2 * n
            indices = np.empty(nnz, dtype=np.int32)
            if k_dense:
                indices[:start] = np.tile(ar, k_dense)
            indices[start : start + n] = ar
            indices[start + n : start + 2 * n] = ar

            struct = (indices, indptr)
            _AEQ_STRUCT_CACHE[skey] = struct

        indices, indptr = struct
        nnz = indices.size
        data = np.empty(nnz, dtype=np.float64)

        # Dense blocks: +X, -X, optional intercept
        # Fill column-major so each column corresponds to a feature.
        if p:
            Xf = np.ravel(X, order="F")  # length n*p
            pn = p * n
            data[:pn] = Xf
            data[pn : 2 * pn] = -Xf
            pos = 2 * pn
        else:
            pos = 0

        if fit_intercept:
            data[pos : pos + n] = 1.0
            pos += n

        # u and v diagonal blocks
        data[pos : pos + n] = 1.0
        data[pos + n : pos + 2 * n] = -1.0

        A_eq = csc_matrix((data, indices, indptr), shape=(n, m), copy=False)

        # Objective: q*sum(u) + (1-q)*sum(v). Coefs have 0 weight since alpha=0.
        c = np.zeros(m, dtype=np.float64)
        u0 = k_dense
        v0 = u0 + n
        c[u0:v0] = q
        c[v0 : v0 + n] = 1.0 - q

        # Objective: q*sum(u) + (1-q)*sum(v). Coefs have 0 weight since alpha=0.
        c = np.zeros(m, dtype=np.float64)
        u0 = 2 * p + ib
        v0 = u0 + n
        c[u0:v0] = q
        c[v0 : v0 + n] = 1.0 - q

        # Bounds (SciPy-compatible): list of (low, high) tuples.
        bkey = (p, n, ib)
        bounds = _BOUNDS_CACHE.get(bkey)
        if bounds is None:
            bounds = [(0.0, None)] * m
            if fit_intercept:
                bounds = list(bounds)
                bounds[2 * p] = (None, None)  # free intercept
            _BOUNDS_CACHE[bkey] = bounds

        res = linprog(
            c,
            A_eq=A_eq,
            b_eq=y,
            bounds=bounds,
            method="highs",
        )

        if res.x is None or res.status != 0:
            coef = np.zeros(p, dtype=np.float64)
            intercept = float(np.quantile(y, q)) if fit_intercept else 0.0
        else:
            x = res.x
            coef = x[:p] - x[p : 2 * p]
            intercept = float(x[2 * p]) if fit_intercept else 0.0

        preds = X @ coef + intercept

        return {
            "coef": coef.tolist(),
            "intercept": [intercept],
            "predictions": preds.tolist(),
        }