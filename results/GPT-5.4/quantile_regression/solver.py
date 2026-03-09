from typing import Any, cast

import numpy as np
from scipy import sparse
from scipy.optimize import _linprog_highs, _linprog_util

class Solver:
    def __init__(self) -> None:
        self._rows_cache: dict[int, np.ndarray] = {}
        self._dense_rows_cache: dict[tuple[int, int], np.ndarray] = {}
        self._indptr_cache: dict[tuple[int, int], np.ndarray] = {}
        self._bounds_cache: dict[int, np.ndarray] = {}
        self._empty_a_cache: dict[int, sparse.csc_matrix] = {}

    def _rows(self, n: int) -> np.ndarray:
        rows = self._rows_cache.get(n)
        if rows is None:
            rows = np.arange(n, dtype=np.int32)
            self._rows_cache[n] = rows
        return rows

    def _dense_rows(self, n: int, m: int) -> np.ndarray:
        key = (n, m)
        dense_rows = self._dense_rows_cache.get(key)
        if dense_rows is None:
            dense_rows = np.tile(self._rows(n), m)
            self._dense_rows_cache[key] = dense_rows
        return dense_rows

    def _indptr(self, n: int, m: int) -> np.ndarray:
        key = (n, m)
        indptr = self._indptr_cache.get(key)
        if indptr is None:
            n_dense = n * m
            n_cols = 2 * m + 2 * n
            indptr = np.empty(n_cols + 1, dtype=np.int32)
            indptr[: 2 * m + 1] = np.arange(0, 2 * n_dense + 1, n, dtype=np.int32)
            indptr[2 * m :] = 2 * n_dense + np.arange(0, 2 * n + 1, dtype=np.int32)
            self._indptr_cache[key] = indptr
        return indptr

    def _bounds(self, n_cols: int) -> np.ndarray:
        bounds = self._bounds_cache.get(n_cols)
        if bounds is None:
            bounds = np.empty((n_cols, 2), dtype=np.float64)
            bounds[:, 0] = 0.0
            bounds[:, 1] = np.inf
            self._bounds_cache[n_cols] = bounds
        return bounds

    def _empty_a(self, n_cols: int) -> sparse.csc_matrix:
        empty_a = self._empty_a_cache.get(n_cols)
        if empty_a is None:
            empty_a = sparse.csc_matrix((0, n_cols), dtype=np.float64)
            self._empty_a_cache[n_cols] = empty_a
        return empty_a

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        X = np.array(problem["X"], dtype=np.float64, order="F")
        y = np.asarray(problem["y"], dtype=np.float64)
        q = float(problem["quantile"])
        fit_intercept = bool(problem["fit_intercept"])

        n_samples, n_features = X.shape
        n_params = n_features + int(fit_intercept)
        n_dense = n_samples * n_params
        n_nz = 2 * n_dense + 2 * n_samples
        n_cols = 2 * n_params + 2 * n_samples

        c = np.empty(n_cols, dtype=np.float64)
        c[: 2 * n_params] = 0.0
        c[2 * n_params : 2 * n_params + n_samples] = q
        c[2 * n_params + n_samples :] = 1.0 - q

        data = np.empty(n_nz, dtype=np.float64)
        flat_x = X.ravel(order="F")
        if fit_intercept:
            data[:n_samples] = 1.0
            data[n_samples:n_dense] = flat_x
            data[n_dense : n_dense + n_samples] = -1.0
            data[n_dense + n_samples : 2 * n_dense] = -flat_x
        else:
            data[:n_dense] = flat_x
            data[n_dense : 2 * n_dense] = -flat_x
        data[2 * n_dense : 2 * n_dense + n_samples] = 1.0
        data[2 * n_dense + n_samples :] = -1.0

        rows = self._rows(n_samples)
        dense_rows = self._dense_rows(n_samples, n_params)
        indices = np.empty(n_nz, dtype=np.int32)
        indices[:n_dense] = dense_rows
        indices[n_dense : 2 * n_dense] = dense_rows
        indices[2 * n_dense : 2 * n_dense + n_samples] = rows
        indices[2 * n_dense + n_samples :] = rows

        a_eq = sparse.csc_matrix(
            (data, indices, self._indptr(n_samples, n_params)),
            shape=(n_samples, n_cols),
        )

        lp = _linprog_util._LPProblem(
            c,
            self._empty_a(n_cols),
            np.empty(0, dtype=np.float64),
            a_eq,
            y,
            self._bounds(n_cols),
            None,
            None,
        )
        res = cast(dict[str, Any], _linprog_highs._linprog_highs(lp, solver=None))
        sol = np.asarray(res["x"], dtype=np.float64)
        params = sol[:n_params] - sol[n_params : 2 * n_params]

        if fit_intercept:
            intercept = float(params[0])
            coef = params[1:]
        else:
            intercept = 0.0
            coef = params

        predictions = X @ coef + intercept
        return {
            "coef": coef.tolist(),
            "intercept": [intercept],
            "predictions": predictions.tolist(),
        }