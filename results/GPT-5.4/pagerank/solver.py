from __future__ import annotations

from itertools import chain
from typing import Any

import numpy as np
from numba import njit

@njit(cache=True)
def _pagerank_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    outdeg: np.ndarray,
    dangling_idx: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    n = outdeg.shape[0]
    inv_n = 1.0 / n
    teleport = (1.0 - alpha) * inv_n
    threshold = n * tol

    x = np.empty(n, dtype=np.float64)
    xnew = np.empty(n, dtype=np.float64)
    for i in range(n):
        x[i] = inv_n

    for it in range(max_iter):
        dangling_sum = 0.0
        for k in range(dangling_idx.shape[0]):
            dangling_sum += x[dangling_idx[k]]

        base = alpha * dangling_sum * inv_n + teleport
        for j in range(n):
            xnew[j] = base

        for i in range(n):
            deg = outdeg[i]
            if deg != 0:
                contrib = alpha * x[i] / deg
                start = indptr[i]
                end = indptr[i + 1]
                for p in range(start, end):
                    xnew[indices[p]] += contrib

        err = 0.0
        for j in range(n):
            err += abs(xnew[j] - x[j])

        if err < threshold:
            return xnew, it + 1

        x, xnew = xnew, x

    return x, -1

class Solver:
    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

        dummy_indptr = np.array([0, 1], dtype=np.int64)
        dummy_indices = np.array([0], dtype=np.int32)
        dummy_outdeg = np.array([1], dtype=np.int32)
        dummy_dangling = np.empty(0, dtype=np.int64)
        _pagerank_numba(dummy_indptr, dummy_indices, dummy_outdeg, dummy_dangling, alpha, 1, tol)

    def solve(self, problem, **kwargs) -> Any:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        alpha = float(kwargs.get("alpha", self.alpha))
        max_iter = int(kwargs.get("max_iter", self.max_iter))
        tol = float(kwargs.get("tol", self.tol))

        outdeg = np.fromiter((len(neigh) for neigh in adj_list), dtype=np.int32, count=n)
        m = int(outdeg.sum())
        if m == 0:
            inv_n = 1.0 / n
            return {"pagerank_scores": [inv_n] * n}

        indptr = np.empty(n + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(outdeg, dtype=np.int64, out=indptr[1:])
        indices = np.fromiter(chain.from_iterable(adj_list), dtype=np.int32, count=m)
        dangling_idx = np.flatnonzero(outdeg == 0).astype(np.int64)

        scores, iters = _pagerank_numba(
            indptr,
            indices,
            outdeg,
            dangling_idx,
            alpha,
            max_iter,
            tol,
        )
        if iters < 0:
            return {"pagerank_scores": [0.0] * n}
        return {"pagerank_scores": scores.tolist()}