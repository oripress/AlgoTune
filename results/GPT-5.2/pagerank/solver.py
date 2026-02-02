from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

if njit is not None:

    @njit(cache=True)
    def _transpose_out_to_in(
        out_indptr: np.ndarray,
        out_indices: np.ndarray,
        in_indptr: np.ndarray,
        in_indices: np.ndarray,
    ) -> None:
        """
        Build incoming adjacency (CSR by destination) from outgoing CSR.
        For each edge (src -> dst), append src to incoming list of dst.
        """
        n = out_indptr.size - 1
        cursor = in_indptr[:-1].copy()
        for src in range(n):
            start = out_indptr[src]
            end = out_indptr[src + 1]
            for k in range(start, end):
                dst = out_indices[k]
                pos = cursor[dst]
                in_indices[pos] = src
                cursor[dst] = pos + 1

    @njit(cache=True)
    def _pagerank_incoming(
        in_indptr: np.ndarray,
        in_indices: np.ndarray,
        dangling: np.ndarray,
        inv_outdeg: np.ndarray,
        alpha: float,
        max_iter: int,
        tol: float,
    ) -> np.ndarray:
        n = inv_outdeg.size
        inv_n = 1.0 / n
        base = (1.0 - alpha) * inv_n
        thresh = n * tol

        x_old = np.empty(n, dtype=np.float64)
        x_new = np.empty(n, dtype=np.float64)

        for i in range(n):
            x_old[i] = inv_n

        x_res = x_old
        for _ in range(max_iter):
            danglesum = 0.0
            for t in range(dangling.size):
                danglesum += x_old[dangling[t]]
            add = base + alpha * danglesum * inv_n

            err = 0.0
            for i in range(n):
                s = 0.0
                start = in_indptr[i]
                end = in_indptr[i + 1]
                for k in range(start, end):
                    src = in_indices[k]
                    s += x_old[src] * inv_outdeg[src]
                xi = alpha * s + add
                x_new[i] = xi
                diff = xi - x_old[i]
                if diff < 0.0:
                    diff = -diff
                err += diff

            x_res = x_new
            if err < thresh:
                break

            tmp = x_old
            x_old = x_new
            x_new = tmp

        ssum = 0.0
        for i in range(n):
            ssum += x_res[i]
        if ssum > 0.0:
            invs = 1.0 / ssum
            for i in range(n):
                x_res[i] *= invs
        else:
            for i in range(n):
                x_res[i] = inv_n
        return x_res

class Solver:
    """
    Fast PageRank solver (power iteration) on an adjacency list.

    Matches NetworkX pagerank defaults used by the reference:
      - alpha=0.85
      - max_iter=100
      - tol=1e-6
      - uniform personalization and dangling weights
      - L1 convergence check: err < n * tol
    """

    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1.0e-6):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Compile numba during init (init time not counted).
        if njit is not None:
            out_indptr = np.array([0, 1], dtype=np.int64)
            out_indices = np.array([0], dtype=np.int32)
            in_indptr = np.array([0, 1], dtype=np.int64)
            in_indices = np.empty(1, dtype=np.int32)
            _transpose_out_to_in(out_indptr, out_indices, in_indptr, in_indices)

            dangling = np.array([], dtype=np.int32)
            inv_outdeg = np.array([1.0], dtype=np.float64)
            _pagerank_incoming(in_indptr, in_indices, dangling, inv_outdeg, 0.85, 1, 1.0e-6)

    @staticmethod
    def _build_incoming(
        adj_list: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(adj_list)

        outdeg = np.fromiter((len(nei) for nei in adj_list), dtype=np.int32, count=n)

        out_indptr = np.empty(n + 1, dtype=np.int64)
        out_indptr[0] = 0
        np.cumsum(outdeg, out=out_indptr[1:])
        m = int(out_indptr[-1])

        out_indices = np.empty(m, dtype=np.int32)
        pos = 0
        for neigh in adj_list:
            deg = len(neigh)
            if deg:
                out_indices[pos : pos + deg] = neigh
                pos += deg

        if m:
            indeg = np.bincount(out_indices, minlength=n).astype(np.int32, copy=False)
        else:
            indeg = np.zeros(n, dtype=np.int32)

        in_indptr = np.empty(n + 1, dtype=np.int64)
        in_indptr[0] = 0
        np.cumsum(indeg, out=in_indptr[1:])
        in_indices = np.empty(m, dtype=np.int32)

        if njit is not None and m:
            _transpose_out_to_in(out_indptr, out_indices, in_indptr, in_indices)
        elif m:
            cursor = in_indptr[:-1].copy()
            for src, neigh in enumerate(adj_list):
                for dst in neigh:
                    p = cursor[dst]
                    in_indices[p] = src
                    cursor[dst] = p + 1

        inv_outdeg = np.zeros(n, dtype=np.float64)
        np.divide(1.0, outdeg, out=inv_outdeg, where=outdeg != 0)

        dangling = np.flatnonzero(outdeg == 0).astype(np.int32, copy=False)
        return in_indptr, in_indices, dangling, inv_outdeg

    def solve(self, problem: dict[str, list[list[int]]], **kwargs: Any) -> dict[str, list[float]]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        alpha = float(kwargs.get("alpha", self.alpha))
        max_iter = int(kwargs.get("max_iter", self.max_iter))
        tol = float(kwargs.get("tol", self.tol))

        in_indptr, in_indices, dangling, inv_outdeg = self._build_incoming(adj_list)

        if njit is not None:
            r = _pagerank_incoming(in_indptr, in_indices, dangling, inv_outdeg, alpha, max_iter, tol)
            return {"pagerank_scores": r.tolist()}

        # Fallback without numba.
        inv_n = 1.0 / n
        r = np.full(n, inv_n, dtype=np.float64)
        base = (1.0 - alpha) * inv_n
        thresh = n * tol

        for _ in range(max_iter):
            r_last = r
            danglesum = float(r_last[dangling].sum()) if dangling.size else 0.0
            add = base + alpha * danglesum * inv_n

            r = np.empty_like(r_last)
            for i in range(n):
                s = 0.0
                for k in range(in_indptr[i], in_indptr[i + 1]):
                    src = in_indices[k]
                    s += r_last[src] * inv_outdeg[src]
                r[i] = alpha * s + add

            if float(np.abs(r - r_last).sum()) < thresh:
                break

        ssum = float(r.sum())
        if ssum > 0.0 and np.isfinite(ssum):
            r /= ssum
        else:
            r[:] = inv_n

        return {"pagerank_scores": r.tolist()}