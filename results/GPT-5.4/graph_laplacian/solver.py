from typing import Any

import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

@njit(cache=True)
def _try_fast_laplacian(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    n: int,
    normed: bool,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    empty_f = np.empty(0, dtype=np.float64)
    empty_i = np.empty(0, dtype=np.int64)

    if indptr.size != n + 1:
        return False, empty_f, empty_i, empty_i
    if n == 0:
        return True, empty_f, empty_i, np.array([0], dtype=np.int64)
    if indptr[0] != 0 or indptr[-1] != data.size:
        return False, empty_f, empty_i, empty_i

    degrees = np.empty(n, dtype=np.float64)
    diag_count = 0

    for i in range(n):
        start = indptr[i]
        end = indptr[i + 1]
        if end < start:
            return False, empty_f, empty_i, empty_i

        prev = -1
        s = 0.0
        for p in range(start, end):
            j = indices[p]
            v = data[p]
            if j < 0 or j >= n or j <= prev or j == i or v == 0.0:
                return False, empty_f, empty_i, empty_i
            prev = j
            s += v

        degrees[i] = s
        if s != 0.0:
            diag_count += 1
        elif normed and end > start:
            return False, empty_f, empty_i, empty_i

    if normed:
        for i in range(n):
            if degrees[i] < 0.0:
                return False, empty_f, empty_i, empty_i
        for p in range(data.size):
            if degrees[indices[p]] <= 0.0:
                return False, empty_f, empty_i, empty_i

    total = data.size + diag_count
    out_data = np.empty(total, dtype=np.float64)
    out_indices = np.empty(total, dtype=np.int64)
    out_indptr = np.empty(n + 1, dtype=np.int64)
    out_indptr[0] = 0

    inv = np.empty(n, dtype=np.float64)
    if normed:
        for i in range(n):
            if degrees[i] > 0.0:
                inv[i] = 1.0 / np.sqrt(degrees[i])
            else:
                inv[i] = 0.0

    dst = 0
    for i in range(n):
        start = indptr[i]
        end = indptr[i + 1]
        d = degrees[i]

        if d != 0.0:
            lo = start
            hi = end
            while lo < hi:
                mid = (lo + hi) >> 1
                if indices[mid] < i:
                    lo = mid + 1
                else:
                    hi = mid
            ins = lo

            for p in range(start, ins):
                j = indices[p]
                out_indices[dst] = j
                if normed:
                    out_data[dst] = -data[p] * inv[i] * inv[j]
                else:
                    out_data[dst] = -data[p]
                dst += 1

            out_indices[dst] = i
            out_data[dst] = 1.0 if normed else d
            dst += 1

            for p in range(ins, end):
                j = indices[p]
                out_indices[dst] = j
                if normed:
                    out_data[dst] = -data[p] * inv[i] * inv[j]
                else:
                    out_data[dst] = -data[p]
                dst += 1
        else:
            for p in range(start, end):
                out_indices[dst] = indices[p]
                out_data[dst] = -data[p]
                dst += 1

        out_indptr[i + 1] = dst

    return True, out_data, out_indices, out_indptr

class Solver:
    __slots__ = ("_csr_matrix", "_laplacian")

    def __init__(self) -> None:
        self._csr_matrix = sp.csr_matrix
        self._laplacian = csgraph_laplacian
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        empty_p = np.array([0], dtype=np.int64)
        _try_fast_laplacian(empty_f, empty_i, empty_p, 0, False)
        _try_fast_laplacian(empty_f, empty_i, empty_p, 0, True)

    @staticmethod
    def _failure(shape: Any) -> dict[str, dict[str, Any]]:
        return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}

    def solve(self, problem, **kwargs) -> Any:
        try:
            shp = problem["shape"]
            shape = (int(shp[0]), int(shp[1]))
            if shape[0] != shape[1]:
                return self._failure(shape)

            data = np.asarray(problem["data"], dtype=np.float64)
            indices = np.asarray(problem["indices"], dtype=np.int64)
            indptr = np.asarray(problem["indptr"], dtype=np.int64)
            normed = bool(problem["normed"])
        except Exception:
            shape = problem.get("shape", (0, 0)) if isinstance(problem, dict) else (0, 0)
            return self._failure(shape)

        try:
            ok, out_data, out_indices, out_indptr = _try_fast_laplacian(
                data, indices, indptr, shape[0], normed
            )
            if ok:
                return {
                    "laplacian": {
                        "data": out_data,
                        "indices": out_indices,
                        "indptr": out_indptr,
                        "shape": shape,
                    }
                }

            graph_csr = self._csr_matrix((data, indices, indptr), shape=shape)
            lap = self._laplacian(graph_csr, normed=normed)
            if not isinstance(lap, sp.csr_matrix):
                lap = lap.tocsr()
            lap.eliminate_zeros()
            return {
                "laplacian": {
                    "data": lap.data,
                    "indices": lap.indices,
                    "indptr": lap.indptr,
                    "shape": lap.shape,
                }
            }
        except Exception:
            return self._failure(shape)