from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None

if nb is not None:

    @nb.njit(cache=True)
    def _dp_small_assignment(
        n: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        size = 1 << n
        inf = 1e300

        dp = np.empty(size, dtype=np.float64)
        dp.fill(inf)
        dp[0] = 0.0

        parent_col = np.empty(size, dtype=np.int64)
        parent_col.fill(-1)

        active = np.empty(size, dtype=np.int64)
        active[0] = 0
        active_len = 1

        seen = np.zeros(size, dtype=np.uint8)

        for r in range(n):
            ndp = np.empty(size, dtype=np.float64)
            ndp.fill(inf)

            next_active = np.empty(size, dtype=np.int64)
            next_len = 0

            start = indptr[r]
            end = indptr[r + 1]

            for i in range(active_len):
                mask = active[i]
                base = dp[mask]

                for k in range(start, end):
                    c = indices[k]
                    if c < 0 or c >= n:
                        return False, np.empty(0, dtype=np.int64)
                    bit = 1 << c
                    if (mask & bit) == 0:
                        nmask = mask | bit
                        cand = base + data[k]
                        if cand < ndp[nmask]:
                            ndp[nmask] = cand
                            parent_col[nmask] = c
                        if seen[nmask] == 0:
                            seen[nmask] = 1
                            next_active[next_len] = nmask
                            next_len += 1

            if next_len == 0:
                return False, np.empty(0, dtype=np.int64)

            for i in range(next_len):
                seen[next_active[i]] = 0

            dp = ndp
            active = next_active
            active_len = next_len

        full_mask = size - 1
        if dp[full_mask] >= inf * 0.5:
            return False, np.empty(0, dtype=np.int64)

        cols = np.empty(n, dtype=np.int64)
        mask = full_mask
        for r in range(n - 1, -1, -1):
            c = parent_col[mask]
            if c < 0:
                return False, np.empty(0, dtype=np.int64)
            cols[r] = c
            mask ^= 1 << c

        return True, cols

class Solver:
    def __init__(self) -> None:
        self._rows_cache = {}

        if nb is not None:
            indptr = np.array([0, 1], dtype=np.int64)
            indices = np.array([0], dtype=np.int64)
            data = np.array([0.0], dtype=np.float64)
            _dp_small_assignment(1, indptr, indices, data)

    def _empty(self) -> dict[str, dict[str, list[int]]]:
        return {"assignment": {"row_ind": [], "col_ind": []}}

    def _rows(self, n: int) -> list[int]:
        rows = self._rows_cache.get(n)
        if rows is None:
            rows = list(range(n))
            self._rows_cache[n] = rows
        return rows

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        try:
            shape = problem["shape"]
            n = int(shape[0])
            m = int(shape[1])
            if n != m or n < 0:
                return self._empty()

            if n == 0:
                return self._empty()

            indices = np.asarray(problem["indices"], dtype=np.int64)
            indptr = np.asarray(problem["indptr"], dtype=np.int64)
            data_arr = np.asarray(problem["data"])

            if indptr.shape[0] != n + 1:
                return self._empty()

            nnz = indices.shape[0]
            if data_arr.shape[0] != nnz:
                return self._empty()

            if nnz < n:
                return self._empty()
        except Exception:
            return self._empty()

        try:
            # Fast forced-permutation case: exactly one available column per row.
            if nnz == n:
                if np.all(indptr[1:] - indptr[:-1] == 1):
                    cols = indices[indptr[:-1]]
                    if cols.shape[0] == n:
                        if n == 1:
                            c0 = int(cols[0])
                            if c0 == 0:
                                return {
                                    "assignment": {
                                        "row_ind": [0],
                                        "col_ind": [0],
                                    }
                                }
                            return self._empty()

                        if int(cols.min()) >= 0 and int(cols.max()) < n:
                            seen = np.zeros(n, dtype=np.uint8)
                            ok = True
                            for c in cols:
                                if seen[c]:
                                    ok = False
                                    break
                                seen[c] = 1
                            if ok:
                                return {
                                    "assignment": {
                                        "row_ind": self._rows(n),
                                        "col_ind": cols.tolist(),
                                    }
                                }

            if nb is not None and n <= 16:
                if nnz == 0:
                    return self._empty()
                data64 = data_arr.astype(np.float64, copy=False)
                ok, cols = _dp_small_assignment(n, indptr, indices, data64)
                if ok:
                    return {
                        "assignment": {
                            "row_ind": self._rows(n),
                            "col_ind": cols.tolist(),
                        }
                    }
                return self._empty()

            mat = sp.csr_matrix((data_arr, indices, indptr), shape=(n, n), copy=False)
            row_ind, col_ind = min_weight_full_bipartite_matching(mat)
            return {
                "assignment": {
                    "row_ind": row_ind.tolist(),
                    "col_ind": col_ind.tolist(),
                }
            }
        except Exception:
            return self._empty()