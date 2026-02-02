from __future__ import annotations

from typing import Any

class Solver:
    def __init__(self) -> None:
        import numpy as np
        import scipy.sparse as sp
        from scipy.optimize import linear_sum_assignment
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching

        self._np = np
        self._sp_csr = sp.csr_matrix
        self._lsa = linear_sum_assignment
        self._mwfbm = min_weight_full_bipartite_matching

        self._arange_cache: dict[int, Any] = {}

    def _get_arange(self, n: int, dtype: Any):
        arr = self._arange_cache.get(n)
        if arr is None or arr.dtype != dtype:
            arr = self._np.arange(n, dtype=dtype)
            self._arange_cache[n] = arr
        return arr

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        np = self._np
        try:
            n = int(problem["shape"][0])

            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            nnz = len(data)

            # Dense fast-path (only when we can cheaply verify canonical dense CSR layout).
            if nnz == n * n:
                # Fast indptr check: [0, n, 2n, ..., n*n]
                indptr_a = np.asarray(indptr, dtype=np.int64)
                if indptr_a.size == n + 1 and indptr_a[0] == 0 and indptr_a[-1] == nnz:
                    if np.all(indptr_a[1:] - indptr_a[:-1] == n):
                        idx_a = np.asarray(indices, dtype=np.int64).reshape(n, n)
                        expected = self._get_arange(n, idx_a.dtype)
                        if (idx_a == expected).all():
                            cost = np.asarray(data, dtype=np.float64).reshape(n, n)
                            row_ind, col_ind = self._lsa(cost)
                            return {"assignment": {"row_ind": row_ind, "col_ind": col_ind}}

            mat = self._sp_csr((data, indices, indptr), shape=(n, n))
            row_ind, col_ind = self._mwfbm(mat)
            return {"assignment": {"row_ind": row_ind, "col_ind": col_ind}}
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}