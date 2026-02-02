from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm as _sparse_expm

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        A = problem["matrix"]
        if not sparse.isspmatrix_csc(A):
            A = A.tocsc()

        n = int(A.shape[0])
        if n == 0:
            return sparse.csc_matrix((0, 0), dtype=np.float64)

        nnz = int(A.nnz)
        if nnz == 0:
            return sparse.identity(n, format="csc", dtype=np.float64)

        # Fast path: diagonal matrices.
        # In CSC, diagonal implies at most 1 stored entry per column and, when present,
        # its row index equals the column index.
        indptr = A.indptr
        counts = np.diff(indptr)
        if nnz <= n and counts.size == n and counts.max(initial=0) <= 1:
            cols = np.flatnonzero(counts).astype(A.indices.dtype, copy=False)
            if cols.size == nnz and np.array_equal(A.indices, cols):
                diag = np.zeros(n, dtype=np.float64)
                diag[cols] = A.data
                return sparse.diags(np.exp(diag), 0, format="csc")

        # General case: use SciPy's sparse expm (reference behavior).
        return _sparse_expm(A)