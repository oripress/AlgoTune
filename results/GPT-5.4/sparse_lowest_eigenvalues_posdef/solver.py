from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        mat = problem["matrix"]
        if not sparse.isspmatrix_csr(mat):
            mat = mat.tocsr(copy=False)

        k = int(problem["k"])
        n = mat.shape[0]

        # Large ARPACK overhead dominates for small systems.
        if k >= n or n <= 48 or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return vals[:k].tolist()

        # Cheap exact fast path for diagonal CSR matrices.
        nnz = mat.nnz
        if nnz == n:
            indptr = mat.indptr
            if np.all(indptr[1:] - indptr[:-1] == 1) and np.array_equal(mat.indices, np.arange(n)):
                vals = np.sort(mat.data.real)
                return vals[:k].tolist()

        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                return_eigenvectors=False,
                tol=1e-7,
                maxiter=n * 100,
                ncv=min(n - 1, max(2 * k + 1, 12)),
            )
            vals.sort()
            return vals.tolist()
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())
            return vals[:k].tolist()