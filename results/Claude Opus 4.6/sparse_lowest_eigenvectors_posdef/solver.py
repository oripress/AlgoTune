from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        mat = problem["matrix"]
        if not sparse.issparse(mat):
            mat = sparse.csr_matrix(mat)
        k = int(problem["k"])
        n = mat.shape[0]

        # Dense path for tiny systems or k too close to n
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Sparse Lanczos without shift-invert (same as reference but optimized params)
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return [float(v) for v in np.sort(np.real(vals))]