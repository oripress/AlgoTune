from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Dense path for tiny systems or k too close to n
        if k >= n or n < 2 * k + 1:
            arr = mat.toarray()
            vals = eigh(arr, eigvals_only=True, subset_by_index=[0, k-1], driver='evr')
            return [float(v) for v in vals]
        
        # For small-to-medium matrices, dense with subset is very efficient
        # The MRRR algorithm (evr driver) is O(n^2) for computing k eigenvalues
        if n <= 800:
            arr = mat.toarray()
            vals = eigh(arr, eigvals_only=True, subset_by_index=[0, k-1], driver='evr')
            return [float(v) for v in vals]

        # For larger sparse matrices use eigsh with SA mode (fastest for PSD)
        try:
            ncv = min(n - 1, max(2 * k + 1, 4 * k, 25))
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                return_eigenvectors=False,
                maxiter=n * 100,
                ncv=ncv,
                tol=1e-8,
            )
            return [float(v) for v in np.sort(np.real(vals))]
        except Exception:
            pass
        
        # Fallback to SM mode
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
            arr = mat.toarray()
            vals = eigh(arr, eigvals_only=True, subset_by_index=[0, k-1], driver='evr')

        return [float(v) for v in np.sort(np.real(vals))]