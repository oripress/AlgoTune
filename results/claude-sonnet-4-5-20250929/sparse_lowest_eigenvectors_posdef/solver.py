from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import eigsh
class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"]
        # Handle both list and sparse matrix inputs - optimize conversion
        if isinstance(mat, list):
            mat = sparse.csr_matrix(mat)
        elif not sparse.issparse(mat):
            mat = sparse.csr_matrix(mat)
        elif mat.format != "csr":
            mat = mat.tocsr()
        
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Dense path for small matrices where dense is faster
        if n <= 100 or k >= n or n < 2 * k + 1:
            dense = mat.toarray()
            # Use np.linalg.eigh with overwrite for speed
            vals = np.linalg.eigh(dense, UPLO='L')[0][:k]
            return [float(v) for v in vals]

        # Sparse path with optimized parameters
        try:
            # Optimize ncv based on matrix size
            if n < 200:
                ncv = min(n - 1, max(4 * k, 25))
            else:
                ncv = min(n - 1, max(3 * k, 20))
            
            vals = eigsh(
                mat,
                k=k,
                which="SM",  # smallest magnitude
                return_eigenvectors=False,
                maxiter=min(n * 20, 5000),  # More aggressive cap
                tol=1e-5,  # Slightly relaxed tolerance for speed
                ncv=ncv,
            )
        except Exception:
            vals = linalg.eigh(mat.toarray(), subset_by_index=[0, k-1], eigvals_only=True)

        return [float(v) for v in np.sort(np.real(vals))]