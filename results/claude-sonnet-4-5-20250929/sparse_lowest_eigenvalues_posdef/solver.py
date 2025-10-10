from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"]
        if isinstance(mat, list):
            mat = sparse.csr_matrix(mat)
        elif hasattr(mat, 'asformat'):
            mat = mat.asformat("csr")
        k: int = int(problem["k"])
        n = mat.shape[0]
        
        # Dense path for tiny systems or k too close to n
        if k >= n or n < 2 * k + 1:
            dense = mat.toarray()
            vals = linalg.eigh(dense, subset_by_index=[0, k-1], check_finite=False, overwrite_a=True)[0]
            return vals.tolist()
        
        # For matrices up to ~150, dense with subset is faster than sparse
        if n < 150:
            dense = mat.toarray()
            vals = linalg.eigh(dense, subset_by_index=[0, k-1], check_finite=False, overwrite_a=True)[0]
            return vals.tolist()
        
        # Sparse Lanczos with highly optimized parameters
        try:
            # Very large ncv for fastest convergence
            ncv = min(n - 1, max(10 * k, 80))
            vals = eigsh(
                mat,
                k=k,
                which="SA",  # Smallest algebraic - already sorted
                return_eigenvectors=False,
                maxiter=n * 20,
                tol=1e-6,
                ncv=ncv,
            )
            # SA returns sorted values, just convert to list
            return vals.tolist()
        except Exception:
            dense = mat.toarray()
            vals = linalg.eigh(dense, subset_by_index=[0, k-1], check_finite=False, overwrite_a=True)[0]
            return vals.tolist()