from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat: sparse.spmatrix = problem["matrix"]
        k: int = int(problem["k"])
        n = mat.shape[0]
        
        # Early CSR conversion to avoid repeated conversions
        if not sparse.isspmatrix_csr(mat):
            mat = mat.tocsr()
        
        # Use dense for small matrices or when k is close to n
        # Optimized threshold based on matrix size
        if k >= n - 1 or n <= 50:
            vals = np.linalg.eigvalsh(mat.toarray())
            return vals[:k].tolist()
        
        # For PSD matrices, 'SA' (smallest algebraic) is more efficient than 'SM'
        # Optimize ncv and maxiter for better performance
        ncv = min(n - 1, max(2*k + 1, min(20, n//2)))  # Optimal subspace size
        
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",  # Smallest algebraic - better for PSD
                return_eigenvectors=False,
                tol=1e-9,  # Tighter tolerance but still fast
                maxiter=min(n * 10, 1000),  # Reduced iterations
                ncv=ncv
            )
            # eigsh with 'SA' returns sorted values, no need to sort again
            return vals.tolist()
        except Exception:
            # Fallback to dense
            vals = np.linalg.eigvalsh(mat.toarray())
            return vals[:k].tolist()