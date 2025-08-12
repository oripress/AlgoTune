from typing import Any
import numpy as np
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        """Find k smallest eigenvalues of a sparse positive semi-definite matrix."""
        mat = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]
        
        # Dense path for small matrices or when k is close to n
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]
        
        # Sparse eigenvalue computation with optimized parameters
        try:
            # Use SA (smallest algebraic) for PSD matrices - often faster than SM
            # Reduce iterations and tolerance for speed
            vals = eigsh(
                mat,
                k=k,
                which="SA",  # Smallest algebraic - faster for PSD matrices
                return_eigenvectors=False,
                maxiter=min(n * 50, 2000),  # Reduced iterations
                tol=1e-8,  # Slightly relaxed tolerance
                ncv=min(n - 1, max(2 * k + 1, min(3 * k, 25)))  # Optimized ncv
            )
            return [float(v) for v in np.sort(vals)]
        except Exception:
            # Fallback to dense
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
            return [float(v) for v in vals]