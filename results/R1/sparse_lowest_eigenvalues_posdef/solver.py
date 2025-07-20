from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k: int = int(problem["k"])
        n = mat.shape[0]
        
        # Efficient symmetric check using sparse matrix operations
        if (mat != mat.T).sum() > 0:
            mat = (mat + mat.T) * 0.5

        # Dense path for tiny systems or k too close to n
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Hybrid approach: use shift-invert for ill-conditioned matrices
        try:
            # Estimate condition number using diagonal
            diag = mat.diagonal()
            min_diag = diag.min()
            max_diag = diag.max()
            
            # Use shift-invert if matrix appears ill-conditioned
            if min_diag < 1e-5 or max_diag/min_diag > 1e6:
                vals = eigsh(
                    mat,
                    k=k,
                    sigma=0.0,
                    which="SA",
                    return_eigenvectors=False,
                    maxiter=n * 30,  # Reduced iterations for shift-invert
                    tol=1e-6,        # Allow earlier convergence
                    ncv=min(n - 1, max(2 * k + 1, 20)),
                )
            else:
                # Standard Lanczos for well-conditioned matrices
                vals = eigsh(
                    mat,
                    k=k,
                    which="SM",
                    return_eigenvectors=False,
                    maxiter=n * 100,  # Reduced iterations for standard Lanczos
                    tol=1e-6,         # Allow earlier convergence
                    ncv=min(n - 1, max(2 * k + 1, 20)),
                )
        except Exception:
            # Last-resort dense fallback
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return [float(v) for v in np.sort(np.real(vals))]