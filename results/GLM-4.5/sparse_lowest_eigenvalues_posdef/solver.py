from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        # Use CSC format which can be more efficient for certain operations
        mat: sparse.spmatrix = problem["matrix"].asformat("csc")
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Dense path for tiny systems or k too close to n
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Optimized sparse Lanczos with reduced maxiter and explicit tolerance
        try:
            # Try with more aggressive parameters for speed
            vals = eigsh(
                mat,
                k=k,
                which="SM",  # smallest magnitude eigenvalues
                return_eigenvectors=False,
                maxiter=n * 15,  # Even more aggressive reduction
                tol=1e-6,        # Explicit tolerance matching required precision
                ncv=min(n - 1, max(k + 10, 2 * k)),  # More aggressive ncv
            )
        except Exception:
            # Fallback with more conservative parameters
            try:
                vals = eigsh(
                    mat,
                    k=k,
                    which="SM",
                    return_eigenvectors=False,
                    maxiter=n * 20,
                    tol=1e-6,
                    ncv=min(n - 1, max(2 * k + 1, 20)),
                )
            except Exception:
                # Last-resort dense fallback (rare)
                vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return [float(v) for v in np.sort(np.real(vals))]