from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu, LinearOperator

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        mat = problem["matrix"]
        k = 5  # Fixed per problem statement
        n = mat.shape[0]
        
        # Dense fallback for small matrices
        if n <= 100:
            dense_mat = mat.toarray()
            vals = np.linalg.eigvalsh(dense_mat)
            return [float(v) for v in vals[:k]]
        
        # Optimized shift-invert for moderate matrices
        if 100 < n <= 500:
            try:
                sigma = 1e-10  # Shift near zero
                A_shifted = mat - sigma * sparse.eye(n)
                lu = splu(A_shifted.tocsc())
                
                Minv = LinearOperator((n, n), matvec=lu.solve)
                
                vals = eigsh(
                    mat,
                    k=k,
                    sigma=sigma,
                    which='LM',  # Largest magnitude of inverse = smallest eigenvalues
                    Minv=Minv,
                    return_eigenvectors=False,
                    maxiter=50,  # Fewer iterations needed
                    ncv=min(n, 20),
                    tol=1e-6
                )
                return [float(v) for v in np.sort(vals)]
            except Exception:
                pass  # Fall through to next method
        
        # Optimized parameters for larger matrices
        ncv_val = min(n - 1, 25)  # Reduced ncv for better performance
        max_iter = min(n * 5, 1000)
        
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",  # smallest algebraic eigenvalues
                return_eigenvectors=False,
                maxiter=max_iter,
                ncv=ncv_val,
                tol=1e-5
            )
            return [float(v) for v in np.sort(vals)]
        except Exception:
            dense_mat = mat.toarray()
            vals = np.linalg.eigvalsh(dense_mat)
            return [float(v) for v in vals[:k]]