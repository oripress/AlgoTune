from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        mat = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]

        # Dense path for small matrices or k too close to n
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]
        
        # Dense for small matrices is faster than sparse solvers
        if n <= 80:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Use SA (smallest algebraic) - more efficient for PSD matrices
        ncv = min(n - 1, max(2 * k + 1, 20))
        
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                return_eigenvectors=False,
                tol=1e-7,
                maxiter=n * 100,
                ncv=ncv,
            )
            return [float(v) for v in np.sort(np.real(vals))]
        except Exception:
            pass

        # Fallback to SM (what reference uses)
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=ncv,
            )
            return [float(v) for v in np.sort(np.real(vals))]
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
            return [float(v) for v in vals]