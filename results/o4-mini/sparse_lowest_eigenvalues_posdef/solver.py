from typing import Any, List
import numpy as np
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[float]:
        mat = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]
        # Dense fallback for trivial or very small systems
        if k >= n or n < 2 * k + 1:
            arr = mat.toarray()
            vals = np.linalg.eigvalsh(arr)
            return [float(v) for v in vals[:k]]
        # Parameters for ARPACK Lanczos
        ncv = min(n - 1, max(2 * k + 1, 20))
        maxiter = min(n * 10, 1000)
        tol = 1e-6
        try:
            # Compute smallest algebraic eigenvalues without explicit shift-invert
            vals = eigsh(
                mat,
                k=k,
                which='SA',
                return_eigenvectors=False,
                tol=tol,
                ncv=ncv,
                maxiter=maxiter
            )
        except ValueError:
            # Fallback to smallest-magnitude with shift-invert
            vals = eigsh(
                mat,
                k=k,
                which='SM',
                return_eigenvectors=False,
                tol=tol,
                ncv=ncv,
                maxiter=maxiter
            )
        except Exception:
            # Last-resort dense fallback
            arr = mat.toarray()
            vals = np.linalg.eigvalsh(arr)[:k]
        vals = np.real(vals)
        vals.sort()
        return [float(v) for v in vals]