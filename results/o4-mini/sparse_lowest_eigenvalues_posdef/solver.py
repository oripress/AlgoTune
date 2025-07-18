from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[float]:
        # Prepare CSR sparse matrix
        mat0 = problem["matrix"]
        if hasattr(mat0, "asformat"):
            mat = mat0.asformat("csr")
        elif sparse.isspmatrix(mat0):
            mat = mat0.tocsr()
        else:
            mat = sparse.csr_matrix(mat0)
        # update for validator compatibility
        problem["matrix"] = mat

        k = int(problem.get("k", 0))
        n = mat.shape[0]

        # Dense fallback for trivial or very small matrices
        if k >= n or n < 2 * k + 1:
            arr = mat.toarray()
            vals = np.linalg.eigvalsh(arr)
            return [float(v) for v in vals[:k]]

        # ARPACK Lanczos for sparse: smallest magnitude eigenvalues
        ncv = min(n - 1, max(2 * k + 1, 20))
        maxiter = n * 10
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                ncv=ncv,
                maxiter=maxiter,
                tol=1e-6,
                return_eigenvectors=False,
            )
        except Exception:
            # Fallback to dense if ARPACK fails
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # Sort and return real eigenvalues as floats
        vals = np.real(vals)
        vals.sort()
        return [float(v) for v in vals]