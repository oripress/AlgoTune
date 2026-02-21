from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"]
        if not sparse.issparse(mat):
            mat = sparse.csr_matrix(mat)
        else:
            mat = mat.asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]
        from scipy.linalg import eigh

        if k >= n or n < 1000:
            if k >= n:
                vals = np.linalg.eigvalsh(mat.toarray())
                return [float(v) for v in vals[:k]]
            else:
                vals = eigh(mat.toarray(), eigvals_only=True, subset_by_index=(0, k - 1))
                return [float(v) for v in vals]

        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                tol=4.5e-4,
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, 25),
            )
        except Exception:
            vals = eigh(mat.toarray(), eigvals_only=True, subset_by_index=(0, k - 1))

        return [float(v) for v in np.sort(np.real(vals))]