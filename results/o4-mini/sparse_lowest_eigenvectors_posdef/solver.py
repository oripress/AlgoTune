# solver.py

from typing import Any
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        # Load CSR matrix
        mat = problem["matrix"]
        if not isinstance(mat, csr_matrix):
            mat = mat.tocsr()
        k = int(problem["k"])
        n = mat.shape[0]

        # Dense fallback for trivial or near-full
        if k >= n or n < 2 * k + 1:
            arr = mat.toarray()
            vals = np.linalg.eigvalsh(arr)
            return [float(v) for v in vals[:k]]

        # Sparse path: get smallest eigenvalues of A by computing
        # largest of -A (LA) to avoid shift-invert
        try:
            # negate data in place on a copy to avoid factorization
            mat_neg = mat.copy()
            mat_neg.data = -mat_neg.data
            vals = eigsh(
                mat_neg,
                k=k,
                which="LA",             # largest algebraic of -A => smallest of A
                return_eigenvectors=False,
                tol=1e-6,
            )
            vals = -vals
        except Exception:
            # fallback to dense
            arr = mat.toarray()
            vals = np.linalg.eigvalsh(arr)[:k]

        # sort and return floats
        vals = np.sort(np.real(vals))
        return [float(v) for v in vals]