from __future__ import annotations
from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Compute the k smallest eigenvalues of a sparse positive semi‑definite matrix.

        Parameters
        ----------
        problem: dict
            - "matrix": a scipy sparse matrix (any format)
            - "k": number of eigenvalues to return (int)

        Returns
        -------
        List[float]
            The k smallest eigenvalues sorted in ascending order.
        """
        # Extract matrix; keep CSR if already in that format to avoid overhead
        mat_input = problem["matrix"]
        if isinstance(mat_input, sparse.csr_matrix):
            mat = mat_input
        else:
            mat = mat_input.asformat("csr")
        k: int = int(problem["k"])
        n = mat.shape[0]

        # If k is large relative to matrix size, fall back to dense eigenvalue computation
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in np.sort(vals)[:k]]

        # Primary path – sparse Lanczos algorithm for smallest algebraic eigenvalues
        # Primary path – sparse Lanczos algorithm for smallest algebraic eigenvalues
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",               # smallest algebraic (better for PSD)
                return_eigenvectors=False,
                tol=1e-4,                 # relaxed tolerance accelerates solve
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            # Fallback to dense computation if eigsh fails
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # Ensure real values, sort, and convert to Python floats
        vals = np.real(vals)
        vals.sort()
        return [float(v) for v in vals]