from __future__ import annotations
from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Compute the k smallest eigenvalues of a sparse positive semi-definite matrix.

        Parameters
        ----------
        problem : dict
            Contains:
                - "matrix": a scipy.sparse matrix (any format)
                - "k": number of smallest eigenvalues to return (int)

        Returns
        -------
        List[float]
            The k smallest eigenvalues sorted in ascending order.
        """
        # Ensure CSR format for efficient operations
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Use dense eigenvalue computation for very small matrices or when k is large
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # If the matrix is relatively dense, dense path is usually faster
        density = mat.nnz / (n * n)
        # Use dense solver for very dense matrices or modest sizes (≤500)
        if density > 0.3 or n <= 500:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Sparse path: use Lanczos algorithm with tighter convergence settings
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",               # smallest magnitude eigenvalues
                return_eigenvectors=False,
                maxiter=min(n * 50, 500),   # fewer iterations for speed
                ncv=min(n - 1, max(2 * k + 1, 20)),
                tol=1e-5,                 # relaxed tolerance for faster convergence
            )
        except Exception:
            # Fallback to dense computation if Lanczos fails
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # Ensure sorted ascending order and convert to Python floats
        vals = np.sort(np.real(vals))
        return [float(v) for v in vals[:k]]