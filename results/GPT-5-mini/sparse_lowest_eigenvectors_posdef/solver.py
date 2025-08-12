from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


class Solver:
    def solve(self, problem: dict, **kwargs) -> List[float]:
        """
        Compute the k smallest eigenvalues (ascending) of a square sparse
        positive semi-definite matrix provided in CSR or convertible form.

        Returns:
            List[float]: k smallest eigenvalues sorted in ascending order.
        """
        if not isinstance(problem, dict):
            raise ValueError("problem must be a dict with 'matrix' and 'k'")
        if "matrix" not in problem or "k" not in problem:
            raise ValueError("problem must contain 'matrix' and 'k'")

        A = problem["matrix"]
        k = int(problem["k"])

        if k <= 0:
            return []

        # Prefer fast CSR conversion if available (some harness matrices expose .asformat)
        mat = None
        try:
            if hasattr(A, "asformat") and callable(getattr(A, "asformat")):
                mat = A.asformat("csr")
        except Exception:
            mat = None

        # Fallback conversions
        if mat is None:
            if sparse.issparse(A):
                mat = A if sparse.isspmatrix_csr(A) else A.tocsr()
            else:
                arr = np.asarray(A)
                if arr.ndim != 2:
                    raise ValueError("Input matrix must be 2-dimensional")
                mat = sparse.csr_matrix(arr)

        n, m = mat.shape

        # Non-square -> dense eigenvalues
        if n != m:
            vals = np.linalg.eigvals(mat.toarray())
            vals = np.sort(np.real(vals))
            return [float(v) for v in vals[:k]]

        # Dense path for tiny systems or k too close to n (match reference logic)
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            vals = np.sort(np.real(vals))
            return [float(v) for v in vals[:k]]

        # Sparse Lanczos (ARPACK) for smallest eigenvalues
        try:
            ncv = min(n - 1, max(2 * k + 1, 20))
            if ncv <= k:
                ncv = k + 1
            maxiter = max(1000, n * 200)

            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                ncv=ncv,
                maxiter=maxiter,
            )
            vals = np.sort(np.real(vals))
        except Exception:
            # Fallback dense
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
            vals = np.sort(np.real(vals))

        return [float(v) for v in vals[:k]]