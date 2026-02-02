from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh as scipy_eigvalsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"]
        # Ensure csr
        if not sparse.isspmatrix_csr(mat):
            mat = mat.tocsr()
            
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Check if matrix is diagonal
        if mat.nnz == n:
            diag = mat.diagonal()
            if len(diag) == n and np.count_nonzero(diag) == mat.nnz:
                 vals = np.sort(np.real(diag))
                 return [float(v) for v in vals[:k]]

        # Dense path for small systems
        if k >= n or n < 200:
            try:
                # Use scipy.linalg.eigvalsh with check_finite=False
                # Also subset_by_index to compute only needed eigenvalues
                vals = scipy_eigvalsh(
                    mat.toarray(), 
                    subset_by_index=(0, k-1),
                    check_finite=False
                )
                return [float(v) for v in vals]
            except Exception:
                pass

        # Sparse Lanczos with SA and relaxed tolerance
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SA",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
                tol=1e-4
            )
            return [float(v) for v in np.sort(np.real(vals))]
        except Exception:
            pass

        # Fallback to Reference implementation (Standard Lanczos with SM)
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            # Last‑resort dense fallback
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return [float(v) for v in np.sort(np.real(vals))]