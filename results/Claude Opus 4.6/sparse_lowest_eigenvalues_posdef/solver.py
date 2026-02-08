from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse import issparse
from scipy.linalg import eigh

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        mat = problem["matrix"]
        if not issparse(mat):
            mat = sparse.csr_matrix(mat)
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Dense path for tiny systems or k too close to n
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # For moderate sized matrices, use dense eigh with subset
        if n <= 1500:
            dense = mat.toarray()
            vals = eigh(dense, eigvals_only=True, subset_by_index=[0, k - 1], driver='evr')
            return [float(v) for v in vals]

        # For larger matrices: try LOBPCG (very fast for PSD)
        mat_csr = mat.asformat("csr")
        try:
            np.random.seed(42)
            X = np.random.randn(n, k)
            # Diagonal preconditioner
            diag_vals = mat_csr.diagonal().copy()
            diag_vals[diag_vals < 1e-10] = 1.0
            inv_diag = 1.0 / diag_vals

            def precond(x):
                return inv_diag[:, None] * x if x.ndim > 1 else inv_diag * x

            M = sparse.linalg.LinearOperator((n, n), matvec=lambda x: inv_diag * x,
                                              matmat=lambda x: inv_diag[:, None] * x)
            vals, _ = lobpcg(mat_csr, X, M=M, largest=False, tol=1e-7, maxiter=500, verbosityLevel=0)
            return [float(v) for v in np.sort(np.real(vals))]
        except Exception:
            pass

        # Fallback: shift-invert eigsh
        try:
            vals = eigsh(
                mat_csr,
                k=k,
                sigma=-1e-5,
                which="LM",
                return_eigenvectors=False,
                tol=1e-7,
            )
            return [float(v) for v in np.sort(np.real(vals))]
        except Exception:
            pass

        # Last fallback: dense
        dense = mat_csr.toarray()
        vals = eigh(dense, eigvals_only=True, subset_by_index=[0, k - 1], driver='evr')
        return [float(v) for v in vals]