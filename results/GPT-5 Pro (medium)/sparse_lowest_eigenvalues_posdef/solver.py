from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the k smallest eigenvalues (ascending) of a sparse positive semi-definite matrix.

        Strategy:
        - Dense path for small problems or when k is close to n (matches reference behavior).
        - Fast diagonal path if the matrix is detected as diagonal.
        - Sparse Lanczos (eigsh) with which='SM' to match the reference precisely.
        """
        mat = problem["matrix"]
        k = int(problem["k"])

        # Handle dense matrices if provided unexpectedly
        if not sparse.isspmatrix(mat):
            arr = np.asarray(mat)
            vals = np.linalg.eigvalsh(arr)
            return [float(v) for v in vals[:k]]

        # Ensure CSR format (cheap if already CSR)
        if not sparse.isspmatrix_csr(mat):
            mat = mat.tocsr()

        n = mat.shape[0]

        # Trivial cases
        if k <= 0:
            return []

        # Fast path: diagonal matrix detection (CSR)
        # Check that each row has exactly one nonzero positioned at the diagonal.
        # This is O(n) and avoids dense computations for diagonal matrices.
        if mat.nnz == n:
            indptr = mat.indptr
            indices = mat.indices
            data = mat.data
            is_diagonal = True
            for i in range(n):
                start, end = indptr[i], indptr[i + 1]
                if end - start != 1 or indices[start] != i:
                    is_diagonal = False
                    break
            if is_diagonal:
                diag = np.real(data.copy())
                diag.sort()
                return [float(v) for v in diag[:k]]

        # Dense path for tiny systems or when k is close to n (same threshold as reference)
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Sparse Lanczos without shift-invert; match reference which='SM'
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",  # smallest magnitude eigenvalues (matches reference)
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),  # ensure k < ncv < n (matches reference)
            )
        except Exception:
            # Last-resort dense fallback
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        vals = np.real(vals)
        vals.sort()
        return [float(v) for v in vals[:k]]