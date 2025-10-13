from typing import Any, Dict

import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> sparse.spmatrix:
        """
        Compute the matrix exponential of a sparse CSC matrix using efficient paths for
        trivial cases and falling back to scipy.sparse.linalg.expm otherwise.

        The output is guaranteed to be in CSC format to match the validator expectations.
        """
        A = problem["matrix"]

        # Ensure CSC format for predictable behavior and performance
        if not sparse.isspmatrix_csc(A):
            A = A.tocsc()

        n, m = A.shape
        if n != m:
            raise ValueError("Input matrix must be square.")

        # Fast path: zero matrix (including explicit zero entries)
        if A.nnz == 0 or (A.nnz > 0 and not np.any(A.data)):
            return sparse.identity(n, format="csc", dtype=A.dtype)

        # Fast path: diagonal matrix
        # Check if all nonzeros in each column are on the diagonal
        indptr = A.indptr
        indices = A.indices
        is_diagonal = True
        for j in range(n):
            start, end = indptr[j], indptr[j + 1]
            col_rows = indices[start:end]
            if col_rows.size and np.any(col_rows != j):
                is_diagonal = False
                break

        if is_diagonal:
            diag_vals = A.diagonal()
            exp_diag = np.exp(diag_vals)
            return sparse.diags(exp_diag, offsets=0, format="csc", dtype=A.dtype)

        # General case: delegate to scipy's sparse expm
        result = expm(A)

        # Ensure CSC output for validator structural comparison
        if not sparse.isspmatrix_csc(result):
            result = csc_matrix(result)

        return result