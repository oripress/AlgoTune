import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm as sparse_expm
from scipy.linalg import expm as dense_expm
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Compute the matrix exponential exp(A) for the square matrix provided under key "matrix".
        Returns a scipy.sparse.csc_matrix (dtype float64).

        Strategy:
        - Handle trivial zero and diagonal matrices quickly.
        - For small or relatively dense matrices, use dense expm (often faster).
        - Otherwise use scipy.sparse.linalg.expm to preserve sparsity.
        """
        if not isinstance(problem, dict):
            raise ValueError("Expected problem to be a dict with key 'matrix'")

        A = problem.get("matrix")
        if A is None:
            raise ValueError("Problem dictionary missing 'matrix' key")

        # Normalize input to CSC sparse for inspections and computations
        if sparse.isspmatrix_csc(A):
            A_csc = A.copy()
        elif sparse.isspmatrix(A):
            A_csc = A.tocsc()
        else:
            # Accept dense ndarray or other matrix-like
            A_csc = sparse.csc_matrix(A, dtype=np.float64)

        n_rows, n_cols = A_csc.shape
        if n_rows != n_cols:
            raise ValueError("Input matrix must be square")
        n = n_rows

        # Empty matrix case
        if n == 0:
            return sparse.csc_matrix((0, 0), dtype=np.float64)

        # Quick special-case: zero matrix -> exp(0) = I
        if A_csc.nnz == 0:
            return sparse.identity(n, format="csc", dtype=np.float64)

        # Diagonal matrix check: if all nonzeros are on the diagonal
        diag = A_csc.diagonal()
        if A_csc.nnz == np.count_nonzero(diag):
            # Exponential of diagonal matrix is diagonal of elementwise exp
            exp_diag = np.exp(diag)
            return sparse.diags(exp_diag, offsets=0, format="csc", dtype=np.float64)

        # Heuristic: decide whether to use dense or sparse path.
        density = A_csc.nnz / float(n * n)

        # Tunable thresholds: small matrices or moderately dense matrices use dense path
        DENSE_N_THRESHOLD = 200
        DENSE_DENSITY_THRESHOLD = 0.02

        use_dense = (n <= DENSE_N_THRESHOLD) or (density > DENSE_DENSITY_THRESHOLD)

        if use_dense:
            try:
                arr = A_csc.toarray()
                exp_arr = dense_expm(arr)
                exp_sparse = sparse.csc_matrix(exp_arr, dtype=np.float64)
                return exp_sparse
            except Exception:
                # Fall back to sparse implementation on any failure (memory, numerical, etc.)
                pass

        # Default: use sparse expm
        exp_sparse = sparse_expm(A_csc)

        # Ensure CSC format and float64 dtype
        if not sparse.isspmatrix_csc(exp_sparse):
            exp_sparse = exp_sparse.tocsc()
        if exp_sparse.dtype != np.float64:
            exp_sparse = exp_sparse.astype(np.float64)
        return exp_sparse