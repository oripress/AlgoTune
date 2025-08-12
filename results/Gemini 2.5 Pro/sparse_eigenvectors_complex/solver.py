from typing import Any
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solve the eigenvalue problem for the given square sparse matrix.
        The solution returned is a list of the eigenvectors with the largest `k` eigenvalues sorted in descending order by their modulus.

        :param problem: A dictionary representing the sparse eigenvalue problem.
        :return: List of eigenvectors sorted in descending order by the modulus of the corresponding eigenvalue.
        """
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]

        if k == 0 or N == 0:
            return []

        def solve_dense(matrix, num_eigs):
            """A robust helper function to solve the problem using a dense solver."""
            if sparse.issparse(matrix):
                A_dense = matrix.toarray()
            else:
                A_dense = np.asarray(matrix)
            
            if np.issubdtype(A_dense.dtype, np.integer):
                A_dense = A_dense.astype(np.float64)

            eigenvalues, eigenvectors = np.linalg.eig(A_dense)
            pairs = sorted(zip(eigenvalues, eigenvectors.T), key=lambda p: -np.abs(p[0]))
            return [p[1] for p in pairs[:num_eigs]]

        # The sparse solver `eigs` has strict preconditions. We must check them
        # before attempting to call it.
        # Precondition 1: The number of eigenvalues `k` must be less than N-1.
        # Precondition 2: The number of basis vectors `ncv` must be less than N.
        ncv = max(2 * k + 1, 20)
        if k >= N - 1 or ncv >= N:
            return solve_dense(A, k)

        # If preconditions are met, attempt the sparse solver. It can still fail
        # for other reasons (e.g., non-convergence), so we need a robust fallback.
        try:
            _A = A
            if np.issubdtype(_A.dtype, np.integer):
                _A = _A.astype(np.float64)
            if not sparse.isspmatrix_csr(_A):
                _A = _A.tocsr()
            
            np.random.seed(0)  # For deterministic "random" starting vector
            
            # Pass the pre-validated ncv.
            eigenvalues, eigenvectors = sparse.linalg.eigs(_A, k=k, ncv=ncv)
            
            pairs = sorted(list(zip(eigenvalues, eigenvectors.T)), key=lambda p: -np.abs(p[0]))
            return [p[1] for p in pairs]
        except Exception:
            # If the sparse solver fails for any reason, fall back to the dense solver.
            return solve_dense(A, k)