import numpy as np
import scipy.linalg as la
from typing import Any

class Solver:
    def solve(self, problem: tuple[list[list[float]], list[list[float]]], **kwargs) -> Any:
        """
        Solves the generalized eigenvalue problem A*v = lambda*B*v.
        """
        A_list, B_list = problem
        A = np.array(A_list, dtype=np.float64)
        B = np.array(B_list, dtype=np.float64)
        
        n = A.shape[0]
        if n == 0:
            return ([], [])

        try:
            # This is the "outside the box" optimization.
            # Convert the generalized problem A*v = lambda*B*v into a standard
            # problem C*v = lambda*v by finding C = inv(B)*A. This is often
            # significantly faster if B is well-conditioned.
            # We use la.solve(B, A) which is faster and more stable than
            # computing the inverse of B explicitly.
            # check_finite=False provides a speed boost by skipping validation.
            # The default overwrite_b=False ensures B is not modified, so it's
            # available for the fallback path if this fails.
            C = la.solve(B, A, check_finite=False)
            
            # Solve the standard eigenvalue problem for C.
            # overwrite_a=True allows the solver to destroy C, saving memory.
            eigenvalues, eigenvectors = la.eig(C, overwrite_a=True, check_finite=False)

        except np.linalg.LinAlgError:
            # Fallback to the robust generalized eigenvalue solver if B is singular
            # or ill-conditioned.
            # We can safely use overwrite flags as A and B are not needed after this.
            eigenvalues, eigenvectors = la.eig(A, B, overwrite_a=True, overwrite_b=True, check_finite=False)

        # Vectorized normalization of eigenvectors.
        norms = np.linalg.norm(eigenvectors, axis=0)
        non_zero_mask = norms > 1e-15
        eigenvectors[:, non_zero_mask] /= norms[non_zero_mask]

        # Vectorized sorting using np.argsort on negated complex eigenvalues.
        sort_indices = np.argsort(-eigenvalues)
        
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # Convert final NumPy arrays to the required list format.
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()

        return (eigenvalues_list, eigenvectors_list)