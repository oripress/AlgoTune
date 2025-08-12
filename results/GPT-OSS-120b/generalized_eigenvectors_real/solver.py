import numpy as np
from typing import List, Tuple
from scipy.linalg import eigh

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the generalized eigenvalue problem A x = λ B x where A is symmetric
        and B is symmetric positive definite.

        Returns eigenvalues in descending order and corresponding eigenvectors
        normalized with respect to the B-inner product.
        """
        A, B = problem
        # Use SciPy's optimized generalized eigenvalue solver.
        # It returns eigenvalues in ascending order and eigenvectors that are B‑orthonormal.
        eigenvalues, eigenvectors = eigh(A, B, lower=False, eigvals_only=False,
                                        overwrite_a=False, overwrite_b=False,
                                        check_finite=False)
        # Reverse order to descending eigenvalues
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        # Convert to plain Python lists
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()
        return (eigenvalues_list, eigenvectors_list)