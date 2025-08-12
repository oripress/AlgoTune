import numpy as np
import scipy.linalg as la
from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: Tuple[Any, Any], **kwargs) -> List[complex]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for real matrices A and B.

        Parameters
        ----------
        problem : tuple
            A pair (A, B) where each is an n×n real matrix provided as a list of lists
            or a NumPy array.

        Returns
        -------
        list[complex]
            Eigenvalues sorted in descending order by real part, then by imaginary part.
        """
        A_raw, B_raw = problem

        # Convert inputs to NumPy arrays of dtype float64 for numerical stability.
        A = np.asarray(A_raw, dtype=np.float64)
        B = np.asarray(B_raw, dtype=np.float64)

        # Compute eigenvalues via solving the standard eigenvalue problem for inv(B)·A.
        # This avoids the overhead of the generalized routine and is typically faster.
        # Solve B·X = A for X, i.e., X = inv(B)·A.
        X = la.solve(B, A, assume_a='gen', overwrite_b=True, check_finite=False)

        # Compute eigenvalues of the resulting matrix.
        eigenvalues = np.linalg.eigvals(X)

        # Sort by descending real part, then descending imaginary part using NumPy's lexsort.
        sort_idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        sorted_vals = eigenvalues[sort_idx]

        # Return as a list of native Python complex numbers.
        return sorted_vals.tolist()