import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: Any, **kwargs) -> List[List[complex]]:
        """
        Compute eigenvalues and eigenvectors of a real square matrix.
        Returns the eigenvectors sorted by descending eigenvalue (real part,
        then imaginary part), each normalized to unit Euclidean norm.

        Parameters
        ----------
        problem : array-like
            Real square matrix.

        Returns
        -------
        List[List[complex]]
            List of eigenvectors (as lists of complex numbers) in the required order.
        """
        # Ensure input is a NumPy array of dtype float
        A = np.asarray(problem, dtype=float)
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(A)          # eigvecs columns are eigenvectors
        # Sort indices by descending real part, then descending imaginary part
        # lexsort uses the last key as primary; we want primary = -real, secondary = -imag
        sort_idx = np.lexsort((-eigvals.imag, -eigvals.real))
        # Reorder eigenvectors accordingly
        eigvecs = eigvecs[:, sort_idx]
        # Normalize each eigenvector to unit norm (broadcast division)
        norms = np.linalg.norm(eigvecs, axis=0, keepdims=True)
        eigvecs = eigvecs / norms
        # Convert the normalized eigenvectors (columns) to a list of lists
        result = eigvecs.T.tolist()
        return result