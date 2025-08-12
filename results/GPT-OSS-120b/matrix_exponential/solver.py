import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the matrix exponential exp(A) for a given square matrix A.

        Parameters
        ----------
        problem : dict
            Dictionary containing the key "matrix" with a square list-of-lists
            representing the matrix A.

        Returns
        -------
        dict
            Dictionary with key "exponential" containing the matrix
            exponential as a list of lists (JSON‑compatible).
        """
        # Convert input to NumPy array (float64)
        A = np.asarray(problem["matrix"], dtype=np.float64)

        # Compute matrix exponential using SciPy (highly optimized)
        expA = expm(A)

        # Convert result to plain Python list-of-lists for JSON compatibility
        return {"exponential": expA.tolist()}