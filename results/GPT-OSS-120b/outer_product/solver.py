import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the outer product of two vectors.

        Parameters
        ----------
        problem : tuple[np.ndarray, np.ndarray]
            A tuple containing two 1‑D NumPy arrays of equal length.

        Returns
        -------
        np.ndarray
            An n×n matrix where element (i, j) = vec1[i] * vec2[j].
        """
        vec1, vec2 = problem
        # Use NumPy's highly optimized outer product implementation.
        return np.outer(vec1, vec2)