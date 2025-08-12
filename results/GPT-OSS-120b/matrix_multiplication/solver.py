import numpy as np

class Solver:
    def solve(self, problem: dict) -> any:
        """
        Compute the matrix product C = A · B using NumPy.

        Parameters
        ----------
        problem : dict
            Dictionary with keys "A" and "B", each a list of lists representing
            matrices.

        Returns
        -------
        numpy.ndarray
            The product matrix C as a NumPy array.
        """
        # Convert input lists to NumPy arrays (float64 for speed and precision)
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        # Perform matrix multiplication using highly‑optimized BLAS via NumPy
        return np.dot(A, B).tolist()