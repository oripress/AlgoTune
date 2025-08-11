import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the eigenvalues of a symmetric real matrix.

        Parameters
        ----------
        problem : list[list[float]] or np.ndarray
            Symmetric matrix of shape (n, n).

        Returns
        -------
        list[float]
            Eigenvalues sorted in descending order.
        """
        # Convert input to a NumPy array of type float64.
        A = np.asarray(problem, dtype=np.float64)

        # Validate that the matrix is square.
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input must be a square matrix.")

        # Compute eigenvalues; eigvalsh returns them in ascending order.
        eigvals = np.linalg.eigvalsh(A)

        # Return eigenvalues in descending order as a plain Python list.
        return eigvals[::-1].tolist()