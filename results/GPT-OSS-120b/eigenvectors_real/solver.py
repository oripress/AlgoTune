import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute eigenvalues and orthonormal eigenvectors of a real symmetric matrix.

        Parameters
        ----------
        problem : list[list[float]] or np.ndarray
            Real symmetric matrix.

        Returns
        -------
        tuple[list[float], list[list[float]]]
            Eigenvalues sorted in descending order and corresponding eigenvectors
            (each eigenvector as a list, rows correspond to eigenvectors).
        """
        # Convert input to a NumPy array if it isn't already
        A = np.asarray(problem, dtype=float)

        # Ensure the matrix is 2‑dimensional and square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input must be a square matrix.")

        # Use NumPy's symmetric eigensolver (returns ascending eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Reverse order to get descending eigenvalues
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]   # columns correspond to eigenvectors

        # Convert to required Python list format: eigenvectors as rows
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = [eigenvectors[:, i].tolist() for i in range(eigenvectors.shape[1])]

        return eigenvalues_list, eigenvectors_list