from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square sparse matrix.
        The solution returned is a list of the eigenvectors with the largest `m` eigenvalues sorted in descending order by their modulus.

        :param problem: A dictionary representing the sparse eigenvalue problem.
        :return: List of eigenvectors sorted in descending order by the modulus of the corresponding eigenvalue.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        A = problem["matrix"]
        k = problem["k"]
        # Check if input is already a sparse matrix or needs conversion
        if isinstance(A, sparse.spmatrix):
            A_sparse = A
        else:
            # Convert the input list to a sparse CSR matrix directly
            A_sparse = sparse.csr_matrix(A, dtype=np.complex128)
        N = A_sparse.shape[0]
        # Create a deterministic starting vector
        v0 = np.ones(N, dtype=A_sparse.dtype)  # Use matrix dtype

        # Compute eigenvalues using sparse.linalg.eigs
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            A_sparse,
            k=k,
            v0=v0,  # Add deterministic start vector
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )

        pairs = list(zip(eigenvalues, eigenvectors.T))
        # Sort by descending order of eigenvalue modulus
        pairs.sort(key=lambda pair: -np.abs(pair[0]))

        solution = [pair[1] for pair in pairs]

        return solution