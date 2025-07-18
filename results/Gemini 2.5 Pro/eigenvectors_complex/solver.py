import numpy as np
from typing import Any

class Solver:
    """
    A solver for finding the normalized and sorted eigenvectors of a matrix.
    """
    def solve(self, problem: list[list[float]], **kwargs) -> Any:
        """
        Computes eigenpairs for a real square matrix.

        This implementation uses a fully vectorized NumPy approach for maximum
        efficiency. It leverages NumPy's highly optimized C and Fortran
        backends for all numerical operations, avoiding the overhead of
        Python loops or JIT compilation.

        The steps are:
        1. Compute eigenpairs using `np.linalg.eig`.
        2. Sort the eigenpairs in descending order based on their eigenvalues.
        3. Normalize the sorted eigenvectors in a single vectorized operation
           using broadcasting. A check is included to prevent division by zero.
        4. Convert the final NumPy array to the required list-of-lists format.
        """
        # Step 1: Convert input to a NumPy array and compute eigenpairs.
        # np.linalg.eig is a highly optimized LAPACK call.
        A = np.array(problem, dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Step 2: Sort the eigenpairs in descending order of eigenvalues.
        # np.argsort is an extremely fast sorting algorithm implementation.
        indices = np.argsort(-eigenvalues)
        sorted_eigenvectors = eigenvectors[:, indices]

        # Step 3: Normalize the eigenvectors using a vectorized approach.
        # Calculate all column norms at once. This is a single, fast operation.
        norms = np.linalg.norm(sorted_eigenvectors, axis=0)

        # Avoid division by zero for zero vectors. Where a norm is ~0, the
        # vector is ~0, so dividing by 1.0 results in the correct zero vector.
        norms[norms < 1e-12] = 1.0

        # Normalize all vectors at once using broadcasting. This is much
        # faster than iterating in Python or dispatching to a Numba function.
        normalized_vectors = sorted_eigenvectors / norms

        # Step 4: Convert to the required output format.
        # Ensure the output is complex as per problem requirements.
        # The .T.tolist() method is an efficient way to get the desired
        # list-of-lists structure (list of row vectors).
        return normalized_vectors.astype(np.complex128).T.tolist()