import numpy as np
from scipy.linalg import eigh
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        """
        Solve the generalized eigenvalue problem A·x = λB·x.
        
        Uses scipy.linalg.eigh with built-in support for generalized eigenvalue problems,
        which is more efficient than manually doing Cholesky decomposition and inversion.
        """
        A, B = problem
        
        # Use scipy's optimized generalized eigenvalue solver
        # eigvals_only=True is faster as it doesn't compute eigenvectors
        # eigh returns eigenvalues in ascending order, so we reverse using slicing
        eigenvalues = eigh(A, B, eigvals_only=True, check_finite=False)
        
        # Convert to list in descending order using efficient array slicing
        return eigenvalues[::-1].tolist()