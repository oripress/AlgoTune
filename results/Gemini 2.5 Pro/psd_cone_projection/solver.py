from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves a given positive semidefinite cone projection problem.
        
        Args:
            problem: A dictionary with problem parameter:
                - A: symmetric matrix.
        
        Returns:
            A dictionary containing the problem solution:
                - X: result of projecting A onto PSD cone.
        """
        A = np.array(problem["A"])
        
        # Since A is symmetric, eigh is more efficient than eig.
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Clip eigenvalues to be non-negative.
        eigvals[eigvals < 0] = 0
        
        # Reconstruct the matrix using broadcasting, which is more efficient
        # than creating a diagonal matrix.
        X = (eigvecs * eigvals) @ eigvecs.T
        
        return {"X": X.tolist()}