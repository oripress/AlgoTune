import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
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
        # Use eigh for symmetric matrices - faster than eig
        eigvals, eigvecs = np.linalg.eigh(A)
        # Zero out negative eigenvalues
        eigvals[eigvals < 0] = 0
        # More efficient reconstruction without creating diagonal matrix
        X = (eigvecs * eigvals) @ eigvecs.T
        return {"X": X}