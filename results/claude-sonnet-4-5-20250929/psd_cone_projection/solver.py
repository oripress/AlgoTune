import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solves a given positive semidefinite cone projection problem.
        
        Args:
            problem: A dictionary with problem parameter:
                - A: symmetric matrix.
        
        Returns:
            A dictionary containing the problem solution:
                - X: result of projecting A onto PSD cone.
        """
        A = problem["A"]
        # Use eigh for symmetric matrices - it's faster than eig
        eigvals, eigvecs = np.linalg.eigh(A)
        # Filter out negative eigenvalues
        pos_mask = eigvals > 0
        if not np.any(pos_mask):
            # All eigenvalues are negative, return zero matrix
            return {"X": np.zeros_like(A)}
        eigvals_pos = eigvals[pos_mask]
        eigvecs_pos = eigvecs[:, pos_mask]
        # Reconstruct using only positive eigenvalues
        X = (eigvecs_pos * eigvals_pos) @ eigvecs_pos.T
        return {"X": X}