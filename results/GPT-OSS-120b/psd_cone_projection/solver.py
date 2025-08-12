import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Projects a symmetric matrix onto the PSD cone by zeroing negative eigenvalues.
        Args:
            problem: dict with key "A" containing a symmetric matrix as list of lists.
        Returns:
            dict with key "X" containing the projected matrix as list of lists.
        """
        # Convert input to a NumPy array (ensure float dtype)
        A = np.asarray(problem["A"], dtype=float)
        # Compute eigen-decomposition once
        eigvals, eigvecs = np.linalg.eigh(A)
        if np.min(eigvals) >= 0:
            # Already positive semidefinite – return original matrix as NumPy array
            return {"X": A}
        # Zero out negative eigenvalues
        eigvals = np.maximum(eigvals, 0.0)
        # Reconstruct the projected matrix
        X = (eigvecs * eigvals) @ eigvecs.T
        # Return the projected matrix as NumPy array
        return {"X": X}