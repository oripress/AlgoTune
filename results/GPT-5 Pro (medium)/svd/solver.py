from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute the SVD of the input matrix A.

        Returns:
            A dictionary with:
              - "U": (n, k) left singular vectors
              - "S": (k,) singular values
              - "V": (m, k) right singular vectors
            where k = min(n, m)
        """
        # Convert input to a NumPy array with float64 dtype (needed for numerical stability/accuracy)
        A = np.asarray(problem["matrix"], dtype=np.float64)

        # Use economy-size SVD for performance and correct shapes
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        # V is the transpose of Vt
        V = Vt.T

        return {"U": U, "S": s, "V": V}