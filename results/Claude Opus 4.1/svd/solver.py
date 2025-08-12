import numpy as np
from scipy import linalg
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the SVD problem using scipy's optimized SVD implementation.
        """
        A = np.array(problem["matrix"], dtype=np.float64)
        
        # Use scipy's SVD which can be faster than numpy's
        # lapack_driver='gesdd' is often faster for general matrices
        U, s, Vh = linalg.svd(A, full_matrices=False, lapack_driver='gesdd')
        V = Vh.T  # Convert Vh to V so that A = U * diag(s) * V^T
        
        return {"U": U, "S": s, "V": V}