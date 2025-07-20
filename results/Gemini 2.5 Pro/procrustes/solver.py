import numpy as np
from scipy.linalg import svd
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solve the OPP instance by first computing M = B A^T and
        computing its singular value decomposition: M = U S V^T.
        The final solution is obtained via G = U V^T.
        """
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}
        
        try:
            A_np = np.array(A, dtype=np.float64)
            B_np = np.array(B, dtype=np.float64)
        except ValueError:
            return {}

        if A_np.shape != B_np.shape or len(A_np.shape) != 2:
            return {}
            
        # Compute M = B A^T
        M = B_np @ A_np.T
        
        # Compute SVD of M
        U, _, Vt = svd(M)
        
        # Compute G = U V^T
        G = U @ Vt
        
        solution = {"solution": G.tolist()}
        
        return solution