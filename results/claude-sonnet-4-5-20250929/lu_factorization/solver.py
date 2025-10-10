import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve the LU factorization problem with minimal conversion overhead.
        """
        # Directly work with the matrix - avoid unnecessary copies
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.asarray(A, dtype=np.float64)
        
        # Compute LU decomposition
        P, L, U = lu(A)
        
        # Direct list conversion (fastest way)
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}