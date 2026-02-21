import numpy as np
import scipy.linalg
from typing import Any

class Solver:
    def __init__(self):
        self.eye_cache = {}

    def get_eye(self, n: int) -> np.ndarray:
        if n not in self.eye_cache:
            self.eye_cache[n] = np.eye(n, dtype=float)
        return self.eye_cache[n]

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        A = np.array(problem["A"], dtype=float)
        n = A.shape[0]
        
        # Fast necessary conditions for stability
        if abs(np.trace(A)) >= n or abs(np.linalg.det(A)) >= 1.0:
            return {"is_stable": False, "P": None}
            
        # Check stability
        eigenvalues = np.linalg.eigvals(A)
        # Fast check for max absolute eigenvalue
        if np.any(np.abs(eigenvalues) >= 1.0 - 1e-10):
            return {"is_stable": False, "P": None}
        try:
            # Solve discrete Lyapunov equation: A.T @ P @ A - P + I = 0
            # scipy.linalg.solve_discrete_lyapunov uses the Bartels-Stewart algorithm
            # Let's try to use scipy.linalg._solve_sylvester.solve_discrete_lyapunov directly
            # Actually, scipy.linalg.solve_discrete_lyapunov is a wrapper around solve_sylvester or solve_discrete_are
            # Let's just use scipy.linalg.solve_discrete_lyapunov
            P = scipy.linalg.solve_discrete_lyapunov(A.T, self.get_eye(n))
            
            # Ensure symmetry
            P = (P + P.T) * 0.5
            
            return {"is_stable": True, "P": P.tolist()}
        except Exception:
            return {"is_stable": False, "P": None}