import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from typing import Any, Dict
class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines asymptotic stability of a discrete-time LTI system and,
        if stable, computes a Lyapunov matrix P satisfying A.T @ P @ A - P = -I.

        Args:
            problem: Dictionary with key "A" containing an n×n matrix (list or np.ndarray).

        Returns:
            dict with keys:
                - "is_stable": bool indicating stability.
                - "P": n×n list of lists (Lyapunov matrix) if stable, else None.
        """
        A = np.array(problem["A"], dtype=float)
        n = A.shape[0]

        # Quick eigenvalue magnitude check for stability
        eigvals = np.linalg.eigvals(A)
        if not np.all(np.abs(eigvals) < 1.0):
            return {"is_stable": False, "P": None}
        # Solve discrete Lyapunov equation using SciPy's implementation
        P = solve_discrete_lyapunov(A.T, np.eye(n))

        return {"is_stable": True, "P": P.tolist()}