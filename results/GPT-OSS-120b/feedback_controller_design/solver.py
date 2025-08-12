import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design a static state feedback controller for a discrete-time LTI system.

        Returns a dictionary with:
            - is_stabilizable: bool
            - K: list (m x n) feedback gain matrix (or None)
            - P: list (n x n) Lyapunov matrix (or None)
        """
        # Convert inputs to numpy arrays
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)

        n = A.shape[0]
        m = B.shape[1]

        # ---------- Discrete-time LQR (R=I, Q=I) via fast Riccati iteration ----------
        Q = np.eye(n, dtype=np.float64)
        R = np.eye(m, dtype=np.float64)

        # Initialize P
        P = Q.copy()
        max_iter = 50          # few iterations are enough for typical sizes
        tol = 1e-9
        for _ in range(max_iter):
            S = R + B.T @ P @ B
            # Compute next P without forming K each step
            P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(S, B.T @ P @ A) + Q
            # Ensure symmetry
            P_next = (P_next + P_next.T) * 0.5
            if np.linalg.norm(P_next - P, ord='fro') < tol:
                P = P_next
                break
            P = P_next

        # Compute optimal gain K from final P
        S = R + B.T @ P @ B
        K = -np.linalg.solve(S, B.T @ P @ A)

        # Verify stability (optional safety check)
        closed_loop = A + B @ K
        eigs = np.linalg.eigvals(closed_loop)
        if np.any(np.abs(eigs) >= 1.0):
            return {"is_stabilizable": False, "K": None, "P": None}

        return {
            "is_stabilizable": True,
            "K": K.tolist(),
            "P": P.tolist(),
        }
        return {
            "is_stabilizable": True,
            "K": K.tolist(),
            "P": P.tolist(),
        }