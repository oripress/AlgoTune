from typing import Any
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solve discrete-time finite-horizon LQR via backward Riccati recursion
        with Cholesky-based solves for speed and numerical stability.

        Input (problem): dict with keys "A","B","Q","R","P","T","x0".
        Output: dict with key "U" containing optimal control sequence shape (T, m).
        """
        # Parse inputs as numpy arrays (float)
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        Q = np.asarray(problem["Q"], dtype=float)
        R = np.asarray(problem["R"], dtype=float)
        S = np.asarray(problem["P"], dtype=float)  # terminal cost
        T = int(problem["T"])
        x0 = np.asarray(problem["x0"], dtype=float).reshape(-1)

        # Dimensions
        n, m = B.shape

        # Preallocate feedback gains K (T, m, n)
        if T > 0:
            K = np.empty((T, m, n), dtype=float)
        else:
            K = np.empty((0, m, n), dtype=float)

        # Local references for speed
        Bt = B.T
        A_mat = A
        Q_mat = Q
        R_mat = R

        # Backward Riccati recursion
        for t in range(T - 1, -1, -1):
            M1 = R_mat + Bt @ S @ B      # (m x m), should be positive definite
            M2 = Bt @ S @ A_mat          # (m x n)
            # Prefer Cholesky solve for speed/stability
            try:
                c_and_lower = cho_factor(M1, check_finite=False)
                Kt = cho_solve(c_and_lower, M2, check_finite=False)
            except Exception:
                # Fallback to standard solve, then pseudo-inverse
                try:
                    Kt = np.linalg.solve(M1, M2)
                except np.linalg.LinAlgError:
                    Kt = np.linalg.pinv(M1) @ M2
            K[t] = Kt
            Acl = A_mat - B @ Kt
            S = Q_mat + Kt.T @ R_mat @ Kt + Acl.T @ S @ Acl
            # Enforce symmetry to avoid numerical drift
            S = 0.5 * (S + S.T)

        # Forward simulation to compute optimal controls
        U = np.zeros((T, m), dtype=float)
        x = x0.copy()
        for t in range(T):
            u = -K[t] @ x
            U[t] = u
            x = A_mat @ x + B @ u

        return {"U": U}