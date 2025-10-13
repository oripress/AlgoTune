from typing import Any, Dict
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve as sp_solve

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute optimal control sequence via backward Riccati recursion.

        Returns dict with key "U" (shape (T, m)).
        """
        # Parse inputs
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        Q = np.asarray(problem["Q"], dtype=float)
        R = np.asarray(problem["R"], dtype=float)
        P = np.asarray(problem["P"], dtype=float)
        T = int(problem["T"])
        x0 = np.asarray(problem["x0"], dtype=float).reshape(-1)

        n, m = B.shape

        # Backward Riccati recursion storing only current S and K sequence
        K = np.empty((T, m, n), dtype=float)
        S = P.copy()
        At = A.T
        Bt = B.T

        for t in range(T - 1, -1, -1):
            # Precompute SA, SB
            SA = S @ A
            SB = S @ B

            # M1 = R + B^T S B, M2 = B^T S A
            M1 = R + Bt @ SB
            M2 = Bt @ SA

            # Solve M1 * Kt = M2 efficiently (prefer Cholesky)
            try:
                c, lower = cho_factor(M1, lower=True, check_finite=False)
                Kt = cho_solve((c, lower), M2, check_finite=False)
            except Exception:
                try:
                    Kt = sp_solve(M1, M2, assume_a="sym", check_finite=False)
                except Exception:
                    # Robust fallback
                    Kt = np.linalg.pinv(M1) @ M2

            K[t] = Kt

            # Riccati update: S = Q + A^T S A - (B^T S A)^T (R + B^T S B)^{-1} (B^T S A)
            S = Q + At @ SA - M2.T @ Kt
            # Symmetrize to improve numerical stability
            S = 0.5 * (S + S.T)

        # Forward simulation to get U
        U = np.empty((T, m), dtype=float)
        x = x0
        for t in range(T):
            u = -K[t].dot(x)
            U[t] = u
            x = A.dot(x)
            x += B.dot(u)

        return {"U": U}