from __future__ import annotations

from typing import Any, Dict

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve, solve as linalg_solve

class Solver:
    def solve(self, problem: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        """
        Solve the finite-horizon discrete-time LQR via backward Riccati recursion.

        Input keys: "A", "B", "Q", "R", "P", "T", "x0"
        Returns: {"U": ndarray of shape (T, m)}
        """
        # Convert inputs to numpy arrays (float64)
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        Q = np.asarray(problem["Q"], dtype=float)
        R = np.asarray(problem["R"], dtype=float)
        P = np.asarray(problem["P"], dtype=float)
        T = int(problem["T"])
        x0 = np.asarray(problem["x0"], dtype=float)
        if x0.ndim > 1:
            x = x0.reshape(-1)
        else:
            x = x0.copy()

        n, m = B.shape
        At = A.T
        Bt = B.T

        # Backward Riccati recursion to compute gains K[t]
        K = np.empty((T, m, n), dtype=float)
        S = P.copy()

        if m == 1:
            # Specialized fast scalar-control path
            b = B[:, 0]  # (n,)
            r = float(R[0, 0])
            for t in range(T - 1, -1, -1):
                # Intermediates
                SA = S @ A                 # (n, n)
                sB = S @ b                 # (n,)
                denom = r + float(b @ sB)  # scalar
                # M2 = b.T @ SA -> shape (n,)
                m2 = b @ SA                # (n,)
                # Kt row vector: (1, n)
                Kt_row = m2 / denom
                K[t, 0, :] = Kt_row
                # Riccati update: S = Q + A.T @ SA - (1/denom) * outer(m2, m2)
                S = Q + At @ SA - (np.outer(m2, m2) / denom)
                S = 0.5 * (S + S.T)
        else:
            # General multi-input path with Cholesky-based solve
            for t in range(T - 1, -1, -1):
                SA = S @ A           # (n, n)
                SB = S @ B           # (n, m)
                M2 = Bt @ SA         # (m, n) == B.T @ S @ A
                M1 = R + Bt @ SB     # (m, m) == R + B.T @ S @ B

                # Solve M1 * Kt = M2 for Kt (SPD system); prefer Cholesky
                try:
                    c, lower = cho_factor(M1, lower=True, check_finite=False)
                    Kt = cho_solve((c, lower), M2, check_finite=False)
                except Exception:
                    try:
                        Kt = linalg_solve(M1, M2, assume_a="pos", check_finite=False)
                    except LinAlgError:
                        # Robust fallback in case of ill-conditioning
                        Kt = np.linalg.pinv(M1) @ M2

                K[t] = Kt

                # Riccati update: S = Q + A.T @ S @ A - M2.T @ Kt
                S = Q + At @ SA - M2.T @ Kt
                S = 0.5 * (S + S.T)

        # Forward simulation to compute control sequence
        U = np.empty((T, m), dtype=float)
        for t in range(T):
            # u = -K[t] @ x
            u = -(K[t] @ x)
            U[t, :] = u
            # x = A @ x + B @ u
            x = A @ x + B @ u

        # Ensure finiteness
        if not np.isfinite(U).all():
            U = np.zeros((T, m), dtype=float)

        return {"U": U}