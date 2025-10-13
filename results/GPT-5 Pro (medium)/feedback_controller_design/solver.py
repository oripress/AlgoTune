from typing import Any, Dict, Optional

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Fast stabilizing static feedback design for discrete-time LTI systems.

        Strategy:
        - Try discrete-time LQR (DARE) with Q=I, R=I to get a stabilizing K.
        - If successful and closed-loop is Schur, compute P from discrete Lyapunov:
              (A+BK)^T P (A+BK) - P = -I
          which guarantees the Lyapunov inequality with strict negativity.
        - If LQR fails (rare for stabilizable systems) or closed-loop not Schur,
          fall back to solving the reference LMI via CVXPY/CLARABEL to match the
          reference stabilizability decision and produce K, P accordingly.

        Returns dict:
        - is_stabilizable: bool
        - K: list (m x n) or None
        - P: list (n x n) or None
        """
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)

        # Ensure dimensions are consistent and B is 2-D
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        n = A.shape[0]
        if A.shape[1] != n or B.shape[0] != n:
            # Invalid dimensions; report non-stabilizable
            return {"is_stabilizable": False, "K": None, "P": None}
        m = B.shape[1]

        # Helper to package output
        def pack(is_stab: bool, K: Optional[np.ndarray], P: Optional[np.ndarray]) -> Dict[str, Any]:
            if not is_stab:
                return {"is_stabilizable": False, "K": None, "P": None}
            # Convert to lists, ensure finite
            if K is None or P is None:
                return {"is_stabilizable": False, "K": None, "P": None}
            if not (np.all(np.isfinite(K)) and np.all(np.isfinite(P))):
                return {"is_stabilizable": False, "K": None, "P": None}
            return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}

        # Attempt fast LQR-based approach
        try:
            Qw = np.eye(n)
            Rw = np.eye(m)
            # Solve DARE for LQR
            P_dare = solve_discrete_are(A, B, Qw, Rw)
            # K_lqr yields u = -K_lqr x; our convention u = K x
            BT_P = B.T @ P_dare
            G = Rw + BT_P @ B
            # Solve G K_lqr = B^T P A
            K_lqr = np.linalg.solve(G, BT_P @ A)
            K = -K_lqr  # because we use u = K x
            Acl = A + B @ K

            # Check stability of closed-loop
            eigs = np.linalg.eigvals(Acl)
            if np.any(np.abs(eigs) >= 1.0 - 1e-12):
                raise RuntimeError("Closed-loop from LQR not strictly Schur.")

            # Compute strict Lyapunov P via discrete Lyapunov with Qp = I
            Qp = np.eye(n)
            # solve_discrete_lyapunov solves A X A^T - X + Q = 0
            # We need (A+BK)^T P (A+BK) - P = -Qp, so use A = (A+BK)^T
            P = solve_discrete_lyapunov(Acl.T, Qp)
            # Symmetrize for numerical safety
            P = 0.5 * (P + P.T)

            # Ensure P positive definite
            evP = np.linalg.eigvalsh(P)
            if np.any(evP <= 1e-12):
                # Add small regularization if needed
                reg = (1e-10 - np.min(evP)) if np.min(evP) < 1e-10 else 0.0
                if reg > 0:
                    P += reg * np.eye(n)

            return pack(True, K, P)
        except Exception:
            # Fall back to CVX-based approach mirroring the reference
            try:
                import cvxpy as cp  # lazy import for speed in normal path

                Q = cp.Variable((n, n), symmetric=True)
                L = cp.Variable((m, n))
                # Form (A + B K) via substitution K = L Q^{-1}, but implemented as in reference LMI
                # Using the exact same constraints/solver as the reference to match stabilizability.
                constraints = [
                    cp.bmat(
                        [
                            [Q, Q @ A.T + L.T @ B.T],
                            [A @ Q + B @ L, Q],
                        ]
                    )
                    >> np.eye(2 * n),
                    Q >> np.eye(n),
                ]
                prob = cp.Problem(cp.Minimize(0), constraints)
                prob.solve(solver=cp.CLARABEL)
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    Qv = Q.value
                    Lv = L.value
                    # Recover K and P
                    K_val = Lv @ np.linalg.inv(Qv)
                    P_val = np.linalg.inv(Qv)
                    # Numerical symmetrization
                    P_val = 0.5 * (P_val + P_val.T)

                    # As a safeguard, if closed-loop is not strictly stable due to numerical issues, refine P via Lyapunov
                    Acl = A + B @ K_val
                    eigs = np.linalg.eigvals(Acl)
                    if np.any(np.abs(eigs) >= 1.0 - 1e-12):
                        # Try slight scaling of K to push eigenvalues in
                        # Scale down K a bit
                        scale = 0.999
                        for _ in range(5):
                            K_try = scale * K_val
                            Acl_try = A + B @ K_try
                            eigs_try = np.linalg.eigvals(Acl_try)
                            if np.all(np.abs(eigs_try) < 1.0 - 1e-12):
                                # Recompute P via Lyapunov for strict inequality
                                P_try = solve_discrete_lyapunov(Acl_try.T, np.eye(n))
                                P_try = 0.5 * (P_try + P_try.T)
                                return pack(True, K_try, P_try)
                            scale *= 0.999
                        # If still not stable, declare non-stabilizable to match reference robustness
                        return pack(False, None, None)
                    else:
                        # Compute strict P via Lyapunov to satisfy is_solution's inequality robustly
                        P_lyap = solve_discrete_lyapunov(Acl.T, np.eye(n))
                        P_lyap = 0.5 * (P_lyap + P_lyap.T)
                        return pack(True, K_val, P_lyap)
                else:
                    return pack(False, None, None)
            except Exception:
                # If even CVX fallback fails, return non-stabilizable
                return pack(False, None, None)