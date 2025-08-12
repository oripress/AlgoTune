from typing import Any, Optional, Tuple

import numpy as np
from numpy.linalg import eigvals, matrix_rank
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

class Solver:
    @staticmethod
    def _is_schur_stable(A: np.ndarray, tol: float = 1e-10) -> bool:
        try:
            return np.all(np.abs(eigvals(A)) < 1.0 - tol)
        except Exception:
            return False

    @staticmethod
    def _stable_lyapunov_P(A: np.ndarray) -> np.ndarray:
        """
        For Schur-stable A, compute P solving A^T P A - P = -I.
        """
        n = A.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=float)
        Q = np.eye(n, dtype=float)
        P = solve_discrete_lyapunov(A.T, Q)
        P = np.real_if_close(0.5 * (P + P.T))
        return P

    @staticmethod
    def _lqr_gain(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stabilizing LQR gain K and Riccati matrix P for Q=I, R=I.
        Returns:
            K (m x n), P (n x n)
        """
        n = A.shape[0]
        m = B.shape[1]
        Q = np.eye(n, dtype=float)
        R = np.eye(m, dtype=float)
        P = solve_discrete_are(A, B, Q, R)
        RB = R + B.T @ P @ B
        K_lqr = np.linalg.solve(RB, B.T @ P @ A)
        K = -K_lqr  # our convention u = K x
        P = np.real_if_close(0.5 * (P + P.T))
        return K, P

    @staticmethod
    def _fallback_sdp(A: np.ndarray, B: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Fallback to the SDP using CVXPY. Returns (ok, K, P).
        """
        try:
            import cvxpy as cp  # lazy import
        except Exception:
            return False, None, None

        n, m = A.shape[0], B.shape[1]
        Qv = cp.Variable((n, n), symmetric=True)
        L = cp.Variable((m, n))
        constraints = [
            cp.bmat(
                [
                    [Qv, Qv @ A.T + L.T @ B.T],
                    [A @ Qv + B @ L, Qv],
                ]
            )
            >> np.eye(2 * n),
            Qv >> np.eye(n),
        ]
        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            solver_choice = getattr(cp, "CLARABEL", None) or cp.SCS
            prob.solve(solver=solver_choice, verbose=False)
        except Exception:
            return False, None, None

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return False, None, None

        Qval = Qv.value
        Lval = L.value
        if Qval is None or Lval is None:
            return False, None, None
        try:
            Qinv = np.linalg.inv(Qval)
        except np.linalg.LinAlgError:
            return False, None, None
        K = Lval @ Qinv
        P = Qinv
        K = np.real_if_close(K)
        P = np.real_if_close(0.5 * (P + P.T))
        return True, K, P

    def solve(self, problem, **kwargs) -> Any:
        """
        Fast solver for static state feedback stabilization of discrete-time LTI systems.

        Strategy:
        - If A is Schur-stable: return K=0 and P from Lyapunov equation.
        - Else (m > 0): try LQR (DARE) to compute stabilizing K,P.
        - If LQR fails, fall back to SDP via CVXPY; if that fails, declare not stabilizable.

        Returns dict with keys: is_stabilizable (bool), K (list or None), P (list or None).
        """
        # Parse inputs safely
        try:
            A = np.array(problem["A"], dtype=float)
            B = np.array(problem["B"], dtype=float)
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"is_stabilizable": False, "K": None, "P": None}
        if B.ndim != 2 or B.shape[0] != A.shape[0]:
            return {"is_stabilizable": False, "K": None, "P": None}

        n = A.shape[0]
        m = B.shape[1]

        # No-input case
        if m == 0:
            if self._is_schur_stable(A):
                P = self._stable_lyapunov_P(A)
                K = np.zeros((0, n), dtype=float)
                return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
            return {"is_stabilizable": False, "K": None, "P": None}

        # If already stable, trivial controller
        if self._is_schur_stable(A):
            P = self._stable_lyapunov_P(A)
            K = np.zeros((m, n), dtype=float)
            return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}

        # Try LQR for fast stabilization (covers stabilizable systems with Q=I,R=I)
        try:
            K, P = self._lqr_gain(A, B)
            K = np.real_if_close(K)
            P = np.real_if_close(0.5 * (P + P.T))
            if not (np.all(np.isfinite(K)) and np.all(np.isfinite(P))):
                raise RuntimeError("Non-finite K or P")
            return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
        except Exception:
            # Fall back to SDP for robustness (rare)
            ok, K_sdp, P_sdp = self._fallback_sdp(A, B)
            if not ok or K_sdp is None or P_sdp is None:
                return {"is_stabilizable": False, "K": None, "P": None}
            if not (np.all(np.isfinite(K_sdp)) and np.all(np.isfinite(P_sdp))):
                return {"is_stabilizable": False, "K": None, "P": None}
            return {"is_stabilizable": True, "K": K_sdp.tolist(), "P": P_sdp.tolist()}