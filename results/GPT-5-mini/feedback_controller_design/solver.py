from typing import Any, Dict
import numpy as np
from scipy import linalg as la

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Design a static state-feedback controller u = K x for discrete-time LTI system:
            x[k+1] = A x[k] + B u[k]

        Returns:
            {"is_stabilizable": bool, "K": m x n list or None, "P": n x n list or None}
        """
        # Basic validation & parsing
        if not isinstance(problem, dict) or "A" not in problem or "B" not in problem:
            return {"is_stabilizable": False, "K": None, "P": None}
        try:
            A = np.array(problem["A"], dtype=float)
            B = np.array(problem["B"], dtype=float)
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"is_stabilizable": False, "K": None, "P": None}
        n = A.shape[0]

        # Normalize B to 2D
        if B.ndim == 1:
            if B.size == n:
                B = B.reshape((n, 1))
            elif B.size == 0:
                B = B.reshape((n, 0))
            else:
                return {"is_stabilizable": False, "K": None, "P": None}
        if B.ndim != 2 or B.shape[0] != n:
            return {"is_stabilizable": False, "K": None, "P": None}
        m = B.shape[1]

        # Check finite entries
        if not np.isfinite(A).all() or (B.size and not np.isfinite(B).all()):
            return {"is_stabilizable": False, "K": None, "P": None}

        # Compute eigenvalues (used multiple times)
        try:
            eigs = np.linalg.eigvals(A)
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}

        # No-input case: must already be stable
        if m == 0:
            if np.all(np.abs(eigs) < 1.0 - 1e-12):
                Q = np.eye(n)
                try:
                    # Solve A^T P A - P = -Q  -> solve_discrete_lyapunov with A.T
                    P = la.solve_discrete_lyapunov(A.T, Q)
                except Exception:
                    # fallback series summation
                    P = np.zeros((n, n))
                    Ak = np.eye(n)
                    for _ in range(2000):
                        P_next = P + Ak.T @ Q @ Ak
                        if np.linalg.norm(P_next - P, ord="fro") < 1e-12:
                            P = P_next
                            break
                        P = P_next
                        Ak = Ak @ A
                    P = (P + P.T) / 2
                # Return empty K (no inputs)
                return {"is_stabilizable": True, "K": [], "P": P.tolist()}
            else:
                return {"is_stabilizable": False, "K": None, "P": None}

        # PBH test for stabilizability: for eigenvalues with |lambda| >= 1, rank([A - lambda I, B]) must be n
        I = np.eye(n)
        for lam in eigs:
            if np.abs(lam) >= 1.0 - 1e-12:
                M = np.hstack((A - lam * I, B))
                if np.linalg.matrix_rank(M) < n:
                    return {"is_stabilizable": False, "K": None, "P": None}

        # Prepare weighting matrices for DARE (simple identity choices)
        Q = np.eye(n)
        R = np.eye(m)
        tiny = 1e-9

        # Primary fast approach: solve discrete algebraic Riccati equation (DARE)
        try:
            P = la.solve_discrete_are(A, B, Q, R)
            P = (P + P.T) / 2
            S = R + B.T @ P @ B
            try:
                K_lqr = np.linalg.solve(S, B.T @ P @ A)
            except np.linalg.LinAlgError:
                K_lqr = np.linalg.solve(S + tiny * np.eye(m), B.T @ P @ A)
            # LQR gives u = -K_lqr x; we must return u = K x so set K = -K_lqr
            K = -K_lqr

            # Remove negligible imaginary parts
            if np.iscomplexobj(P) and np.max(np.abs(P.imag)) < 1e-8:
                P = P.real
            if np.iscomplexobj(K) and np.max(np.abs(K.imag)) < 1e-8:
                K = K.real

            # Ensure correct shape for K (m x n)
            K = np.atleast_2d(np.array(K, dtype=float))
            if K.shape[0] != m:
                K = K.reshape((m, n))

            # Verify closed-loop stability and Lyapunov condition; if not satisfied, fall through to fallback
            Acl = A + B @ K
            eig_cl = np.linalg.eigvals(Acl)
            if np.any(np.abs(eig_cl) >= 1.0 + 1e-10):
                raise RuntimeError("DARE did not stabilize")

            eigP = np.linalg.eigvals(P)
            if np.any(np.real(eigP) <= 1e-12):
                raise RuntimeError("P not positive definite")

            S_mat = Acl.T @ P @ Acl - P
            if np.any(np.linalg.eigvals(S_mat) > 1e-8):
                # Try to compute a Lyapunov P for the closed-loop instead
                try:
                    P2 = la.solve_discrete_lyapunov(Acl.T, np.eye(n))
                    P = (P2 + P2.T) / 2
                except Exception:
                    raise RuntimeError("Lyapunov condition failed")
            return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
        except Exception:
            # Fallback 1: pole placement (if available)
            try:
                # Choose desired poles strictly inside unit disk:
                # take original eigenvalues and shrink their magnitude to 0.8 times (but inside)
                desired = []
                for lam in eigs:
                    if np.abs(lam) < 1e-8:
                        desired.append(0.5)
                    else:
                        desired.append(0.8 * lam / max(1.0, np.abs(lam)))
                desired = np.array(desired, dtype=complex)

                # Lazy import to avoid hard dependency when not needed
                from scipy.signal import place_poles

                pp = place_poles(A, B, desired)
                # place_poles returns Kp such that A - B Kp has desired poles
                Kp = np.array(pp.gain_matrix, dtype=float)
                # We want A + B K = A - B Kp  => K = -Kp
                K = -Kp
                K = np.atleast_2d(K)
                if K.shape[0] != m:
                    K = K.reshape((m, n))

                # Compute Lyapunov P for closed-loop Acl^T P Acl - P = -I
                Acl = A + B @ K
                try:
                    P = la.solve_discrete_lyapunov(Acl.T, np.eye(n))
                except Exception:
                    # fallback series summation
                    P = np.zeros((n, n))
                    Ak = np.eye(n)
                    for _ in range(2000):
                        P_next = P + Ak.T @ np.eye(n) @ Ak
                        if np.linalg.norm(P_next - P, ord="fro") < 1e-12:
                            P = P_next
                            break
                        P = P_next
                        Ak = Ak @ Acl
                    P = (P + P.T) / 2

                eig_cl = np.linalg.eigvals(Acl)
                if np.any(np.abs(eig_cl) >= 1.0 + 1e-10):
                    raise RuntimeError("Pole placement failed to stabilize")

                return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
            except Exception:
                # Fallback 2: iterative Riccati/value iteration
                try:
                    P = np.eye(n)
                    tol = 1e-9
                    max_iter = 2000
                    for _ in range(max_iter):
                        S = R + B.T @ P @ B
                        try:
                            Sinv = np.linalg.inv(S)
                        except np.linalg.LinAlgError:
                            Sinv = np.linalg.inv(S + tiny * np.eye(m))
                        K_lqr = Sinv @ B.T @ P @ A
                        P_next = A.T @ P @ A - A.T @ P @ B @ K_lqr + Q
                        P_next = (P_next + P_next.T) / 2
                        if np.linalg.norm(P_next - P, ord="fro") < tol:
                            P = P_next
                            break
                        P = P_next
                    else:
                        return {"is_stabilizable": False, "K": None, "P": None}

                    S = R + B.T @ P @ B
                    try:
                        K_lqr = np.linalg.solve(S, B.T @ P @ A)
                    except np.linalg.LinAlgError:
                        K_lqr = np.linalg.solve(S + tiny * np.eye(m), B.T @ P @ A)
                    K = -K_lqr
                    K = np.atleast_2d(np.array(K, dtype=float))
                    if K.shape[0] != m:
                        K = K.reshape((m, n))

                    Acl = A + B @ K
                    eig_cl = np.linalg.eigvals(Acl)
                    if np.any(np.abs(eig_cl) >= 1.0 + 1e-8):
                        return {"is_stabilizable": False, "K": None, "P": None}

                    # Try to compute Lyapunov P for the closed-loop
                    try:
                        P2 = la.solve_discrete_lyapunov(Acl.T, np.eye(n))
                        P = (P2 + P2.T) / 2
                    except Exception:
                        P = (P + P.T) / 2

                    return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
                except Exception:
                    return {"is_stabilizable": False, "K": None, "P": None}