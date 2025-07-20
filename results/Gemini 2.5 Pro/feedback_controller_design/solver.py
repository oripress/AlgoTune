from typing import Any
import numpy as np
from scipy.linalg import eigvals, LinAlgError, solve_discrete_are, solve_discrete_lyapunov, cholesky, qr

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        n, m = A.shape[0], B.shape[1]

        # Step 1: Fast stability check with eigvals only.
        try:
            # Using check_finite=False skips validation, which is safe for this problem.
            eigenvalues = eigvals(A, check_finite=False)
        except LinAlgError:
            # If eigenvalue computation fails, system is likely ill-conditioned.
            return {"is_stabilizable": False, "K": None, "P": None}

        # Step 2: Path for already-stable systems (fastest).
        if np.max(np.abs(eigenvalues)) < 1.0 - 1e-9:
            try:
                P = solve_discrete_lyapunov(A, np.eye(n))
                cholesky(P, lower=True)
                K = np.zeros((m, n))
                return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
            except (LinAlgError, np.linalg.LinAlgError):
                # Lyapunov solver can fail for systems near stability boundary.
                # Let the subsequent, more robust paths handle this.
                pass

        # Step 3: Path for unstable systems - try fast DARE solver.
        try:
            P = solve_discrete_are(A, B, np.eye(n), np.eye(m), balanced=False)
            cholesky(P, lower=True)
            K = -np.linalg.inv(np.eye(m) + B.T @ P @ B) @ (B.T @ P @ A)
            return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
        except (LinAlgError, np.linalg.LinAlgError):
            # DARE failed. Proceed to definitive stabilizability check before LMI.
            pass

        # Step 4: DARE failed. Use PBH test to check for non-stabilizability.
        # This avoids the slow LMI solver for provably non-stabilizable systems.
        unstable_eigenvalues = eigenvalues[np.abs(eigenvalues) >= 1.0 - 1e-9]
        for lam in unstable_eigenvalues:
            pbh_matrix = np.hstack([A - lam * np.eye(n), B])
            # Use QR decomposition for a faster rank check than SVD-based matrix_rank
            R = qr(pbh_matrix, mode='r', check_finite=False)
            if np.sum(np.abs(np.diag(R)) > 1e-9) < n:
                return {"is_stabilizable": False, "K": None, "P": None}

        # Step 5: PBH test passed, so system is stabilizable.
        # DARE must have failed for numerical reasons. Fallback to robust LMI solver.
        try:
            import cvxpy as cp
            
            Q_var = cp.Variable((n, n), symmetric=True)
            L = cp.Variable((m, n))
            epsilon = 1e-7
            constraints = [
                cp.bmat([
                    [Q_var, (A @ Q_var + B @ L).T],
                    [A @ Q_var + B @ L, Q_var]
                ]) >> epsilon * np.eye(2 * n),
                Q_var >> epsilon * np.eye(n)
            ]
            prob = cp.Problem(cp.Minimize(0), constraints)
            prob.solve(solver=cp.CLARABEL)

            if prob.status in ["optimal", "optimal_inaccurate"]:
                Q_val = Q_var.value
                L_val = L.value
                if Q_val is not None and L_val is not None:
                    P_val = np.linalg.inv(Q_val)
                    K_val = L_val @ P_val
                    return {"is_stabilizable": True, "K": K_val.tolist(), "P": P_val.tolist()}
        except Exception:
            pass

        # If all paths fail, classify as not stabilizable.
        return {"is_stabilizable": False, "K": None, "P": None}