from typing import Any
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_lyapunov

class Solver:
    def _verify_solution(self, A, P, p_tol=1e-10, s_tol=1e-10):
        """Verifies if P is a valid Lyapunov matrix against checker tolerances."""
        if P is None:
            return False
        
        P_sym = (P + P.T) / 2.0
        try:
            if np.min(np.linalg.eigvalsh(P_sym)) < p_tol:
                return False
        except np.linalg.LinAlgError:
            return False

        S = A.T @ P_sym @ A - P_sym
        S_sym = (S + S.T) / 2.0
        try:
            if np.max(np.linalg.eigvalsh(S_sym)) > -s_tol:
                return False
        except np.linalg.LinAlgError:
            return False
            
        return True

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        A = np.array(problem["A"])
        n = A.shape[0]
        I = np.eye(n)

        # --- Fast Path: Direct Lyapunov Solver ---
        # This is much faster than SDP and works for most stable systems.
        try:
            # A necessary condition for stability. If violated, no need to solve.
            if np.max(np.abs(np.linalg.eigvals(A))) >= 1.0:
                return {"is_stable": False, "P": None}
            
            # Solve A'PA - P = -I directly.
            P_fast = solve_discrete_lyapunov(A, I)
            
            # Verify if the solution is numerically robust enough for the checker.
            if self._verify_solution(A, P_fast):
                P_fast = (P_fast + P_fast.T) / 2.0
                return {"is_stable": True, "P": P_fast.tolist()}
        except np.linalg.LinAlgError:
            # This solver can fail for systems near the stability boundary.
            # We will fall back to the robust SDP solver in this case.
            pass

        # --- Robust Fallback: SDP Solver ---
        # This is slower but handles the ill-conditioned cases the fast path misses.
        try:
            P = cp.Variable((n, n), symmetric=True)
            constraints = [P >> 0, A.T @ P @ A - P << -I]
            prob = cp.Problem(cp.Minimize(0), constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                P_sol = P.value
                if P_sol is not None:
                    # Adjust solution to guarantee strict positive definiteness
                    P_final = P_sol + 1e-6 * I
                    P_final = (P_final + P_final.T) / 2.0
                    return {"is_stable": True, "P": P_final.tolist()}
            
            return {"is_stable": False, "P": None}
        except Exception:
            return {"is_stable": False, "P": None}