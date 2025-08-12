from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the Lyapunov stability analysis problem using optimized methods.

        Args:
            problem: A dictionary containing the system matrix A.

        Returns:
            A dictionary containing:
                - is_stable: Boolean indicating if the system is asymptotically stable
                - P: The Lyapunov matrix P (if stable)
        """
        A = np.array(problem["A"], dtype=np.float64)
        n = A.shape[0]
        
        # Quick checks for obvious instability (much faster than eigenvalues)
        # For 1x1 matrix, just check if |a| >= 1
        if n == 1:
            if abs(A[0, 0]) >= 1.0:
                return {"is_stable": False, "P": None}
            else:
                P = np.array([[1.0 / (1.0 - A[0, 0]**2)]])
                return {"is_stable": True, "P": P.tolist()}
        
        # For 2x2 matrix, check trace and determinant conditions
        elif n == 2:
            trace = np.trace(A)
            det = np.linalg.det(A)
            # For 2x2 discrete system, stability requires:
            # 1. |trace| < 1 + det
            # 2. |det| < 1
            if abs(det) >= 1.0 or abs(trace) >= 1.0 + det:
                return {"is_stable": False, "P": None}
        
        # Try to solve Lyapunov equation directly - this will fail if system is unstable
        try:
            # Use scipy's discrete Lyapunov equation solver
            from scipy import linalg
            # Solve A^T P A - P = -I
            P = linalg.solve_discrete_lyapunov(A.T, np.eye(n))
            
            # Quick check without computing all eigenvalues
            # Check if P is positive definite using Cholesky
            try:
                np.linalg.cholesky(P)
                # Check if A^T P A - P is negative definite
                S = A.T @ P @ A - P
                # For negative definite, -S should be positive definite
                try:
                    np.linalg.cholesky(-S)
                    return {"is_stable": True, "P": P.tolist()}
                except:
                    pass
            except:
                pass
        except:
            pass
        
        # Fast spectral radius estimation using power iteration
        # This is much faster than computing all eigenvalues
        try:
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            for _ in range(20):  # Usually converges in < 20 iterations
                v_new = A @ v
                v_new_norm = np.linalg.norm(v_new)
                if v_new_norm > 1.01:  # If we exceed 1, system is unstable
                    return {"is_stable": False, "P": None}
                v = v_new / v_new_norm
            
            # Estimate spectral radius from last iteration
            spectral_radius = np.linalg.norm(A @ v)
            if spectral_radius >= 0.99:  # Conservative threshold
                # If close to 1, fall back to eigenvalue computation
                eigenvalues = np.linalg.eigvals(A)
                if np.any(np.abs(eigenvalues) >= 1.0):
                    return {"is_stable": False, "P": None}
            else:
                # Spectral radius is safely less than 1, system is stable
                # We can return here since we already tried Lyapunov solver
                pass
        except:
            # Fall back to eigenvalue computation if power iteration fails
            eigenvalues = np.linalg.eigvals(A)
            if np.any(np.abs(eigenvalues) >= 1.0):
                return {"is_stable": False, "P": None}
        
        # Fall back to SDP if Lyapunov solver failed but system appears stable
        P_var = cp.Variable((n, n), symmetric=True)
        constraints = [P_var >> np.eye(n), A.T @ P_var @ A - P_var << -np.eye(n)]
        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return {"is_stable": True, "P": P_var.value.tolist()}
            else:
                return {"is_stable": False, "P": None}
        except Exception:
            return {"is_stable": False, "P": None}