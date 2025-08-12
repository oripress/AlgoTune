from typing import Any
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from numba import jit
class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the Lyapunov stability analysis problem using semidefinite programming.
        
        Args:
            problem: A dictionary containing the system matrix A.
            
        Returns:
            A dictionary containing:
                - is_stable: Boolean indicating if the system is asymptotically stable
                - P: The Lyapunov matrix P (if stable)
        """
        # Convert to numpy array with appropriate dtype
        A = np.asarray(problem["A"], dtype=np.float64)
        n = A.shape[0]
        
        # Handle edge cases
        if n == 0:
            return {"is_stable": True, "P": []}
        
        # For 1x1 matrices, we can directly check
        if n == 1:
            a = A[0, 0]
            if abs(a) >= 1:
                return {"is_stable": False, "P": None}
            else:
                # For a stable 1x1 system, P = 1/(1-a^2) works
                p_val = 1.0 / (1.0 - a * a)
                return {"is_stable": True, "P": [[p_val]]}
        
        # For all other matrices, check stability using eigenvalues
        try:
            # Check spectral radius using eigenvalues
            # For better performance, use eigvals for small matrices and eigs for large ones
            if n <= 10:
                eigenvals = np.linalg.eigvals(A)
            else:
                # For larger matrices, use iterative method
                eigenvals = np.linalg.eigvals(A)
            spectral_radius = np.max(np.abs(eigenvals))
            
            if spectral_radius >= 1:
                return {"is_stable": False, "P": None}
            
            # System is stable, solve Lyapunov equation
            Q = np.eye(n)
            P = solve_discrete_lyapunov(A.T, Q)
            return {"is_stable": True, "P": P.tolist()}
        except np.linalg.LinAlgError:
            # Numerical issues with eigenvalue computation
            return {"is_stable": False, "P": None}
        except Exception:
            # Other numerical issues
            return {"is_stable": False, "P": None}
    
    def fixed_point_iteration(self, A, Q, max_iter=100, tol=1e-16):
        """Solve Lyapunov equation using fixed-point iteration."""
        n = A.shape[0]
        P = np.eye(n)
        I = np.eye(n)
        for i in range(max_iter):
            P_new = np.dot(np.dot(A.T, P), A) + I
            # Check convergence using relative error
            if np.allclose(P, P_new, rtol=1e-12, atol=1e-12):
                P = P_new
                break
            P = P_new
        # Ensure symmetry
        P = (P + P.T) / 2
        return P
    
    
    def compute_spectral_radius(self, A, max_iter=20):
        """Compute spectral radius using inverse iteration for the largest eigenvalue."""
        n = A.shape[0]
        
        # For small matrices, use direct method
        if n <= 10:
            # For small matrices, compute directly
            try:
                eigenvals = np.linalg.eigvals(A)
                return np.max(np.abs(eigenvals))
            except:
                # Fallback to power method
                return self._power_method(A, max_iter)
        else:
            # For larger matrices, use power method
            return self._power_method(A, max_iter)
    
    
    @jit(nopython=True)
    def _power_method(self, A, max_iter=20):
        """Compute dominant eigenvalue using power method."""
        n = A.shape[0]
        x = np.random.rand(n)
        x = x / np.linalg.norm(x)
        
        for _ in range(max_iter):
            # Apply matrix
            Ax = np.dot(A, x)
            # Apply transpose
            ATAx = np.dot(A.T, Ax)
            # Normalize
            norm = np.linalg.norm(ATAx)
            if norm == 0:
                return 0
            x = ATAx / norm
            
        # Return the estimated spectral radius
        return np.sqrt(np.linalg.norm(np.dot(A.T, np.dot(A, x))))