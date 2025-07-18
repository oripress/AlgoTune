import numpy as np
from scipy import linalg
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the Lyapunov stability analysis problem.
        
        For discrete-time systems, stability requires all eigenvalues 
        of A to have magnitude less than 1.
        """
        A = np.array(problem["A"])
        
        # Check stability via eigenvalues
        eigenvalues = np.linalg.eigvals(A)
        max_magnitude = np.max(np.abs(eigenvalues))
        
        if max_magnitude >= 1.0 - 1e-10:
            # System is not stable
            return {"is_stable": False, "P": None}
        
        # System is stable, solve discrete Lyapunov equation
        # A^T P A - P = -Q, where Q = I
        # This gives us P such that A^T P A - P = -I < 0
        try:
            Q = np.eye(A.shape[0])
            P = linalg.solve_discrete_lyapunov(A.T, Q)
            
            # Verify P is positive definite
            eigvals_P = np.linalg.eigvals(P)
            if np.min(eigvals_P) <= 1e-10:
                return {"is_stable": False, "P": None}
            
            return {"is_stable": True, "P": P.tolist()}
            
        except Exception:
            # If Lyapunov equation solving fails, system is not stable
            return {"is_stable": False, "P": None}