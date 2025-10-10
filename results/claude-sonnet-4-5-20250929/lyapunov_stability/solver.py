import numpy as np
from scipy.linalg import solve_discrete_lyapunov

class Solver:
    def solve(self, problem, **kwargs):
        """
        Optimized Lyapunov stability analysis using direct solver.
        """
        A = np.array(problem["A"])
        n = A.shape[0]
        
        # Quick stability check: all eigenvalues must have |λ| < 1
        eigenvalues = np.linalg.eigvals(A)
        max_eig_mag = np.max(np.abs(eigenvalues))
        
        if max_eig_mag >= 1.0:
            return {"is_stable": False, "P": None}
        
        # Solve discrete Lyapunov equation: A^T P A - P = -I
        try:
            P = solve_discrete_lyapunov(A.T, np.eye(n))
            
            return {"is_stable": True, "P": P}
        except:
            return {"is_stable": False, "P": None}