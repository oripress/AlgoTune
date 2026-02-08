import numpy as np
from scipy.linalg import solve_discrete_lyapunov

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64)
        n = A.shape[0]
        
        # Fast stability check using spectral radius
        # For small matrices, eigvals is fast enough
        eigenvalues = np.linalg.eigvals(A)
        max_abs = 0.0
        for ev in eigenvalues:
            a = abs(ev)
            if a >= 1.0:
                return {"is_stable": False, "P": None}
        
        # Solve discrete Lyapunov equation: P - A^T P A = I
        P = solve_discrete_lyapunov(A.T, np.eye(n))
        # Ensure symmetry
        np.add(P, P.T, out=P)
        P *= 0.5
        return {"is_stable": True, "P": P}