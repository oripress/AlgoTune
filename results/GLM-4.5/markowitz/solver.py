from typing import Any
import numpy as np
try:
    import osqp
    import scipy.sparse as sp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]] | None:
        μ = np.asarray(problem["μ"], dtype=float)
        Σ = np.asarray(problem["Σ"], dtype=float)
        γ = float(problem["γ"])
        n = μ.size
        
        if not OSQP_AVAILABLE or n == 0:
            return None
            
        try:
            # Convert to OSQP format
            # Objective: 0.5 * w^T (2γΣ) w + (-μ)^T w
            P = 2 * γ * Σ
            q = -μ
            
            # Constraints: sum(w) = 1, w >= 0
            A_eq = np.ones((1, n))
            b_eq = np.array([1.0])
            A_ineq = -np.eye(n)  # -w <= 0 => w >= 0
            b_ineq = np.zeros(n)
            
            # Combine constraints
            A = np.vstack([A_eq, A_ineq])
            l = np.hstack([b_eq, -np.inf * np.ones(n)])  # lower bounds
            u = np.hstack([b_eq, b_ineq])  # upper bounds
            
            # Convert to sparse matrices
            P_sparse = sp.csc_matrix(P)
            A_sparse = sp.csc_matrix(A)
            
            # Setup and solve
            model = osqp.OSQP()
            model.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, 
                       verbose=False, max_iter=1000, eps_abs=1e-6, eps_rel=1e-6)
            result = model.solve()
            
            if result.info.status == 'solved':
                return {"w": result.x.tolist()}
        except:
            pass
            
        return None