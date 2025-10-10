import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]] | None | str]:
        a = np.array(problem["source_weights"], dtype=np.float64)
        b = np.array(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])
        
        try:
            # Sinkhorn algorithm
            n, m = M.shape
            
            # Pre-compute kernel
            K = np.exp(-M / reg)
            
            # Initialize dual variables
            u = np.ones(n, dtype=np.float64)
            v = np.ones(m, dtype=np.float64)
            
            # Sinkhorn iterations
            max_iter = 1000
            tol = 1e-9
            
            for _ in range(max_iter):
                u_prev = u.copy()
                
                # Update u and v
                v = b / (K.T @ u)
                u = a / (K @ v)
                
                # Check convergence
                if np.max(np.abs(u - u_prev)) < tol:
                    break
            
            # Compute transport plan
            G = u[:, None] * K * v[None, :]
            
            if not np.isfinite(G).all():
                raise ValueError("Non-finite values in transport plan")
            
            return {"transport_plan": G, "error_message": None}
        
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}