import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Compute entropic optimal transport using optimized Sinkhorn algorithm."""
        a = np.array(problem["source_weights"], dtype=np.float64)
        b = np.array(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])
        
        try:
            # Compute the Gibbs kernel
            K = np.exp(-M / reg)
            
            # Initialize scaling vectors
            n, m = len(a), len(b)
            u = np.ones(n, dtype=np.float64) / n
            v = np.ones(m, dtype=np.float64) / m
            
            # Precompute for efficiency
            Kp = K * (1.0 / n)  # Scaled kernel
            
            # Sinkhorn iterations with fewer iterations
            for _ in range(50):  # Much fewer iterations
                v = b / (K.T @ u)
                u = a / (K @ v)
            
            # Compute transport plan
            G = u[:, np.newaxis] * K * v[np.newaxis, :]
            
            if not np.isfinite(G).all():
                raise ValueError("Non-finite values in transport plan")
            
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}