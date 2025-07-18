import numpy as np
import ot

class Solver:
    def solve(self, problem, **kwargs):
        try:
            a = np.array(problem["source_weights"], dtype=np.float64)
            b = np.array(problem["target_weights"], dtype=np.float64)
            M = np.array(problem["cost_matrix"], dtype=np.float64)
            reg = float(problem["reg"])
            
            # Precompute kernel matrix
            K = np.exp(-M / reg)
            
            # Use logarithmic stabilization for numerical stability
            u = np.ones_like(a)
            v = np.ones_like(b)
            for i in range(1000):  # Max iterations
                v = b / (K.T @ u)
                u = a / (K @ v)
                
                # Check convergence
                if np.max(np.abs(u * (K @ v) - a)) < 1e-8:
                    break
            
            # Compute transport plan
            P = u[:, None] * K * v[None, :]
            
            return {"transport_plan": P}
        
        except Exception as e:
            return {"transport_plan": None, "error_message": str(e)}