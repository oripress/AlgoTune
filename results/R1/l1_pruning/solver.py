import numpy as np
import numba

class Solver:
    def solve(self, problem, **kwargs):
        v = np.array(problem["v"], dtype=np.float64)
        k = problem["k"]
        n = len(v)
        
        # Handle edge cases immediately
        if k <= 0 or n == 0:
            return {"solution": [0.0] * n}
            
        u = np.abs(v)
        total = u.sum()
        if total <= k:
            return {"solution": v.tolist()}
        
        # Optimized projection computation
        w = self._compute_projection(v, u, k)
        return {"solution": w.tolist()}
    
    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def _compute_projection(v, u, k):
        n = len(u)
        lo = 0.0
        hi = np.max(u)
        theta = 0.0
        
        # Binary search with tolerance-based termination
        max_iter = 100
        tol = 1e-6
        for _ in range(max_iter):
            theta = (lo + hi) * 0.5
            s = 0.0
            for i in range(n):
                val = u[i]
                # Fused operation: compute and conditionally add
                if val > theta:
                    s += val - theta
                    
            # Check for convergence
            if abs(s - k) < tol:
                break
                
            # Update search bounds
            if s < k:
                hi = theta
            else:
                lo = theta
                
            # Check range convergence
            if hi - lo < tol:
                break
                
        # Optimized soft thresholding
        w = np.empty_like(v)
        for i in range(n):
            val = u[i]
            if val > theta:
                w[i] = np.sign(v[i]) * (val - theta)
            else:
                w[i] = 0.0
        return w