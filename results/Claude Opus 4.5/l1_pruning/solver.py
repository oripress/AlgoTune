import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        v = np.array(problem["v"], dtype=np.float64).flatten()
        k = float(problem["k"])
        
        n = len(v)
        if n == 0:
            return {"solution": []}
        
        u = np.abs(v)
        
        # Early exit if L1 norm is within constraint
        if np.sum(u) <= k:
            return {"solution": v.tolist()}
        
        # Sort in descending order
        mu = np.sort(u)[::-1]
        
        # Compute cumulative sum and thresholds (vectorized)
        cumsum = np.cumsum(mu)
        indices = np.arange(1.0, n + 1.0)
        thresholds = (cumsum - k) / indices
        
        # Find first index where mu[j] < threshold[j]
        mask = mu < thresholds
        
        if np.any(mask):
            j = np.argmax(mask)
            theta = thresholds[j]
        else:
            theta = 0.0
        
        # Compute result
        w = np.maximum(u - theta, 0.0) * np.sign(v)
        
        return {"solution": w.tolist()}