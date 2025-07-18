import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the L1 proximal operator problem:
        min_w ‖v-w‖² s.t. ‖w‖₁ ≤ k
        
        Using an optimized version of the algorithm from https://doi.org/10.1109/CVPR.2018.00890
        """
        v = np.array(problem["v"], dtype=np.float64)
        k = problem["k"]
        
        # Work with absolute values
        u = np.abs(v)
        
        # If sum of absolute values is <= k, no thresholding needed
        if u.sum() <= k:
            return {"solution": v.tolist()}
        
        # Sort in descending order using the fastest sorting algorithm
        mu = np.sort(u, kind='heapsort')[::-1]
        
        # Vectorized computation to find threshold
        n = len(mu)
        cumsum = np.cumsum(mu)
        j_range = np.arange(1, n + 1)
        thresholds = (cumsum - k) / j_range
        
        # Find the first index where mu[j] < threshold[j]
        mask = mu < thresholds
        if mask.any():
            j = np.argmax(mask)
            theta = thresholds[j]
        else:
            theta = 0.0
        
        # Apply soft thresholding vectorized
        w = np.where(u > theta, (u - theta) * np.sign(v), 0.0)
        
        return {"solution": w.tolist()}