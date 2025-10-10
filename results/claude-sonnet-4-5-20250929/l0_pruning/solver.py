import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the L0 proximal operator problem efficiently.
        
        Find w that minimizes ||v-w||² subject to ||w||₀ ≤ k
        Solution: keep k elements with largest absolute values, zero out the rest.
        """
        v = np.array(problem["v"])
        k = problem["k"]
        
        # Ensure v is 1D
        v = v.flatten()
        
        # Use argpartition to find k largest absolute values (O(n) average)
        abs_v = np.abs(v)
        indx = np.argpartition(abs_v, -k)[-k:]
        
        # Create solution with only k largest elements
        pruned = np.zeros_like(v)
        pruned[indx] = v[indx]
        
        return {"solution": pruned.tolist()}