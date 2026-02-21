import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list]:
        v = problem["v"]
        k = problem["k"]
        
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float64)
        elif v.ndim > 1:
            v = v.flatten()
            
        n = v.size
        if k <= 0:
            return {"solution": [0.0] * n}
        if k >= n:
            return {"solution": v.tolist()}
            
        # Use argpartition on the absolute values
        idx = np.argpartition(np.abs(v), n - k)[n - k:]
        
        pruned = np.zeros(n, dtype=v.dtype)
        pruned[idx] = v[idx]
        
        return {"solution": pruned.tolist()}