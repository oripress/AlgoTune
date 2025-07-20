from typing import Any
import numpy as np
from mpmath import mp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Count zeta zeros along critical strip with imaginary part <= t.
        
        Using exact computation with mpmath for accuracy.
        """
        t = problem["t"]
        
        # Always use exact computation
        result = mp.nzeros(t)
        
        return {"result": result}