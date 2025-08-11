from typing import Any
from mpmath import mp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Count zeta zeros using mpmath with optimized precision."""
        t = problem["t"]
        
        # Dynamically adjust precision based on t
        # Lower precision for larger t means faster computation
        if t <= 100:
            mp.dps = 15
        elif t <= 1000:
            mp.dps = 20
        elif t <= 10000:
            mp.dps = 25
        else:
            mp.dps = 30
        
        # Use mp.nzeros which counts zeros up to height t
        result = mp.nzeros(t)
        
        return {"result": result}