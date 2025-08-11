from typing import Any
import math

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        t = problem["t"]
        
        # For small t, use high precision
        if t <= 50:
            import mpmath
            mpmath.mp.dps = 13
            result = mpmath.nzeros(t)
            return {"result": int(result)}
        
        # For medium t, use medium precision
        elif t <= 500:
            import mpmath
            mpmath.mp.dps = 8
            result = mpmath.nzeros(t)
            return {"result": int(result)}
        
        # For large t, use low precision
        elif t <= 5000:
            import mpmath
            mpmath.mp.dps = 6
            result = mpmath.nzeros(t)
            return {"result": int(result)}
        
        # For very large t, use minimal precision
        else:
            import mpmath
            mpmath.mp.dps = 4
            result = mpmath.nzeros(t)
            return {"result": int(result)}