from typing import Any, Dict
from mpmath import mp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Count zeta zeros in the critical strip with imaginary part <= t.

        Uses mpmath.mp.nzeros to match the reference implementation exactly.
        """
        t = float(problem["t"])
        result = int(mp.nzeros(t))
        return {"result": result}