from typing import Any, Dict
import numpy as np

# Always use the compiled Cython extension for water-filling
from waterfill_ext import waterfill as _waterfill_ext

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Extract inputs
        alpha = np.asarray(problem.get("alpha", []), dtype=float)
        P_total = float(problem.get("P_total", 0.0))
        n = alpha.size

        # Handle trivial or invalid cases
        if n == 0 or P_total <= 0.0:
            return {"x": np.full(n, np.nan), "Capacity": np.nan}

        # Delegate to the high-performance Cython implementation
        x, cap = _waterfill_ext(alpha, P_total)
        return {"x": x, "Capacity": cap}