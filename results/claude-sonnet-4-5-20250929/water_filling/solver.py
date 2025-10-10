import numpy as np
from typing import Any
from solver_cy import water_filling_core

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Analytic water-filling solution with Cython optimization."""
        alpha = np.asarray(problem["alpha"], dtype=np.float64)
        P_total = float(problem["P_total"])
        n = alpha.size
        
        if n == 0 or P_total <= 0:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        x_opt, capacity = water_filling_core(alpha, P_total)
        
        return {"x": x_opt.tolist(), "Capacity": float(capacity)}