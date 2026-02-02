import numpy as np
from typing import Any
import fast_solver

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.array(problem.get("v"), dtype=np.float64)
        k = float(problem.get("k"))
        
        # Ensure v is a 1D array
        if v.ndim > 1:
            v = v.flatten()
            
        w = fast_solver.solve_cython(v, k)
        
        return {"solution": w.tolist()}