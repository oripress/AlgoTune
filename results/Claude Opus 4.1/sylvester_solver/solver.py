from typing import Any
import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve the Sylvester equation AX + XB = Q with optimized memory layout."""
        # Ensure C-contiguous arrays for better cache performance
        A = np.ascontiguousarray(problem["A"])
        B = np.ascontiguousarray(problem["B"])
        Q = np.ascontiguousarray(problem["Q"])
        
        # Use scipy's optimized solver
        X = solve_sylvester(A, B, Q)
        
        return {"X": X}