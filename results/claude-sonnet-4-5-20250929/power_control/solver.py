from typing import Any
import numpy as np
import highspy

class Solver:
    def __init__(self):
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        G = np.asarray(problem["G"], dtype=np.float64)
        σ = np.asarray(problem["σ"], dtype=np.float64)
        P_min = np.asarray(problem["P_min"], dtype=np.float64)
        P_max = np.asarray(problem["P_max"], dtype=np.float64)
        S_min = float(problem["S_min"])
        n = G.shape[0]
        
        # Clear previous model
        self.h.clear()
        
        # Add all variables with bounds at once
        self.h.addVars(n, P_min, P_max)
        
        # Objective: minimize sum(P)
        indices = np.arange(n, dtype=np.int32)
        ones = np.ones(n, dtype=np.float64)
        self.h.changeColsCost(n, indices, ones)
        
        # Build constraint matrix more efficiently
        A = -S_min * G
        np.fill_diagonal(A, G.diagonal() * (1 + S_min))
        b = S_min * σ
        
        # Add all constraints at once
        nn = n * n
        starts = np.arange(0, nn + 1, n, dtype=np.int32)
        indices_flat = np.tile(indices, n)
        values_flat = A.ravel()
        upper = np.full(n, highspy.kHighsInf, dtype=np.float64)
        self.h.addRows(n, b, upper, nn, starts, indices_flat, values_flat)
        
        # Solve
        self.h.run()
        
        solution = self.h.getSolution()
        return {"P": list(solution.col_value), "objective": float(self.h.getInfo().objective_function_value)}