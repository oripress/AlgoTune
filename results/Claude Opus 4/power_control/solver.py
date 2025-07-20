from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve the power control optimization problem using direct LP solver."""
        G = np.asarray(problem["G"], float)
        σ = np.asarray(problem["σ"], float)
        P_min = np.asarray(problem["P_min"], float)
        P_max = np.asarray(problem["P_max"], float)
        S_min = float(problem["S_min"])
        n = G.shape[0]
        
        # Objective: minimize sum(P) = minimize c^T * P where c = [1, 1, ..., 1]
        c = np.ones(n)
        
        # Inequality constraints: A_ub @ P <= b_ub
        # We need:
        # 1. -P <= -P_min  (i.e., P >= P_min)
        # 2. P <= P_max
        # 3. -A @ P <= -S_min * σ  (i.e., A @ P >= S_min * σ)
        
        # Build constraint matrix A for SINR constraints
        diag_G = np.diag(G)
        A_sinr = np.diag(diag_G * (1 + S_min)) - S_min * G
        
        # Stack all inequality constraints
        A_ub = np.vstack([
            -np.eye(n),      # -P <= -P_min
            np.eye(n),       # P <= P_max
            -A_sinr          # -A @ P <= -S_min * σ
        ])
        
        b_ub = np.hstack([
            -P_min,          # -P_min
            P_max,           # P_max
            -S_min * σ       # -S_min * σ
        ])
        
        # Solve using highs method (fastest LP solver in scipy)
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs-ds')
        
        if not result.success:
            raise ValueError(f"Solver failed: {result.message}")
        
        return {"P": result.x.tolist()}