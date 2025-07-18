from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def __init__(self):
        # Pre-import to save time during solve
        pass
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]] | None:
        """Solve the Markowitz portfolio optimization problem."""
        μ = np.asarray(problem["μ"], dtype=float)
        Σ = np.asarray(problem["Σ"], dtype=float)
        γ = float(problem["γ"])
        n = μ.size
        
        # Create optimization problem
        w = cp.Variable(n)
        obj = cp.Maximize(μ @ w - γ * cp.quad_form(w, cp.psd_wrap(Σ)))
        cons = [cp.sum(w) == 1, w >= 0]
        
        try:
            cp.Problem(obj, cons).solve()
        except cp.error.SolverError:
            return None
        
        if w.value is None or not np.isfinite(w.value).all():
            return None
        
        return {"w": w.value.tolist()}