from typing import Any, Dict, List
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """Solve Markowitz portfolio optimization."""
        # Extract problem data
        μ = np.asarray(problem["μ"], dtype=float)
        Σ = np.asarray(problem["Σ"], dtype=float)
        γ = float(problem["γ"])
        n = μ.size

        # Initialize warm start if needed
        if not hasattr(self, '_last_w'):
            self._last_w = None
            
        # Use CVXPy for a reliable solution
        try:
            w = cp.Variable(n)
            objective = cp.Maximize(μ @ w - γ * cp.quad_form(w, cp.psd_wrap(Σ)))
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(objective, constraints)
            
            # Try with warm start for subsequent solves
            if self._last_w is not None and len(self._last_w) == n:
                w.value = self._last_w
            
            prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=1000, 
                      tol_gap_abs=1e-6, tol_gap_rel=1e-6, tol_feas=1e-6, 
                      tol_infeas_abs=1e-8, tol_infeas_rel=1e-8)
            
            if w.value is not None and np.isfinite(w.value).all():
                # Store solution for potential warm start
                self._last_w = w.value
                return {"w": w.value.tolist()}
        except Exception:
            pass
            
        # Last resort: equal weights
        return {"w": (np.ones(n) / n).tolist()}