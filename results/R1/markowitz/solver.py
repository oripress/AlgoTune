import cvxpy as cp
import numpy as np

class Solver:
    def __init__(self):
        # Initialize variables for warm-starting
        self.last_w = None
        self.last_n = 0
        
    def solve(self, problem, **kwargs):
        # Validate and extract input
        try:
            μ = np.asarray(problem["μ"], dtype=float)
            Σ = np.asarray(problem["Σ"], dtype=float)
            γ = float(problem["γ"])
            n = μ.size
            
            # Validate inputs
            if Σ.shape != (n, n) or γ <= 0:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        
        # Handle trivial case: all non-positive returns
        if np.all(μ <= 0):
            idx = np.argmax(μ)
            w = np.zeros(n)
            w[idx] = 1.0
            return {"w": w.tolist()}

        # Setup optimization problem
        w = cp.Variable(n)
        
        # Use warm-start if dimensions match
        if self.last_w is not None and self.last_n == n:
            w.value = self.last_w
        
        # Formulate objective: maximize risk-adjusted return
        objective = cp.Maximize(μ @ w - γ * cp.quad_form(w, cp.psd_wrap(Σ)))
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        
        # Try HiGHS solver first
        try:
            prob.solve(solver='HIGHS', warm_start=True)
        except cp.error.SolverError:
            return None
        
        # Validate solution
        if w.value is None or not np.isfinite(w.value).all():
            return None
        
        # Store solution for warm-starting next call
        self.last_w = w.value.copy()
        self.last_n = n
        
        return {"w": w.value.tolist()}