from typing import Any
import numpy as np
import scipy.sparse as sp
try:
    import osqp
except ImportError:
    osqp = None

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]] | None:
        μ = np.asarray(problem["μ"], dtype=float)
        Σ = np.asarray(problem["Σ"], dtype=float)
        γ = float(problem["γ"])
        n = μ.size

        # Convert to OSQP format: minimize (1/2)x^T P x + q^T x
        # Our problem: maximize μ^T w - γ * w^T Σ w
        # -> minimize -μ^T w + γ * w^T Σ w
        # -> minimize (1/2)w^T (2γΣ) w + (-μ)^T w
        P = sp.csc_matrix(2 * γ * Σ)
        q = -μ
        
        # Constraints: sum(w) = 1, w >= 0
        # A w = l <= A w <= u
        # [1 1 ... 1] w = 1
        # I w >= 0
        A = sp.vstack([
            sp.csc_matrix(np.ones((1, n))),  # sum constraint
            sp.eye(n)  # lower bound constraints
        ])
        l = np.hstack([1.0, np.zeros(n)])  # lower bounds
        u = np.hstack([1.0, np.inf * np.ones(n)])  # upper bounds
        
        # Setup OSQP
        if osqp is None:
            # Fallback to cvxpy if osqp not available
            import cvxpy as cp
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
        
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
        res = prob.solve()
        
        if res.x is None or not np.isfinite(res.x).all():
            return None
        
        return {"w": res.x.tolist()}