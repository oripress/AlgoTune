from typing import Any, Dict
import numpy as np
from scipy.optimize import minimize

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Solve the Markowitz portfolio optimization:
            maximize μ^T w - γ * w^T Σ w
            subject to sum(w)=1, w>=0
        Uses SLSQP from scipy.optimize for a convex QP.
        """
        μ = np.array(problem["μ"], dtype=float)
        Σ = np.array(problem["Σ"], dtype=float)
        γ = float(problem["γ"])
        n = μ.size

        # Objective: minimize f(w) = γ * w^T Σ w - μ^T w
        def obj(w: np.ndarray) -> float:
            return γ * w.dot(Σ.dot(w)) - μ.dot(w)

        # Gradient: ∇f(w) = 2γ Σ w - μ
        def jac(w: np.ndarray) -> np.ndarray:
            return 2.0 * γ * Σ.dot(w) - μ

        # Bounds and equality constraint
        bounds = [(0.0, None)] * n
        cons = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0,
            'jac': lambda w: np.ones(n, dtype=float)
        }

        # Initial guess: uniform weights
        w0 = np.ones(n, dtype=float) / n

        try:
            res = minimize(
                obj, w0, jac=jac,
                bounds=bounds, constraints=cons,
                method='SLSQP',
                options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
            )
            w = res.x
        except Exception:
            # fallback to uniform if solver fails
            w = w0

        # Enforce feasibility and numerical stability
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s > 0.0:
            w /= s
        else:
            w = np.ones(n, dtype=float) / n

        return {'w': w.tolist()}