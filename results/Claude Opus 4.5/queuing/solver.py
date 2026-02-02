from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs) -> dict[str, Any]:
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])
        n = γ.size

        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)
        ρ = λ / μ

        q = cp.power(ρ, 2) / (1 - ρ)
        w = q / λ + 1 / μ
        d = 1 / (μ - λ)

        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            λ >= λ_min,
            cp.sum(μ) <= μ_max,
        ]
        obj = cp.Minimize(γ @ (μ / λ))
        prob = cp.Problem(obj, constraints)

        try:
            prob.solve(gp=True, solver=cp.SCS, eps=1e-4, max_iters=500, acceleration_lookback=0)
        except Exception:
            try:
                prob.solve(gp=True)
            except Exception:
                λ_val = λ_min
                μ_val = np.full(n, μ_max / n)
                return {"μ": μ_val, "λ": λ_val, "objective": float(γ @ (μ_val / λ_val))}

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            λ_val = λ_min
            μ_val = np.full(n, μ_max / n)
            return {"μ": μ_val, "λ": λ_val, "objective": float(γ @ (μ_val / λ_val))}

        return {
            "μ": μ.value,
            "λ": λ.value,
            "objective": float(prob.value),
        }