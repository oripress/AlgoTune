import numpy as np
import cvxpy as cp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Optimize a Markovian M/M/1 queuing system."""
        # Extract problem parameters
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])
        n = γ.size

        # Define optimization variables
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)
        ρ = λ / μ  # server load

        # Calculate queue metrics
        q = cp.power(ρ, 2) / (1 - ρ)
        w = q / λ + 1 / μ
        d = 1 / (μ - λ)

        # Define constraints
        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            λ >= λ_min,
            cp.sum(μ) <= μ_max,
        ]
        
        # Define objective
        obj = cp.Minimize(γ @ (μ / λ))
        prob = cp.Problem(obj, constraints)

        # Solve the problem - try GP first for better performance
        try:
            prob.solve(gp=True)
        except cp.error.DGPError:
            try:
                prob.solve()
            except cp.error.DCPError:
                # Fallback heuristic
                λ_val = λ_min
                μ_val = np.full(n, μ_max / n)
                obj_val = float(γ @ (μ_val / λ_val))
                return {"μ": μ_val, "λ": λ_val, "objective": obj_val}

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed with status {prob.status}")

        return {
            "μ": μ.value,
            "λ": λ.value,
            "objective": float(prob.value),
        }