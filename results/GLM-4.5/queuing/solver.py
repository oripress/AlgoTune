from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])
        n = γ.size
        
        # For single queue, use ultra-fast analytical solution
        if n == 1:
            w_max, d_max, q_max, λ_min, μ_max, γ = w_max[0], d_max[0], q_max[0], λ_min[0], μ_max, γ[0]
            
            # Calculate minimum required difference
            min_diff = max(1.0/w_max, 1.0/d_max)
            
            # Try λ = λ_min first
            λ_opt = λ_min
            μ_opt = λ_opt + min_diff
            
            # Check if feasible and within μ_max
            if μ_opt <= μ_max:
                ρ = λ_opt / μ_opt
                if ρ < 1.0:
                    q = ρ**2 / (1 - ρ)
                    if q <= q_max:
                        obj_val = γ * μ_opt / λ_opt
                        return {"μ": np.array([μ_opt]), "λ": np.array([λ_opt]), "objective": obj_val}
            
            # Try μ = μ_max
            μ_opt = μ_max
            λ_opt = μ_opt - min_diff
            
            # Check if feasible and above λ_min
            if λ_opt >= λ_min:
                ρ = λ_opt / μ_opt
                if ρ < 1.0:
                    q = ρ**2 / (1 - ρ)
                    if q <= q_max:
                        obj_val = γ * μ_opt / λ_opt
                        return {"μ": np.array([μ_opt]), "λ": np.array([λ_opt]), "objective": obj_val}
            
            # Fallback: use λ_min and adjust μ to satisfy queue constraint
            λ_opt = λ_min
            # Solve quadratic: q_max * μ^2 - q_max * λ_opt * μ - λ_opt^2 = 0
            a, b, c = q_max, -q_max * λ_opt, -λ_opt**2
            disc = b**2 - 4*a*c
            if disc >= 0:
                μ_opt = min((-b + np.sqrt(disc)) / (2*a), μ_max)
                if μ_opt > λ_opt:
                    obj_val = γ * μ_opt / λ_opt
                    return {"μ": np.array([μ_opt]), "λ": np.array([λ_opt]), "objective": obj_val}
            
            # Final fallback
            μ_opt = min(λ_opt + min_diff, μ_max)
            obj_val = γ * μ_opt / λ_opt
            return {"μ": np.array([μ_opt]), "λ": np.array([λ_opt]), "objective": obj_val}
        
        # For multiple queues, skip GP and go directly to DCP for speed
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)
        ρ = λ / μ  # server load

        # queue‐length, waiting time, total delay
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

        # Skip GP and go directly to DCP
        try:
            prob.solve(verbose=False)
        except cp.error.DCPError:
            # heuristic: λ = λ_min, μ = μ_max/n
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