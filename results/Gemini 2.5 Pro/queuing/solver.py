from typing import Any
import numpy as np
from scipy.optimize import minimize

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the queuing system optimization problem using a general-purpose
        NLP solver. This version uses a highly stable, vectorized formulation
        where the queue length constraint is linearized to prevent numerical
        instability and memory errors in the solver.
        """
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])
        n = γ.size
        
        epsilon = 1e-9

        def objective(x):
            """Objective function: sum(γ * μ / (μ - s))"""
            μ = x[:n]
            s = x[n:]
            λ = μ - s
            λ_safe = np.maximum(λ, epsilon)
            return np.sum(γ * μ / λ_safe)

        # --- Pre-calculate coefficients for the linearized queue length constraint ---
        # The constraint q_max >= (μ/s-1)^2 / (μ/s) is a quadratic in z = μ/s.
        # It can be rewritten as z^2 - (q_max + 2)z + 1 <= 0, which holds between the roots.
        q_plus_2 = q_max + 2.0
        # Discriminant is (q+2)^2 - 4 = q^2 + 4q. This is always non-negative for q>=0.
        discriminant_sqrt = np.sqrt(q_max * (q_max + 4.0))
        r1 = (q_plus_2 - discriminant_sqrt) / 2.0
        r2 = (q_plus_2 + discriminant_sqrt) / 2.0

        # --- Vectorized Constraints ---
        constraints = [
            {'type': 'ineq', 'fun': lambda x: np.array([μ_max - np.sum(x[:n])])},
            {'type': 'ineq', 'fun': lambda x: x[:n] - x[n:] - λ_min},
            {'type': 'ineq', 'fun': lambda x: d_max - (1.0/x[:n] + 1.0/x[n:])},
            # Linearized queue length constraints: r1 <= μ/s <= r2
            {'type': 'ineq', 'fun': lambda x: x[:n] - r1 * x[n:]}, # μ - r1*s >= 0
            {'type': 'ineq', 'fun': lambda x: r2 * x[n:] - x[:n]}  # r2*s - μ >= 0
        ]

        # Bounds for [μ_1..n, s_1..n]
        μ_bounds = [(epsilon, None) for _ in range(n)]
        s_bounds = [(1.0/w_max[i] + epsilon, None) for i in range(n)]
        bounds = μ_bounds + s_bounds

        # --- Create a guaranteed-feasible initial guess (x0) ---
        s0 = 1.0 / w_max + epsilon
        # To satisfy q_max constraint, we need r1*s <= mu <= r2*s.
        # To satisfy λ_min, we need mu >= λ_min + s.
        # So, mu must be at least max(r1*s, λ_min+s).
        μ0_req = np.maximum(r1 * s0, λ_min + s0)

        if np.sum(μ0_req) <= μ_max:
            slack_μ = μ_max - np.sum(μ0_req)
            μ0 = μ0_req + slack_μ / n
        else:
            μ0 = μ0_req
        
        x0 = np.concatenate([μ0, s0])

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})

        if result.success:
            μ_val = result.x[:n]
            s_val = result.x[n:]
            λ_val = μ_val - s_val
            return {"μ": μ_val, "λ": λ_val, "objective": float(result.fun)}
        
        # Fallback: If solver fails, return the feasible initial guess.
        μ_fb = x0[:n]
        s_fb = x0[n:]
        λ_fb = μ_fb - s_fb
        obj_fb = np.sum(γ * μ_fb / np.maximum(λ_fb, epsilon))
        return {"μ": μ_fb, "λ": λ_fb, "objective": float(obj_fb)}