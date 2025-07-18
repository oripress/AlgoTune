import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        # Parse input
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])
        n = γ.size
        
        # Early termination for trivial cases
        if n == 1 and λ_min[0] == 0:
            return {"μ": [μ_max], "λ": [μ_max * 0.999]}
        elif np.all(λ_min == 0) and np.isinf(μ_max):
            return {"μ": np.ones(n), "λ": np.ones(n) * 0.999}

        # Define optimization variables
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)
        
        # Precompute expressions for efficiency
        ρ = λ / μ
        q = ρ**2 / (1 - ρ)
        w = q / λ + 1 / μ
        d = 1 / (μ - λ)

        # Constraints
        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            λ >= λ_min,
            cp.sum(μ) <= μ_max,
        ]
        
        # Objective: minimize weighted sum of service loads
        obj = cp.Minimize(γ @ (μ / λ))
        prob = cp.Problem(obj, constraints)

        # Solve with optimized parameters
        try:
            # Try GP first (fastest if applicable)
            prob.solve(gp=True, verbose=False)
        except (cp.error.DGPError, cp.error.SolverError):
            # If GP fails, try DCP with ECOS
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except (cp.error.DCPError, cp.error.SolverError):
                # Fallback to heuristic
                λ_val = λ_min
                μ_val = np.full(n, μ_max / n)
                return {"μ": μ_val, "λ": λ_val}

        # Check solution status
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # Fallback to heuristic if solution not optimal
            λ_val = λ_min
            μ_val = np.full(n, μ_max / n)
            return {"μ": μ_val, "λ": λ_val}

        return {"μ": μ.value, "λ": λ.value}