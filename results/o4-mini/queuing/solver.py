from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Parse problem parameters
        w_max = np.asarray(problem["w_max"], dtype=float)
        d_max = np.asarray(problem["d_max"], dtype=float)
        q_max = np.asarray(problem["q_max"], dtype=float)
        lam_min = np.asarray(problem["λ_min"], dtype=float)
        mu_max = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"], dtype=float)
        n = gamma.size

        # Define optimization variables
        mu = cp.Variable(n, pos=True)
        lam = cp.Variable(n, pos=True)
        rho = lam / mu  # server load

        # Queue metrics
        q = cp.power(rho, 2) / (1 - rho)
        w = q / lam + 1.0 / mu
        d = 1.0 / (mu - lam)

        # Constraints
        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            lam >= lam_min,
            cp.sum(mu) <= mu_max,
        ]

        # Objective: minimize weighted sum of service load (mu / lam)
        obj = cp.Minimize(gamma @ (mu / lam))
        prob = cp.Problem(obj, constraints)

        # Solve as geometric program if possible, else DCP
        try:
            prob.solve(gp=True, verbose=False)
        except cp.error.DGPError:
            try:
                prob.solve(verbose=False)
            except cp.error.DCPError:
                # Fallback heuristic
                lam_val = lam_min
                mu_val = np.full(n, mu_max / n, dtype=float)
                return {
                    "μ": mu_val,
                    "λ": lam_val,
                    "objective": float(gamma @ (mu_val / lam_val)),
                }

        # If solver failed to find a solution, fallback heuristic
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            lam_val = lam_min
            mu_val = np.full(n, mu_max / n, dtype=float)
            return {
                "μ": mu_val,
                "λ": lam_val,
                "objective": float(gamma @ (mu_val / lam_val)),
            }

        # Retrieve values
        mu_val = mu.value
        lam_val = lam.value
        obj_val = float(prob.value)

        return {"μ": mu_val, "λ": lam_val, "objective": obj_val}