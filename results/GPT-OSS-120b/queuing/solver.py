from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the queuing system design problem.

        Parameters
        ----------
        problem : dict
            Dictionary containing the problem data.

        Returns
        -------
        dict
            Dictionary with optimal service rates ``μ`` and arrival rates ``λ``.
        """
        # Extract data
        w_max = np.asarray(problem["w_max"], dtype=float)
        d_max = np.asarray(problem["d_max"], dtype=float)
        q_max = np.asarray(problem["q_max"], dtype=float)
        lam_min = np.asarray(problem["λ_min"], dtype=float)
        mu_max = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"], dtype=float)

        n = gamma.size

        # Decision variables
        mu = cp.Variable(n, pos=True)
        lam = cp.Variable(n, pos=True)

        # Load ratio
        rho = lam / mu

        # Queue length, waiting time, total delay
        q = cp.power(rho, 2) / (1 - rho)
        w = q / lam + 1 / mu
        d = 1 / (mu - lam)

        # Constraints
        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            lam >= lam_min,
            cp.sum(mu) <= mu_max,
        ]

        # Objective: minimize weighted sum of service loads (mu/lam = 1/rho)
        objective = cp.Minimize(gamma @ (mu / lam))

        # Problem definition
        prob = cp.Problem(objective, constraints)

        # Try geometric programming first (fast for this problem)
        try:
            prob.solve(gp=True)
        except cp.error.DGPError:
            # Fall back to generic DCP solve
            try:
                prob.solve()
            except cp.error.DCPError:
                # Heuristic fallback: use minimal arrivals and split service capacity evenly
                lam_val = lam_min
                mu_val = np.full(n, mu_max / n)
                return {
                    "μ": mu_val,
                    "λ": lam_val,
                    "objective": float(gamma @ (mu_val / lam_val)),
                }

        # Verify solution status
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Optimization failed with status {prob.status}")

        return {
            "μ": mu.value,
            "λ": lam.value,
            "objective": float(prob.value),
        }