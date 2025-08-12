from typing import Any, Dict, List
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the rocket landing optimization problem using CVXPY.

        Parameters
        ----------
        problem : dict
            Dictionary containing problem parameters:
            - p0: initial position (list of 3 floats)
            - v0: initial velocity (list of 3 floats)
            - p_target: target landing position (list of 3 floats)
            - g: gravitational acceleration (float)
            - m: mass of the rocket (float)
            - h: time step (float)
            - K: number of time steps (int)
            - F_max: maximum thrust magnitude (float)
            - gamma: fuel consumption coefficient (float)

        Returns
        -------
        dict
            Dictionary with keys:
            - "position": list of K+1 positions (each a list of 3 floats)
            - "velocity": list of K+1 velocities (each a list of 3 floats)
            - "thrust": list of K thrust vectors (each a list of 3 floats)
            - "fuel_consumption": total fuel consumption (float)
        """
        # Extract parameters
        p0 = np.asarray(problem["p0"], dtype=float)
        v0 = np.asarray(problem["v0"], dtype=float)
        p_target = np.asarray(problem["p_target"], dtype=float)

        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # Decision variables
        V = cp.Variable((K + 1, 3))  # velocity
        P = cp.Variable((K + 1, 3))  # position
        F = cp.Variable((K, 3))      # thrust

        constraints = []

        # Initial conditions
        constraints.append(V[0] == v0)
        constraints.append(P[0] == p0)

        # Terminal conditions
        constraints.append(V[K] == np.zeros(3))
        constraints.append(P[K] == p_target)

        # Height constraint
        constraints.append(P[:, 2] >= 0)

        # Dynamics: velocity
        constraints.append(V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m))
        constraints.append(V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g))

        # Dynamics: position
        constraints.append(P[1:] == P[:-1] + (h / 2) * (V[:-1] + V[1:]))

        # Thrust magnitude limit
        constraints.append(cp.norm(F, axis=1) <= F_max)

        # Objective: minimize total fuel consumption
        fuel = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel)

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, warm_start=True,
                       max_iters=2000, abstol=1e-6, reltol=1e-6, feastol=1e-6, verbose=False)
        except Exception:
            prob.solve()  # fallback to default solver

        # Verify solution status
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {
                "position": [],
                "velocity": [],
                "thrust": [],
                "fuel_consumption": float("inf"),
            }

        # Return solution
        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }