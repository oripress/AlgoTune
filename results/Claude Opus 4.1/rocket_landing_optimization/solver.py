import cvxpy as cp
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the rocket landing optimization problem using CVXPY with optimized ECOS.
        
        :param problem: Dictionary with problem parameters
        :return: Dictionary with position, velocity, and thrust trajectories
        """
        # Extract problem parameters with type safety
        p0 = np.array(problem["p0"], dtype=np.float64)
        v0 = np.array(problem["v0"], dtype=np.float64)
        p_target = np.array(problem["p_target"], dtype=np.float64)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # Variables
        V = cp.Variable((K + 1, 3))  # Velocity
        P = cp.Variable((K + 1, 3))  # Position
        F = cp.Variable((K, 3))  # Thrust

        # Constraints list
        constraints = []

        # Initial conditions
        constraints.append(V[0] == v0)
        constraints.append(P[0] == p0)

        # Terminal conditions
        constraints.append(V[K] == np.zeros(3))  # Zero final velocity
        constraints.append(P[K] == p_target)  # Target position

        # Height constraint (always positive)
        constraints.append(P[:, 2] >= 0)

        # Dynamics for velocity (split for numerical stability)
        constraints.append(V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m))
        constraints.append(V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g))

        # Dynamics for position
        constraints.append(P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]))

        # Maximum thrust constraint
        constraints.append(cp.norm(F, 2, axis=1) <= F_max)

        # Objective: minimize fuel consumption
        fuel_consumption = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_consumption)

        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        
        # Optimized ECOS parameters for speed
        ecos_params = {
            'verbose': False,
            'abstol': 1e-6,  # Slightly relaxed for speed
            'reltol': 1e-5,  # Slightly relaxed for speed
            'feastol': 1e-6,  # Slightly relaxed for speed
            'abstol_inacc': 5e-5,
            'reltol_inacc': 5e-5,
            'feastol_inacc': 1e-4,
            'max_iters': 100,  # Reduced for speed
        }
        
        # Try solving with optimized ECOS
        try:
            prob.solve(solver=cp.ECOS, **ecos_params)
        except Exception:
            # If still fails, return empty result quickly
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}

        # Check if solution is valid
        if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and P.value is not None and V.value is not None and F.value is not None:
            # Return solution
            return {
                "position": P.value.tolist(),
                "velocity": V.value.tolist(),
                "thrust": F.value.tolist(),
                "fuel_consumption": float(prob.value),
            }
        
        # If solving failed, return empty lists
        return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}