import numpy as np
import cvxpy as cp
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Extract problem parameters
        p0 = np.array(problem["p0"], dtype=float)
        v0 = np.array(problem["v0"], dtype=float)
        p_target = np.array(problem["p_target"], dtype=float)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # Decision variables
        V = cp.Variable((K + 1, 3))
        P = cp.Variable((K + 1, 3))
        F = cp.Variable((K, 3))

        # Constraints
        constraints = []
        # Initial conditions
        constraints.append(V[0] == v0)
        constraints.append(P[0] == p0)
        # Terminal conditions
        constraints.append(V[K] == np.zeros(3))
        constraints.append(P[K] == p_target)
        # Height must be non-negative
        constraints.append(P[:, 2] >= 0)
        # Velocity dynamics
        constraints.append(V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m))
        constraints.append(V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g))
        # Position dynamics (trapezoidal integration)
        constraints.append(P[1:] == P[:-1] + (h / 2) * (V[:-1] + V[1:]))
        # Thrust limits
        constraints.append(cp.norm(F, 2, axis=1) <= F_max)

        # Objective: minimize fuel consumption
        fuel_consumption = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_consumption)

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Return solution
        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }