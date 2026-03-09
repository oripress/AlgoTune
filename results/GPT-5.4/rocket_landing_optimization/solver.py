from typing import Any

import cvxpy as cp
import numpy as np

class Solver:
    def __init__(self) -> None:
        self._solver = cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None

    def solve(self, problem, **kwargs) -> Any:
        p0 = np.array(problem["p0"], dtype=float)
        v0 = np.array(problem["v0"], dtype=float)
        p_target = np.array(problem["p_target"], dtype=float)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        V = cp.Variable((K + 1, 3))
        P = cp.Variable((K + 1, 3))
        F = cp.Variable((K, 3))

        constraints = [
            V[0] == v0,
            P[0] == p0,
            V[K] == np.zeros(3),
            P[K] == p_target,
            P[:, 2] >= 0,
            V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m),
            V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g),
            P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]),
            cp.norm(F, 2, axis=1) <= F_max,
        ]

        fuel_consumption = gamma * cp.sum(cp.norm(F, axis=1))
        prob = cp.Problem(cp.Minimize(fuel_consumption), constraints)

        try:
            if self._solver is None:
                prob.solve()
            else:
                prob.solve(solver=self._solver, warm_start=True, verbose=False)
        except Exception:
            return {"position": [], "velocity": [], "thrust": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {"position": [], "velocity": [], "thrust": []}

        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }