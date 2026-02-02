from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        p0 = np.array(problem["p0"], dtype=np.float64)
        v0 = np.array(problem["v0"], dtype=np.float64)
        p_target = np.array(problem["p_target"], dtype=np.float64)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        V = cp.Variable((K + 1, 3))
        P = cp.Variable((K + 1, 3))
        F = cp.Variable((K, 3))

        h_m = h / m
        hg = h * g
        h2 = h / 2.0
        
        constraints = [
            V[0] == v0,
            P[0] == p0,
            V[K] == 0,
            P[K] == p_target,
            P[:, 2] >= 0,
            V[1:, :2] == V[:-1, :2] + h_m * F[:, :2],
            V[1:, 2] == V[:-1, 2] + h_m * F[:, 2] - hg,
            P[1:] == P[:-1] + h2 * (V[:-1] + V[1:]),
            cp.norm(F, 2, axis=1) <= F_max
        ]

        objective = cp.Minimize(gamma * cp.sum(cp.norm(F, axis=1)))
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False, tol_gap_abs=1e-6, tol_gap_rel=1e-6)
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
                prob.solve(solver=cp.ECOS, verbose=False)
        except:
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except:
                return {"position": [], "velocity": [], "thrust": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {"position": [], "velocity": [], "thrust": []}

        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }