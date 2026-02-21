from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    _cache = {}

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        K = int(problem["K"])
        
        if K not in self._cache:
            print("CACHE MISS")
            p0_param = cp.Parameter(3)
            v0_param = cp.Parameter(3)
            p_target_param = cp.Parameter(3)
            
            h_m_param = cp.Parameter(nonneg=True)
            h_g_param = cp.Parameter()
            h_2_param = cp.Parameter(nonneg=True)
            F_max_param = cp.Parameter(nonneg=True)
            gamma_param = cp.Parameter(nonneg=True)

            V = cp.Variable((K + 1, 3))
            P = cp.Variable((K + 1, 3))
            F = cp.Variable((K, 3))

            constraints = [
                V[0] == v0_param,
                P[0] == p0_param,
                V[K] == np.zeros(3),
                P[K] == p_target_param,
                P[:, 2] >= 0,
                V[1:, :2] == V[:-1, :2] + h_m_param * F[:, :2],
                V[1:, 2] == V[:-1, 2] + h_m_param * F[:, 2] - h_g_param,
                P[1:] == P[:-1] + h_2_param * (V[:-1] + V[1:]),
                cp.norm(F, 2, axis=1) <= F_max_param
            ]

            fuel_consumption = gamma_param * cp.sum(cp.norm(F, axis=1))
            objective = cp.Minimize(fuel_consumption)

            prob = cp.Problem(objective, constraints)
            
            self._cache[K] = {
                "prob": prob,
                "p0_param": p0_param,
                "v0_param": v0_param,
                "p_target_param": p_target_param,
                "h_m_param": h_m_param,
                "h_g_param": h_g_param,
                "h_2_param": h_2_param,
                "F_max_param": F_max_param,
                "gamma_param": gamma_param,
                "P": P,
                "V": V,
                "F": F
            }

        cached = self._cache[K]
        prob = cached["prob"]
        
        h = float(problem["h"])
        m = float(problem["m"])
        g = float(problem["g"])
        
        cached["p0_param"].value = np.array(problem["p0"])
        cached["v0_param"].value = np.array(problem["v0"])
        cached["p_target_param"].value = np.array(problem["p_target"])
        cached["h_m_param"].value = h / m
        cached["h_g_param"].value = h * g
        cached["h_2_param"].value = h / 2.0
        cached["F_max_param"].value = float(problem["F_max"])
        cached["gamma_param"].value = float(problem["gamma"])

        try:
            prob.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            return {"position": [], "velocity": [], "thrust": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or cached["P"].value is None:
            return {"position": [], "velocity": [], "thrust": []}

        return {
            "position": cached["P"].value.tolist(),
            "velocity": cached["V"].value.tolist(),
            "thrust": cached["F"].value.tolist(),
            "fuel_consumption": float(prob.value),
        }