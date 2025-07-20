import cvxpy as cp
import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        num_conditions = int(problem["num_conditions"])
        conditions = problem["conditions"]
        # Define shared design variables
        A = cp.Variable(pos=True, name="A")
        S = cp.Variable(pos=True, name="S")
        # Condition-specific variables
        V = [cp.Variable(pos=True, name=f"V_{i}") for i in range(num_conditions)]
        W = [cp.Variable(pos=True, name=f"W_{i}") for i in range(num_conditions)]
        Re = [cp.Variable(pos=True, name=f"Re_{i}") for i in range(num_conditions)]
        C_D = [cp.Variable(pos=True, name=f"C_D_{i}") for i in range(num_conditions)]
        C_L = [cp.Variable(pos=True, name=f"C_L_{i}") for i in range(num_conditions)]
        C_f = [cp.Variable(pos=True, name=f"C_f_{i}") for i in range(num_conditions)]
        W_w = [cp.Variable(pos=True, name=f"W_w_{i}") for i in range(num_conditions)]
        constraints: List[Any] = []
        total_drag = 0
        for i in range(num_conditions):
            c = conditions[i]
            CDA0 = float(c["CDA0"])
            C_Lmax = float(c["C_Lmax"])
            N_ult = float(c["N_ult"])
            S_wetratio = float(c["S_wetratio"])
            V_min = float(c["V_min"])
            W_0 = float(c["W_0"])
            W_W_coeff1 = float(c["W_W_coeff1"])
            W_W_coeff2 = float(c["W_W_coeff2"])
            e = float(c["e"])
            k = float(c["k"])
            mu = float(c["mu"])
            rho = float(c["rho"])
            tau = float(c["tau"])
            drag_i = 0.5 * rho * V[i] ** 2 * C_D[i] * S
            total_drag += drag_i
            constraints += [
                C_D[i] >= CDA0 / S + k * C_f[i] * S_wetratio + C_L[i] ** 2 / (np.pi * A * e),
                C_f[i] >= 0.074 / Re[i] ** 0.2,
                Re[i] * mu >= rho * V[i] * cp.sqrt(S / A),
                W_w[i] >= W_W_coeff2 * S + W_W_coeff1 * N_ult * A ** (3 / 2) * cp.sqrt(W_0 * W[i]) / tau,
                W[i] >= W_0 + W_w[i],
                W[i] <= 0.5 * rho * V[i] ** 2 * C_L[i] * S,
                2 * W[i] / (rho * V_min ** 2 * S) <= C_Lmax
            ]
        objective = cp.Minimize(total_drag / num_conditions)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(gp=True)
        except Exception:
            pass
        if A.value is None or prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
        condition_results: List[Dict[str, Any]] = []
        for i in range(num_conditions):
            c = conditions[i]
            Vv = float(V[i].value)
            Wv = float(W[i].value)
            W_wv = float(W_w[i].value)
            C_Lv = float(C_L[i].value)
            C_Dv = float(C_D[i].value)
            C_fv = float(C_f[i].value)
            Rev = float(Re[i].value)
            drag_v = float(0.5 * c["rho"] * Vv ** 2 * C_Dv * S.value)
            condition_results.append({
                "condition_id": c["condition_id"],
                "V": Vv, "W": Wv, "W_w": W_wv,
                "C_L": C_Lv, "C_D": C_Dv, "C_f": C_fv,
                "Re": Rev, "drag": drag_v
            })
        return {
            "A": float(A.value),
            "S": float(S.value),
            "avg_drag": float(prob.value),
            "condition_results": condition_results
        }