from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        num_conditions = int(problem.get("num_conditions", 0))
        conditions = problem.get("conditions", [])

        # Shared design variables
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

        constraints = []
        total_drag = 0
        # Build constraints and drag expression
        for i in range(num_conditions):
            cond = conditions[i]
            CDA0 = float(cond["CDA0"])
            C_Lmax = float(cond["C_Lmax"])
            N_ult = float(cond["N_ult"])
            S_wet = float(cond["S_wetratio"])
            V_min = float(cond["V_min"])
            W_0 = float(cond["W_0"])
            WW1 = float(cond["W_W_coeff1"])
            WW2 = float(cond["W_W_coeff2"])
            e = float(cond["e"])
            k = float(cond["k"])
            mu = float(cond["mu"])
            rho = float(cond["rho"])
            tau = float(cond["tau"])

            # drag in condition i
            drag_i = 0.5 * rho * V[i]**2 * C_D[i] * S
            total_drag += drag_i

            # constraints
            constraints += [
                C_D[i] >= CDA0 / S
                             + k * C_f[i] * S_wet
                             + C_L[i]**2 / (np.pi * A * e),
                C_f[i] >= 0.074 / Re[i]**0.2,
                Re[i] * mu >= rho * V[i] * cp.sqrt(S / A),
                W_w[i] >= WW2 * S
                             + WW1 * N_ult * A**1.5 * cp.sqrt(W_0 * W[i]) / tau,
                W[i] >= W_0 + W_w[i],
                W[i] <= 0.5 * rho * V[i]**2 * C_L[i] * S,
                2 * W[i] / (rho * V_min**2 * S) <= C_Lmax,
            ]

        # Objective: average drag
        objective = cp.Minimize(total_drag / max(num_conditions, 1))
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(gp=True)
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # gather results
            condition_results = []
            for i in range(num_conditions):
                cond = conditions[i]
                val_V = float(V[i].value)
                val_W = float(W[i].value)
                val_Ww = float(W_w[i].value)
                val_CL = float(C_L[i].value)
                val_CD = float(C_D[i].value)
                val_Cf = float(C_f[i].value)
                val_Re = float(Re[i].value)
                val_drag = float(0.5 * cond["rho"] * val_V**2 * val_CD * float(S.value))
                condition_results.append({
                    "condition_id": cond["condition_id"],
                    "V": val_V,
                    "W": val_W,
                    "W_w": val_Ww,
                    "C_L": val_CL,
                    "C_D": val_CD,
                    "C_f": val_Cf,
                    "Re": val_Re,
                    "drag": val_drag
                })

            return {
                "A": float(A.value),
                "S": float(S.value),
                "avg_drag": float(prob.value),
                "condition_results": condition_results,
            }
        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}