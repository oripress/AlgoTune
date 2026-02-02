from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]

        # Define shared design variables
        A = cp.Variable(pos=True)
        S = cp.Variable(pos=True)

        # Define condition-specific variables
        V = [cp.Variable(pos=True) for _ in range(num_conditions)]
        W = [cp.Variable(pos=True) for _ in range(num_conditions)]
        Re = [cp.Variable(pos=True) for _ in range(num_conditions)]
        C_D = [cp.Variable(pos=True) for _ in range(num_conditions)]
        C_L = [cp.Variable(pos=True) for _ in range(num_conditions)]
        C_f = [cp.Variable(pos=True) for _ in range(num_conditions)]
        W_w = [cp.Variable(pos=True) for _ in range(num_conditions)]

        constraints = []
        total_drag = 0
        pi = np.pi

        for i in range(num_conditions):
            c = conditions[i]
            rho = c["rho"]
            CDA0 = c["CDA0"]
            C_Lmax = c["C_Lmax"]
            N_ult = c["N_ult"]
            S_wetratio = c["S_wetratio"]
            V_min = c["V_min"]
            W_0 = c["W_0"]
            W_W_coeff1 = c["W_W_coeff1"]
            W_W_coeff2 = c["W_W_coeff2"]
            e = c["e"]
            k = c["k"]
            mu = c["mu"]
            tau = c["tau"]

            total_drag += 0.5 * rho * V[i]**2 * C_D[i] * S
            
            constraints.append(C_D[i] >= CDA0/S + k*C_f[i]*S_wetratio + C_L[i]**2/(pi*A*e))
            constraints.append(C_f[i] >= 0.074/Re[i]**0.2)
            constraints.append(Re[i]*mu >= rho*V[i]*cp.sqrt(S/A))
            constraints.append(W_w[i] >= W_W_coeff2*S + W_W_coeff1*N_ult*(A**1.5)*cp.sqrt(W_0*W[i])/tau)
            constraints.append(W[i] >= W_0 + W_w[i])
            constraints.append(W[i] <= 0.5*rho*V[i]**2*C_L[i]*S)
            constraints.append(2*W[i]/(rho*V_min**2*S) <= C_Lmax)

        prob = cp.Problem(cp.Minimize(total_drag/num_conditions), constraints)
        
        try:
            prob.solve(gp=True, solver=cp.ECOS, abstol=1e-7, reltol=1e-6, feastol=1e-7)
            
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            A_val = float(A.value)
            S_val = float(S.value)
            
            condition_results = []
            for i in range(num_conditions):
                rho_i = conditions[i]["rho"]
                V_val = float(V[i].value)
                CD_val = float(C_D[i].value)
                condition_results.append({
                    "condition_id": conditions[i]["condition_id"],
                    "V": V_val,
                    "W": float(W[i].value),
                    "W_w": float(W_w[i].value),
                    "C_L": float(C_L[i].value),
                    "C_D": CD_val,
                    "C_f": float(C_f[i].value),
                    "Re": float(Re[i].value),
                    "drag": 0.5 * rho_i * V_val**2 * CD_val * S_val,
                })

            return {
                "A": A_val,
                "S": S_val,
                "avg_drag": float(prob.value),
                "condition_results": condition_results,
            }
        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}