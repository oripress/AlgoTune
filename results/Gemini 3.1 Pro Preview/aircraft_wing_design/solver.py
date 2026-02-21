import cvxpy as cp
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]
        
        if num_conditions == 0:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
            
        # Extract parameters into numpy arrays
        CDA0 = np.array([c["CDA0"] for c in conditions])
        C_Lmax = np.array([c["C_Lmax"] for c in conditions])
        N_ult = np.array([c["N_ult"] for c in conditions])
        S_wetratio = np.array([c["S_wetratio"] for c in conditions])
        V_min = np.array([c["V_min"] for c in conditions])
        W_0 = np.array([c["W_0"] for c in conditions])
        W_W_coeff1 = np.array([c["W_W_coeff1"] for c in conditions])
        W_W_coeff2 = np.array([c["W_W_coeff2"] for c in conditions])
        e = np.array([c["e"] for c in conditions])
        k = np.array([c["k"] for c in conditions])
        mu = np.array([c["mu"] for c in conditions])
        rho = np.array([c["rho"] for c in conditions])
        tau = np.array([c["tau"] for c in conditions])
        
        # Log-transformed shared variables
        a = cp.Variable(name="a")
        s = cp.Variable(name="s")
        
        # Log-transformed vectorized variables
        v = cp.Variable(num_conditions, name="v")
        w = cp.Variable(num_conditions, name="w")
        re = cp.Variable(num_conditions, name="re")
        cd = cp.Variable(num_conditions, name="cd")
        cl = cp.Variable(num_conditions, name="cl")
        cf = cp.Variable(num_conditions, name="cf")
        ww = cp.Variable(num_conditions, name="ww")
        
        # Constraints
        constraints = [
            # C_D >= CDA0 / S + k * C_f * S_wetratio + C_L**2 / (np.pi * A * e)
            cp.log_sum_exp(cp.vstack([
                np.log(CDA0) - s - cd,
                np.log(k * S_wetratio) + cf - cd,
                np.log(1 / (np.pi * e)) + 2 * cl - a - cd
            ]), axis=0) <= 0,
            
            # C_f >= 0.074 / Re**0.2
            cf >= np.log(0.074) - 0.2 * re,
            
            # Re * mu >= rho * V * sqrt(S / A)
            re + np.log(mu) >= np.log(rho) + v + 0.5 * s - 0.5 * a,
            
            # W_w >= W_W_coeff2 * S + W_W_coeff1 * N_ult * A**(3/2) * sqrt(W_0 * W) / tau
            cp.log_sum_exp(cp.vstack([
                np.log(W_W_coeff2) + s - ww,
                np.log(W_W_coeff1 * N_ult * np.sqrt(W_0) / tau) + 1.5 * a + 0.5 * w - ww
            ]), axis=0) <= 0,
            
            # W >= W_0 + W_w
            cp.log_sum_exp(cp.vstack([
                np.log(W_0) - w,
                ww - w
            ]), axis=0) <= 0,
            
            # W <= 0.5 * rho * V**2 * C_L * S
            w <= np.log(0.5 * rho) + 2 * v + cl + s,
            
            # 2 * W / (rho * V_min**2 * S) <= C_Lmax
            w - s + np.log(2 / (rho * V_min**2 * C_Lmax)) <= 0
        ]
        
        # Objective
        # Minimize average drag: sum(0.5 * rho * V**2 * C_D * S) / num_conditions
        log_drag = np.log(0.5 * rho) + 2 * v + cd + s
        objective = cp.Minimize(cp.sum(cp.exp(log_drag)) / num_conditions)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9, feastol=1e-9, ignore_dpp=True)
            
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or a.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
                
            # Recover original variables
            A_val = np.exp(a.value)
            S_val = np.exp(s.value)
            V_val = np.exp(v.value)
            W_val = np.exp(w.value)
            W_w_val = np.exp(ww.value)
            C_L_val = np.exp(cl.value)
            C_D_val = np.exp(cd.value)
            C_f_val = np.exp(cf.value)
            Re_val = np.exp(re.value)
            
            condition_results = []
            for i in range(num_conditions):
                v_i = float(V_val[i]) if num_conditions > 1 else float(V_val)
                w_i = float(W_val[i]) if num_conditions > 1 else float(W_val)
                ww_i = float(W_w_val[i]) if num_conditions > 1 else float(W_w_val)
                cl_i = float(C_L_val[i]) if num_conditions > 1 else float(C_L_val)
                cd_i = float(C_D_val[i]) if num_conditions > 1 else float(C_D_val)
                cf_i = float(C_f_val[i]) if num_conditions > 1 else float(C_f_val)
                re_i = float(Re_val[i]) if num_conditions > 1 else float(Re_val)
                
                condition_results.append({
                    "condition_id": conditions[i]["condition_id"],
                    "V": v_i,
                    "W": w_i,
                    "W_w": ww_i,
                    "C_L": cl_i,
                    "C_D": cd_i,
                    "C_f": cf_i,
                    "Re": re_i,
                    "drag": float(0.5 * rho[i] * v_i**2 * cd_i * S_val)
                })
                
            return {
                "A": float(A_val),
                "S": float(S_val),
                "avg_drag": float(prob.value),
                "condition_results": condition_results
            }
        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}