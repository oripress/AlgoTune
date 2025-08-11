from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def __init__(self):
        # Pre-configure ECOS solver settings for optimal GP performance
        self.gp_kwargs = {
            'gp': True,
            'verbose': False,
            'solver': cp.ECOS,
            'max_iters': 100,
            'abstol': 1e-6,
            'reltol': 1e-5,
            'feastol': 1e-6
        }
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Optimized aircraft wing design solver using geometric programming.
        """
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]

        # Shared design variables (no names for speed)
        A = cp.Variable(pos=True)
        S = cp.Variable(pos=True)

        # Pre-allocate all condition-specific variables
        n = num_conditions
        V = [cp.Variable(pos=True) for _ in range(n)]
        W = [cp.Variable(pos=True) for _ in range(n)]
        Re = [cp.Variable(pos=True) for _ in range(n)]
        C_D = [cp.Variable(pos=True) for _ in range(n)]
        C_L = [cp.Variable(pos=True) for _ in range(n)]
        C_f = [cp.Variable(pos=True) for _ in range(n)]
        W_w = [cp.Variable(pos=True) for _ in range(n)]

        # Pre-allocate constraints
        constraints = []
        
        # Build objective as list first
        drag_sum = 0
        
        # Vectorized parameter extraction
        params = []
        for c in conditions:
            params.append((
                c["CDA0"], c["C_Lmax"], c["N_ult"], c["S_wetratio"],
                c["V_min"], c["W_0"], c["W_W_coeff1"], c["W_W_coeff2"],
                c["e"], c["k"], c["mu"], c["rho"], c["tau"]
            ))

        # Build constraints and objective
        for i in range(n):
            CDA0, C_Lmax, N_ult, S_wetratio, V_min, W_0, W_W_coeff1, W_W_coeff2, e, k, mu, rho, tau = params[i]
            
            # Objective term
            drag_sum += 0.5 * rho * V[i]**2 * C_D[i] * S
            
            # Constraints (7 per condition)
            pi_A_e = np.pi * A * e
            constraints.extend([
                C_D[i] >= CDA0/S + k*C_f[i]*S_wetratio + C_L[i]**2/pi_A_e,
                C_f[i] >= 0.074 / Re[i]**0.2,
                Re[i] * mu >= rho * V[i] * cp.sqrt(S/A),
                W_w[i] >= W_W_coeff2*S + W_W_coeff1*N_ult*cp.power(A, 1.5)*cp.sqrt(W_0*W[i])/tau,
                W[i] >= W_0 + W_w[i],
                W[i] <= 0.5 * rho * V[i]**2 * C_L[i] * S,
                2*W[i]/(rho * V_min**2 * S) <= C_Lmax
            ])

        # Create problem
        objective = cp.Minimize(drag_sum / n)
        prob = cp.Problem(objective, constraints)
        
        try:
            # Solve with optimized GP settings
            prob.solve(**self.gp_kwargs)
            
            # Check solution validity
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                # Fallback to default GP solver
                prob.solve(gp=True, verbose=False)
                
            if A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Extract solution values once
            A_val = float(A.value)
            S_val = float(S.value)
            avg_drag_val = float(prob.value)
            
            # Build results
            results = []
            for i in range(n):
                V_val = float(V[i].value)
                W_val = float(W[i].value)
                W_w_val = float(W_w[i].value)
                C_L_val = float(C_L[i].value)
                C_D_val = float(C_D[i].value)
                C_f_val = float(C_f[i].value)
                Re_val = float(Re[i].value)
                
                rho = conditions[i]["rho"]
                drag_val = 0.5 * rho * V_val * V_val * C_D_val * S_val
                
                results.append({
                    "condition_id": conditions[i]["condition_id"],
                    "V": V_val,
                    "W": W_val,
                    "W_w": W_w_val,
                    "C_L": C_L_val,
                    "C_D": C_D_val,
                    "C_f": C_f_val,
                    "Re": Re_val,
                    "drag": drag_val
                })

            return {
                "A": A_val,
                "S": S_val,
                "avg_drag": avg_drag_val,
                "condition_results": results
            }

        except:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}