from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the aircraft wing design optimization problem using CVXPY.
        """
        # Extract problem parameters
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]

        # Define shared design variables
        # Define shared design variables (avoid string names for speed)
        A = cp.Variable(pos=True)  # aspect ratio
        S = cp.Variable(pos=True)  # wing area (m²)

        # Define condition-specific variables (no names for speed)
        V = [cp.Variable(pos=True) for _ in range(num_conditions)]
        W = [cp.Variable(pos=True) for _ in range(num_conditions)]
        Re = [cp.Variable(pos=True) for _ in range(num_conditions)]
        C_D = [cp.Variable(pos=True) for _ in range(num_conditions)]
        C_L = [cp.Variable(pos=True) for _ in range(num_conditions)]
        C_f = [cp.Variable(pos=True) for _ in range(num_conditions)]
        W_w = [cp.Variable(pos=True) for _ in range(num_conditions)]

        # Define constraints
        constraints = []
        total_drag = 0
        
        # Pre-compute constants
        pi = np.pi

        # Process each flight condition
        for i in range(num_conditions):
            condition = conditions[i]

            # Extract and pre-compute condition-specific parameters
            CDA0 = float(condition["CDA0"])
            C_Lmax = float(condition["C_Lmax"])
            N_ult = float(condition["N_ult"])
            S_wetratio = float(condition["S_wetratio"])
            V_min = float(condition["V_min"])
            W_0 = float(condition["W_0"])
            W_W_coeff1 = float(condition["W_W_coeff1"])
            W_W_coeff2 = float(condition["W_W_coeff2"])
            e = float(condition["e"])
            k = float(condition["k"])
            mu = float(condition["mu"])
            rho = float(condition["rho"])
            tau = float(condition["tau"])
            
            # Pre-compute constant expressions
            pi_A_e = pi * A * e
            k_S_wetratio = k * S_wetratio
            half_rho = 0.5 * rho
            rho_V_min_sq = rho * V_min ** 2
            W_W_coeff1_N_ult = W_W_coeff1 * N_ult
            sqrt_W_0 = np.sqrt(W_0)

            # Calculate drag for this condition
            drag_i = half_rho * V[i] ** 2 * C_D[i] * S
            total_drag += drag_i

            # Condition-specific constraints (batch append for efficiency)
            constraints.extend([
                C_D[i] >= CDA0 / S + C_f[i] * k_S_wetratio + C_L[i] ** 2 / pi_A_e,
                C_f[i] >= 0.074 / Re[i] ** 0.2,
                Re[i] * mu >= rho * V[i] * cp.sqrt(S / A),
                W_w[i] >= W_W_coeff2 * S + W_W_coeff1_N_ult * (A ** 1.5) * cp.sqrt(W[i]) * sqrt_W_0 / tau,
                W[i] >= W_0 + W_w[i],
                W[i] <= half_rho * V[i] ** 2 * C_L[i] * S,
                2 * W[i] / (rho_V_min_sq * S) <= C_Lmax
            ])

        # Define the objective: minimize average drag
        objective = cp.Minimize(total_drag / num_conditions)

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            # Solve using GP with optimized settings
            prob.solve(gp=True, verbose=False)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Collect results for each condition
            condition_results = []
            for i in range(num_conditions):
                condition_results.append({
                    "condition_id": conditions[i]["condition_id"],
                    "V": float(V[i].value),
                    "W": float(W[i].value),
                    "W_w": float(W_w[i].value),
                    "C_L": float(C_L[i].value),
                    "C_D": float(C_D[i].value),
                    "C_f": float(C_f[i].value),
                    "Re": float(Re[i].value),
                    "drag": float(0.5 * conditions[i]["rho"] * V[i].value ** 2 * C_D[i].value * S.value),
                })

            # Return optimal values
            return {
                "A": float(A.value),
                "S": float(S.value),
                "avg_drag": float(prob.value),
                "condition_results": condition_results,
            }

        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}