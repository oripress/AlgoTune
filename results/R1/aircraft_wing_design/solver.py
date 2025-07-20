import cvxpy as cp
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]
        
        # Precompute all constants as numpy arrays
        CDA0 = np.array([c['CDA0'] for c in conditions])
        C_Lmax = np.array([c['C_Lmax'] for c in conditions])
        N_ult = np.array([c['N_ult'] for c in conditions])
        S_wetratio = np.array([c['S_wetratio'] for c in conditions])
        V_min = np.array([c['V_min'] for c in conditions])
        W_0 = np.array([c['W_0'] for c in conditions])
        W_W_coeff1 = np.array([c['W_W_coeff1'] for c in conditions])
        W_W_coeff2 = np.array([c['W_W_coeff2'] for c in conditions])
        e = np.array([c['e'] for c in conditions])
        k = np.array([c['k'] for c in conditions])
        mu = np.array([c['mu'] for c in conditions])
        rho = np.array([c['rho'] for c in conditions])
        tau = np.array([c['tau'] for c in conditions])
        
        # Shared variables with reasonable bounds
        A = cp.Variable(pos=True, name="A")  # aspect ratio (5-20)
        S = cp.Variable(pos=True, name="S")  # wing area (10-50 m²)
        
        # Define condition-specific variables
        V = [cp.Variable(pos=True, name=f"V_{i}") for i in range(num_conditions)]
        W = [cp.Variable(pos=True, name=f"W_{i}") for i in range(num_conditions)]
        Re = [cp.Variable(pos=True, name=f"Re_{i}") for i in range(num_conditions)]
        C_D = [cp.Variable(pos=True, name=f"C_D_{i}") for i in range(num_conditions)]
        C_L = [cp.Variable(pos=True, name=f"C_L_{i}") for i in range(num_conditions)]
        C_f = [cp.Variable(pos=True, name=f"C_f_{i}") for i in range(num_conditions)]
        W_w = [cp.Variable(pos=True, name=f"W_w_{i}") for i in range(num_conditions)]

        # Define constraints
        constraints = []

        # Objective: minimize average drag across all conditions
        total_drag = 0

        for i in range(num_conditions):
            # Calculate drag for this condition
            drag_i = 0.5 * rho[i] * V[i] ** 2 * C_D[i] * S
            total_drag += drag_i

            # Condition-specific constraints
            constraints.append(
                C_D[i] >= CDA0[i] / S + k[i] * C_f[i] * S_wetratio[i] + C_L[i] ** 2 / (np.pi * A * e[i])
            )
            constraints.append(C_f[i] >= 0.074 * Re[i] ** -0.2)
            constraints.append(Re[i] * mu[i] >= rho[i] * V[i] * cp.sqrt(S / A))
            constraints.append(
                W_w[i] >= W_W_coeff2[i] * S + 
                W_W_coeff1[i] * N_ult[i] * (A ** 1.5) * cp.sqrt(W_0[i] * W[i]) / tau[i]
            )
            constraints.append(W[i] >= W_0[i] + W_w[i])
            constraints.append(W[i] <= 0.5 * rho[i] * V[i] ** 2 * C_L[i] * S)
            constraints.append(2 * W[i] / (rho[i] * V_min[i]**2 * S) <= C_Lmax[i])

        # Define the objective: minimize average drag across all conditions
        objective = cp.Minimize(total_drag / num_conditions)

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            # Use MOSEK if available for better performance
            if cp.MOSEK in cp.installed_solvers():
                prob.solve(gp=True, solver=cp.MOSEK, verbose=False)
            else:
                # Use ECOS with increased iterations and better numerical settings
                prob.solve(gp=True, solver=cp.ECOS, verbose=False,
                          max_iters=10000, abstol=1e-6, reltol=1e-6, feastol=1e-6)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Collect results for each condition
            condition_results = []
            for i in range(num_conditions):
                drag_i = 0.5 * rho[i] * V[i].value**2 * C_D[i].value * S.value
                condition_results.append(
                    {
                        "condition_id": conditions[i]["condition_id"],
                        "V": float(V[i].value),
                        "W": float(W[i].value),
                        "W_w": float(W_w[i].value),
                        "C_L": float(C_L[i].value),
                        "C_D": float(C_D[i].value),
                        "C_f": float(C_f[i].value),
                        "Re": float(Re[i].value),
                        "drag": float(drag_i),
                    }
                )

            # Return optimal values
            return {
                "A": float(A.value),
                "S": float(S.value),
                "avg_drag": float(prob.value),
                "condition_results": condition_results,
            }

        except cp.SolverError:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}