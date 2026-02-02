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
        A = cp.Variable(pos=True, name="A")  # aspect ratio
        S = cp.Variable(pos=True, name="S")  # wing area (m²)

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

        # Process each flight condition
        for i in range(num_conditions):
            condition = conditions[i]

            # Extract condition-specific parameters
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

            # Calculate drag for this condition
            drag_i = 0.5 * rho * V[i] ** 2 * C_D[i] * S
            total_drag += drag_i

            # Condition-specific constraints
            constraints.append(
                C_D[i] >= CDA0 / S + k * C_f[i] * S_wetratio + C_L[i] ** 2 / (np.pi * A * e)
            )
            constraints.append(C_f[i] >= 0.074 / Re[i] ** 0.2)
            constraints.append(Re[i] * mu >= rho * V[i] * cp.sqrt(S / A))
            constraints.append(
                W_w[i] >= W_W_coeff2 * S + W_W_coeff1 * N_ult * (A ** (3 / 2)) * cp.sqrt(W_0 * W[i]) / tau
            )
            constraints.append(W[i] >= W_0 + W_w[i])
            constraints.append(W[i] <= 0.5 * rho * V[i] ** 2 * C_L[i] * S)
            constraints.append(2 * W[i] / (rho * V_min**2 * S) <= C_Lmax)

        # Define the objective
        objective = cp.Minimize(total_drag / num_conditions)

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(gp=True)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Collect results
            condition_results = []
            for i in range(num_conditions):
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
                        "drag": float(0.5 * conditions[i]["rho"] * V[i].value ** 2 * C_D[i].value * S.value),
                    }
                )

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