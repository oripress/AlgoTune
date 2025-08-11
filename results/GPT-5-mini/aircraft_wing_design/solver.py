from typing import Any, Dict, List
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the aircraft wing design optimization problem using CVXPY in GP mode.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal design variables and per-condition results
        """
        # Standard failure output (matches reference expectations)
        fail = {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

        # Parse input safely
        try:
            num_conditions = int(problem["num_conditions"])
            conditions = problem["conditions"]
            if num_conditions <= 0 or not isinstance(conditions, list) or len(conditions) != num_conditions:
                return fail
        except Exception:
            return fail

        # Define shared design variables
        try:
            A = cp.Variable(pos=True, name="A")  # aspect ratio
            S = cp.Variable(pos=True, name="S")  # wing area (m²)
        except TypeError:
            # Fallback if pos=True not supported in this cvxpy version
            A = cp.Variable(name="A")
            S = cp.Variable(name="S")

        # Define condition-specific variables
        V = [cp.Variable(pos=True, name=f"V_{i}") for i in range(num_conditions)]
        W = [cp.Variable(pos=True, name=f"W_{i}") for i in range(num_conditions)]
        Re = [cp.Variable(pos=True, name=f"Re_{i}") for i in range(num_conditions)]
        C_D = [cp.Variable(pos=True, name=f"C_D_{i}") for i in range(num_conditions)]
        C_L = [cp.Variable(pos=True, name=f"C_L_{i}") for i in range(num_conditions)]
        C_f = [cp.Variable(pos=True, name=f"C_f_{i}") for i in range(num_conditions)]
        W_w = [cp.Variable(pos=True, name=f"W_w_{i}") for i in range(num_conditions)]

        constraints: List[Any] = []

        # Objective: minimize average drag across all conditions
        total_drag = 0

        # Process each flight condition
        for i in range(num_conditions):
            condition = conditions[i]

            # Extract condition-specific parameters
            try:
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
            except Exception:
                return fail

            # Ensure positivity if pos=True unsupported (safe even if pos=True used)
            constraints += [
                A >= 1e-12,
                S >= 1e-12,
                V[i] >= 1e-6,
                W[i] >= 1e-6,
                Re[i] >= 1e-6,
                C_D[i] >= 1e-12,
                C_L[i] >= 1e-12,
                C_f[i] >= 1e-12,
                W_w[i] >= 1e-6,
            ]

            # Calculate drag for this condition
            drag_i = 0.5 * rho * cp.power(V[i], 2) * C_D[i] * S
            total_drag += drag_i

            # Condition-specific constraints
            constraints.append(
                C_D[i] >= CDA0 / S + k * C_f[i] * S_wetratio + C_L[i] ** 2 / (np.pi * A * e)
            )  # drag coefficient model
            constraints.append(C_f[i] >= 0.074 / Re[i] ** 0.2)  # skin friction model
            constraints.append(Re[i] * mu >= rho * V[i] * cp.sqrt(S / A))  # Reynolds number definition
            constraints.append(
                W_w[i]
                >= W_W_coeff2 * S + W_W_coeff1 * N_ult * (A ** (3 / 2)) * cp.sqrt(W_0 * W[i]) / tau
            )  # wing weight model
            constraints.append(W[i] >= W_0 + W_w[i])  # total weight
            constraints.append(W[i] <= 0.5 * rho * cp.power(V[i], 2) * C_L[i] * S)  # lift equals weight
            constraints.append(2 * W[i] / (rho * V_min**2 * S) <= C_Lmax)  # stall constraint

        # Define the objective: minimize average drag across all conditions
        objective = cp.Minimize(total_drag / float(num_conditions))

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            # Use GP mode only; if unavailable or fails, return failure to avoid invalid non-GP solutions
            prob.solve(gp=True, verbose=False)
        except Exception:
            return fail

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
            return fail

        # Collect results for each condition
        condition_results = []
        try:
            for i in range(num_conditions):
                condition_results.append(
                    {
                        "condition_id": int(conditions[i].get("condition_id", i)),
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
        except Exception:
            return fail

        # Return optimal values
        return {
            "A": float(A.value),
            "S": float(S.value),
            "avg_drag": float(prob.value),
            "condition_results": condition_results,
        }