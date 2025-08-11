from typing import Any, Dict, List
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the aircraft wing design optimization problem using a vectorized CVXPY GP formulation.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal design variables and minimum drag
        """
        try:
            num_conditions: int = int(problem["num_conditions"])
            conditions: List[Dict[str, Any]] = problem["conditions"]

            if num_conditions <= 0 or not conditions:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Extract condition parameters as numpy arrays for vectorized modeling
            cond_ids = np.array([int(cond["condition_id"]) for cond in conditions], dtype=int)
            CDA0 = np.asarray([float(cond["CDA0"]) for cond in conditions], dtype=float)
            C_Lmax = np.asarray([float(cond["C_Lmax"]) for cond in conditions], dtype=float)
            N_ult = np.asarray([float(cond["N_ult"]) for cond in conditions], dtype=float)
            S_wetratio = np.asarray([float(cond["S_wetratio"]) for cond in conditions], dtype=float)
            V_min = np.asarray([float(cond["V_min"]) for cond in conditions], dtype=float)
            W_0 = np.asarray([float(cond["W_0"]) for cond in conditions], dtype=float)
            W_W_coeff1 = np.asarray([float(cond["W_W_coeff1"]) for cond in conditions], dtype=float)
            W_W_coeff2 = np.asarray([float(cond["W_W_coeff2"]) for cond in conditions], dtype=float)
            e = np.asarray([float(cond["e"]) for cond in conditions], dtype=float)
            k = np.asarray([float(cond["k"]) for cond in conditions], dtype=float)
            mu = np.asarray([float(cond["mu"]) for cond in conditions], dtype=float)
            rho = np.asarray([float(cond["rho"]) for cond in conditions], dtype=float)
            tau = np.asarray([float(cond["tau"]) for cond in conditions], dtype=float)

            # Shared design variables
            A = cp.Variable(pos=True, name="A")  # aspect ratio
            S = cp.Variable(pos=True, name="S")  # wing area

            n = num_conditions

            # Condition-specific variables (vectorized)
            V = cp.Variable(n, pos=True, name="V")      # cruising speed
            W = cp.Variable(n, pos=True, name="W")      # total weight
            Re = cp.Variable(n, pos=True, name="Re")    # Reynolds number
            C_D = cp.Variable(n, pos=True, name="C_D")  # drag coefficient
            C_L = cp.Variable(n, pos=True, name="C_L")  # lift coefficient
            C_f = cp.Variable(n, pos=True, name="C_f")  # skin friction coefficient
            W_w = cp.Variable(n, pos=True, name="W_w")  # wing weight

            constraints = []

            # Drag vector for all conditions: 0.5 * rho * V^2 * C_D * S
            drag_vec = 0.5 * S * cp.multiply(rho, cp.multiply(cp.power(V, 2), C_D))

            # Drag coefficient model:
            # C_D >= CDA0/S + k*C_f*S_wetratio + C_L^2/(pi*A*e)
            constraints.append(
                C_D
                >= CDA0 / S
                + cp.multiply(k * S_wetratio, C_f)
                + (cp.power(C_L, 2) / (np.pi * A * e))
            )

            # Skin friction model: C_f >= 0.074 / Re^0.2
            constraints.append(C_f >= 0.074 / cp.power(Re, 0.2))

            # Reynolds number: Re * mu >= rho * V * sqrt(S/A)
            constraints.append(cp.multiply(Re, mu) >= cp.multiply(rho, V) * cp.sqrt(S / A))

            # Wing weight model:
            # W_w >= W_W_coeff2*S + W_W_coeff1*N_ult*A^(3/2)*sqrt(W_0*W)/tau
            constraints.append(
                W_w
                >= cp.multiply(W_W_coeff2, S)
                + cp.power(A, 1.5)
                * cp.multiply((W_W_coeff1 * N_ult) / tau, cp.sqrt(cp.multiply(W_0, W)))
            )

            # Total weight: W >= W_0 + W_w
            constraints.append(W >= W_0 + W_w)

            # Lift equals weight upper bound: W <= 0.5 * rho * V^2 * C_L * S
            lift_vec = 0.5 * S * cp.multiply(rho, cp.multiply(cp.power(V, 2), C_L))
            constraints.append(W <= lift_vec)

            # Stall constraint: 2*W/(rho * V_min^2 * S) <= C_Lmax
            constraints.append(2.0 * W / (rho * cp.power(V_min, 2) * S) <= C_Lmax)

            # Objective: minimize average drag
            objective = cp.Minimize(cp.sum(drag_vec) / n)

            # Solve GP
            prob = cp.Problem(objective, constraints)
            prob.solve(gp=True)

            # Check solver status
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            A_val = float(A.value)
            S_val = float(S.value)

            # Extract per-condition values
            V_val = np.asarray(V.value, dtype=float).ravel()
            W_val = np.asarray(W.value, dtype=float).ravel()
            W_w_val = np.asarray(W_w.value, dtype=float).ravel()
            C_L_val = np.asarray(C_L.value, dtype=float).ravel()
            C_D_val = np.asarray(C_D.value, dtype=float).ravel()
            C_f_val = np.asarray(C_f.value, dtype=float).ravel()
            Re_val = np.asarray(Re.value, dtype=float).ravel()

            # Compute drag for each condition
            drag_vals = 0.5 * rho * (V_val ** 2) * C_D_val * S_val
            condition_results = []
            for i in range(n):
                condition_results.append(
                    {
                        "condition_id": int(cond_ids[i]),
                        "V": float(V_val[i]),
                        "W": float(W_val[i]),
                        "W_w": float(W_w_val[i]),
                        "C_L": float(C_L_val[i]),
                        "C_D": float(C_D_val[i]),
                        "C_f": float(C_f_val[i]),
                        "Re": float(Re_val[i]),
                        "drag": float(drag_vals[i]),
                    }
                )

            return {
                "A": A_val,
                "S": S_val,
                "avg_drag": float(prob.value),
                "condition_results": condition_results,
            }

        except cp.SolverError:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}