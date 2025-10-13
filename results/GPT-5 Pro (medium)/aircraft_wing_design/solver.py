from __future__ import annotations

from typing import Any, Dict, List

import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the aircraft wing design optimization problem using a vectorized CVXPY GP model.

        This implementation:
        - Uses vectorized variables for per-condition quantities to reduce overhead.
        - Eliminates C_D and W_w variables by substituting their optimal expressions, reducing problem size.
        - Solves with ECOS in GP mode for speed, with fallback to default if needed.

        Returns a dictionary matching the required output format.
        """

        try:
            num_conditions = int(problem["num_conditions"])
            conditions = problem["conditions"]
            if num_conditions <= 0 or len(conditions) != num_conditions:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Extract parameters into numpy arrays (vectorized)
            # Preserve order of conditions as given
            cond_ids = np.array([int(c["condition_id"]) for c in conditions], dtype=int)
            CDA0 = np.array([float(c["CDA0"]) for c in conditions], dtype=float)
            C_Lmax = np.array([float(c["C_Lmax"]) for c in conditions], dtype=float)
            N_ult = np.array([float(c["N_ult"]) for c in conditions], dtype=float)
            S_wetratio = np.array([float(c["S_wetratio"]) for c in conditions], dtype=float)
            V_min = np.array([float(c["V_min"]) for c in conditions], dtype=float)
            W_0 = np.array([float(c["W_0"]) for c in conditions], dtype=float)
            W_W_coeff1 = np.array([float(c["W_W_coeff1"]) for c in conditions], dtype=float)
            W_W_coeff2 = np.array([float(c["W_W_coeff2"]) for c in conditions], dtype=float)
            e = np.array([float(c["e"]) for c in conditions], dtype=float)
            k = np.array([float(c["k"]) for c in conditions], dtype=float)
            mu = np.array([float(c["mu"]) for c in conditions], dtype=float)
            rho = np.array([float(c["rho"]) for c in conditions], dtype=float)
            tau = np.array([float(c["tau"]) for c in conditions], dtype=float)

            n = num_conditions

            # Shared design variables
            A = cp.Variable(pos=True, name="A")  # aspect ratio
            S = cp.Variable(pos=True, name="S")  # wing area

            # Condition-specific variables (vectorized)
            V = cp.Variable(n, pos=True, name="V")   # cruising speed
            W = cp.Variable(n, pos=True, name="W")   # total aircraft weight
            Re = cp.Variable(n, pos=True, name="Re")  # Reynolds number
            C_L = cp.Variable(n, pos=True, name="C_L")  # lift coefficient
            C_f = cp.Variable(n, pos=True, name="C_f")  # skin friction coefficient

            constraints = []

            # Helper expressions
            # Drag coefficient expression (C_D) eliminated as a variable, computed as lower bound expression
            # C_D = CDA0/S + k*C_f*S_wetratio + C_L^2/(pi*A*e)
            CD_term1 = CDA0 / S
            CD_term2 = cp.multiply(k * S_wetratio, C_f)
            CD_term3 = cp.power(C_L, 2) / (np.pi * A)
            CD_term3 = cp.multiply(CD_term3, 1.0 / e)  # element-wise divide by e
            C_D_expr = CD_term1 + CD_term2 + CD_term3  # vector

            # Drag per condition: 0.5 * rho * V^2 * C_D * S
            drag_vec = 0.5 * S * cp.multiply(rho, cp.multiply(cp.power(V, 2), C_D_expr))
            total_drag = cp.sum(drag_vec)

            # Constraints:

            # Skin friction model: C_f >= 0.074 / Re^0.2
            constraints.append(C_f >= 0.074 / cp.power(Re, 0.2))

            # Reynolds number definition: Re * mu >= rho * V * sqrt(S/A)
            sqrt_S_over_A = cp.power(S / A, 0.5)  # scalar
            constraints.append(cp.multiply(Re, mu) >= cp.multiply(rho, cp.multiply(V, sqrt_S_over_A)))

            # Wing/total weight model (eliminate W_w): W >= W_0 + W_W_coeff2*S + W_W_coeff1*N_ult*A^(3/2)*sqrt(W_0*W)/tau
            A_32 = cp.power(A, 1.5)  # scalar
            sqrt_W0W = cp.power(cp.multiply(W_0, W), 0.5)  # vector
            wing_weight_term = cp.multiply(W_W_coeff1 * N_ult / tau, cp.multiply(A_32, sqrt_W0W))  # vector
            constraints.append(
                W >= W_0 + W_W_coeff2 * S + wing_weight_term
            )

            # Lift equals weight inequality: W <= 0.5 * rho * V^2 * C_L * S
            constraints.append(
                W <= 0.5 * S * cp.multiply(rho, cp.multiply(cp.power(V, 2), C_L))
            )

            # Stall constraint: 2*W/(rho * V_min^2 * S) <= C_Lmax
            denom = cp.multiply(rho, np.power(V_min, 2))  # vector constant
            lhs_stall = 2.0 * cp.multiply(W, cp.power(denom, -1.0)) / S
            constraints.append(lhs_stall <= C_Lmax)

            # Objective
            objective = cp.Minimize(total_drag / n)

            prob = cp.Problem(objective, constraints)

            # Solve with ECOS if available; fallback otherwise
            solved = False
            try:
                prob.solve(gp=True, solver=cp.ECOS, verbose=False)
                solved = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            except Exception:
                solved = False

            if not solved:
                # Fallback to default solver for GP
                prob.solve(gp=True, verbose=False)
                solved = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

            if not solved or A.value is None or S.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Extract values
            A_val = float(A.value)
            S_val = float(S.value)
            V_val = np.asarray(V.value, dtype=float)
            W_val = np.asarray(W.value, dtype=float)
            Re_val = np.asarray(Re.value, dtype=float)
            C_L_val = np.asarray(C_L.value, dtype=float)
            C_f_val = np.asarray(C_f.value, dtype=float)

            # Compute dependent quantities for reporting
            # C_D computed from expression at solution
            C_D_val = (CDA0 / S_val) + (k * S_wetratio) * C_f_val + (C_L_val ** 2) / (np.pi * A_val * e)
            # Wing weight (W_w) from its constraint lower bound (at optimum, active)
            W_w_val = (W_W_coeff2 * S_val) + (W_W_coeff1 * N_ult) * (A_val ** 1.5) * np.sqrt(W_0 * W_val) / tau
            # Drag per condition
            drag_val = 0.5 * rho * (V_val ** 2) * C_D_val * S_val

            condition_results: List[Dict[str, Any]] = []
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
                        "drag": float(drag_val[i]),
                    }
                )

            avg_drag = float(np.mean(drag_val))

            return {
                "A": float(A_val),
                "S": float(S_val),
                "avg_drag": float(avg_drag),
                "condition_results": condition_results,
            }

        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}