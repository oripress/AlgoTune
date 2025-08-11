from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the aircraft wing design optimization problem using CVXPY with optimized mathematical formulation.

        For multi-point design problems (n > 1), this finds a single wing design
        that performs well across all flight conditions.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal design variables and minimum drag
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Extract problem parameters
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]
        
        # Pre-compute constants
        pi = np.pi
        inv_pi = 1.0 / pi
        
        # Define shared design variables
        A = cp.Variable(pos=True, name="A")  # aspect ratio
        S = cp.Variable(pos=True, name="S")  # wing area (m²)
        
        # Pre-compute commonly used expressions
        sqrt_S_over_A = cp.sqrt(S / A)
        A_pow_3_2 = A ** (3 / 2)
        inv_A = 1.0 / A
        inv_S = 1.0 / S

        # Define condition-specific variables - eliminate some through substitution
        V = [cp.Variable(pos=True, name=f"V_{i}") for i in range(num_conditions)]
        W = [cp.Variable(pos=True, name=f"W_{i}") for i in range(num_conditions)]
        Re = [cp.Variable(pos=True, name=f"Re_{i}") for i in range(num_conditions)]
        C_L = [cp.Variable(pos=True, name=f"C_L_{i}") for i in range(num_conditions)]
        C_f = [cp.Variable(pos=True, name=f"C_f_{i}") for i in range(num_conditions)]
        
        # Eliminate C_D and W_w variables through substitution in constraints
        # C_D will be computed directly from the drag model
        # W_w will be computed directly from the wing weight model

        # Define constraints
        constraints = []

        # Objective: minimize average drag across all conditions
        total_drag = 0

        # Process each flight condition with optimized mathematical formulation
        for i in range(num_conditions):
            condition = conditions[i]

            # Extract condition-specific parameters with pre-computation
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
            
            # Pre-compute derived constants for efficiency
            half_rho = 0.5 * rho
            rho_V_min_sq = rho * V_min**2
            inv_rho_V_min_sq = 1.0 / rho_V_min_sq
            k_S_wetratio = k * S_wetratio
            inv_e = 1.0 / e
            inv_tau = 1.0 / tau
            W_W_coeff1_N_ult = W_W_coeff1 * N_ult
            W_W_coeff1_N_ult_inv_tau = W_W_coeff1_N_ult * inv_tau
            const_0_074 = 0.074
            sqrt_W_0 = np.sqrt(W_0)

            # Calculate drag for this condition using optimized formulation
            V_sq = V[i] * V[i]
            C_L_sq = C_L[i] * C_L[i]
            Re_pow_0_2 = Re[i] ** 0.2
            
            # Substitute C_D directly from drag model: C_D = CDA0/S + k*C_f*S_wetratio + C_L^2/(pi*A*e)
            # Reformulate for better numerical stability
            C_D_substituted = CDA0 * inv_S + k_S_wetratio * C_f[i] + C_L_sq * inv_pi * inv_e * inv_A
            
            # Optimized drag calculation - factor out common terms
            drag_i = half_rho * S * V_sq * C_D_substituted
            total_drag += drag_i

            # Skin friction model: C_f >= 0.074/Re^0.2
            # Reformulate as C_f * Re^0.2 >= 0.074 for better numerical properties
            constraints.append(C_f[i] * Re_pow_0_2 >= const_0_074)
            
            # Reynolds number definition: Re * mu >= rho * V * sqrt(S/A)
            # Reformulate for better numerical stability
            constraints.append(Re[i] >= (rho / mu) * V[i] * sqrt_S_over_A)
            
            # Wing weight model - substitute W_w directly
            # Reformulate for better numerical stability
            sqrt_W = cp.sqrt(W[i])
            W_w_substituted = W_W_coeff2 * S + W_W_coeff1_N_ult_inv_tau * A_pow_3_2 * sqrt_W_0 * sqrt_W
            
            # Total weight: W >= W_0 + W_w
            constraints.append(W[i] >= W_0 + W_w_substituted)
            
            # Lift equals weight: W <= 0.5 * rho * V^2 * C_L * S
            # Reformulate for better numerical stability
            constraints.append(W[i] <= half_rho * S * V_sq * C_L[i])
            
            # Stall constraint: 2*W/(rho*V_min^2*S) <= C_Lmax
            # Reformulate for better numerical stability
            constraints.append(2 * W[i] * inv_rho_V_min_sq * inv_S <= C_Lmax)

        # Define the objective: minimize average drag across all conditions
        objective = cp.Minimize(total_drag / num_conditions)

        # Solve the problem with optimized solver settings
        prob = cp.Problem(objective, constraints)
        try:
            # Use optimized solver settings for geometric programming
            # Use CLARABEL solver which is often faster for geometric programs
            prob.solve(gp=True, solver=cp.CLARABEL, verbose=False, max_iter=100, tol_gap_abs=1e-6, tol_gap_rel=1e-6)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                # Fallback to default solver
                prob.solve(gp=True, verbose=False)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Collect results for each condition - need to compute C_D and W_w from the model
            condition_results = []
            S_val = S.value
            A_val = A.value
            
            # Pre-compute common values for result processing
            inv_S_val = 1.0 / S_val
            inv_A_val = 1.0 / A_val
            
            for i in range(num_conditions):
                condition = conditions[i]
                V_val = V[i].value
                W_val = W[i].value
                C_L_val = C_L[i].value
                C_f_val = C_f[i].value
                Re_val = Re[i].value
                
                # Extract parameters for this condition
                CDA0 = float(condition["CDA0"])
                k = float(condition["k"])
                S_wetratio = float(condition["S_wetratio"])
                e = float(condition["e"])
                rho = float(condition["rho"])
                W_W_coeff2 = float(condition["W_W_coeff2"])
                W_W_coeff1 = float(condition["W_W_coeff1"])
                N_ult = float(condition["N_ult"])
                tau = float(condition["tau"])
                W_0 = float(condition["W_0"])
                
                # Pre-compute derived constants
                half_rho = 0.5 * rho
                k_S_wetratio = k * S_wetratio
                inv_e = 1.0 / e
                inv_tau = 1.0 / tau
                W_W_coeff1_N_ult = W_W_coeff1 * N_ult
                W_W_coeff1_N_ult_inv_tau = W_W_coeff1_N_ult * inv_tau
                sqrt_W_0 = np.sqrt(W_0)
                
                # Compute C_D from the model
                C_D_val = CDA0 * inv_S_val + k_S_wetratio * C_f_val + (C_L_val * C_L_val) * inv_pi * inv_e * inv_A_val
                
                # Compute W_w from the model
                W_w_val = W_W_coeff2 * S_val + W_W_coeff1_N_ult_inv_tau * (A_val ** (3 / 2)) * sqrt_W_0 * np.sqrt(W_val)
                
                # Compute drag
                drag_val = half_rho * V_val * V_val * C_D_val * S_val
                
                condition_results.append({
                    "condition_id": condition["condition_id"],
                    "V": float(V_val),
                    "W": float(W_val),
                    "W_w": float(W_w_val),
                    "C_L": float(C_L_val),
                    "C_D": float(C_D_val),
                    "C_f": float(C_f_val),
                    "Re": float(Re_val),
                    "drag": float(drag_val),
                })

            # Return optimal values
            return {
                "A": float(A_val),
                "S": float(S_val),
                "avg_drag": float(prob.value),
                "condition_results": condition_results,
            }

        except cp.SolverError:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}