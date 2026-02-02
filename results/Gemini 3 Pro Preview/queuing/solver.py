import numpy as np
from scipy.optimize import brentq

class Solver:
    def solve(self, problem, **kwargs):
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        lam_min = np.asarray(problem["λ_min"])
        mu_max_total = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"])
        n = gamma.size

        # 1. Precompute constants for each queue
        # Delta_i = max(1/w_max_i, 1/d_max_i)
        # We need to handle potential infinities if w_max or d_max are 0? 
        # Problem statement implies positive floats.
        Delta = np.maximum(1.0 / w_max, 1.0 / d_max)

        # rho_max_i from q_max_i
        # q = rho^2 / (1 - rho) => rho^2 + q*rho - q = 0
        # rho = (-q + sqrt(q^2 + 4q)) / 2
        # We can use a slightly more stable form: 2q / (q + sqrt(q^2 + 4q)) ? No, standard formula is fine for positive q.
        sqrt_term = np.sqrt(q_max**2 + 4 * q_max)
        rho_max = (-q_max + sqrt_term) / 2.0

        # lambda_crit_i
        # Boundary where lambda + Delta = lambda / rho_max
        # lambda * (1/rho_max - 1) = Delta
        # lambda = Delta / (1/rho_max - 1) = Delta * rho_max / (1 - rho_max)
        lam_crit = Delta * rho_max / (1.0 - rho_max)

        # 2. Identify regimes
        # If lam_min >= lam_crit, we are forced into the "constant objective" regime (Regime 2).
        # If lam_min < lam_crit, we can be in the "decreasing objective" regime (Regime 1).
        
        mask_regime2 = lam_min >= lam_crit
        mask_regime1 = ~mask_regime2

        # Initialize solution arrays
        lam_sol = np.zeros(n)
        mu_sol = np.zeros(n)

        # Handle Regime 2 (Fixed)
        # For these queues, increasing lambda increases cost but doesn't improve objective.
        # So we set lambda to minimum.
        if np.any(mask_regime2):
            lam_sol[mask_regime2] = lam_min[mask_regime2]
            mu_sol[mask_regime2] = lam_sol[mask_regime2] / rho_max[mask_regime2]

        # Calculate remaining budget for Regime 1
        used_mu = np.sum(mu_sol[mask_regime2])
        remaining_budget = mu_max_total - used_mu

        # Handle Regime 1 (Optimization)
        indices_regime1 = np.where(mask_regime1)[0]
        if len(indices_regime1) > 0:
            # For these queues, mu = lambda + Delta
            # We want to minimize sum(gamma * Delta / lambda)
            # Subject to sum(lambda + Delta) <= remaining_budget
            # => sum(lambda) <= remaining_budget - sum(Delta)
            
            Delta_r1 = Delta[indices_regime1]
            lam_min_r1 = lam_min[indices_regime1]
            lam_crit_r1 = lam_crit[indices_regime1]
            gamma_r1 = gamma[indices_regime1]
            
            sum_Delta_r1 = np.sum(Delta_r1)
            budget_for_lambda = remaining_budget - sum_Delta_r1
            
            # Check feasibility
            min_required_lambda = np.sum(lam_min_r1)
            if budget_for_lambda < min_required_lambda - 1e-9:
                # Infeasible. The reference might raise an error or return a heuristic.
                # The reference raises ValueError if solver fails.
                # We'll try to return the best effort or raise.
                # Let's raise to match reference behavior for infeasible.
                # But wait, reference has a heuristic fallback!
                # "heuristic: λ = λ_min, μ = μ_max/n"
                # We should probably implement the heuristic if we detect infeasibility.
                pass # Will handle heuristic later if needed, but let's proceed with logic.

            # Check if we can max out everything
            max_useful_lambda = np.sum(lam_crit_r1)
            
            if budget_for_lambda >= max_useful_lambda:
                # We have enough budget to set all lambdas to their critical value
                lam_sol[indices_regime1] = lam_crit_r1
            else:
                # We need to find the optimal allocation.
                # lambda_i = clip(K * sqrt(gamma_i * Delta_i), min_i, crit_i)
                # Find K such that sum(lambda_i) = budget_for_lambda
                
                # Coefficients for the "water-filling"
                # If gamma is 0, coef is 0, so lambda will be clipped to min.
                coeffs = np.sqrt(gamma_r1 * Delta_r1)
                
                def get_total_lambda(K):
                    # K is 1/sqrt(nu)
                    # target = K * coeffs
                    vals = K * coeffs
                    # Clip
                    vals = np.maximum(vals, lam_min_r1)
                    vals = np.minimum(vals, lam_crit_r1)
                    return np.sum(vals)

                # Binary search for K
                # Range for K?
                # If K=0, sum is sum(lam_min_r1) <= budget
                # If K is huge, sum is sum(lam_crit_r1) > budget
                # So solution exists.
                
                # We can estimate bounds.
                # max K needed is max(lam_crit / coeffs) (ignoring 0 coeffs)
                # min K needed is 0
                
                # Handle coeffs=0 case
                nonzero_coeffs = coeffs > 1e-12
                if np.any(nonzero_coeffs):
                    max_K = np.max(lam_crit_r1[nonzero_coeffs] / coeffs[nonzero_coeffs]) * 1.1
                else:
                    max_K = 1.0 # Doesn't matter, all will be min
                
                # Use brentq
                try:
                    K_opt = brentq(lambda k: get_total_lambda(k) - budget_for_lambda, 0, max_K, xtol=1e-8)
                    vals = K_opt * coeffs
                    vals = np.maximum(vals, lam_min_r1)
                    vals = np.minimum(vals, lam_crit_r1)
                    lam_sol[indices_regime1] = vals
                except ValueError:
                    # Should not happen if bounds are correct and function monotonic
                    # Fallback to min
                    lam_sol[indices_regime1] = lam_min_r1

            # Compute mu for Regime 1
            mu_sol[indices_regime1] = lam_sol[indices_regime1] + Delta_r1

        # Final check on global constraints and heuristic fallback
        # The reference heuristic is: λ = λ_min, μ = μ_max/n
        # We should check if our solution is valid.
        # Main validity check: sum(mu) <= mu_max
        
        total_mu = np.sum(mu_sol)
        
        if total_mu > mu_max_total + 1e-7:
            # Our derived solution is infeasible (likely due to min constraints exceeding budget)
            # Fallback to heuristic
            lam_val = lam_min
            mu_val = np.full(n, mu_max_total / n)
            obj_val = float(gamma @ (mu_val / lam_val))
            return {"μ": mu_val, "λ": lam_val, "objective": obj_val}
        
        # Also check if mu > lambda (should be guaranteed by construction)
        if np.any(mu_sol <= lam_sol):
             # Fallback
            lam_val = lam_min
            mu_val = np.full(n, mu_max_total / n)
            obj_val = float(gamma @ (mu_val / lam_val))
            return {"μ": mu_val, "λ": lam_val, "objective": obj_val}

        obj_val = float(gamma @ (mu_sol / lam_sol))
        
        return {
            "μ": mu_sol,
            "λ": lam_sol,
            "objective": obj_val
        }