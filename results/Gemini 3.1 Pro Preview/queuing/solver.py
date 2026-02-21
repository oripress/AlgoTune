from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        w_max = np.asarray(problem["w_max"], dtype=float)
        d_max = np.asarray(problem["d_max"], dtype=float)
        q_max = np.asarray(problem["q_max"], dtype=float)
        lam_min = np.asarray(problem["λ_min"], dtype=float)
        mu_max = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"], dtype=float)
        n = gamma.size

        c = np.minimum(w_max, d_max)
        rho_max = (-q_max + np.sqrt(q_max**2 + 4.0 * q_max)) / 2.0
        lam_cross = (1.0 / c) / (1.0 / rho_max - 1.0)

        active = lam_min < lam_cross
        
        lam = np.copy(lam_min)
        mu = np.where(lam < lam_cross, lam + 1.0 / c, lam / rho_max)
        
        if np.sum(mu) > mu_max + 1e-6:
            # Fallback heuristic from reference
            lam_val = lam_min
            mu_val = np.full(n, mu_max / n)
            obj_val = float(np.sum(gamma * (mu_val / lam_val)))
            return {"μ": mu_val, "λ": lam_val, "objective": obj_val}
            
        budget = mu_max - np.sum(1.0 / c[active]) - np.sum(mu[~active])
        
        if budget <= np.sum(lam_min[active]):
            pass
        elif budget >= np.sum(lam_cross[active]):
            lam[active] = lam_cross[active]
        else:
            B = np.sqrt(gamma[active] / c[active])
            L = lam_min[active]
            U = lam_cross[active]
            
            breakpoints = np.concatenate([L / B, U / B])
            breakpoints = np.unique(breakpoints)
            breakpoints.sort()
            
            low = 0
            high = len(breakpoints) - 1
            
            while low < high - 1:
                mid = (low + high) // 2
                x = breakpoints[mid]
                if np.sum(np.clip(x * B, L, U)) <= budget:
                    low = mid
                else:
                    high = mid
                    
            x0 = breakpoints[low]
            x1 = breakpoints[high]
            f0 = np.sum(np.clip(x0 * B, L, U))
            f1 = np.sum(np.clip(x1 * B, L, U))
            
            if f1 == f0:
                x = x0
            else:
                x = x0 + (budget - f0) / (f1 - f0) * (x1 - x0)
                
            lam[active] = np.clip(x * B, L, U)
            
        mu = np.where(lam < lam_cross, lam + 1.0 / c, lam / rho_max)
        obj = float(np.sum(gamma * mu / lam))
        
        return {"μ": mu, "λ": lam, "objective": obj}