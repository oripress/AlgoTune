import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

class Solver:
    def solve(self, problem, **kwargs):
        # Use copy=False to avoid copy if already array
        a = np.array(problem["source_weights"], dtype=np.float64, copy=False)
        b = np.array(problem["target_weights"], dtype=np.float64, copy=False)
        M = np.array(problem["cost_matrix"], dtype=np.float64, copy=False)
        
        n = len(a)
        m = len(b)
        
        # Check if it is an assignment problem (uniform weights)
        if n == m:
            # Check if weights are uniform.
            if np.ptp(a) < 1e-9 and np.ptp(b) < 1e-9:
                row_ind, col_ind = linear_sum_assignment(M)
                G = np.zeros((n, n), dtype=np.float64)
                G[row_ind, col_ind] = a[0]
                return {"transport_plan": G}

        # Pruning optimization for sparse weights
        mask_a = a > 1e-15
        mask_b = b > 1e-15
        
        if not mask_a.all() or not mask_b.all():
            a_r = a[mask_a]
            b_r = b[mask_b]
            
            if len(a_r) == 0 or len(b_r) == 0:
                return {"transport_plan": np.zeros((n, m))}
            
            M_r = np.ascontiguousarray(M[np.ix_(mask_a, mask_b)])
            G_r = ot.lp.emd(a_r, b_r, M_r, check_marginals=False)
            
            G = np.zeros((n, m), dtype=np.float64)
            G[np.ix_(mask_a, mask_b)] = G_r
            return {"transport_plan": G}

        M = np.ascontiguousarray(M)
        G = ot.lp.emd(a, b, M, check_marginals=False)
        
        return {"transport_plan": G}