import numpy as np
import ot

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the EMD problem using ot.lp.emd."""
        a = np.asarray(problem["source_weights"], dtype=np.float64)
        b = np.asarray(problem["target_weights"], dtype=np.float64)
        M = np.asarray(problem["cost_matrix"], dtype=np.float64)
        
        # Ensure M is C-contiguous as required by ot.lp.emd
        if not M.flags['C_CONTIGUOUS']:
            M = np.ascontiguousarray(M)
        
        # Compute the optimal transport plan
        G = ot.lp.emd(a, b, M, check_marginals=False)
        
        return {"transport_plan": G}