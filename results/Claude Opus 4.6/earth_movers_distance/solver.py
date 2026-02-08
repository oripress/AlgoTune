import numpy as np
import ot

class Solver:
    def solve(self, problem, **kwargs):
        a = np.asarray(problem["source_weights"], dtype=np.float64)
        b = np.asarray(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        
        G = ot.lp.emd(a, b, M, check_marginals=False)
        
        return {"transport_plan": G}