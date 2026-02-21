import numpy as np
from ot.lp.emd_wrap import emd_c

class Solver:
    def solve(self, problem: dict, **kwargs):
        a = np.array(problem["source_weights"], dtype=np.float64, copy=False)
        b = np.array(problem["target_weights"], dtype=np.float64, copy=False)
        M = np.array(problem["cost_matrix"], dtype=np.float64, order="C", copy=False)
        
        G, cost, u, v, result_code = emd_c(a, b, M, 100000, 1)
        
        return {"transport_plan": G}