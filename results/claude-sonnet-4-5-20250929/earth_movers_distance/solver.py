import numpy as np
import ot

class Solver:
    def solve(self, problem):
        a = np.asarray(problem["source_weights"], dtype=np.float64)
        b = np.asarray(problem["target_weights"], dtype=np.float64)
        M = np.asarray(problem["cost_matrix"], dtype=np.float64)
        
        # Ensure proper memory layout for optimal performance
        M = np.ascontiguousarray(M)
        
        # Use ot.emd (network simplex) which is faster than ot.lp.emd
        G = ot.emd(a, b, M)
        
        return {"transport_plan": G}