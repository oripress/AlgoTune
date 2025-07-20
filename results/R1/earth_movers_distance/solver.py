import ot
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Extract inputs directly without any intermediate steps
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]
        
        # Compute optimal transport plan with minimal overhead
        # Using POT's low-level interface with check_marginals=False
        return {"transport_plan": ot.lp.emd(a, b, M, check_marginals=False)}