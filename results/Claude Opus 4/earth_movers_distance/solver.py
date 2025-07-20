import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the EMD problem."""
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]
        
        # For now, let's use the reference implementation to ensure correctness
        import ot
        M_cont = np.ascontiguousarray(M, dtype=np.float64)
        G = ot.lp.emd(a, b, M_cont, check_marginals=False)
        
        return {"transport_plan": G}