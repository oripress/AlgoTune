import osqp
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem, **kwargs):
        # Extract problem data
        P = problem.get("P", problem.get("Q"))
        q = problem["q"]
        G = problem["G"]
        h = problem["h"]
        A = problem["A"]
        b = problem["b"]
        
        # Convert to sparse matrices
        P_sparse = sparse.csc_matrix(P)
        G_sparse = sparse.csc_matrix(G)
        A_sparse = sparse.csc_matrix(A)
        
        # Ensure P is symmetric (required for quad_form)
        P_sparse = (P_sparse + P_sparse.T) / 2
        
        # Setup OSQP problem
        prob = osqp.OSQP()
        prob.setup(
            P=P_sparse, 
            q=np.array(q), 
            A=sparse.vstack([G_sparse, A_sparse], format='csc'), 
            l=np.hstack([-np.inf * np.ones(len(h)), np.array(b)]),
            u=np.hstack([np.array(h), np.array(b)]),
            verbose=False,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=200000
        )
        
        # Solve problem
        results = prob.solve()
        
        if results.info.status_val not in (1, 2):  # 1 = solved, 2 = solved inaccurate
            raise ValueError(f"Solver failed with status: {results.info.status}")
            
        return {"solution": results.x.tolist()}